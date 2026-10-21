// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "entropy_roundtrip.h"

#include <algorithm>
#include <cstddef>
#include <numeric>

#include <cuda_runtime.h>

#include "ac_context.h"
#include "entropy.h"
#include "vardct_layout.h"

namespace cujpegxl {
namespace {

// The device AC histogram and prefix code span all clusters (cluster*256+symbol).
constexpr std::size_t AC_HIST_SPAN = AC_NUM_CLUSTERS * AC_HISTOGRAM_SIZE;

template <typename T>
bool upload(const std::vector<T>& host, T** device) {
    if (cudaMalloc(device, host.size() * sizeof(T)) != cudaSuccess) {
        return false;
    }
    return cudaMemcpy(*device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) ==
           cudaSuccess;
}

struct NonzeroCounts {
    std::vector<std::int32_t> exact{};
    std::vector<std::int32_t> grid{};
};

// Host counterpart of the counts quantize_residual produces. The LLF corner is
// skipped explicitly: the test patterns may hold values there, which the coder
// never tokenizes, while quantize_residual always zeroes it.
NonzeroCounts nonzero_counts(const std::vector<std::int16_t>& ac,
                             const std::vector<std::int8_t>& acs, std::size_t bw, std::size_t bh) {
    const std::size_t blocks{bw * bh};
    const std::size_t plane{blocks * COEFFS_PER_BLOCK};
    NonzeroCounts nz{.exact = std::vector<std::int32_t>(3 * blocks, 0),
                     .grid = std::vector<std::int32_t>(3 * blocks, 0)};
    for (std::size_t by{0}; by < bh; ++by) {
        for (std::size_t bx{0}; bx < bw; ++bx) {
            const std::size_t block{by * bw + bx};
            const int side{acs.empty() ? 8 : acs[block]};
            if (side == ACS_COVERED) {
                continue;
            }
            const int cx{side / 8};
            const std::int32_t covered{cx * cx};
            for (std::size_t c{0}; c < 3; ++c) {
                std::int32_t count{0};
                for (int raw{0}; raw < side * side; ++raw) {
                    if (raw / side < cx && raw % side < cx) {
                        continue;
                    }
                    const std::size_t slot{
                        covered_plane_slot(side, bx, by, bw, static_cast<std::size_t>(raw))};
                    count += ac[c * plane + slot] != 0 ? 1 : 0;
                }
                nz.exact[c * blocks + block] = count;
                for (int q{0}; q < covered; ++q) {
                    const std::size_t at{(by + q / cx) * bw + (bx + q % cx)};
                    nz.grid[c * blocks + at] = (count + covered - 1) / covered;
                }
            }
        }
    }
    return nz;
}

}  // namespace

bool ac_encode_device(const std::vector<std::int32_t>& q, const std::vector<std::int8_t>& acs,
                      std::size_t width, std::size_t height, const std::vector<std::uint8_t>& depth,
                      const std::vector<std::uint16_t>& bits, std::size_t out_capacity,
                      AcDeviceResult& out) {
    const std::size_t num_groups{ac_num_groups(width, height)};

    // The covered-block layout is consumed directly (int32 -> int16, three planes
    // of blocks*64 slots each).
    std::vector<std::int16_t> ac(q.size());
    for (std::size_t i{0}; i < q.size(); ++i) {
        ac[i] = static_cast<std::int16_t>(q[i]);
    }

    const NonzeroCounts nz{nonzero_counts(ac, acs, width / 8, height / 8)};

    std::int16_t* d_ac{nullptr};
    std::int8_t* d_acs{nullptr};
    std::uint8_t* d_depth{nullptr};
    std::uint16_t* d_bits{nullptr};
    std::uint32_t* d_hist{nullptr};
    std::int32_t* d_nz_grid{nullptr};
    std::int32_t* d_nz_exact{nullptr};
    std::uint32_t* d_sizes{nullptr};
    std::uint32_t* d_offsets{nullptr};
    std::uint8_t* d_out{nullptr};

    bool ok{upload(ac, &d_ac) && upload(acs, &d_acs) && upload(depth, &d_depth) &&
            upload(bits, &d_bits) &&
            cudaMalloc(&d_hist, AC_HIST_SPAN * sizeof(std::uint32_t)) == cudaSuccess &&
            upload(nz.grid, &d_nz_grid) && upload(nz.exact, &d_nz_exact) &&
            cudaMalloc(&d_sizes, num_groups * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_offsets, num_groups * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_out, out_capacity) == cudaSuccess &&
            cudaMemset(d_out, AC_OUT_SENTINEL, out_capacity) == cudaSuccess};

    ok = ok && ac_build_histogram(d_ac, d_acs, d_nz_grid, d_nz_exact, width, height, d_hist);
    ok = ok && ac_encode_groups(d_ac, d_acs, d_nz_grid, d_nz_exact, width, height, d_depth, d_bits,
                                d_out, out_capacity, d_sizes, d_offsets);

    if (ok) {
        out.histogram.assign(AC_HIST_SPAN, 0);
        out.group_sizes.assign(num_groups, 0);
        out.group_offsets.assign(num_groups, 0);
        ok = cudaMemcpy(out.histogram.data(), d_hist, AC_HIST_SPAN * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.group_sizes.data(), d_sizes, num_groups * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.group_offsets.data(), d_offsets, num_groups * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    const std::size_t total_bytes{
        std::accumulate(out.group_sizes.begin(), out.group_sizes.end(), std::size_t{0})};
    if (ok) {
        out.stream.assign(std::min(total_bytes, out_capacity), 0);
        ok = cudaMemcpy(out.stream.data(), d_out, out.stream.size(), cudaMemcpyDeviceToHost) ==
             cudaSuccess;
    }

    cudaFree(d_ac);
    cudaFree(d_acs);
    cudaFree(d_depth);
    cudaFree(d_bits);
    cudaFree(d_hist);
    cudaFree(d_nz_grid);
    cudaFree(d_nz_exact);
    cudaFree(d_sizes);
    cudaFree(d_offsets);
    cudaFree(d_out);
    return ok;
}

}  // namespace cujpegxl
