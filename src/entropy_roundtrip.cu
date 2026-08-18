// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "entropy_roundtrip.h"

#include <cuda_runtime.h>

#include "ac_context.h"
#include "entropy.h"

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

}  // namespace

bool ac_encode_device(const std::vector<std::int32_t>& q, const std::vector<std::int8_t>& acs,
                      std::size_t width, std::size_t height, const std::vector<std::uint8_t>& depth,
                      const std::vector<std::uint16_t>& bits, AcDeviceResult& out) {
    const std::size_t num_groups{ac_num_groups(width, height)};
    const std::size_t blocks{(width / 8) * (height / 8)};
    const std::size_t capacity{q.size() * 8 + 4096};

    // The covered-block layout is consumed directly (int32 -> int16, three planes
    // of blocks*64 slots each).
    std::vector<std::int16_t> ac(q.size());
    for (std::size_t i{0}; i < q.size(); ++i) {
        ac[i] = static_cast<std::int16_t>(q[i]);
    }

    std::int16_t* d_ac{nullptr};
    std::int8_t* d_acs{nullptr};
    std::uint8_t* d_depth{nullptr};
    std::uint16_t* d_bits{nullptr};
    std::uint32_t* d_hist{nullptr};
    std::uint32_t* d_sizes{nullptr};
    std::uint32_t* d_offsets{nullptr};
    std::uint8_t* d_out{nullptr};

    bool ok{upload(ac, &d_ac) && upload(acs, &d_acs) && upload(depth, &d_depth) &&
            upload(bits, &d_bits) &&
            cudaMalloc(&d_hist, AC_HIST_SPAN * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_sizes, num_groups * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_offsets, num_groups * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_out, capacity) == cudaSuccess};
    (void)blocks;

    std::size_t total_bytes{0};
    ok = ok && ac_build_histogram(d_ac, d_acs, width, height, d_hist);
    ok = ok && ac_encode_groups(d_ac, d_acs, width, height, d_depth, d_bits, d_out, capacity,
                                d_sizes, d_offsets, &total_bytes);

    if (ok) {
        out.histogram.assign(AC_HIST_SPAN, 0);
        out.group_sizes.assign(num_groups, 0);
        out.group_offsets.assign(num_groups, 0);
        out.stream.assign(total_bytes, 0);
        ok = cudaMemcpy(out.histogram.data(), d_hist, AC_HIST_SPAN * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.group_sizes.data(), d_sizes, num_groups * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.group_offsets.data(), d_offsets, num_groups * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.stream.data(), d_out, total_bytes, cudaMemcpyDeviceToHost) ==
                 cudaSuccess;
    }

    cudaFree(d_ac);
    cudaFree(d_acs);
    cudaFree(d_depth);
    cudaFree(d_bits);
    cudaFree(d_hist);
    cudaFree(d_sizes);
    cudaFree(d_offsets);
    cudaFree(d_out);
    return ok;
}

}  // namespace cujpegxl
