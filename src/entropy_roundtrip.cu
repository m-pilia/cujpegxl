// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "entropy_roundtrip.h"

#include <cstddef>

#include <cuda_runtime.h>

#include "ac_clustering.h"
#include "coeff_order_reference.h"
#include "entropy.h"
#include "vardct_layout.h"

namespace cujpegxl {
namespace {

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

template <typename T>
bool download(const T* device, std::size_t count, std::vector<T>& host) {
    host.assign(count, T{});
    return count == 0 || cudaMemcpy(host.data(), device, count * sizeof(T),
                                    cudaMemcpyDeviceToHost) == cudaSuccess;
}

}  // namespace

bool ac_tokens_device(const std::vector<std::int32_t>& q, const std::vector<std::int8_t>& acs,
                      std::size_t width, std::size_t height, std::size_t batches,
                      AcDeviceTokens& out) {
    const std::size_t num_groups{ac_num_groups(width, height)};
    const std::size_t layout_words{3 * num_groups + 2};

    // The covered-block layout is consumed directly (int32 -> int16, three planes
    // of blocks*64 slots each).
    std::vector<std::int16_t> ac(q.size());
    for (std::size_t i{0}; i < q.size(); ++i) {
        ac[i] = static_cast<std::int16_t>(q[i]);
    }

    const NonzeroCounts nz{nonzero_counts(ac, acs, width / 8, height / 8)};
    const std::size_t plane{q.size() / 3};
    const CoeffOrderStats stats{
        coeff_order_stats_reference({q.data(), q.data() + plane, q.data() + 2 * plane},
                                    acs.empty() ? nullptr : acs.data(), width / 8, height / 8)};

    std::int16_t* d_ac{nullptr};
    std::int8_t* d_acs{nullptr};
    CoeffOrderStats* d_stats{nullptr};
    CoeffOrders* d_orders{nullptr};
    std::uint32_t* d_hist{nullptr};
    std::uint8_t* d_cluster_scratch{nullptr};
    AcClustering* d_clustering{nullptr};
    std::int32_t* d_nz_grid{nullptr};
    std::int32_t* d_nz_exact{nullptr};
    std::uint64_t* d_item_sizes{nullptr};
    std::uint8_t* d_layout_scratch{nullptr};
    std::uint64_t* d_chunk_base{nullptr};
    std::uint32_t* d_layout{nullptr};
    std::uint16_t* d_entries{nullptr};
    std::uint64_t* d_raw{nullptr};

    bool ok{upload(ac, &d_ac) && upload(acs, &d_acs) &&
            cudaMalloc(&d_stats, sizeof(CoeffOrderStats)) == cudaSuccess &&
            cudaMemcpy(d_stats, &stats, sizeof(CoeffOrderStats), cudaMemcpyHostToDevice) ==
                cudaSuccess &&
            cudaMalloc(&d_orders, sizeof(CoeffOrders)) == cudaSuccess &&
            cudaMalloc(&d_hist, AC_NUM_CONTEXTS * AC_CONTEXT_ALPHABET * sizeof(std::uint32_t)) ==
                cudaSuccess &&
            cudaMalloc(&d_cluster_scratch, ac_cluster_scratch_bytes()) == cudaSuccess &&
            cudaMalloc(&d_clustering, sizeof(AcClustering)) == cudaSuccess &&
            upload(nz.grid, &d_nz_grid) && upload(nz.exact, &d_nz_exact) &&
            cudaMalloc(&d_item_sizes, nz.exact.size() * sizeof(std::uint64_t)) == cudaSuccess &&
            cudaMalloc(&d_layout_scratch, ac_token_layout_scratch_bytes(width, height)) ==
                cudaSuccess &&
            cudaMalloc(&d_chunk_base, ac_token_chunks(width, height) * sizeof(std::uint64_t)) ==
                cudaSuccess &&
            cudaMalloc(&d_layout, layout_words * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_entries, ac_max_tokens(width, height) * sizeof(std::uint16_t)) ==
                cudaSuccess &&
            cudaMalloc(&d_raw, ac_max_raw_words(width, height) * sizeof(std::uint64_t)) ==
                cudaSuccess &&
            cudaMemset(d_raw, 0xA5, ac_max_raw_words(width, height) * sizeof(std::uint64_t)) ==
                cudaSuccess};

    ok = ok && ac_coeff_orders(d_stats, d_orders, nullptr);
    ok = ok && ac_build_histogram(d_ac, d_acs, d_nz_grid, d_nz_exact, d_orders, width, height,
                                  d_hist, d_item_sizes, nullptr);
    ok = ok && ac_token_layout(d_item_sizes, width, height, d_layout_scratch, d_chunk_base,
                               d_layout, nullptr);
    ok = ok && ac_cluster_contexts(d_hist, d_cluster_scratch, d_clustering, nullptr);
    for (std::size_t b{0}; ok && b < batches; ++b) {
        const std::size_t begin{b * num_groups / batches};
        const std::size_t end{(b + 1) * num_groups / batches};
        ok = begin == end ||
             ac_pack_tokens(d_ac, d_acs, d_nz_grid, d_nz_exact, d_orders, width, height,
                            d_clustering->context_map, d_item_sizes, d_chunk_base, d_layout, begin,
                            end, d_entries, d_raw, nullptr);
    }
    ok = ok &&
         cudaMemcpy(&out.orders, d_orders, sizeof(CoeffOrders), cudaMemcpyDeviceToHost) ==
             cudaSuccess &&
         download(d_hist, AC_NUM_CONTEXTS * AC_CONTEXT_ALPHABET, out.context_histograms) &&
         cudaMemcpy(&out.clustering, d_clustering, sizeof(AcClustering), cudaMemcpyDeviceToHost) ==
             cudaSuccess &&
         download(d_layout, layout_words, out.layout) &&
         download(d_entries, out.layout[num_groups], out.entries) &&
         download(d_raw, out.layout[2 * num_groups + 1], out.raw);

    cudaFree(d_ac);
    cudaFree(d_acs);
    cudaFree(d_stats);
    cudaFree(d_orders);
    cudaFree(d_hist);
    cudaFree(d_cluster_scratch);
    cudaFree(d_clustering);
    cudaFree(d_nz_grid);
    cudaFree(d_nz_exact);
    cudaFree(d_item_sizes);
    cudaFree(d_layout_scratch);
    cudaFree(d_chunk_base);
    cudaFree(d_layout);
    cudaFree(d_entries);
    cudaFree(d_raw);
    return ok;
}

bool ac_coeff_orders_device(const CoeffOrderStats& stats, CoeffOrders& out) {
    CoeffOrderStats* d_stats{nullptr};
    CoeffOrders* d_orders{nullptr};
    const bool ok{cudaMalloc(&d_stats, sizeof(CoeffOrderStats)) == cudaSuccess &&
                  cudaMemcpy(d_stats, &stats, sizeof(CoeffOrderStats), cudaMemcpyHostToDevice) ==
                      cudaSuccess &&
                  cudaMalloc(&d_orders, sizeof(CoeffOrders)) == cudaSuccess &&
                  ac_coeff_orders(d_stats, d_orders, nullptr) &&
                  cudaMemcpy(&out, d_orders, sizeof(CoeffOrders), cudaMemcpyDeviceToHost) ==
                      cudaSuccess};
    cudaFree(d_stats);
    cudaFree(d_orders);
    return ok;
}

bool ac_cluster_contexts_device(const std::vector<std::uint32_t>& context_histograms,
                                AcClustering& out) {
    std::uint32_t* d_hist{nullptr};
    std::uint8_t* d_scratch{nullptr};
    AcClustering* d_clustering{nullptr};
    const bool ok{upload(context_histograms, &d_hist) &&
                  cudaMalloc(&d_scratch, ac_cluster_scratch_bytes()) == cudaSuccess &&
                  cudaMalloc(&d_clustering, sizeof(AcClustering)) == cudaSuccess &&
                  ac_cluster_contexts(d_hist, d_scratch, d_clustering, nullptr) &&
                  cudaMemcpy(&out, d_clustering, sizeof(AcClustering), cudaMemcpyDeviceToHost) ==
                      cudaSuccess};
    cudaFree(d_hist);
    cudaFree(d_scratch);
    cudaFree(d_clustering);
    return ok;
}

}  // namespace cujpegxl
