// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_AC_CLUSTERING_H_
#define CUJPEGXL_SRC_AC_CLUSTERING_H_

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "ac_context.h"

// Data-driven clustering of the AC contexts into the histograms the AC groups
// are coded with: starting from the fixed map ac_initial_cluster,
// AC_CLUSTER_LLOYD_ITERATIONS parallel Lloyd iterations move every context to
// the cluster that codes it in the fewest bits, by cross-entropy with smoothed
// probabilities. Before iteration AC_CLUSTER_REFILL_ITERATION, the empty
// clusters are refilled, since the Lloyd iterations cannot fill them: each
// non-empty cluster nominates its context with the largest ac_refill_excess, and
// the nominees with the largest excess move alone to the empty clusters. Costs
// are in integer fixed point, so the device and the host reference cluster
// identically.

namespace cujpegxl {

// An AC symbol is below 64: a coefficient token codes the PackSigned value of an
// int16 (at most 65535, token 63), a non-zero-count token at most 1008.
inline constexpr std::size_t AC_CONTEXT_ALPHABET = 64;
inline constexpr std::size_t AC_MAX_CLUSTERS = 32;
inline constexpr int AC_CLUSTER_LLOYD_ITERATIONS = 4;
inline constexpr int AC_CLUSTER_REFILL_ITERATION = 2;
static_assert(AC_CLUSTER_REFILL_ITERATION > 0 &&
              AC_CLUSTER_REFILL_ITERATION < AC_CLUSTER_LLOYD_ITERATIONS);

// Costs are bits in AC_COST_FRAC_BITS fixed point.
inline constexpr int AC_COST_FRAC_BITS = 20;

// The luma/chroma split of the block contexts, the non-zero-count contexts in
// four bands of the predicted count and the coefficient contexts in twelve of
// the zero-density context: 2 * 4 + 2 * 12 clusters.
CUJPEGXL_ACC_HD inline std::uint32_t ac_initial_cluster(std::uint32_t context) {
    const std::uint32_t count_contexts{AC_NUM_BLOCK_CTX * AC_NON_ZERO_BUCKETS};
    if (context < count_contexts) {
        const std::uint32_t nonzero_bucket{context / AC_NUM_BLOCK_CTX};
        const std::uint32_t channel_group{context % AC_NUM_BLOCK_CTX < 7 ? 0u : 1u};
        const std::uint32_t band{
            nonzero_bucket < 4 ? 0u : (nonzero_bucket < 12 ? 1u : (nonzero_bucket < 24 ? 2u : 3u))};
        return channel_group * 4 + band;
    }
    const std::uint32_t rel{context - count_contexts};
    const std::uint32_t channel_group{rel / AC_ZERO_DENSITY_COUNT < 7 ? 0u : 1u};
    const std::uint32_t zdc{rel % AC_ZERO_DENSITY_COUNT};
    const std::uint32_t band{zdc < 32    ? 0u
                             : zdc < 64  ? 1u
                             : zdc < 96  ? 2u
                             : zdc < 128 ? 3u
                             : zdc < 160 ? 4u
                             : zdc < 200 ? 5u
                             : zdc < 240 ? 6u
                             : zdc < 280 ? 7u
                             : zdc < 328 ? 8u
                             : zdc < 376 ? 9u
                             : zdc < 424 ? 10u
                                         : 11u};
    return 8 + channel_group * 12 + band;
}

// log2(1 + i / AC_LOG2_TABLE_SIZE) in fixed point, interpolated linearly.
inline constexpr int AC_LOG2_TABLE_BITS = 10;
inline constexpr std::size_t AC_LOG2_TABLE_SIZE = std::size_t{1} << AC_LOG2_TABLE_BITS;
using AcLog2Table = std::array<std::int32_t, AC_LOG2_TABLE_SIZE + 1>;

inline AcLog2Table ac_log2_table() {
    AcLog2Table table{};
    for (std::size_t i{0}; i <= AC_LOG2_TABLE_SIZE; ++i) {
        table[i] = static_cast<std::int32_t>(
            std::lround(std::log2(1.0 + static_cast<double>(i) / AC_LOG2_TABLE_SIZE) *
                        static_cast<double>(1 << AC_COST_FRAC_BITS)));
    }
    return table;
}

CUJPEGXL_ACC_HD inline int ac_floor_log2(std::uint32_t x) {
#if defined(__CUDA_ARCH__)
    return 31 - __clz(x);
#else
    return 31 - __builtin_clz(x);
#endif
}

CUJPEGXL_ACC_HD inline int ac_lowest_bit(std::uint64_t mask) {
#if defined(__CUDA_ARCH__)
    return __ffsll(static_cast<long long>(mask)) - 1;
#else
    return __builtin_ctzll(mask);
#endif
}

// log2(x) for x >= 1.
CUJPEGXL_ACC_HD inline std::int32_t ac_fixed_log2(std::uint32_t x, const std::int32_t* table) {
    constexpr int REM_BITS{12};
    const int e{ac_floor_log2(x)};
    const std::uint32_t frac{(x << (31 - e)) & 0x7FFFFFFFu};
    const std::uint32_t idx{frac >> (31 - AC_LOG2_TABLE_BITS)};
    const std::int32_t rem{static_cast<std::int32_t>(
        (frac >> (31 - AC_LOG2_TABLE_BITS - REM_BITS)) & ((1u << REM_BITS) - 1))};
    const std::int32_t lo{table[idx]};
    return (e << AC_COST_FRAC_BITS) + lo + (((table[idx + 1] - lo) * rem) >> REM_BITS);
}

// x times a log2 >= 0: a 32 x 32 -> 64-bit product.
CUJPEGXL_ACC_HD inline std::int64_t ac_times_log2(std::uint32_t x, std::int32_t log2) {
    return static_cast<std::int64_t>(static_cast<std::uint64_t>(x) *
                                     static_cast<std::uint32_t>(log2));
}

// A histogram of AC_CONTEXT_ALPHABET bins with its total and the mask of its
// non-zero bins.
struct AcHistogramRef {
    const std::uint32_t* bins;
    std::uint32_t total;
    std::uint64_t mask;
};

// The log2 terms of a cluster's smoothed probabilities (n + 1/2) / (total +
// alphabet / 2), doubled to stay integers, so that a symbol the cluster lacks
// costs finitely. `alphabet` is one past the highest symbol of any context of
// the frame.
CUJPEGXL_ACC_HD inline std::int32_t ac_smoothed_bin_log2(std::uint32_t count,
                                                         const std::int32_t* table) {
    return ac_fixed_log2(2 * count + 1, table);
}

CUJPEGXL_ACC_HD inline std::int32_t ac_smoothed_total_log2(std::uint32_t total,
                                                           std::uint32_t alphabet,
                                                           const std::int32_t* table) {
    return ac_fixed_log2(2 * total + alphabet, table);
}

// The size of `a` entropy-coded with the distribution of a cluster, given the
// ac_smoothed_total_log2 and ac_smoothed_bin_log2 of the cluster.
CUJPEGXL_ACC_HD inline std::int64_t ac_coding_cost(const AcHistogramRef& a,
                                                   std::int32_t cluster_total_log2,
                                                   const std::int32_t* cluster_bin_log2) {
    std::int64_t cost{ac_times_log2(a.total, cluster_total_log2)};
    for (std::uint64_t m{a.mask}; m != 0; m &= m - 1) {
        const int s{ac_lowest_bit(m)};
        cost -= ac_times_log2(a.bins[s], cluster_bin_log2[s]);
    }
    return cost;
}

// The bits `a` loses coded with a cluster's distribution instead of its own: its
// ac_coding_cost minus its entropy, at least 0, in AC_EXCESS_FRAC_BITS fixed
// point. Below 2^45 for the 2^27 tokens of the largest frames.
inline constexpr int AC_EXCESS_FRAC_BITS = 13;

CUJPEGXL_ACC_HD inline std::uint64_t ac_refill_excess(const AcHistogramRef& a,
                                                      std::int32_t cluster_total_log2,
                                                      const std::int32_t* cluster_bin_log2,
                                                      const std::int32_t* table) {
    std::int64_t entropy{ac_times_log2(a.total, ac_fixed_log2(a.total, table))};
    for (std::uint64_t m{a.mask}; m != 0; m &= m - 1) {
        const int s{ac_lowest_bit(m)};
        entropy -= ac_times_log2(a.bins[s], ac_fixed_log2(a.bins[s], table));
    }
    const std::int64_t excess{ac_coding_cost(a, cluster_total_log2, cluster_bin_log2) - entropy};
    return excess > 0
               ? static_cast<std::uint64_t>(excess) >> (AC_COST_FRAC_BITS - AC_EXCESS_FRAC_BITS)
               : 0;
}

// A refill nominee's key, ordered by the largest excess then the lowest context;
// never 0.
inline constexpr int AC_REFILL_CONTEXT_BITS = 13;
static_assert(AC_NUM_CONTEXTS < (1 << AC_REFILL_CONTEXT_BITS));

CUJPEGXL_ACC_HD inline std::uint64_t ac_refill_key(std::uint64_t excess, std::uint32_t context) {
    return excess << AC_REFILL_CONTEXT_BITS |
           (((std::uint64_t{1} << AC_REFILL_CONTEXT_BITS) - 1) - context);
}

CUJPEGXL_ACC_HD inline std::uint32_t ac_refill_key_context(std::uint64_t key) {
    return static_cast<std::uint32_t>(((std::uint64_t{1} << AC_REFILL_CONTEXT_BITS) - 1) -
                                      (key & ((std::uint64_t{1} << AC_REFILL_CONTEXT_BITS) - 1)));
}

// The clustering of a frame's AC contexts. Clusters are numbered by their first
// context, and a context without tokens takes the previous context's cluster
// (cluster 0 before the first context with tokens), which keeps the signalled
// context map small.
struct AcClustering {
    // Cluster c's histogram at c * AC_CONTEXT_ALPHABET.
    std::uint32_t histograms[AC_MAX_CLUSTERS * AC_CONTEXT_ALPHABET];
    std::uint32_t num_clusters;
    std::uint8_t context_map[AC_NUM_CONTEXTS];
};

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_AC_CLUSTERING_H_
