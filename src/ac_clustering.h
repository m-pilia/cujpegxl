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
// are coded with: libjxl kFast's farthest-first seeding, a parallel assignment
// of every context to its nearest seed, then AC_CLUSTER_LLOYD_ITERATIONS Lloyd
// iterations by cross-entropy. Costs are in integer fixed point, so the device
// and the host reference cluster identically.

namespace cujpegxl {

// An AC symbol is below 64: a coefficient token codes the PackSigned value of an
// int16 (at most 65535, token 63), a non-zero-count token at most 1008.
inline constexpr std::size_t AC_CONTEXT_ALPHABET = 64;
inline constexpr std::size_t AC_MAX_CLUSTERS = 32;
inline constexpr int AC_CLUSTER_LLOYD_ITERATIONS = 2;

// Costs are bits in AC_COST_FRAC_BITS fixed point.
inline constexpr int AC_COST_FRAC_BITS = 20;
// libjxl kMinDistanceForDistinct: seeding stops when the farthest context is
// closer than this to its nearest seed.
inline constexpr std::int64_t AC_MIN_SEED_DISTANCE = std::int64_t{48} << AC_COST_FRAC_BITS;

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
CUJPEGXL_ACC_HD inline std::int64_t ac_fixed_log2(std::uint32_t x, const std::int32_t* table) {
    constexpr int REM_BITS{12};
    const int e{ac_floor_log2(x)};
    const std::uint32_t frac{(x << (31 - e)) & 0x7FFFFFFFu};
    const std::uint32_t idx{frac >> (31 - AC_LOG2_TABLE_BITS)};
    const std::int32_t rem{
        static_cast<std::int32_t>((frac >> (31 - AC_LOG2_TABLE_BITS - REM_BITS)) &
                                  ((1u << REM_BITS) - 1))};
    const std::int32_t lo{table[idx]};
    return (static_cast<std::int64_t>(e) << AC_COST_FRAC_BITS) + lo +
           (((table[idx + 1] - lo) * rem) >> REM_BITS);
}

CUJPEGXL_ACC_HD inline std::int64_t ac_xlog2x(std::uint32_t x, const std::int32_t* table) {
    return x == 0 ? 0 : static_cast<std::int64_t>(x) * ac_fixed_log2(x, table);
}

// A histogram of AC_CONTEXT_ALPHABET bins with its total and the mask of its
// non-zero bins.
struct AcHistogramRef {
    const std::uint32_t* bins;
    std::uint32_t total;
    std::uint64_t mask;
};

// libjxl HistogramDistance: the entropy-coded size of `a` and `b` merged minus
// their sizes apart. Only the bins non-zero in both contribute beyond the totals.
CUJPEGXL_ACC_HD inline std::int64_t ac_merge_cost(const AcHistogramRef& a,
                                                  const AcHistogramRef& b,
                                                  const std::int32_t* table) {
    std::int64_t shared{0};
    for (std::uint64_t m{a.mask & b.mask}; m != 0; m &= m - 1) {
        const int s{ac_lowest_bit(m)};
        shared += ac_xlog2x(a.bins[s] + b.bins[s], table) - ac_xlog2x(a.bins[s], table) -
                  ac_xlog2x(b.bins[s], table);
    }
    return ac_xlog2x(a.total + b.total, table) - ac_xlog2x(a.total, table) -
           ac_xlog2x(b.total, table) - shared;
}

// The size of `a` entropy-coded with the distribution of `cluster`, which must
// hold every symbol of `a`; up to a term of `a` alone.
CUJPEGXL_ACC_HD inline std::int64_t ac_coding_cost(const AcHistogramRef& a,
                                                   const AcHistogramRef& cluster,
                                                   const std::int32_t* table) {
    std::int64_t cost{static_cast<std::int64_t>(a.total) * ac_fixed_log2(cluster.total, table)};
    for (std::uint64_t m{a.mask}; m != 0; m &= m - 1) {
        const int s{ac_lowest_bit(m)};
        cost -= static_cast<std::int64_t>(a.bins[s]) * ac_fixed_log2(cluster.bins[s], table);
    }
    return cost;
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
