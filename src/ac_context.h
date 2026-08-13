// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_AC_CONTEXT_H_
#define CUJPEGXL_SRC_AC_CONTEXT_H_

#include <cstddef>
#include <cstdint>

// Mirror of libjxl's VarDCT AC context model (lib/jxl/ac_context.h and the
// dec_group.cc DecodeACVarBlock loop) for the default BlockCtxMap. The encoder
// must assign every AC token the exact context id the decoder recomputes, so
// this header is the single source of truth shared by the host reference, the
// production assembler, and the device kernels. Default map: no DC/QF
// thresholds, so the block context depends only on channel and transform order.

namespace cujpegxl {

#if defined(__CUDACC__)
#define CUJPEGXL_ACC_HD __host__ __device__
#else
#define CUJPEGXL_ACC_HD
#endif

inline constexpr int AC_NUM_ORDERS = 13;         // kNumOrders
inline constexpr int AC_NUM_BLOCK_CTX = 15;      // default BlockCtxMap num_ctxs
inline constexpr int AC_NON_ZERO_BUCKETS = 37;   // kNonZeroBuckets
inline constexpr int AC_ZERO_DENSITY_COUNT = 458;  // kZeroDensityContextCount
inline constexpr int AC_NUM_CONTEXTS =
    AC_NUM_BLOCK_CTX * (AC_NON_ZERO_BUCKETS + AC_ZERO_DENSITY_COUNT);  // 7425

// libjxl kDefaultCtxMap: 3 channel groups x kNumOrders, clustering the large
// transforms together. Row order is the permuted channel index (Y, X, B).
inline constexpr std::uint8_t AC_DEFAULT_CTX_MAP[3 * AC_NUM_ORDERS] = {
    0, 1, 2, 2, 3, 3, 4, 5, 6, 6, 6, 6, 6,       //
    7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,  //
    7, 8, 9, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14,  //
};

inline constexpr std::uint16_t AC_COEFF_FREQ_CONTEXT[64] = {
    0xBAD, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
    15,    15, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 21, 22, 22,
    23,    23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26,
    27,    27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30,
};

inline constexpr std::uint16_t AC_COEFF_NUM_NONZERO_CONTEXT[64] = {
    0xBAD, 0,   31,  62,  62,  93,  93,  93,  93,  123, 123, 123, 123,
    152,   152, 152, 152, 152, 152, 152, 152, 180, 180, 180, 180, 180,
    180,   180, 180, 180, 180, 180, 180, 206, 206, 206, 206, 206, 206,
    206,   206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
    206,   206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206,
};

// libjxl kStrategyOrder for the square strategies we emit: DCT (side 8) -> 0,
// DCT16X16 -> 2, DCT32X32 -> 3.
CUJPEGXL_ACC_HD inline int ac_strategy_order(int side) {
    return side == 16 ? 2 : (side == 32 ? 3 : 0);
}

// Block context for physical channel `c` (0=X, 1=Y, 2=B) and transform order
// `ord`. Default map has no DC/QF thresholds, so this is the whole Context().
CUJPEGXL_ACC_HD inline int ac_block_context(int c, int ord) {
    const int idx{(c < 2 ? (c ^ 1) : 2) * AC_NUM_ORDERS + ord};
    return AC_DEFAULT_CTX_MAP[idx];
}

// Context for the block's non-zero-count symbol, from the neighbor-predicted
// non-zero count and the block context.
CUJPEGXL_ACC_HD inline std::uint32_t ac_nonzero_context(std::uint32_t predicted_nonzeros,
                                                        int block_ctx) {
    std::uint32_t nz{predicted_nonzeros};
    if (nz >= 64) {
        nz = 64;
    }
    const std::uint32_t ctx{nz < 8 ? nz : 4 + nz / 2};
    return ctx * AC_NUM_BLOCK_CTX + static_cast<std::uint32_t>(block_ctx);
}

// Base context id for a block's AC coefficient symbols.
CUJPEGXL_ACC_HD inline std::uint32_t ac_zero_density_offset(int block_ctx) {
    return static_cast<std::uint32_t>(AC_NUM_BLOCK_CTX * AC_NON_ZERO_BUCKETS +
                                      AC_ZERO_DENSITY_COUNT * block_ctx);
}

// Within-block context for the AC coefficient at scan index `k` (raster scan
// position), given the remaining non-zeros and the previous coefficient flag.
CUJPEGXL_ACC_HD inline std::uint32_t ac_zero_density_context(std::uint32_t nonzeros_left,
                                                             std::uint32_t k,
                                                             std::uint32_t covered_blocks,
                                                             std::uint32_t log2_covered_blocks,
                                                             std::uint32_t prev) {
    nonzeros_left = (nonzeros_left + covered_blocks - 1) >> log2_covered_blocks;
    k >>= log2_covered_blocks;
    return (AC_COEFF_NUM_NONZERO_CONTEXT[nonzeros_left] + AC_COEFF_FREQ_CONTEXT[k]) * 2 + prev;
}

// Number of AC entropy clusters (histograms) and the fixed, deterministic map
// from the 7425-context space to a cluster. Chosen to separate the dominant
// distribution axes within the JXL simple context-map budget (<=8 clusters):
// non-zero-count vs coefficient symbols, luma vs chroma block context, and a
// coarse coefficient frequency/density band. Data-driven clustering is a later
// refinement; this map is O(1) so it adds no per-frame clustering cost.
inline constexpr int AC_NUM_CLUSTERS = 8;

CUJPEGXL_ACC_HD inline int ac_cluster(std::uint32_t ctx) {
    const std::uint32_t count_ctxs{AC_NUM_BLOCK_CTX * AC_NON_ZERO_BUCKETS};  // 555
    if (ctx < count_ctxs) {
        const int block_ctx{static_cast<int>(ctx % AC_NUM_BLOCK_CTX)};
        return block_ctx < 7 ? 0 : 1;  // luma / chroma non-zero-count
    }
    const std::uint32_t rel{ctx - count_ctxs};
    const int block_ctx{static_cast<int>(rel / AC_ZERO_DENSITY_COUNT)};
    const std::uint32_t zdc{rel % AC_ZERO_DENSITY_COUNT};
    const int base{block_ctx < 7 ? 2 : 5};  // luma coeff 2..4, chroma coeff 5..7
    const int band{zdc < 64 ? 0 : (zdc < 200 ? 1 : 2)};
    return base + band;
}

// libjxl PredictFromTopAndLeft over the group-local normalized non-zero grid.
// `has_left`/`has_top` are false at the group's left column / top row, where the
// default (32) or the single available neighbor is used.
CUJPEGXL_ACC_HD inline std::uint32_t ac_predict_nonzeros(bool has_left, bool has_top,
                                                         std::int32_t top, std::int32_t left) {
    if (!has_left) {
        return has_top ? static_cast<std::uint32_t>(top) : 32u;
    }
    if (!has_top) {
        return static_cast<std::uint32_t>(left);
    }
    return static_cast<std::uint32_t>((top + left + 1) / 2);
}

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_AC_CONTEXT_H_
