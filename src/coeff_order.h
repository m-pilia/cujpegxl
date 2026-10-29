// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_COEFF_ORDER_H_
#define CUJPEGXL_SRC_COEFF_ORDER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(__CUDACC__)
#define CUJPEGXL_CO_HD __host__ __device__
#else
#define CUJPEGXL_CO_HD
#endif

namespace cujpegxl {

// Natural (scan) coefficient order for a square DCT block of side `block_dim`
// (8, 16, or 32), matching libjxl's AcStrategy::ComputeNaturalCoeffOrder for the
// square DCT8/DCT16/DCT32 strategies. Returns `block_dim*block_dim` entries where
// result[k] is the transposed-raster index (fx*block_dim + fy) of the k-th
// coefficient in scan order. The first (block_dim/8)^2 entries are the low-
// frequency block that libjxl reinterprets as the block's DC image (the LLF).
std::vector<std::uint32_t> natural_coeff_order(std::size_t block_dim);

// The frame's coefficient orders, one per square DCT side (8, 16, 32) and
// channel (0=X, 1=Y, 2=B), as libjxl's ComputeCoeffOrder: the natural order
// stably sorted by each position's zero count over the first-blocks of that
// side, quantized by coeff_order_key so that the order changes less.

inline constexpr std::size_t COEFF_ORDER_SIDES = 3;
inline constexpr std::size_t COEFF_ORDER_ENTRIES = 3 * (64 + 256 + 1024);

CUJPEGXL_CO_HD constexpr std::size_t coeff_order_side_index(int side) {
    return side == 16 ? 1 : (side == 32 ? 2 : 0);
}

// First entry of the (side, channel) order in the COEFF_ORDER_ENTRIES tables.
CUJPEGXL_CO_HD constexpr std::size_t coeff_order_offset(int side, int c) {
    const std::size_t size{static_cast<std::size_t>(side) * static_cast<std::size_t>(side)};
    const std::size_t side_base{side == 16 ? std::size_t{3 * 64}
                                           : (side == 32 ? std::size_t{3 * (64 + 256)} : 0)};
    return side_base + static_cast<std::size_t>(c) * size;
}

// `nonzeros` at coeff_order_offset(side, c) + raster index, over the
// first-blocks of the side, of which there are `blocks[coeff_order_side_index]`.
struct CoeffOrderStats {
    std::uint32_t nonzeros[COEFF_ORDER_ENTRIES];
    std::uint32_t blocks[COEFF_ORDER_SIDES];
};

// libjxl's floor(zeros / side + 0.1), in integers.
CUJPEGXL_CO_HD constexpr std::uint32_t coeff_order_key(std::uint32_t blocks, std::uint32_t nonzeros,
                                                       int side) {
    const std::uint32_t s{static_cast<std::uint32_t>(side)};
    return (10 * (blocks - nonzeros) + s) / (10 * s);
}

// `order[coeff_order_offset(side, c) + k]` is the raster index of the k-th
// coefficient, and `lehmer` at the same index the Lehmer code of the order as a
// permutation of the natural one, which AcGlobal signals.
struct CoeffOrders {
    std::uint16_t order[COEFF_ORDER_ENTRIES];
    std::uint16_t lehmer[COEFF_ORDER_ENTRIES];
};

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_COEFF_ORDER_H_
