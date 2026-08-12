// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_COEFF_ORDER_H_
#define CUJPEGXL_SRC_COEFF_ORDER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cujpegxl {

// Natural (scan) coefficient order for a square DCT block of side `block_dim`
// (8, 16, or 32), matching libjxl's AcStrategy::ComputeNaturalCoeffOrder for the
// square DCT8/DCT16/DCT32 strategies. Returns `block_dim*block_dim` entries where
// result[k] is the transposed-raster index (fx*block_dim + fy) of the k-th
// coefficient in scan order. The first (block_dim/8)^2 entries are the low-
// frequency block that libjxl reinterprets as the block's DC image (the LLF).
std::vector<std::uint32_t> natural_coeff_order(std::size_t block_dim);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_COEFF_ORDER_H_
