// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_DCT_MMA_CHECK_H_
#define CUJPEGXL_SRC_DCT_MMA_CHECK_H_

#include <optional>
#include <vector>

namespace cujpegxl {

// The tensor-core DCT32 of each row-major 32x32 tile in `tiles`, as
// coefficients fx * 32 + fy per tile; std::nullopt on a CUDA error. Requires
// device code compiled for sm_87.
std::optional<std::vector<float>> dct32_mma_coefficients(const std::vector<float>& tiles);

// The tensor-core region transform of side n (8 or 16) of each row-major 16x16
// region in `regions`, as coefficients fx * 16 + fy per region, where a DCT8
// of block (dx, dy) has its coefficient (fx % 8, fy % 8) at fx = 8 dx + fx % 8,
// fy = 8 dy + fy % 8; std::nullopt on a CUDA error. Requires device code
// compiled for sm_87.
std::optional<std::vector<float>> region_mma_coefficients(const std::vector<float>& regions, int n);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_DCT_MMA_CHECK_H_
