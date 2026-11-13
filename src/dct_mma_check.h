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

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_DCT_MMA_CHECK_H_
