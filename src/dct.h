// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_DCT_H_
#define CUJPEGXL_SRC_DCT_H_

#include <cstddef>

namespace cujpegxl {

// Forward 8x8 DCT of a planar XYB float image, matching libjxl's VarDCT DCT8:
// the orthonormal 2D DCT-II divided by 8, laid out per block in libjxl's raster
// order (coefficient index = horizontal_freq * 8 + vertical_freq).
//
// `xyb` is three tightly packed width*height float planes. `coeffs` receives, per
// plane, blocks in raster order with 64 coefficients each:
// coeffs[c*width*height + (by*(width/8)+bx)*64 + k]. width and height must be
// multiples of 8. Both pointers are device addresses. Returns false on a CUDA
// error.
bool forward_dct8(const float* xyb, std::size_t width, std::size_t height, float* coeffs);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_DCT_H_
