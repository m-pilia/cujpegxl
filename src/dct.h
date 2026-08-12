// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_DCT_H_
#define CUJPEGXL_SRC_DCT_H_

#include <cstddef>

namespace cujpegxl {

// Forward NxN DCT of a planar XYB float image, matching libjxl's VarDCT square
// transforms: the orthonormal 2D DCT-II divided by N, laid out per block in
// libjxl's transposed raster order (coefficient index = horizontal_freq * N +
// vertical_freq), the same convention libjxl's TransformFromPixels emits and the
// DCT8 path already uses.
//
// `xyb` is three tightly packed width*height float planes. `coeffs` receives, per
// plane, NxN blocks in raster order with N*N coefficients each:
// coeffs[c*width*height + (by*(width/N)+bx)*N*N + k]. width and height must be
// multiples of N. Both pointers are device addresses. Returns false on a CUDA
// error.
bool forward_dct8(const float* xyb, std::size_t width, std::size_t height, float* coeffs);
bool forward_dct16(const float* xyb, std::size_t width, std::size_t height, float* coeffs);
bool forward_dct32(const float* xyb, std::size_t width, std::size_t height, float* coeffs);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_DCT_H_
