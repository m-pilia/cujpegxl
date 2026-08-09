// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_QUANT_H_
#define CUJPEGXL_SRC_QUANT_H_

#include <cstddef>
#include <cstdint>

namespace cujpegxl {

// Uniform quantization of DCT8 coefficients (no adaptive quant, no CfL). Each
// coefficient is divided by its per-channel dequant weight times the global
// `distance` step and rounded half-to-even:
//   q = round(coeff / (weight[c][k] * distance)).
//
// `coeffs` and `q` share the layout of forward_dct8: three planes, blocks in
// raster order, 64 coefficients each. width and height must be multiples of 8.
// Both pointers are device addresses. Returns false on a CUDA error.
//
// NOTE: the distance->step mapping here is the plain linear form; matching
// cjxl's Butteraugli-calibrated global scale is finalized once djxl closes the
// conformance loop.
bool quantize_dct8(const float* coeffs, std::size_t width, std::size_t height, float distance,
                   std::int32_t* q);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_QUANT_H_
