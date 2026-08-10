// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_QUANT_H_
#define CUJPEGXL_SRC_QUANT_H_

#include <cstddef>
#include <cstdint>

namespace cujpegxl {

// Uniform quantization of DCT8 coefficients (uniform quant field, no adaptive
// AQ, no CfL). `distance` is a Butteraugli distance mapped to libjxl's quantizer
// exactly as cjxl does for a flat field (see quant_calibration.h): DC uses the
// separate DC quantizer (kInvDCQuant x global_scale_float x quant_dc); AC (k>=1)
// divides by the DCT8 dequant weight times the per-block AC step
// (inv_global_scale / raw_quant_field). These match the global_scale/quant_dc/
// raw_quant_field that quant_params_for_distance signals, so djxl dequantizes
// the coefficients back to what the encoder intended.
//
// `coeffs` and `q` share the layout of forward_dct8: three planes, blocks in
// raster order, 64 coefficients each. width and height must be multiples of 8.
// Both pointers are device addresses. Returns false on a CUDA error.
bool quantize_dct8(const float* coeffs, std::size_t width, std::size_t height, float distance,
                   std::int32_t* q);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_QUANT_H_
