// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_QUANT_H_
#define CUJPEGXL_SRC_QUANT_H_

#include <cstddef>
#include <cstdint>

namespace cujpegxl {

// Quantization of DCT8 coefficients with a per-block adaptive quant field (no
// CfL). `distance` sets the frame-constant DC and global scales (see
// quant_calibration.h); the per-block AC step comes from `quant_field`: DC uses
// the separate DC quantizer (kInvDCQuant x global_scale_float x quant_dc); AC
// (k>=1) multiplies by (quant_field[block] x global_scale_float) / dequant
// weight. These match the global_scale/quant_dc signaled in DcGlobal and the
// per-block quant field signaled in AcMetadata, so djxl dequantizes the
// coefficients back to what the encoder intended.
//
// `coeffs` and `q` share the layout of forward_dct8: three planes, blocks in
// raster order, 64 coefficients each. `quant_field` is the per-block quant
// integer buffer (width/8 * height/8, block raster order). width and height must
// be multiples of 8. All pointers are device addresses. Returns false on a CUDA
// error.
bool quantize_dct8(const float* coeffs, std::size_t width, std::size_t height, float distance,
                   const std::int32_t* quant_field, std::int32_t* q);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_QUANT_H_
