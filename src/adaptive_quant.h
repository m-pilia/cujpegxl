// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_ADAPTIVE_QUANT_H_
#define CUJPEGXL_SRC_ADAPTIVE_QUANT_H_

#include <cstddef>
#include <cstdint>

namespace cujpegxl {

// Tier-0 adaptive quantization field: a per-8x8-block quant multiplier derived
// from local image content, transcribing libjxl's AdaptiveQuantizationMap
// (enc_adaptive_quantization.cc) structure and constants. The per-pixel diff /
// masking metric and the ComputeMask + HF/gamma/blue modulations are reproduced
// faithfully; libjxl's SIMD fast-math approximations (FastLog2f/FastPow2f) are
// replaced by standard libm calls, and the frame's global scale is taken as a
// constant from the distance (calibrate_quant) rather than from the field's
// median/MAD, so the field carries all the adaptive variation.
//
// `xyb` is the planar XYB float image (three tightly packed width*height planes,
// X then Y then B) produced by nv12_to_xyb. `quant_field` receives the per-block
// quant integers in [1, 256], (width/8)*(height/8) entries in block raster
// order — the same buffer consumed by quantize_dct8 and the AcMetadata encoder.
// width and height must be multiples of 8. Both pointers are device addresses.
// Deterministic. Returns false on a CUDA error.
bool compute_quant_field(const float* xyb, std::size_t width, std::size_t height, float distance,
                         std::int32_t* quant_field);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_ADAPTIVE_QUANT_H_
