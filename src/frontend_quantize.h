// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_FRONTEND_QUANTIZE_H_
#define CUJPEGXL_SRC_FRONTEND_QUANTIZE_H_

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>

namespace cujpegxl {

// CfL estimation: estimates one signed-int8 Y-to-X and Y-to-B factor per
// 64x64 color tile from the FP16 covered-block coefficients. Each first-block's
// AC coefficients (the block's covered-block slots, excluding its low-frequency
// corner) contribute to the regression of its top-left color tile. `coeffs` are
// three channel-major FP16 covered-block planes (see variable_forward_dct); `acs`
// the per-8x8 transform signal. `ytox_map`/`ytob_map` (ceil(width/64) *
// ceil(height/64) int8) receive the maps. Deterministic. Returns false on a CUDA
// error. The host variant runs the identical per-tile core for validation.
bool estimate_cfl_covered(const __half* coeffs, const std::int8_t* acs, std::size_t width,
                          std::size_t height, std::int8_t* ytox_map, std::int8_t* ytob_map);
void estimate_cfl_covered_host(const __half* coeffs, const std::int8_t* acs, std::size_t width,
                               std::size_t height, std::int8_t* ytox_map, std::int8_t* ytob_map);

// Residual + quantize + DC-via-LLF: quantizes the FP16 covered-block
// coefficients into int16 AC (covered-block layout) and int32 DC (per 8x8 block,
// LLF-derived), applying per-tile chroma-from-luma to the AC and the base
// correlation to the DC, using the baked per-size dequant matrices.
//
// Per first-block of side N: Y AC quantized (roundtrip drives the residuals); X
// AC residual = X - ytox_ratio(map)*Yroundtrip; B AC residual = B -
// ytob_ratio(map)*Yroundtrip; the LLF corner of each channel feeds dc_from_llf,
// quantized with the DC quantizer (X/Y direct, B minus base*Y_dc_roundtrip). The
// per-block quantizer strength is quant_field[block] * global_scale. `coeffs`,
// `acs`, `ytox_map`, `ytob_map`, `quant_field` are device inputs; `ac`
// (three int16 covered-block planes) and `dc` (three int32 planes, one per 8x8
// block) the outputs. Deterministic. Returns false on a CUDA error.
bool quantize_residual(const __half* coeffs, const std::int8_t* acs, const std::int8_t* ytox_map,
                       const std::int8_t* ytob_map, const std::int32_t* quant_field,
                       std::size_t width, std::size_t height, float distance, std::int16_t* ac,
                       std::int32_t* dc);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_FRONTEND_QUANTIZE_H_
