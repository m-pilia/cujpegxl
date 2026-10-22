// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_FRONTEND_QUANTIZE_H_
#define CUJPEGXL_SRC_FRONTEND_QUANTIZE_H_

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>

namespace cujpegxl {

// Chroma-from-luma estimation + residual + quantize + DC-via-LLF, one pass per
// 64x64 color tile: estimates one signed-int8 Y-to-X and Y-to-B factor per tile
// and quantizes the FP16 covered-block coefficients into int16 AC
// (covered-block layout) and int32 DC (per 8x8 block, LLF-derived), applying the
// tile's chroma-from-luma to the AC and the base correlation to the DC, using
// the baked per-size dequant matrices.
//
// CfL: each first-block's AC coefficients (the block's covered-block slots,
// excluding its low-frequency corner) contribute to the regression of its
// top-left color tile; `ytox_map`/`ytob_map` (ceil(width/64) * ceil(height/64)
// int8) receive the maps.
//
// Per first-block of side N: Y AC quantized (roundtrip drives the residuals); X
// AC residual = X - ytox_ratio(map)*Yroundtrip; B AC residual = B -
// ytob_ratio(map)*Yroundtrip; the LLF corner of each channel feeds dc_from_llf
// (evaluated in FP32), quantized with the DC quantizer (X/Y direct, B minus
// base*Y_dc_roundtrip). AC coefficients round to the nearest integer outside the
// dead zone of `ac_deadzone_threshold` and to 0 inside it. The
// per-block quantizer strength is quant_field[block] * global_scale. `coeffs`
// (three channel-major FP16 covered-block planes, see encode_frontend), `acs`
// (the per-8x8 transform signal) and `quant_field` are device inputs; `ac`
// (three int16 covered-block planes) and `dc` (three int32 planes, one per 8x8
// block) the outputs.
//
// Also produces the AC non-zero counts consumed by the AC entropy coder, as
// three int32 planes of one entry per 8x8 block in the channel order of `ac`:
// `nz_exact` holds each first-block's number of non-zero quantized AC
// coefficients (covered positions are zero), and `nz_grid` that count
// normalized per covered 8x8 block, (nz + covered - 1) >> log2(covered), at
// every position the transform covers (the neighbor-prediction input).
// Deterministic. Returns false on a CUDA error.
bool quantize_residual(const __half* coeffs, const std::int8_t* acs,
                       const std::int32_t* quant_field, std::size_t width, std::size_t height,
                       float distance, std::int8_t* ytox_map, std::int8_t* ytob_map,
                       std::int16_t* ac, std::int32_t* dc, std::int32_t* nz_exact,
                       std::int32_t* nz_grid);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_FRONTEND_QUANTIZE_H_
