// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_CFL_H_
#define CUJPEGXL_SRC_CFL_H_

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace cujpegxl {

// Chroma-from-luma color model and per-tile estimation, matching libjxl's
// ColorCorrelation. The X and B AC coefficients are decorrelated against Y:
// the decoder reconstructs X = dq_x + ytox_ratio(m) * dq_y and
// B = dq_b + ytob_ratio(m) * dq_y, so the encoder subtracts the same multiple of
// the roundtrip-quantized Y before quantizing. One signed-int8 factor `m` is
// carried per 64x64 color tile (8x8 blocks) in the AcMetadata YtoX/YtoB channels.

#if defined(__CUDACC__)
#define CUJPEGXL_CFL_HD __host__ __device__
#else
#define CUJPEGXL_CFL_HD
#endif

inline constexpr int CFL_COLOR_FACTOR = 84;                        // libjxl kDefaultColorFactor
inline constexpr float CFL_COLOR_SCALE = 1.0f / CFL_COLOR_FACTOR;  // libjxl color_scale_
inline constexpr float CFL_BASE_X = 0.0f;                          // base_correlation_x
inline constexpr float CFL_BASE_B = 1.0f;  // base_correlation_b (kYToBRatio)
inline constexpr int CFL_MAP_MIN = -128;   // ImageSB (signed int8) range
inline constexpr int CFL_MAP_MAX = 127;

// Matches libjxl ColorCorrelation::YtoXRatio/YtoBRatio bit-for-bit (base plus map
// times the precomputed reciprocal color scale).
CUJPEGXL_CFL_HD inline float cfl_ytox_ratio(int m) {
    return CFL_BASE_X + static_cast<float>(m) * CFL_COLOR_SCALE;
}
CUJPEGXL_CFL_HD inline float cfl_ytob_ratio(int m) {
    return CFL_BASE_B + static_cast<float>(m) * CFL_COLOR_SCALE;
}

CUJPEGXL_CFL_HD inline int cfl_clamp_map(int m) {
    return m < CFL_MAP_MIN ? CFL_MAP_MIN : (m > CFL_MAP_MAX ? CFL_MAP_MAX : m);
}

// Quantizes a correlation factor (delta over the channel's base) to a signed map
// value: m = round(delta * color_factor), clamped to the int8 range. ytox uses
// base 0, ytob base 1, so the estimator passes the regression slope and the
// slope-minus-one respectively.
CUJPEGXL_CFL_HD inline int cfl_quantize_factor(float delta_over_base) {
    return cfl_clamp_map(static_cast<int>(lrintf(delta_over_base * CFL_COLOR_FACTOR)));
}

// Converts the least-squares regression sums of a tile to color-map values:
// Y-to-X from the slope Sxy/Syy, Y-to-B from the residual-over-base slope
// Sby/Syy. A degenerate (zero-energy) tile maps to the base.
CUJPEGXL_CFL_HD inline void cfl_maps_from_sums(double sxy, double syy, double sby, int* ytox_map,
                                               int* ytob_map) {
    const float fx{syy > 0.0 ? static_cast<float>(sxy / syy) : 0.0f};
    const float db{syy > 0.0 ? static_cast<float>(sby / syy) : 0.0f};
    *ytox_map = cfl_quantize_factor(fx);
    *ytob_map = cfl_quantize_factor(db);
}

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_CFL_H_
