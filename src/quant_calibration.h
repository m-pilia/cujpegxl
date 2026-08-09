// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_QUANT_CALIBRATION_H_
#define CUJPEGXL_SRC_QUANT_CALIBRATION_H_

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace cujpegxl {

// libjxl DC quantization reciprocals (kInvDCQuant), channels ordered X, Y, B to
// match the quantizer input plane order. DC is quantized with these, separately
// from the DCT8 AC dequant matrix.
inline constexpr float DC_INV_QUANT[3] = {4096.0f, 512.0f, 256.0f};

// libjxl default base chroma-from-luma Y-to-B correlation (kYToBRatio). The
// decoder adds dequant_y * Y_TO_B_RATIO to every B coefficient (DC and AC), so
// the encoder must subtract the roundtrip Y coefficient from B before
// quantizing; otherwise djxl's default correlation reconstructs B at ~2x. The
// Y-to-X default is 0 (no-op).
inline constexpr float Y_TO_B_RATIO = 1.0f;

// libjxl distance->quant constants (lib/jxl/enc_adaptive_quantization.cc and
// lib/jxl/quantizer.{h,cc}).
inline constexpr float K_AC_QUANT = 0.725f;
inline constexpr float K_DC_QUANT = 1.095924047623553f;
inline constexpr float K_DC_QUANT_POW = 0.83f;
inline constexpr float K_DC_MUL = 0.3f;
inline constexpr float K_QUANT_FIELD_TARGET = 5.0f;
inline constexpr int K_GLOBAL_SCALE_DENOM = 1 << 16;
inline constexpr int K_GLOBAL_SCALE_NUMERATOR = 4096;
inline constexpr int K_QUANT_MAX = 256;

// Serialized quantizer state plus the derived float scales, computed from a
// Butteraugli `distance` exactly as libjxl's encoder does for a uniform quant
// field (no adaptive AQ): base AC field value kAcQuant/distance and DC value
// from InitialQuantDC, folded into a global scale by ComputeGlobalScaleAndQuant.
// See lib/jxl/quantizer.cc and lib/jxl/enc_adaptive_quantization.cc.
struct QuantCalibration {
    std::uint32_t global_scale;
    std::uint32_t quant_dc;
    std::uint32_t raw_quant_field;  // uniform per-block AC quant integer (Q)
    float global_scale_float;       // global_scale / 2^16
    float inv_global_scale;         // 2^16 / global_scale
};

inline QuantCalibration calibrate_quant(float distance) {
    const float quant_ac = K_AC_QUANT / distance;

    const float bt_dc =
        std::max(0.5f * distance,
                 std::min(distance, K_DC_MUL * std::pow((1.0f / K_DC_MUL) * distance,
                                                        K_DC_QUANT_POW)));
    const float quant_dc_float = std::min(K_DC_QUANT / bt_dc, 50.0f);

    // ComputeGlobalScaleAndQuant with a uniform field: median == quant_ac and
    // median-absolute-deviation == 0.
    float scale = static_cast<float>(K_GLOBAL_SCALE_DENOM) * quant_ac / K_QUANT_FIELD_TARGET;
    if (scale < 1.0f) {
        scale = 1.0f;
    }
    if (scale > static_cast<float>(1 << 15)) {
        scale = static_cast<float>(1 << 15);
    }
    int global_scale = static_cast<int>(scale);
    const int scaled_quant_dc =
        static_cast<int>(quant_dc_float * K_GLOBAL_SCALE_NUMERATOR * 1.6f);
    if (global_scale > scaled_quant_dc) {
        global_scale = scaled_quant_dc <= 0 ? 1 : scaled_quant_dc;
    }

    const float inv_global_scale =
        1.0f * static_cast<float>(K_GLOBAL_SCALE_DENOM) / static_cast<float>(global_scale);
    float fval = quant_dc_float * inv_global_scale + 0.5f;
    fval = std::min<float>(static_cast<float>(1 << 16), fval);
    const int quant_dc = static_cast<int>(fval);

    int q = static_cast<int>(std::lround(quant_ac * inv_global_scale));
    q = std::max(1, std::min(K_QUANT_MAX, q));

    QuantCalibration cal{};
    cal.global_scale = static_cast<std::uint32_t>(global_scale);
    cal.quant_dc = static_cast<std::uint32_t>(quant_dc);
    cal.raw_quant_field = static_cast<std::uint32_t>(q);
    cal.global_scale_float = static_cast<float>(global_scale) * (1.0f / K_GLOBAL_SCALE_DENOM);
    cal.inv_global_scale = inv_global_scale;
    return cal;
}

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_QUANT_CALIBRATION_H_
