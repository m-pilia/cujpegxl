// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_QUANT_CALIBRATION_H_
#define CUJPEGXL_SRC_QUANT_CALIBRATION_H_

#include <algorithm>
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define CUJPEGXL_QC_HD __host__ __device__
#else
#define CUJPEGXL_QC_HD
#endif

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

// libjxl's QuantizeBlockAC dead zone (lib/jxl/enc_group.cc) with the thresholds
// of its speed tiers that skip AdjustQuantBlockAC, for one transform side and
// channel: an AC coefficient whose magnitude in quant steps is below the
// threshold quantizes to 0. The threshold is lower in the low-frequency
// quadrant (fx, fy < side / 2), and X and B lower it for transforms covering 4+
// blocks, down to plain rounding (0.5).
struct AcDeadzone {
    // The bits of a raw index fx * side + fy that are set iff fx >= side / 2 or
    // fy >= side / 2.
    std::uint32_t high_quadrant_bits;
    float low_quadrant_threshold;
    float high_quadrant_threshold;

    // `raw` is the coefficient's index fx * side + fy in its transform.
    constexpr float threshold(std::uint32_t raw) const {
        return (raw & high_quadrant_bits) == 0 ? low_quadrant_threshold : high_quadrant_threshold;
    }
};

constexpr AcDeadzone ac_deadzone(std::uint32_t side, std::uint32_t channel) {
    const std::uint32_t high_quadrant_bits{side / 2 * (side + 1)};
    if (channel == 1) {
        return {high_quadrant_bits, 0.56f, 0.62f};
    }
    const std::uint32_t covered{side / 8 * (side / 8)};
    if (covered < 4) {
        return {high_quadrant_bits, 0.58f, 0.62f};
    }
    const auto lowered = [covered](float base) {
        const float t{base - 0.00744f * static_cast<float>(covered)};
        return t < 0.5f ? 0.5f : t;
    };
    return {high_quadrant_bits, lowered(0.58f), lowered(0.62f)};
}

// Serialized quantizer state plus the derived float scales.
struct QuantCalibration {
    std::uint32_t global_scale;
    std::uint32_t quant_dc;
    float global_scale_float;  // global_scale / 2^16
    float inv_global_scale;    // 2^16 / global_scale
};

// libjxl's InitialQuantDC (lib/jxl/enc_adaptive_quantization.cc): the DC
// quantizer of a Butteraugli `distance`.
inline float initial_quant_dc(float distance) {
    const float bt_dc = std::max(
        0.5f * distance,
        std::min(distance, K_DC_MUL * std::pow((1.0f / K_DC_MUL) * distance, K_DC_QUANT_POW)));
    return std::min(K_DC_QUANT / bt_dc, 50.0f);
}

// libjxl's ComputeGlobalScaleAndQuant (lib/jxl/quantizer.cc): the global scale
// puts the adaptive quant field's median minus its median absolute deviation at
// K_QUANT_FIELD_TARGET, capped so that the DC quantizer `quant_dc_float` stays
// representable.
CUJPEGXL_QC_HD inline QuantCalibration calibrate_quant(float quant_dc_float, float quant_median,
                                                       float quant_median_absd) {
    float scale{static_cast<float>(K_GLOBAL_SCALE_DENOM) * (quant_median - quant_median_absd) /
                K_QUANT_FIELD_TARGET};
    scale = scale < 1.0f ? 1.0f : scale;
    scale = scale > static_cast<float>(1 << 15) ? static_cast<float>(1 << 15) : scale;
    std::int32_t global_scale{static_cast<std::int32_t>(scale)};
    const std::int32_t scaled_quant_dc{
        static_cast<std::int32_t>(quant_dc_float * K_GLOBAL_SCALE_NUMERATOR * 1.6f)};
    if (global_scale > scaled_quant_dc) {
        global_scale = scaled_quant_dc <= 0 ? 1 : scaled_quant_dc;
    }

    const float inv_global_scale{static_cast<float>(K_GLOBAL_SCALE_DENOM) /
                                 static_cast<float>(global_scale)};
    float fval{quant_dc_float * inv_global_scale + 0.5f};
    fval = fval > static_cast<float>(1 << 16) ? static_cast<float>(1 << 16) : fval;

    return QuantCalibration{
        .global_scale = static_cast<std::uint32_t>(global_scale),
        .quant_dc = static_cast<std::uint32_t>(fval),
        .global_scale_float = static_cast<float>(global_scale) * (1.0f / K_GLOBAL_SCALE_DENOM),
        .inv_global_scale = inv_global_scale,
    };
}

// The integer quant field value of an adaptive quant field value `qf`.
CUJPEGXL_QC_HD inline std::int32_t quant_field_value(float qf, float inv_global_scale) {
    const std::int32_t qi{static_cast<std::int32_t>(qf * inv_global_scale + 0.5f)};
    return qi < 1 ? 1 : (qi > K_QUANT_MAX ? K_QUANT_MAX : qi);
}

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_QUANT_CALIBRATION_H_
