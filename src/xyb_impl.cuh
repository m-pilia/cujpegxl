// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_XYB_IMPL_CUH_
#define CUJPEGXL_SRC_XYB_IMPL_CUH_

#include <cuda_runtime.h>

// Per-pixel NV12 (BT.709 full-range Y'CbCr 4:2:0) -> XYB opsin transform, shared
// by the standalone nv12_to_xyb kernel (validated against libjxl by
// xyb_conformance_test) and the fused front-end megakernel.

namespace cujpegxl {

// libjxl opsin absorbance matrix (jxl::cms::kOpsinAbsorbanceMatrix) and bias.
// The intensity-target multiplier is 1.0 for 8-bit SDR (intensity_target=255).
constexpr float M00{0.30f};
constexpr float M02{0.078f};
constexpr float M01{1.0f - M02 - M00};
constexpr float M10{0.23f};
constexpr float M12{0.078f};
constexpr float M11{1.0f - M12 - M10};
constexpr float M20{0.24342268924547819f};
constexpr float M21{0.20476744424496821f};
constexpr float M22{1.0f - M20 - M21};
constexpr float OPSIN_BIAS{0.0037930732552754493f};

// BT.709 full-range Y'CbCr -> R'G'B' (exact inverse of the corpus generator's
// forward matrix; Kr=0.2126, Kb=0.0722).
constexpr float CR_TO_R{1.5748f};
constexpr float CB_TO_G{0.1873242729f};
constexpr float CR_TO_G{0.4681242729f};
constexpr float CB_TO_B{1.8556f};

constexpr float CHROMA_CENTER{128.0f / 255.0f};

__device__ inline float srgb_to_linear(float encoded) {
    return encoded <= 0.04045f ? encoded * (1.0f / 12.92f)
                               : powf((encoded + 0.055f) * (1.0f / 1.055f), 2.4f);
}

__device__ inline float clamp01(float v) {
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

// `yv` is the normalized luma sample; `cb_tex`/`cr_tex` are the normalized
// (0..1) upsampled chroma samples (texture read). Outputs the raw ToXYB opsin
// values (not ScaleXYB'd) matching libjxl.
__device__ inline void nv12_pixel_to_xyb(float yv, float cb_tex, float cr_tex, float& out_x,
                                         float& out_y, float& out_b) {
    const float cb{cb_tex - CHROMA_CENTER};
    const float cr{cr_tex - CHROMA_CENTER};

    const float r{clamp01(srgb_to_linear(clamp01(yv + CR_TO_R * cr)))};
    const float g{clamp01(srgb_to_linear(clamp01(yv - CB_TO_G * cb - CR_TO_G * cr)))};
    const float b{clamp01(srgb_to_linear(clamp01(yv + CB_TO_B * cb)))};

    float m0{fmaxf(M00 * r + M01 * g + M02 * b + OPSIN_BIAS, 0.0f)};
    float m1{fmaxf(M10 * r + M11 * g + M12 * b + OPSIN_BIAS, 0.0f)};
    float m2{fmaxf(M20 * r + M21 * g + M22 * b + OPSIN_BIAS, 0.0f)};

    const float neg_cbrt_bias{-cbrtf(OPSIN_BIAS)};
    m0 = cbrtf(m0) + neg_cbrt_bias;
    m1 = cbrtf(m1) + neg_cbrt_bias;
    m2 = cbrtf(m2) + neg_cbrt_bias;

    out_x = 0.5f * (m0 - m1);
    out_y = 0.5f * (m0 + m1);
    out_b = m2;
}

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_XYB_IMPL_CUH_
