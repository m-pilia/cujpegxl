// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_ADAPTIVE_QUANT_IMPL_CUH_
#define CUJPEGXL_SRC_ADAPTIVE_QUANT_IMPL_CUH_

// Scalar device helpers transcribing libjxl's AdaptiveQuantizationMap
// (enc_adaptive_quantization.cc) masking metric and modulation math, shared by
// the fused front-end megakernel. libjxl's SIMD fast-math approximations
// (FastLog2f/FastPow2f) are replaced by standard libm calls; the surrounding
// spatial passes (pre-erosion footprints and the FuzzyErosion neighbourhood) are
// driven by the caller over its own (tile-local, shared-memory) layout.

namespace cujpegxl {

// libjxl adds one cubic root to the opsin gamma to approach butteraugli's gamma.
constexpr float MATCH_GAMMA_OFFSET = 0.019f;
// Per-pixel squared-difference clamp before MaskingSqrt.
constexpr float DIFF_LIMIT = 0.2f;

// RatioOfDerivativesOfCubicRootToSimpleGamma: moves the quant field from jxl's
// opsin space to butteraugli's log-gamma space.
constexpr float K_SG_MUL = 226.77216153508914f;
constexpr float K_SG_MUL2 = 1.0f / 73.377132366608819f;
constexpr float K_SG_LOG2 = 0.693147181f;
constexpr float K_SG_RET_MUL = K_SG_MUL2 * 18.6580932135f * K_SG_LOG2;
constexpr float K_SG_V_OFFSET = 7.7825991679894591f;

template <bool invert>
__device__ inline float ratio_cbrt_gamma(float v) {
    constexpr float eps{1e-2f};
    v = v < 0.0f ? 0.0f : v;
    constexpr float num_mul{K_SG_RET_MUL * 3.0f * K_SG_MUL};
    constexpr float v_offset{K_SG_V_OFFSET * K_SG_LOG2 + eps};
    constexpr float den_mul{K_SG_LOG2 * K_SG_MUL};
    const float v2{v * v};
    const float num{num_mul * v2 + eps};
    const float den{den_mul * v * v2 + v_offset};
    return invert ? num / den : den / num;
}

__device__ inline float masking_sqrt(float v) {
    constexpr float log_offset{27.505837037000106f};
    constexpr float mul{211.66567973503678f * 1e8f};
    return 0.25f * sqrtf(v * sqrtf(mul) + log_offset);
}

__device__ inline float compute_mask(float out_val) {
    constexpr float kBase{-0.7647f};
    constexpr float kMul4{9.4708735624378946f};
    constexpr float kMul2{17.35036561631863f};
    constexpr float kOffset2{302.59587815579727f};
    constexpr float kMul3{6.7943250517376494f};
    constexpr float kOffset3{3.7179635626140772f};
    constexpr float kOffset4{0.25f * kOffset3};
    constexpr float kMul0{0.80061762862741759f};
    const float v1{fmaxf(out_val * kMul0, 1e-3f)};
    const float v2{1.0f / (v1 + kOffset2)};
    const float v3{1.0f / (v1 * v1 + kOffset3)};
    const float v4{1.0f / (v1 * v1 + kOffset4)};
    return kBase + kMul4 * v4 + kMul2 * v2 + kMul3 * v3;
}

__device__ inline void compare_swap(float& lo, float& hi) {
    const float t{fminf(lo, hi)};
    hi = fmaxf(lo, hi);
    lo = t;
}

// Replaces the ascending m0..m3 with the four smallest values, ascending, of
// them and the ascending m0..m3 of lane (lane ^ lane_mask). Both lanes end with
// the same values.
__device__ inline void merge_min4_xor(float& m0, float& m1, float& m2, float& m3, int lane_mask) {
    const float p0{__shfl_xor_sync(0xffffffffu, m0, lane_mask)};
    const float p1{__shfl_xor_sync(0xffffffffu, m1, lane_mask)};
    const float p2{__shfl_xor_sync(0xffffffffu, m2, lane_mask)};
    const float p3{__shfl_xor_sync(0xffffffffu, m3, lane_mask)};
    // Pairing ascending with descending keeps the four smallest as a bitonic
    // sequence, which two compare-swap stages sort.
    m0 = fminf(m0, p3);
    m1 = fminf(m1, p2);
    m2 = fminf(m2, p1);
    m3 = fminf(m3, p0);
    compare_swap(m0, m2);
    compare_swap(m1, m3);
    compare_swap(m0, m1);
    compare_swap(m2, m3);
}

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_ADAPTIVE_QUANT_IMPL_CUH_
