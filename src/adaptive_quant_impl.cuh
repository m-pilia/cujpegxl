// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_ADAPTIVE_QUANT_IMPL_CUH_
#define CUJPEGXL_SRC_ADAPTIVE_QUANT_IMPL_CUH_

#include <cuda_fp16.h>

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
constexpr float K_SG_EPS = 1e-2f;
constexpr float K_SG_NUM_MUL = K_SG_RET_MUL * 3.0f * K_SG_MUL;
constexpr float K_SG_DEN_OFFSET = K_SG_V_OFFSET * K_SG_LOG2 + K_SG_EPS;
constexpr float K_SG_DEN_MUL = K_SG_LOG2 * K_SG_MUL;

template <bool invert>
__device__ inline float ratio_cbrt_gamma(float v) {
    v = v < 0.0f ? 0.0f : v;
    const float v2{v * v};
    const float num{K_SG_NUM_MUL * v2 + K_SG_EPS};
    const float den{K_SG_DEN_MUL * v * v2 + K_SG_DEN_OFFSET};
    return invert ? num / den : den / num;
}

// ratio_cbrt_gamma<true>(a) + ratio_cbrt_gamma<true>(b) with the polynomials in
// half2 arithmetic and an approximate division.
__device__ inline float inverse_ratio_cbrt_gamma_sum(float a, float b) {
    const __half2 v{__hmax2(__floats2half2_rn(a, b), __float2half2_rn(0.0f))};
    const __half2 v2{__hmul2(v, v)};
    const float2 num{__half22float2(
        __hfma2(__float2half2_rn(K_SG_NUM_MUL), v2, __float2half2_rn(K_SG_EPS)))};
    const float2 den{__half22float2(__hfma2(__hmul2(__float2half2_rn(K_SG_DEN_MUL), v), v2,
                                            __float2half2_rn(K_SG_DEN_OFFSET)))};
    return __fdividef(num.x, den.x) + __fdividef(num.y, den.y);
}

constexpr float K_MASKING_LOG_OFFSET = 27.505837037000106f;
constexpr float K_MASKING_MUL = 211.66567973503678f * 1e8f;

__device__ inline float masking_sqrt(float v) {
    return 0.25f * sqrtf(v * sqrtf(K_MASKING_MUL) + K_MASKING_LOG_OFFSET);
}

// The sum over the two pixels (y.x, d.x) and (y.y, d.y) of
// masking_sqrt(min((ratio_cbrt_gamma<false>(y + MATCH_GAMMA_OFFSET) * d)^2, DIFF_LIMIT)),
// in half2 arithmetic with approximate reciprocals and square roots.
__device__ inline float masking_sqrt_pair_sum(float2 y, float2 d) {
    const __half2 v{__hmax2(__hadd2(__float22half2_rn(y), __float2half2_rn(MATCH_GAMMA_OFFSET)),
                            __float2half2_rn(0.0f))};
    const __half2 v2{__hmul2(v, v)};
    const __half2 num{__hfma2(__float2half2_rn(K_SG_NUM_MUL), v2, __float2half2_rn(K_SG_EPS))};
    const __half2 den{__hfma2(__hmul2(__float2half2_rn(K_SG_DEN_MUL), v), v2,
                              __float2half2_rn(K_SG_DEN_OFFSET))};
    const __half2 scaled{__hmul2(__hmul2(den, h2rcp(num)), __float22half2_rn(d))};
    const __half2 clamped{__hmin2(__hmul2(scaled, scaled), __float2half2_rn(DIFF_LIMIT))};
    // masking_sqrt's factor 0.25 is folded into the radicand as 1/16.
    const float2 radicand{__half22float2(
        __hfma2(clamped, __float2half2_rn(sqrtf(K_MASKING_MUL) / 16.0f),
                __float2half2_rn(K_MASKING_LOG_OFFSET / 16.0f)))};
    return radicand.x * rsqrtf(radicand.x) + radicand.y * rsqrtf(radicand.y);
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
