// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "adaptive_quant.h"

#include <cuda_runtime.h>

#include "quant_calibration.h"

namespace cujpegxl {
namespace {

// libjxl adds one cubic root to the opsin gamma to approach butteraugli's gamma.
constexpr float MATCH_GAMMA_OFFSET = 0.019f;
// Per-pixel squared-difference clamp before MaskingSqrt.
constexpr float DIFF_LIMIT = 0.2f;

// RatioOfDerivativesOfCubicRootToSimpleGamma (enc_adaptive_quantization.cc):
// moves the quant field from jxl's opsin space to butteraugli's log-gamma space.
constexpr float K_SG_MUL = 226.77216153508914f;
constexpr float K_SG_MUL2 = 1.0f / 73.377132366608819f;
constexpr float K_SG_LOG2 = 0.693147181f;
constexpr float K_SG_RET_MUL = K_SG_MUL2 * 18.6580932135f * K_SG_LOG2;
constexpr float K_SG_V_OFFSET = 7.7825991679894591f;

template <bool invert>
__device__ float ratio_cbrt_gamma(float v) {
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

__device__ float masking_sqrt(float v) {
    constexpr float log_offset{27.505837037000106f};
    constexpr float mul{211.66567973503678f * 1e8f};
    return 0.25f * sqrtf(v * sqrtf(mul) + log_offset);
}

__device__ float compute_mask(float out_val) {
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

__device__ float clampi(int lo, int hi, int v) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Per-pixel 1x1 Laplacian of the Y (opsin luminance) plane, gamma-corrected,
// clamped and passed through MaskingSqrt (the diff feeding the pre-erosion map).
__device__ float pixel_diff(const float* __restrict__ y_plane, std::size_t w, std::size_t h,
                            std::size_t px, std::size_t py) {
    const std::size_t x1{px > 0 ? px - 1 : px};
    const std::size_t x2{px + 1 < w ? px + 1 : px};
    const std::size_t y1{py > 0 ? py - 1 : py};
    const std::size_t y2{py + 1 < h ? py + 1 : py};
    const float yv{y_plane[py * w + px]};
    const float base{0.25f * (y_plane[y2 * w + px] + y_plane[y1 * w + px] + y_plane[py * w + x1] +
                              y_plane[py * w + x2])};
    const float gammac{ratio_cbrt_gamma<false>(yv + MATCH_GAMMA_OFFSET)};
    float diff{gammac * (yv - base)};
    diff *= diff;
    diff = fminf(diff, DIFF_LIMIT);
    return masking_sqrt(diff);
}

// Pre-erosion map at 4x-subsampled resolution (2x2 cells per 8x8 block): each
// cell is 0.25 * sum of the pixel diffs over its 4x4 pixel footprint.
__global__ void pre_erosion_kernel(const float* __restrict__ xyb, std::size_t w, std::size_t h,
                                   std::size_t cw4, std::size_t ch4, float* __restrict__ pre) {
    const std::size_t idx{blockIdx.x * blockDim.x + threadIdx.x};
    if (idx >= cw4 * ch4) {
        return;
    }
    const float* y_plane{xyb + w * h};
    const std::size_t cx{idx % cw4};
    const std::size_t cy{idx / cw4};
    float sum{0.0f};
    for (std::size_t dy{0}; dy < 4; ++dy) {
        for (std::size_t dx{0}; dx < 4; ++dx) {
            sum += pixel_diff(y_plane, w, h, cx * 4 + dx, cy * 4 + dy);
        }
    }
    pre[idx] = 0.25f * sum;
}

__device__ void sort4(float& a, float& b, float& c, float& d) {
    float t{};
    if (a > b) {
        t = a;
        a = b;
        b = t;
    }
    if (a > c) {
        t = a;
        a = c;
        c = t;
    }
    if (a > d) {
        t = a;
        a = d;
        d = t;
    }
    if (b > c) {
        t = b;
        b = c;
        c = t;
    }
    if (b > d) {
        t = b;
        b = d;
        d = t;
    }
    if (c > d) {
        t = c;
        c = d;
        d = t;
    }
}

__device__ void store_min4(float v, float& m0, float& m1, float& m2, float& m3) {
    if (v < m3) {
        if (v < m0) {
            m3 = m2;
            m2 = m1;
            m1 = m0;
            m0 = v;
        } else if (v < m1) {
            m3 = m2;
            m2 = m1;
            m1 = v;
        } else if (v < m2) {
            m3 = m2;
            m2 = v;
        } else {
            m3 = v;
        }
    }
}

// FuzzyErosion at one pre-erosion cell: weighted sum of the four smallest values
// in its clamped 3x3 neighbourhood.
__device__ float erosion_value(const float* __restrict__ pre, std::size_t cw4, std::size_t ch4,
                               std::size_t x, std::size_t y, float km0, float km1, float km2,
                               float km3) {
    const std::size_t xm1{x > 0 ? x - 1 : x};
    const std::size_t xp1{x + 1 < cw4 ? x + 1 : x};
    const std::size_t ym1{y > 0 ? y - 1 : y};
    const std::size_t yp1{y + 1 < ch4 ? y + 1 : y};
    const float* rowt{pre + ym1 * cw4};
    const float* row{pre + y * cw4};
    const float* rowb{pre + yp1 * cw4};
    float m0{row[x]};
    float m1{row[xm1]};
    float m2{row[xp1]};
    float m3{rowt[xm1]};
    sort4(m0, m1, m2, m3);
    store_min4(rowt[x], m0, m1, m2, m3);
    store_min4(rowt[xp1], m0, m1, m2, m3);
    store_min4(rowb[xm1], m0, m1, m2, m3);
    store_min4(rowb[x], m0, m1, m2, m3);
    store_min4(rowb[xp1], m0, m1, m2, m3);
    return km0 * m0 + km1 * m1 + km2 * m2 + km3 * m3;
}

__device__ float hf_modulation(const float* __restrict__ y_plane, std::size_t w, std::size_t x0,
                               std::size_t y0, float out_val) {
    constexpr float valmin{0.0206f};
    float sum{0.0f};
    for (std::size_t dy{0}; dy < 8; ++dy) {
        const std::size_t row{y0 + dy};
        const std::size_t row_next{dy == 7 ? row : row + 1};
        for (std::size_t dx{0}; dx < 8; ++dx) {
            const float p{y_plane[row * w + x0 + dx]};
            if (dx < 7) {
                const float pr{y_plane[row * w + x0 + dx + 1]};
                sum += fminf(valmin, fabsf(p - pr));
            }
            const float pd{y_plane[row_next * w + x0 + dx]};
            sum += fminf(valmin, fabsf(p - pd));
        }
    }
    return out_val + sum * -0.38f + 0.42f;
}

__device__ float gamma_modulation(const float* __restrict__ x_plane,
                                  const float* __restrict__ y_plane, std::size_t w, std::size_t x0,
                                  std::size_t y0, float out_val) {
    constexpr float kBias{0.16f};
    float overall{0.0f};
    for (std::size_t dy{0}; dy < 8; ++dy) {
        for (std::size_t dx{0}; dx < 8; ++dx) {
            const std::size_t p{(y0 + dy) * w + x0 + dx};
            const float iny{y_plane[p] + kBias};
            const float inx{x_plane[p]};
            overall += ratio_cbrt_gamma<true>(iny - inx);
            overall += ratio_cbrt_gamma<true>(iny + inx);
        }
    }
    overall *= 0.5f / 64.0f;
    return out_val + 0.1005613337192697f * log2f(overall);
}

__device__ float blue_modulation(const float* __restrict__ x_plane,
                                 const float* __restrict__ y_plane,
                                 const float* __restrict__ b_plane, std::size_t w, std::size_t x0,
                                 std::size_t y0, float out_val) {
    constexpr float kLimit{0.027121074570634722f};
    constexpr float kOffset{0.084381641171960495f};
    float sum{0.0f};
    for (std::size_t dy{0}; dy < 8; ++dy) {
        for (std::size_t dx{0}; dx < 8; ++dx) {
            const std::size_t p{(y0 + dy) * w + x0 + dx};
            const float px{x_plane[p]};
            const float pb{b_plane[p]};
            const float pye{y_plane[p] + kOffset + fabsf(px)};
            if (pb > pye) {
                sum += fminf(pb - pye, kLimit);
            }
        }
    }
    if (sum >= 32.0f * kLimit) {
        sum = 64.0f * kLimit - sum;
    }
    constexpr float kMaxLimit{15.398788439047934f};
    if (sum >= kMaxLimit * kLimit) {
        sum = kMaxLimit * kLimit;
    }
    return out_val + sum * 0.14207000358439159f;
}

__global__ void quant_field_kernel(const float* __restrict__ xyb, const float* __restrict__ pre,
                                   std::size_t w, std::size_t h, std::size_t bw, std::size_t bh,
                                   std::size_t cw4, std::size_t ch4, float km0, float km1, float km2,
                                   float km3, float mul_pb, float add_pb, float inv_global_scale,
                                   std::int32_t* __restrict__ quant_field) {
    const std::size_t b{blockIdx.x * blockDim.x + threadIdx.x};
    if (b >= bw * bh) {
        return;
    }
    const std::size_t bx{b % bw};
    const std::size_t by{b / bw};

    float aq{0.0f};
    for (std::size_t sy{0}; sy < 2; ++sy) {
        for (std::size_t sx{0}; sx < 2; ++sx) {
            aq += erosion_value(pre, cw4, ch4, 2 * bx + sx, 2 * by + sy, km0, km1, km2, km3);
        }
    }

    const std::size_t plane{w * h};
    const float* x_plane{xyb};
    const float* y_plane{xyb + plane};
    const float* b_plane{xyb + 2 * plane};
    const std::size_t x0{bx * 8};
    const std::size_t y0{by * 8};

    float ov{compute_mask(aq)};
    ov = hf_modulation(y_plane, w, x0, y0, ov);
    ov = gamma_modulation(x_plane, y_plane, w, x0, y0, ov);
    ov = blue_modulation(x_plane, y_plane, b_plane, w, x0, y0, ov);

    const float qval{expf(ov) * mul_pb + add_pb};
    quant_field[b] =
        static_cast<std::int32_t>(clampi(1, K_QUANT_MAX, static_cast<int>(qval * inv_global_scale +
                                                                          0.5f)));
}

}  // namespace

bool compute_quant_field(const float* xyb, std::size_t width, std::size_t height, float distance,
                         std::int32_t* quant_field) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t cw4{width / 4};
    const std::size_t ch4{height / 4};

    const QuantCalibration cal{calibrate_quant(distance)};
    const float scale{K_AC_QUANT / distance};

    // PerBlockModulations exponent -> multiplicative field mapping.
    const float base_level{0.48f * scale};
    float dampen{1.0f};
    if (distance >= 2.0f) {
        dampen = 1.0f - (distance - 2.0f) / (14.0f - 2.0f);
        if (dampen < 0.0f) {
            dampen = 0.0f;
        }
    }
    const float mul_pb{scale * dampen};
    const float add_pb{(1.0f - dampen) * base_level};

    // FuzzyErosion weights (butteraugli-target dependent), normalized to kTotal.
    const float efm{distance < 2.0f ? (2.0f - distance) * 0.5f : 0.0f};
    float km0{0.125f};
    float km1{0.10f - efm * 0.10f};
    float km2{0.09f - efm * 0.09f};
    float km3{0.06f - efm * 0.06f};
    const float norm{0.29959705784054957f / (km0 + km1 + km2 + km3)};
    km0 *= norm;
    km1 *= norm;
    km2 *= norm;
    km3 *= norm;

    float* pre{nullptr};
    if (cudaMalloc(&pre, cw4 * ch4 * sizeof(float)) != cudaSuccess) {
        return false;
    }

    const unsigned int threads{256};
    const unsigned int pre_blocks{static_cast<unsigned int>((cw4 * ch4 + threads - 1) / threads)};
    pre_erosion_kernel<<<pre_blocks, threads>>>(xyb, width, height, cw4, ch4, pre);
    const unsigned int qf_blocks{static_cast<unsigned int>((bw * bh + threads - 1) / threads)};
    quant_field_kernel<<<qf_blocks, threads>>>(xyb, pre, width, height, bw, bh, cw4, ch4, km0, km1,
                                               km2, km3, mul_pb, add_pb, cal.inv_global_scale,
                                               quant_field);

    const cudaError_t launch{cudaGetLastError()};
    const cudaError_t sync{cudaDeviceSynchronize()};
    cudaFree(pre);
    return launch == cudaSuccess && sync == cudaSuccess;
}

}  // namespace cujpegxl
