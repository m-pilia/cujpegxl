// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frontend_fused.h"

#include <cmath>
#include <mutex>

#include <cuda_runtime.h>

#include "adaptive_quant_impl.cuh"
#include "quant_calibration.h"
#include "quant_weights_dct8.h"
#include "xyb_impl.cuh"

namespace cujpegxl {
namespace {

// Tile geometry: one CUDA block processes TB x TB image blocks. The shared XYB
// region carries a one-pixel halo (PAD) for the adaptive-quant Laplacian; the
// pre-erosion map is at 4x-subsampled resolution (CELLS x CELLS cells).
constexpr int TB = 4;
constexpr int TILE_PX = TB * 8;
constexpr int PAD = TILE_PX + 2;
constexpr int CELLS = 2 * TB;

__constant__ float DCT_A[64];
__constant__ float QUANT_WEIGHTS[3][64];
__constant__ float DC_INV_QUANT_D[3];
__constant__ float Y_TO_B_RATIO_D;

void init_constants() {
    float basis[64];
    for (int k{0}; k < 8; ++k) {
        const double g{k == 0 ? 0.125 : 1.4142135623730951 / 8.0};
        for (int n{0}; n < 8; ++n) {
            basis[k * 8 + n] = static_cast<float>(g * std::cos(M_PI * (n + 0.5) * k / 8.0));
        }
    }
    cudaMemcpyToSymbol(DCT_A, basis, sizeof(basis));
    cudaMemcpyToSymbol(QUANT_WEIGHTS, DCT8_DEQUANT_WEIGHTS, sizeof(DCT8_DEQUANT_WEIGHTS));
    cudaMemcpyToSymbol(DC_INV_QUANT_D, DC_INV_QUANT, sizeof(DC_INV_QUANT));
    cudaMemcpyToSymbol(Y_TO_B_RATIO_D, &Y_TO_B_RATIO, sizeof(float));
}

void ensure_constants() {
    static std::once_flag flag;
    std::call_once(flag, init_constants);
}

__device__ inline float quant_factor(int c, int k, float ac_scale, float dc_scale) {
    return k == 0 ? DC_INV_QUANT_D[c] * dc_scale : ac_scale / QUANT_WEIGHTS[c][k];
}

// HF/gamma/blue modulations over one 8x8 block, reading the shared XYB planes
// (PAD-strided, `x0`/`y0` the block's top-left interior coordinate).
__device__ float hf_modulation(const float* __restrict__ y, int x0, int y0, float out_val) {
    constexpr float valmin{0.0206f};
    float sum{0.0f};
    for (int dy{0}; dy < 8; ++dy) {
        const int row{y0 + dy};
        const int row_next{dy == 7 ? row : row + 1};
        for (int dx{0}; dx < 8; ++dx) {
            const float p{y[row * PAD + x0 + dx]};
            if (dx < 7) {
                sum += fminf(valmin, fabsf(p - y[row * PAD + x0 + dx + 1]));
            }
            sum += fminf(valmin, fabsf(p - y[row_next * PAD + x0 + dx]));
        }
    }
    return out_val + sum * -0.38f + 0.42f;
}

__device__ float gamma_modulation(const float* __restrict__ xp, const float* __restrict__ yp,
                                  int x0, int y0, float out_val) {
    constexpr float kBias{0.16f};
    float overall{0.0f};
    for (int dy{0}; dy < 8; ++dy) {
        for (int dx{0}; dx < 8; ++dx) {
            const int idx{(y0 + dy) * PAD + x0 + dx};
            const float iny{yp[idx] + kBias};
            const float inx{xp[idx]};
            overall += ratio_cbrt_gamma<true>(iny - inx);
            overall += ratio_cbrt_gamma<true>(iny + inx);
        }
    }
    overall *= 0.5f / 64.0f;
    return out_val + 0.1005613337192697f * log2f(overall);
}

__device__ float blue_modulation(const float* __restrict__ xp, const float* __restrict__ yp,
                                 const float* __restrict__ bp, int x0, int y0, float out_val) {
    constexpr float kLimit{0.027121074570634722f};
    constexpr float kOffset{0.084381641171960495f};
    float sum{0.0f};
    for (int dy{0}; dy < 8; ++dy) {
        for (int dx{0}; dx < 8; ++dx) {
            const int idx{(y0 + dy) * PAD + x0 + dx};
            const float px{xp[idx]};
            const float pye{yp[idx] + kOffset + fabsf(px)};
            if (bp[idx] > pye) {
                sum += fminf(bp[idx] - pye, kLimit);
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

// FuzzyErosion at pre-erosion cell (cx, cy): weighted sum of the four smallest
// values in its 3x3 neighbourhood, clamped to the tile (the seam approximation).
__device__ float erosion_cell(const float* __restrict__ pre, int cx, int cy, float km0, float km1,
                              float km2, float km3) {
    const int xm1{cx > 0 ? cx - 1 : cx};
    const int xp1{cx + 1 < CELLS ? cx + 1 : cx};
    const int ym1{cy > 0 ? cy - 1 : cy};
    const int yp1{cy + 1 < CELLS ? cy + 1 : cy};
    float m0{pre[cy * CELLS + cx]};
    float m1{pre[cy * CELLS + xm1]};
    float m2{pre[cy * CELLS + xp1]};
    float m3{pre[ym1 * CELLS + xm1]};
    sort4(m0, m1, m2, m3);
    store_min4(pre[ym1 * CELLS + cx], m0, m1, m2, m3);
    store_min4(pre[ym1 * CELLS + xp1], m0, m1, m2, m3);
    store_min4(pre[yp1 * CELLS + xm1], m0, m1, m2, m3);
    store_min4(pre[yp1 * CELLS + cx], m0, m1, m2, m3);
    store_min4(pre[yp1 * CELLS + xp1], m0, m1, m2, m3);
    return km0 * m0 + km1 * m1 + km2 * m2 + km3 * m3;
}

__global__ void frontend_fused_kernel(const std::uint8_t* __restrict__ luma, std::size_t luma_pitch,
                                      cudaTextureObject_t chroma_tex, std::size_t width,
                                      std::size_t height, std::size_t bw, std::size_t bh, float gsf,
                                      float dc_scale, float km0, float km1, float km2, float km3,
                                      float mul_pb, float add_pb, float inv_global_scale,
                                      std::int32_t* __restrict__ q,
                                      std::int32_t* __restrict__ quant_field) {
    __shared__ float sh_x[PAD * PAD];
    __shared__ float sh_y[PAD * PAD];
    __shared__ float sh_b[PAD * PAD];
    __shared__ float sh_pre[CELLS * CELLS];
    __shared__ int sh_q;

    const std::size_t tbx0{static_cast<std::size_t>(blockIdx.x) * TB};
    const std::size_t tby0{static_cast<std::size_t>(blockIdx.y) * TB};
    const std::size_t px0{tbx0 * 8};
    const std::size_t py0{tby0 * 8};
    const int tid{static_cast<int>(threadIdx.y) * 8 + static_cast<int>(threadIdx.x)};
    const std::size_t plane{width * height};

    // Load the padded tile and compute XYB into shared memory.
    for (int i{tid}; i < PAD * PAD; i += 64) {
        const int lx{i % PAD};
        const int ly{i / PAD};
        long gx{static_cast<long>(px0) + lx - 1};
        long gy{static_cast<long>(py0) + ly - 1};
        gx = gx < 0 ? 0 : (gx >= static_cast<long>(width) ? static_cast<long>(width) - 1 : gx);
        gy = gy < 0 ? 0 : (gy >= static_cast<long>(height) ? static_cast<long>(height) - 1 : gy);
        const float yv{luma[static_cast<std::size_t>(gy) * luma_pitch + gx] * (1.0f / 255.0f)};
        const float2 c{tex2D<float2>(chroma_tex, (gx + 0.5f) * 0.5f, (gy + 0.5f) * 0.5f)};
        nv12_pixel_to_xyb(yv, c.x, c.y, sh_x[i], sh_y[i], sh_b[i]);
    }
    __syncthreads();

    // Pre-erosion cells: 0.25 * sum of the 4x4 pixel diffs (Laplacian of Y).
    for (int ci{tid}; ci < CELLS * CELLS; ci += 64) {
        const int ccx{ci % CELLS};
        const int ccy{ci / CELLS};
        float sum{0.0f};
        for (int dy{0}; dy < 4; ++dy) {
            for (int dx{0}; dx < 4; ++dx) {
                const int lx{ccx * 4 + dx + 1};
                const int ly{ccy * 4 + dy + 1};
                const float yv{sh_y[ly * PAD + lx]};
                const float base{0.25f * (sh_y[(ly + 1) * PAD + lx] + sh_y[(ly - 1) * PAD + lx] +
                                          sh_y[ly * PAD + lx - 1] + sh_y[ly * PAD + lx + 1])};
                float diff{ratio_cbrt_gamma<false>(yv + MATCH_GAMMA_OFFSET) * (yv - base)};
                diff *= diff;
                diff = fminf(diff, DIFF_LIMIT);
                sum += masking_sqrt(diff);
            }
        }
        sh_pre[ci] = 0.25f * sum;
    }
    __syncthreads();

    const int fx{static_cast<int>(threadIdx.x)};
    const int fy{static_cast<int>(threadIdx.y)};
    const int k{fx * 8 + fy};

    for (int lb{0}; lb < TB * TB; ++lb) {
        const int lbx{lb % TB};
        const int lby{lb / TB};
        const std::size_t gbx{tbx0 + lbx};
        const std::size_t gby{tby0 + lby};
        if (gbx >= bw || gby >= bh) {
            continue;
        }
        const int x0{lbx * 8 + 1};  // shared interior origin (+halo)
        const int y0{lby * 8 + 1};

        // Adaptive quant integer (thread 0 computes, broadcasts via shared).
        if (tid == 0) {
            float aq{0.0f};
            for (int sy{0}; sy < 2; ++sy) {
                for (int sx{0}; sx < 2; ++sx) {
                    aq += erosion_cell(sh_pre, 2 * lbx + sx, 2 * lby + sy, km0, km1, km2, km3);
                }
            }
            float ov{compute_mask(aq)};
            ov = hf_modulation(sh_y, x0, y0, ov);
            ov = gamma_modulation(sh_x, sh_y, x0, y0, ov);
            ov = blue_modulation(sh_x, sh_y, sh_b, x0, y0, ov);
            const float qval{expf(ov) * mul_pb + add_pb};
            int qi{static_cast<int>(qval * inv_global_scale + 0.5f)};
            qi = qi < 1 ? 1 : (qi > K_QUANT_MAX ? K_QUANT_MAX : qi);
            sh_q = qi;
            quant_field[gby * bw + gbx] = qi;
        }
        __syncthreads();

        const float ac_scale{static_cast<float>(sh_q) * gsf};
        const std::size_t blk{gby * bw + gbx};

        // Forward DCT8 + quantize, per channel. Thread (fx, fy) owns coefficient
        // k = fx*8 + fy (libjxl raster layout).
        float acc_y{0.0f};
        float acc_x{0.0f};
        float acc_b{0.0f};
        for (int y{0}; y < 8; ++y) {
            const float ay{DCT_A[fy * 8 + y]};
            for (int x{0}; x < 8; ++x) {
                const float axy{ay * DCT_A[fx * 8 + x]};
                const int s{(y0 + y) * PAD + x0 + x};
                acc_y += axy * sh_y[s];
                acc_x += axy * sh_x[s];
                acc_b += axy * sh_b[s];
            }
        }

        const std::int32_t qy{
            static_cast<std::int32_t>(rintf(acc_y * quant_factor(1, k, ac_scale, dc_scale)))};
        q[plane + blk * 64 + k] = qy;
        q[blk * 64 + k] =
            static_cast<std::int32_t>(rintf(acc_x * quant_factor(0, k, ac_scale, dc_scale)));
        const float roundtrip_y{static_cast<float>(qy) / quant_factor(1, k, ac_scale, dc_scale)};
        const float residual{acc_b - Y_TO_B_RATIO_D * roundtrip_y};
        q[2 * plane + blk * 64 + k] =
            static_cast<std::int32_t>(rintf(residual * quant_factor(2, k, ac_scale, dc_scale)));
        __syncthreads();
    }
}

}  // namespace

bool encode_frontend(const std::uint8_t* luma, std::size_t luma_pitch, const std::uint8_t* chroma,
                     std::size_t chroma_pitch, std::size_t width, std::size_t height, float distance,
                     std::int32_t* q, std::int32_t* quant_field) {
    ensure_constants();

    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};

    const QuantCalibration cal{calibrate_quant(distance)};
    const float gsf{cal.global_scale_float};
    const float dc_scale{cal.global_scale_float * static_cast<float>(cal.quant_dc)};
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

    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypePitch2D;
    resource.res.pitch2D.devPtr = const_cast<std::uint8_t*>(chroma);
    resource.res.pitch2D.desc = cudaCreateChannelDesc<uchar2>();
    resource.res.pitch2D.width = width / 2;
    resource.res.pitch2D.height = height / 2;
    resource.res.pitch2D.pitchInBytes = chroma_pitch;

    cudaTextureDesc texture{};
    texture.addressMode[0] = cudaAddressModeClamp;
    texture.addressMode[1] = cudaAddressModeClamp;
    texture.filterMode = cudaFilterModeLinear;
    texture.readMode = cudaReadModeNormalizedFloat;
    texture.normalizedCoords = 0;

    cudaTextureObject_t chroma_tex{};
    if (cudaCreateTextureObject(&chroma_tex, &resource, &texture, nullptr) != cudaSuccess) {
        return false;
    }

    const dim3 block{8, 8};
    const dim3 grid{static_cast<unsigned int>((bw + TB - 1) / TB),
                    static_cast<unsigned int>((bh + TB - 1) / TB)};
    frontend_fused_kernel<<<grid, block>>>(luma, luma_pitch, chroma_tex, width, height, bw, bh, gsf,
                                           dc_scale, km0, km1, km2, km3, mul_pb, add_pb,
                                           cal.inv_global_scale, q, quant_field);

    const cudaError_t launch{cudaGetLastError()};
    const cudaError_t sync{cudaDeviceSynchronize()};
    cudaDestroyTextureObject(chroma_tex);
    return launch == cudaSuccess && sync == cudaSuccess;
}

}  // namespace cujpegxl
