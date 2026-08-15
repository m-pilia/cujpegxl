// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frontend_fused.h"

#include "device_allocation_cache.h"

#include <cmath>
#include <mutex>

#include <cuda_runtime.h>

#include "adaptive_quant_impl.cuh"
#include "entropy.h"
#include "quant_calibration.h"
#include "quant_weights_dct8.h"
#include "xyb_impl.cuh"

namespace cujpegxl {
namespace {

// Tile geometry: one CUDA block processes TB x TB image blocks. The shared XYB
// region carries a two-pixel halo (PAD): the inner pixel feeds the
// adaptive-quant Laplacian, and the full two pixels feed the 5x5
// gaborish-inverse pre-sharpening applied before the DCT. The pre-erosion map
// is at 4x-subsampled resolution (CELLS x CELLS cells).
constexpr int HALO = 2;
constexpr int TB = 4;
constexpr int TILE_PX = TB * 8;
constexpr int PAD = TILE_PX + 2 * HALO;
constexpr int CELLS = 2 * TB;

// GaborishInverse (libjxl enc_gaborish.cc) normalized 5x5 symmetric sharpening,
// evaluated at the encoder's default per-channel strength mul = 1 so all three
// XYB planes share one kernel. Applied to the opsin before the DCT, it cancels
// the decoder's default gaborish smoothing (loop_filter all_default), which
// otherwise blurs away high-frequency detail no quantizer step can recover.
// Weights follow the WeightsSymmetric5 quadrant layout {c, r, R, d, L, D}.
constexpr float GAB_K0 = -0.09495815671340026f;
constexpr float GAB_K1 = -0.041031725066768575f;
constexpr float GAB_K2 = 0.013710004822696948f;
constexpr float GAB_K3 = 0.006510206083837737f;
constexpr float GAB_K4 = -0.0014789063378272242f;
constexpr float GAB_NM = 1.0f / (1.0f + 4.0f * (GAB_K0 + GAB_K1 + GAB_K2 + GAB_K4 + 2.0f * GAB_K3));
constexpr float GAB_WC = GAB_NM;            // center (0,0)
constexpr float GAB_WR = GAB_NM * GAB_K0;   // r: (1,0),(0,1)
constexpr float GAB_WR2 = GAB_NM * GAB_K2;  // R: (2,0),(0,2)
constexpr float GAB_WD = GAB_NM * GAB_K1;   // d: (1,1)
constexpr float GAB_WL = GAB_NM * GAB_K3;   // L: (2,1),(1,2)
constexpr float GAB_WD2 = GAB_NM * GAB_K4;  // D: (2,2)

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

// Full-mask warp sum reduction. Both warps of the 64-thread block are fully
// populated here, so every lane participates.
__device__ inline float warp_reduce_sum(float v) {
    for (int off{16}; off > 0; off >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

// GaborishInverse 5x5 symmetric filter at PAD-space index idx (stride PAD),
// exploiting the {c, r, R, d, L, D} tap symmetry. Requires a two-pixel halo
// around idx.
__device__ inline float gaborish_inverse(const float* __restrict__ p, int idx) {
    const float e{p[idx - 1] + p[idx + 1] + p[idx - PAD] + p[idx + PAD]};
    const float r2{p[idx - 2] + p[idx + 2] + p[idx - 2 * PAD] + p[idx + 2 * PAD]};
    const float dg{p[idx - PAD - 1] + p[idx - PAD + 1] + p[idx + PAD - 1] + p[idx + PAD + 1]};
    const float l8{p[idx - 2 * PAD - 1] + p[idx - 2 * PAD + 1] + p[idx + 2 * PAD - 1] +
                   p[idx + 2 * PAD + 1] + p[idx - PAD - 2] + p[idx - PAD + 2] + p[idx + PAD - 2] +
                   p[idx + PAD + 2]};
    const float d4{p[idx - 2 * PAD - 2] + p[idx - 2 * PAD + 2] + p[idx + 2 * PAD - 2] +
                   p[idx + 2 * PAD + 2]};
    return GAB_WC * p[idx] + GAB_WR * e + GAB_WR2 * r2 + GAB_WD * dg + GAB_WL * l8 + GAB_WD2 * d4;
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
                                      std::int16_t* __restrict__ ac, std::int32_t* __restrict__ dc,
                                      std::int32_t* __restrict__ quant_field) {
    __shared__ float sh_y[PAD * PAD];
    __shared__ float sh_x[PAD * PAD];
    __shared__ float sh_b[PAD * PAD];
    __shared__ float sh_sx[64];  // gaborish-inverse sharpened block (X/Y/B)
    __shared__ float sh_sy[64];
    __shared__ float sh_sb[64];
    __shared__ float sh_pre[CELLS * CELLS];
    __shared__ float sh_red[2][4];
    __shared__ int sh_q;

    const std::size_t tbx0{static_cast<std::size_t>(blockIdx.x) * TB};
    const std::size_t tby0{static_cast<std::size_t>(blockIdx.y) * TB};
    const std::size_t px0{tbx0 * 8};
    const std::size_t py0{tby0 * 8};
    const int tid{static_cast<int>(threadIdx.y) * 8 + static_cast<int>(threadIdx.x)};
    const std::size_t nblk{bw * bh};

    // Load the padded tile and compute XYB into shared memory.
    for (int i{tid}; i < PAD * PAD; i += 64) {
        const int lx{i % PAD};
        const int ly{i / PAD};
        long gx{static_cast<long>(px0) + lx - HALO};
        long gy{static_cast<long>(py0) + ly - HALO};
        gx = gx < 0 ? 0 : (gx >= static_cast<long>(width) ? static_cast<long>(width) - 1 : gx);
        gy = gy < 0 ? 0 : (gy >= static_cast<long>(height) ? static_cast<long>(height) - 1 : gy);
        const float yv{luma[static_cast<std::size_t>(gy) * luma_pitch + gx] * (1.0f / 255.0f)};
        const float2 c{tex2D<float2>(chroma_tex, (gx + 0.5f) * 0.5f, (gy + 0.5f) * 0.5f)};
        float xo{};
        float yo{};
        float bo{};
        nv12_pixel_to_xyb(yv, c.x, c.y, xo, yo, bo);
        sh_y[i] = yo;
        sh_x[i] = xo;
        sh_b[i] = bo;
    }
    __syncthreads();

    // Pre-erosion cells: 0.25 * sum of the 4x4 pixel diffs (Laplacian of Y).
    for (int ci{tid}; ci < CELLS * CELLS; ci += 64) {
        const int ccx{ci % CELLS};
        const int ccy{ci / CELLS};
        float sum{0.0f};
        for (int dy{0}; dy < 4; ++dy) {
            for (int dx{0}; dx < 4; ++dx) {
                const int lx{ccx * 4 + dx + HALO};
                const int ly{ccy * 4 + dy + HALO};
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
    // Separable DCT ownership: thread (fx, fy) produces coefficient with
    // x-frequency fy and y-frequency fx, i.e. k = fy*8 + fx (x-frequency major,
    // libjxl raster). This transpose keeps each coefficient's row-pass inputs
    // (the 8 threads sharing fy) within a single warp for the shuffle gather.
    const int k{fy * 8 + fx};

    for (int lb{0}; lb < TB * TB; ++lb) {
        const int lbx{lb % TB};
        const int lby{lb / TB};
        const std::size_t gbx{tbx0 + lbx};
        const std::size_t gby{tby0 + lby};
        if (gbx >= bw || gby >= bh) {
            continue;
        }
        const int x0{lbx * 8 + HALO};  // shared interior origin (+halo)
        const int y0{lby * 8 + HALO};

        // Adaptive quant integer: each thread reduces its own pixel's modulation
        // contributions across the 64-thread block; thread 0 combines the
        // reduced sums and broadcasts the quant integer via shared memory. Every
        // modulation term is an additive reduction over the block's 64 pixels,
        // so the per-pixel partials can be summed in parallel.
        {
            const int idx{(y0 + fy) * PAD + x0 + fx};
            const float yv{sh_y[idx]};
            const float xv{sh_x[idx]};
            const float bv{sh_b[idx]};

            constexpr float valmin{0.0206f};
            float hf_part{0.0f};
            if (fx < 7) {
                hf_part += fminf(valmin, fabsf(yv - sh_y[idx + 1]));
            }
            if (fy < 7) {
                hf_part += fminf(valmin, fabsf(yv - sh_y[idx + PAD]));
            }

            constexpr float kBias{0.16f};
            const float iny{yv + kBias};
            const float gamma_part{ratio_cbrt_gamma<true>(iny - xv) +
                                   ratio_cbrt_gamma<true>(iny + xv)};

            constexpr float kLimit{0.027121074570634722f};
            constexpr float kOffset{0.084381641171960495f};
            const float pye{yv + kOffset + fabsf(xv)};
            const float blue_part{bv > pye ? fminf(bv - pye, kLimit) : 0.0f};

            float er_part{0.0f};
            if (tid < 4) {
                er_part = erosion_cell(sh_pre, 2 * lbx + (tid & 1), 2 * lby + (tid >> 1), km0, km1,
                                       km2, km3);
            }

            const float hf_red{warp_reduce_sum(hf_part)};
            const float gamma_red{warp_reduce_sum(gamma_part)};
            const float blue_red{warp_reduce_sum(blue_part)};
            const float er_red{warp_reduce_sum(er_part)};
            if ((tid & 31) == 0) {
                const int w{tid >> 5};
                sh_red[w][0] = hf_red;
                sh_red[w][1] = gamma_red;
                sh_red[w][2] = blue_red;
                sh_red[w][3] = er_red;
            }
        }
        __syncthreads();
        if (tid == 0) {
            const float hf_sum{sh_red[0][0] + sh_red[1][0]};
            const float gamma_overall{(sh_red[0][1] + sh_red[1][1]) * (0.5f / 64.0f)};
            float blue_sum{sh_red[0][2] + sh_red[1][2]};
            const float aq{sh_red[0][3] + sh_red[1][3]};

            constexpr float kLimit{0.027121074570634722f};
            if (blue_sum >= 32.0f * kLimit) {
                blue_sum = 64.0f * kLimit - blue_sum;
            }
            constexpr float kMaxLimit{15.398788439047934f};
            if (blue_sum >= kMaxLimit * kLimit) {
                blue_sum = kMaxLimit * kLimit;
            }

            float ov{compute_mask(aq)};
            ov += hf_sum * -0.38f + 0.42f;
            ov += 0.1005613337192697f * log2f(gamma_overall);
            ov += blue_sum * 0.14207000358439159f;

            const float qval{expf(ov) * mul_pb + add_pb};
            int qi{static_cast<int>(qval * inv_global_scale + 0.5f)};
            qi = qi < 1 ? 1 : (qi > K_QUANT_MAX ? K_QUANT_MAX : qi);
            sh_q = qi;
            quant_field[gby * bw + gbx] = qi;
        }
        __syncthreads();

        // GaborishInverse pre-sharpening: replace this block's 8x8 opsin with
        // its 5x5-sharpened values before the DCT, cancelling the decoder's
        // default gaborish smoothing. The AQ field above intentionally used the
        // unsharpened opsin, matching libjxl's ordering (InitialQuantField
        // precedes GaborishInverse).
        {
            const int idx{(y0 + fy) * PAD + x0 + fx};
            sh_sx[fy * 8 + fx] = gaborish_inverse(sh_x, idx);
            sh_sy[fy * 8 + fx] = gaborish_inverse(sh_y, idx);
            sh_sb[fy * 8 + fx] = gaborish_inverse(sh_b, idx);
        }
        __syncthreads();

        const float ac_scale{static_cast<float>(sh_q) * gsf};
        const std::size_t blk{gby * bw + gbx};

        // Separable forward DCT8 + quantize, per channel (see the k definition
        // above). Row pass: thread (fx, fy) reduces spatial row fx against
        // x-frequency fy, producing R[fy][fx]. Column pass: warp shuffles gather
        // R[fy][0..7] (the 8 lanes sharing fy) and reduce against y-frequency fx.
        // This replaces the 64-MAC dense transform with 16 MACs + 8 shuffles.
        float ry{0.0f};
        float rx{0.0f};
        float rb{0.0f};
        for (int x{0}; x < 8; ++x) {
            const float a{DCT_A[fy * 8 + x]};
            const int s{fx * 8 + x};
            ry += a * sh_sy[s];
            rx += a * sh_sx[s];
            rb += a * sh_sb[s];
        }
        float acc_y{0.0f};
        float acc_x{0.0f};
        float acc_b{0.0f};
        const int base_lane{(fy & 3) * 8};
        for (int s{0}; s < 8; ++s) {
            const float a{DCT_A[fx * 8 + s]};
            acc_y += a * __shfl_sync(0xffffffffu, ry, base_lane + s, 32);
            acc_x += a * __shfl_sync(0xffffffffu, rx, base_lane + s, 32);
            acc_b += a * __shfl_sync(0xffffffffu, rb, base_lane + s, 32);
        }

        const std::int32_t qx{
            static_cast<std::int32_t>(rintf(acc_x * quant_factor(0, k, ac_scale, dc_scale)))};
        const std::int32_t qy{
            static_cast<std::int32_t>(rintf(acc_y * quant_factor(1, k, ac_scale, dc_scale)))};
        const float roundtrip_y{static_cast<float>(qy) / quant_factor(1, k, ac_scale, dc_scale)};
        const float residual{acc_b - Y_TO_B_RATIO_D * roundtrip_y};
        const std::int32_t qb{
            static_cast<std::int32_t>(rintf(residual * quant_factor(2, k, ac_scale, dc_scale)))};

        // Split storage: DC (k == 0) stays int32 in the compact DC buffer; the 63
        // AC coefficients narrow to int16 in the packed AC buffer (slot k-1).
        if (k == 0) {
            dc[blk] = qx;
            dc[nblk + blk] = qy;
            dc[2 * nblk + blk] = qb;
        } else {
            const std::size_t ac_off{blk * AC_COEFFS_PER_BLOCK + (k - 1)};
            ac[ac_off] = static_cast<std::int16_t>(qx);
            ac[nblk * AC_COEFFS_PER_BLOCK + ac_off] = static_cast<std::int16_t>(qy);
            ac[2 * nblk * AC_COEFFS_PER_BLOCK + ac_off] = static_cast<std::int16_t>(qb);
        }
        __syncthreads();
    }
}

}  // namespace

bool encode_frontend(const std::uint8_t* luma, std::size_t luma_pitch, const std::uint8_t* chroma,
                     std::size_t chroma_pitch, std::size_t width, std::size_t height,
                     float distance, std::int16_t* ac, std::int32_t* dc,
                     std::int32_t* quant_field) {
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
    frontend_fused_kernel<<<grid, block, 0, encoder_stream()>>>(
        luma, luma_pitch, chroma_tex, width, height, bw, bh, gsf, dc_scale, km0, km1, km2, km3,
        mul_pb, add_pb, cal.inv_global_scale, ac, dc, quant_field);

    const cudaError_t launch{cudaGetLastError()};
    const cudaError_t sync{encoder_stream_synchronize()};
    cudaDestroyTextureObject(chroma_tex);
    return launch == cudaSuccess && sync == cudaSuccess;
}

}  // namespace cujpegxl
