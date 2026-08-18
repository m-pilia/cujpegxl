// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frontend_transform.h"

#include <cmath>

#include <cuda_runtime.h>

#include "gaborish.h"
#include "transform_select.h"
#include "vardct_layout.h"
#include "xyb.h"

namespace cujpegxl {
namespace {

// GaborishInverse 5x5 sharpening of one pixel with image-border clamping.
__device__ inline float gaborish_at(const float* __restrict__ p, std::size_t width,
                                    std::size_t height, long x, long y) {
    const long w{static_cast<long>(width)};
    const long h{static_cast<long>(height)};
    auto at = [&](long dx, long dy) -> float {
        const long xx{x + dx < 0 ? 0 : (x + dx >= w ? w - 1 : x + dx)};
        const long yy{y + dy < 0 ? 0 : (y + dy >= h ? h - 1 : y + dy)};
        return p[yy * w + xx];
    };
    const float e{at(-1, 0) + at(1, 0) + at(0, -1) + at(0, 1)};
    const float r2{at(-2, 0) + at(2, 0) + at(0, -2) + at(0, 2)};
    const float dg{at(-1, -1) + at(1, -1) + at(-1, 1) + at(1, 1)};
    const float l8{at(-2, -1) + at(2, -1) + at(-2, 1) + at(2, 1) + at(-1, -2) + at(1, -2) +
                   at(-1, 2) + at(1, 2)};
    const float d4{at(-2, -2) + at(2, -2) + at(-2, 2) + at(2, 2)};
    return GAB_WC * at(0, 0) + GAB_WR * e + GAB_WR2 * r2 + GAB_WD * dg + GAB_WL * l8 + GAB_WD2 * d4;
}

// Full-image gaborish-inverse pre-sharpening of the three XYB planes.
__global__ void gaborish_kernel(const float* __restrict__ xyb, std::size_t width,
                                std::size_t height, float* __restrict__ out) {
    const std::size_t idx{static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x};
    const std::size_t plane{width * height};
    if (idx >= 3 * plane) {
        return;
    }
    const std::size_t c{idx / plane};
    const std::size_t p{idx % plane};
    out[idx] = gaborish_at(xyb + c * plane, width, height, static_cast<long>(p % width),
                           static_cast<long>(p / width));
}

// Block positions per 32x32 selection region, per side. Transform selection
// (decide_region) partitions the frame into 4x4-block regions, each resolving to
// a single 32-block or quads of 16/8-blocks whose covered blocks never cross the
// region boundary, so one CUDA block per region owns a self-contained set of
// first-blocks.
constexpr int VDCT_REGION = 4;

// One CUDA block per 32x32 selection region. Threads cooperate on each
// first-block in the region: the separable DCT basis and the row-pass results
// live in shared memory and are shared across all coefficients and channels,
// replacing the former one-thread-per-block kernel's per-thread cosf basis
// recompute and 8 KB of spilled local arrays. The per-coefficient arithmetic
// (basis fill order, row/column summation order) is unchanged, so the emitted
// coefficients are bit-identical. Coefficients match forward_dctN's transposed-
// raster layout coeff[fx*N+fy] and scatter as FP16 across covered-block slots.
__global__ void variable_forward_dct_kernel(const float* __restrict__ xyb, std::size_t width,
                                            std::size_t height, std::size_t bw, std::size_t bh,
                                            std::size_t rbw, std::size_t rbh,
                                            const std::int8_t* __restrict__ acs,
                                            __half* __restrict__ coeffs) {
    const std::size_t region{static_cast<std::size_t>(blockIdx.x)};
    if (region >= rbw * rbh) {
        return;
    }
    const std::size_t rbx{(region % rbw) * VDCT_REGION};
    const std::size_t rby{(region / rbw) * VDCT_REGION};

    __shared__ float basis[32 * 32];
    __shared__ float rows[32 * 32];

    const int tid{static_cast<int>(threadIdx.x)};
    const int nthreads{static_cast<int>(blockDim.x)};
    const std::size_t plane{width * height};
    const std::size_t cplane{bw * bh * COEFFS_PER_BLOCK};

    int cur_basis_n{0};
    for (int p{0}; p < VDCT_REGION * VDCT_REGION; ++p) {
        const std::size_t bx{rbx + static_cast<std::size_t>(p % VDCT_REGION)};
        const std::size_t by{rby + static_cast<std::size_t>(p / VDCT_REGION)};
        if (bx >= bw || by >= bh) {
            continue;
        }
        const int side{acs == nullptr ? 8 : acs[by * bw + bx]};
        if (side == ACS_COVERED) {
            continue;
        }
        const int n{side};

        __syncthreads();
        if (n != cur_basis_n) {
            for (int e{tid}; e < n * n; e += nthreads) {
                const int k{e / n};
                const int t{e % n};
                const float g{k == 0 ? 1.0f / n : 1.4142135623730951f / n};
                basis[e] = g * cosf(3.14159265358979323846f * (t + 0.5f) * k / n);
            }
            cur_basis_n = n;
            __syncthreads();
        }

        const std::size_t px0{bx * 8};
        const std::size_t py0{by * 8};
        for (int c{0}; c < 3; ++c) {
            const float* src{xyb + c * plane};
            __half* dst{coeffs + c * cplane};
            // Row pass: rows[fx*n + r] = sum_col basis[fx][col] * pixel[r][col].
            for (int e{tid}; e < n * n; e += nthreads) {
                const int fx{e / n};
                const int r{e % n};
                float s{0.0f};
                const float* prow{src + (py0 + r) * width + px0};
                for (int col{0}; col < n; ++col) {
                    s += basis[fx * n + col] * prow[col];
                }
                rows[e] = s;
            }
            __syncthreads();
            // Column pass: coeff[fx*n + fy] = sum_r basis[fy][r] * rows[fx][r].
            for (int e{tid}; e < n * n; e += nthreads) {
                const int fx{e / n};
                const int fy{e % n};
                float coeff{0.0f};
                for (int r{0}; r < n; ++r) {
                    coeff += basis[fy * n + r] * rows[fx * n + r];
                }
                dst[covered_plane_slot(n, bx, by, bw, static_cast<std::size_t>(e))] =
                    __float2half(coeff);
            }
            __syncthreads();
        }
    }
}

}  // namespace

bool variable_forward_dct(const float* xyb, std::size_t width, std::size_t height,
                          const std::int8_t* acs, __half* coeffs) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t rbw{(bw + VDCT_REGION - 1) / VDCT_REGION};
    const std::size_t rbh{(bh + VDCT_REGION - 1) / VDCT_REGION};
    const unsigned int threads{256};
    const unsigned int blocks{static_cast<unsigned int>(rbw * rbh)};
    variable_forward_dct_kernel<<<blocks, threads>>>(xyb, width, height, bw, bh, rbw, rbh, acs,
                                                     coeffs);
    const cudaError_t launch{cudaGetLastError()};
    const cudaError_t sync{cudaDeviceSynchronize()};
    return launch == cudaSuccess && sync == cudaSuccess;
}

bool frontend_transform(const std::uint8_t* luma, std::size_t luma_pitch,
                        const std::uint8_t* chroma, std::size_t chroma_pitch, std::size_t width,
                        std::size_t height, float distance, __half* coeffs, std::int8_t* acs) {
    const std::size_t plane{width * height};
    float* xyb{nullptr};
    float* sharp{nullptr};
    if (cudaMallocAsync(&xyb, 3 * plane * sizeof(float), 0) != cudaSuccess) {
        return false;
    }
    if (cudaMallocAsync(&sharp, 3 * plane * sizeof(float), 0) != cudaSuccess) {
        cudaFreeAsync(xyb, 0);
        return false;
    }

    bool ok{nv12_to_xyb(luma, luma_pitch, chroma, chroma_pitch, width, height, xyb)};
    if (ok) {
        // Gaborish-inverse pre-sharpen the opsin before selection and the DCT,
        // cancelling the decoder's default gaborish smoothing.
        const unsigned int threads{256};
        const unsigned int blocks{static_cast<unsigned int>((3 * plane + threads - 1) / threads)};
        gaborish_kernel<<<blocks, threads>>>(xyb, width, height, sharp);
        ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
    }
    ok = ok && select_transforms(sharp + plane, width, height, distance, acs);  // Y plane
    ok = ok && variable_forward_dct(sharp, width, height, acs, coeffs);

    cudaFreeAsync(xyb, 0);
    cudaFreeAsync(sharp, 0);
    return ok;
}

}  // namespace cujpegxl
