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

// One thread per 8x8 block position. A first-block computes its side*side DCT of
// each XYB channel (separable, matching forward_dctN's transposed-raster layout
// coeff[fx*N+fy]) and scatters the coefficients as FP16 across its covered
// blocks' slots; covered blocks do nothing. Correctness-first (per-block serial);
// fusion and parallelism within a block are T9.
__global__ void variable_forward_dct_kernel(const float* __restrict__ xyb, std::size_t width,
                                            std::size_t height, std::size_t bw, std::size_t bh,
                                            const std::int8_t* __restrict__ acs,
                                            __half* __restrict__ coeffs) {
    const std::size_t blk{static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x};
    if (blk >= bw * bh) {
        return;
    }
    const int side{acs == nullptr ? 8 : acs[blk]};
    if (side == ACS_COVERED) {
        return;
    }
    const std::size_t bx{blk % bw};
    const std::size_t by{blk / bw};
    const int n{side};

    float basis[32 * 32];
    for (int k{0}; k < n; ++k) {
        const float g{k == 0 ? 1.0f / n : 1.4142135623730951f / n};
        for (int t{0}; t < n; ++t) {
            basis[k * n + t] = g * cosf(3.14159265358979323846f * (t + 0.5f) * k / n);
        }
    }

    const std::size_t plane{width * height};
    const std::size_t cplane{bw * bh * COEFFS_PER_BLOCK};
    const std::size_t px0{bx * 8};
    const std::size_t py0{by * 8};
    for (int c{0}; c < 3; ++c) {
        const float* src{xyb + c * plane};
        // Row pass: rows[fx*n + r] = sum_col basis[fx][col] * pixel[r][col].
        float rows[32 * 32];
        for (int fx{0}; fx < n; ++fx) {
            for (int r{0}; r < n; ++r) {
                float s{0.0f};
                const float* prow{src + (py0 + r) * width + px0};
                for (int col{0}; col < n; ++col) {
                    s += basis[fx * n + col] * prow[col];
                }
                rows[fx * n + r] = s;
            }
        }
        __half* dst{coeffs + c * cplane};
        for (int fx{0}; fx < n; ++fx) {
            for (int fy{0}; fy < n; ++fy) {
                float coeff{0.0f};
                for (int r{0}; r < n; ++r) {
                    coeff += basis[fy * n + r] * rows[fx * n + r];
                }
                const std::size_t raw{static_cast<std::size_t>(fx) * n + fy};
                dst[covered_plane_slot(n, bx, by, bw, raw)] = __float2half(coeff);
            }
        }
    }
}

}  // namespace

bool variable_forward_dct(const float* xyb, std::size_t width, std::size_t height,
                          const std::int8_t* acs, __half* coeffs) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const unsigned int threads{128};
    const unsigned int blocks{static_cast<unsigned int>((bw * bh + threads - 1) / threads)};
    variable_forward_dct_kernel<<<blocks, threads>>>(xyb, width, height, bw, bh, acs, coeffs);
    const cudaError_t launch{cudaGetLastError()};
    const cudaError_t sync{cudaDeviceSynchronize()};
    return launch == cudaSuccess && sync == cudaSuccess;
}

bool frontend_transform_m3(const std::uint8_t* luma, std::size_t luma_pitch,
                           const std::uint8_t* chroma, std::size_t chroma_pitch, std::size_t width,
                           std::size_t height, float distance, __half* coeffs, std::int8_t* acs) {
    const std::size_t plane{width * height};
    float* xyb{nullptr};
    float* sharp{nullptr};
    if (cudaMalloc(&xyb, 3 * plane * sizeof(float)) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&sharp, 3 * plane * sizeof(float)) != cudaSuccess) {
        cudaFree(xyb);
        return false;
    }

    bool ok{nv12_to_xyb(luma, luma_pitch, chroma, chroma_pitch, width, height, xyb)};
    if (ok) {
        // Gaborish-inverse pre-sharpen the opsin before selection and the DCT,
        // cancelling the decoder's default gaborish smoothing.
        const unsigned int threads{256};
        const unsigned int blocks{
            static_cast<unsigned int>((3 * plane + threads - 1) / threads)};
        gaborish_kernel<<<blocks, threads>>>(xyb, width, height, sharp);
        ok = cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
    }
    ok = ok && select_transforms(sharp + plane, width, height, distance, acs);  // Y plane
    ok = ok && variable_forward_dct(sharp, width, height, acs, coeffs);

    cudaFree(xyb);
    cudaFree(sharp);
    return ok;
}

}  // namespace cujpegxl
