// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frontend_transform.h"

#include <cmath>

#include <cuda_runtime.h>

#include "transform_select.h"
#include "vardct_layout.h"
#include "xyb.h"

namespace cujpegxl {
namespace {

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
    if (cudaMalloc(&xyb, 3 * plane * sizeof(float)) != cudaSuccess) {
        return false;
    }

    bool ok{nv12_to_xyb(luma, luma_pitch, chroma, chroma_pitch, width, height, xyb)};
    ok = ok && select_transforms(xyb + plane, width, height, distance, acs);  // Y plane
    ok = ok && variable_forward_dct(xyb, width, height, acs, coeffs);

    cudaFree(xyb);
    return ok;
}

}  // namespace cujpegxl
