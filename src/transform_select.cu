// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "transform_select.h"

#include <mutex>

#include <cuda_runtime.h>

#include "quant_calibration.h"
#include "quant_weights_dct8.h"
#include "transform_select_impl.cuh"

namespace cujpegxl {
namespace {

std::size_t region_grid(std::size_t blocks) {
    return (blocks + 3) / 4;
}

// Separable DCT-II bases (transposed-raster layout basis[k*n+t]) for the three
// candidate sides, precomputed once. The values are derived in double (matching
// block_ac_cost) and stored as float so the cooperative kernel and the host
// reference share bit-identical basis coefficients.
__device__ float g_basis8[8 * 8];
__device__ float g_basis16[16 * 16];
__device__ float g_basis32[32 * 32];

__device__ void fill_basis(float* basis, int n, int i) {
    if (i >= n * n) {
        return;
    }
    const int k{i / n};
    const int t{i % n};
    const double g{k == 0 ? 1.0 / n : 1.4142135623730950488 / n};
    basis[i] = static_cast<float>(g * cos(3.14159265358979323846 * (t + 0.5) * k / n));
}

__global__ void init_select_basis_kernel() {
    const int i{static_cast<int>(threadIdx.x)};
    fill_basis(g_basis8, 8, i);
    fill_basis(g_basis16, 16, i);
    fill_basis(g_basis32, 32, i);
}

void ensure_select_basis() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        init_select_basis_kernel<<<1, 32 * 32>>>();
        cudaDeviceSynchronize();
    });
}

__device__ const float* basis_for(int n) {
    return n == 32 ? g_basis32 : (n == 16 ? g_basis16 : g_basis8);
}

// Cooperative AC cost of one NxN block. All threads stage the basis and run the
// row pass in parallel, then fuse the column pass with quantization: each thread
// accumulates dist/rate for its strided share of coefficients and the block
// reduces the two partials with a power-of-two tree. The strided assignment and
// tree match the host block_ac_cost, so the float cost is bit-identical while the
// former one-thread-per-block matmul + serial quant is gone. The return value is
// meaningful only on thread 0. Every thread must call this with the same
// n/px0/py0 (the internal __syncthreads must not diverge).
__device__ float coop_block_ac_cost(const float* __restrict__ y, std::size_t width, std::size_t px0,
                                    std::size_t py0, int n, float qgsf, float lambda, float* basis,
                                    float* rows, float* pd, float* pr, int tid, int nthreads) {
    __syncthreads();
    const float* src_basis{basis_for(n)};
    for (int e{tid}; e < n * n; e += nthreads) {
        basis[e] = src_basis[e];
    }
    __syncthreads();

    // Row pass: rows[fx*n + r] = sum_c basis[fx][c] * pixel[r][c].
    for (int e{tid}; e < n * n; e += nthreads) {
        const int fx{e / n};
        const int r{e % n};
        float s{0.0f};
        const float* prow{y + (py0 + r) * width + px0};
        for (int c{0}; c < n; ++c) {
            s += basis[fx * n + c] * prow[c];
        }
        rows[e] = s;
    }
    __syncthreads();

    // Column pass fused with quantization: coeff[fx*n + fy] = sum_r basis[fy][r] *
    // rows[fx][r], quantized into per-thread dist/rate partials.
    const int m{n / 8};
    float dist{0.0f};
    float rate{0.0f};
    for (int e{tid}; e < n * n; e += nthreads) {
        const int fx{e / n};
        const int fy{e % n};
        if (fx < m && fy < m) {
            continue;
        }
        float coeff{0.0f};
        for (int r{0}; r < n; ++r) {
            coeff += basis[fy * n + r] * rows[fx * n + r];
        }
        int sx{static_cast<int>(lround(fx * 8.0 / n))};
        int sy{static_cast<int>(lround(fy * 8.0 / n))};
        sx = sx > 7 ? 7 : sx;
        sy = sy > 7 ? 7 : sy;
        const float w{static_cast<float>(DCT8_DEQUANT_WEIGHTS[1][sx * 8 + sy])};
        const float q{rintf(coeff * qgsf / w)};
        const float err{q * w / qgsf - coeff};
        dist += err * err;
        if (q != 0.0f) {
            rate += 1.0f + log2f(1.0f + fabsf(q));
        }
    }
    pd[tid] = dist;
    pr[tid] = rate;
    __syncthreads();
    for (int s{nthreads / 2}; s > 0; s >>= 1) {
        if (tid < s) {
            pd[tid] += pd[tid + s];
            pr[tid] += pr[tid + s];
        }
        __syncthreads();
    }
    return static_cast<float>(n * n) * pd[0] + lambda * pr[0];
}

// One CUDA block per 32x32 selection region. The block evaluates the 21 candidate
// transforms (sixteen 8x8, four 16x16, one 32x32) cooperatively, grouped by side
// to reuse the staged basis, then thread 0 runs the RD split decision on the
// collected float costs in the same order and with the same tie-break epsilons as
// the host decide_region, so device and host produce byte-identical ACS.
__global__ void select_transforms_kernel(const float* __restrict__ y, std::size_t width,
                                         std::size_t bw, std::size_t bh, std::size_t rbw,
                                         std::size_t rbh, float qgsf, float lambda,
                                         std::int8_t* __restrict__ acs) {
    const std::size_t region{static_cast<std::size_t>(blockIdx.x)};
    if (region >= rbw * rbh) {
        return;
    }
    const std::size_t rbx{(region % rbw) * 4};
    const std::size_t rby{(region / rbw) * 4};

    __shared__ float basis[32 * 32];
    __shared__ float rows[32 * 32];
    __shared__ float pd[TRANSFORM_SELECT_THREADS];
    __shared__ float pr[TRANSFORM_SELECT_THREADS];
    __shared__ float cost8[16];
    __shared__ float cost16[4];
    __shared__ float cost32;

    const int tid{static_cast<int>(threadIdx.x)};
    const int nthreads{static_cast<int>(blockDim.x)};
    constexpr float INF{1.0e30f};

    // The 8x8 costs feed both the split decision and, summed per quad, the 16x16
    // comparison, so evaluate all sixteen first (basis staged once for n=8).
    for (int idx{0}; idx < 16; ++idx) {
        const int qy{idx / 8};
        const int qx{(idx / 4) % 2};
        const int dy{(idx / 2) % 2};
        const int dx{idx % 2};
        const std::size_t bbx{rbx + 2 * static_cast<std::size_t>(qx) + dx};
        const std::size_t bby{rby + 2 * static_cast<std::size_t>(qy) + dy};
        const bool exist{bbx < bw && bby < bh};
        const float c{exist ? coop_block_ac_cost(y, width, bbx * 8, bby * 8, 8, qgsf, lambda, basis,
                                                 rows, pd, pr, tid, nthreads)
                            : INF};
        if (tid == 0) {
            cost8[idx] = c;
        }
        __syncthreads();
    }

    for (int q{0}; q < 4; ++q) {
        const int qy{q / 2};
        const int qx{q % 2};
        const std::size_t bx{rbx + 2 * static_cast<std::size_t>(qx)};
        const std::size_t by{rby + 2 * static_cast<std::size_t>(qy)};
        const bool fit16{bx + 2 <= bw && by + 2 <= bh};
        const float c{fit16 ? coop_block_ac_cost(y, width, bx * 8, by * 8, 16, qgsf, lambda, basis,
                                                 rows, pd, pr, tid, nthreads)
                            : INF};
        if (tid == 0) {
            cost16[q] = c;
        }
        __syncthreads();
    }

    const bool fit32{rbx + 4 <= bw && rby + 4 <= bh};
    const float c32{fit32 ? coop_block_ac_cost(y, width, rbx * 8, rby * 8, 32, qgsf, lambda, basis,
                                               rows, pd, pr, tid, nthreads)
                          : INF};
    if (tid != 0) {
        return;
    }
    cost32 = c32;

    float split_cost{0.0f};
    int quad_choice[2][2];
    for (int qy{0}; qy < 2; ++qy) {
        for (int qx{0}; qx < 2; ++qx) {
            const std::size_t bx{rbx + 2 * static_cast<std::size_t>(qx)};
            const std::size_t by{rby + 2 * static_cast<std::size_t>(qy)};
            float sum8{0.0f};
            bool any8{false};
            for (int dy{0}; dy < 2; ++dy) {
                for (int dx{0}; dx < 2; ++dx) {
                    const std::size_t bbx{bx + dx};
                    const std::size_t bby{by + dy};
                    if (bbx < bw && bby < bh) {
                        sum8 += cost8[qy * 8 + qx * 4 + dy * 2 + dx];
                        any8 = true;
                    }
                }
            }
            if (!any8) {
                quad_choice[qy][qx] = -1;
                continue;
            }
            const bool fit16{bx + 2 <= bw && by + 2 <= bh};
            const float c16{cost16[qy * 2 + qx]};
            if (fit16 && c16 <= sum8 + TRANSFORM_SELECT_SPLIT_EPS) {
                quad_choice[qy][qx] = 16;
                split_cost += c16;
            } else {
                quad_choice[qy][qx] = 8;
                split_cost += sum8;
            }
        }
    }

    if (fit32 && cost32 <= split_cost + TRANSFORM_SELECT_SPLIT_EPS) {
        write_first_block(acs, bw, rbx, rby, 32);
        return;
    }
    for (int qy{0}; qy < 2; ++qy) {
        for (int qx{0}; qx < 2; ++qx) {
            const int choice{quad_choice[qy][qx]};
            if (choice < 0) {
                continue;
            }
            const std::size_t bx{rbx + 2 * static_cast<std::size_t>(qx)};
            const std::size_t by{rby + 2 * static_cast<std::size_t>(qy)};
            if (choice == 16) {
                write_first_block(acs, bw, bx, by, 16);
                continue;
            }
            for (int dy{0}; dy < 2; ++dy) {
                for (int dx{0}; dx < 2; ++dx) {
                    const std::size_t bbx{bx + dx};
                    const std::size_t bby{by + dy};
                    if (bbx < bw && bby < bh) {
                        acs[bby * bw + bbx] = 8;
                    }
                }
            }
        }
    }
}

}  // namespace

bool select_transforms(const float* y, std::size_t width, std::size_t height, float distance,
                       std::int8_t* acs) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t rbw{region_grid(bw)};
    const std::size_t rbh{region_grid(bh)};

    const QuantCalibration cal{calibrate_quant(distance)};
    const float qgsf{static_cast<float>(cal.raw_quant_field) * cal.global_scale_float};

    ensure_select_basis();
    const unsigned int threads{256};
    const unsigned int blocks{static_cast<unsigned int>(rbw * rbh)};
    select_transforms_kernel<<<blocks, threads>>>(y, width, bw, bh, rbw, rbh, qgsf,
                                                  TRANSFORM_SELECT_LAMBDA, acs);

    const cudaError_t launch{cudaGetLastError()};
    const cudaError_t sync{cudaDeviceSynchronize()};
    return launch == cudaSuccess && sync == cudaSuccess;
}

void select_transforms_host(const float* y, std::size_t width, std::size_t height, float distance,
                            std::int8_t* acs) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t rbw{region_grid(bw)};
    const std::size_t rbh{region_grid(bh)};

    const QuantCalibration cal{calibrate_quant(distance)};
    const float qgsf{static_cast<float>(cal.raw_quant_field) * cal.global_scale_float};

    for (std::size_t ry{0}; ry < rbh; ++ry) {
        for (std::size_t rx{0}; rx < rbw; ++rx) {
            decide_region(y, width, bw, bh, rx * 4, ry * 4, qgsf, TRANSFORM_SELECT_LAMBDA, acs);
        }
    }
}

}  // namespace cujpegxl
