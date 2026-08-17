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
// candidate sides, precomputed once. Filling them on the device with the exact
// double-precision formula block_ac_cost uses keeps the selection costs, and
// hence the chosen ACS map, bit-identical to the host reference while removing
// the per-block cos() basis recompute that dominated the former kernel.
__device__ double g_basis8[8 * 8];
__device__ double g_basis16[16 * 16];
__device__ double g_basis32[32 * 32];

__device__ void fill_basis(double* basis, int n, int i) {
    if (i >= n * n) {
        return;
    }
    const int k{i / n};
    const int t{i % n};
    const double g{k == 0 ? 1.0 / n : 1.4142135623730950488 / n};
    basis[i] = g * cos(3.14159265358979323846 * (t + 0.5) * k / n);
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

__device__ const double* basis_for(int n) {
    return n == 32 ? g_basis32 : (n == 16 ? g_basis16 : g_basis8);
}

// Device counterpart of block_ac_cost: identical basis values, row/column
// summation order, and dist/rate accumulation order, but sourcing the basis from
// the precomputed table instead of recomputing it per call.
__device__ double dev_block_ac_cost(const float* __restrict__ y, std::size_t width,
                                    std::size_t px0, std::size_t py0, int n, double qgsf,
                                    double lambda) {
    const double* basis{basis_for(n)};
    double rows[32 * 32];
    for (int fx{0}; fx < n; ++fx) {
        for (int r{0}; r < n; ++r) {
            double s{0.0};
            const float* prow{y + (py0 + r) * width + px0};
            for (int c{0}; c < n; ++c) {
                s += basis[fx * n + c] * prow[c];
            }
            rows[fx * n + r] = s;
        }
    }

    const int m{n / 8};
    double dist{0.0};
    double rate{0.0};
    for (int fx{0}; fx < n; ++fx) {
        for (int fy{0}; fy < n; ++fy) {
            if (fx < m && fy < m) {
                continue;
            }
            double coeff{0.0};
            for (int r{0}; r < n; ++r) {
                coeff += basis[fy * n + r] * rows[fx * n + r];
            }
            int sx{static_cast<int>(lround(fx * 8.0 / n))};
            int sy{static_cast<int>(lround(fy * 8.0 / n))};
            sx = sx > 7 ? 7 : sx;
            sy = sy > 7 ? 7 : sy;
            const double w{DCT8_DEQUANT_WEIGHTS[1][sx * 8 + sy]};
            const double q{rint(coeff * qgsf / w)};
            const double e{q * w / qgsf - coeff};
            dist += e * e;
            if (q != 0.0) {
                rate += 1.0 + log2(1.0 + fabs(q));
            }
        }
    }
    return static_cast<double>(n * n) * dist + lambda * rate;
}

// Device counterpart of decide_region: same candidate order, tie-break epsilons,
// and ACS writes, so device and host produce byte-identical ACS.
__device__ void decide_region_dev(const float* __restrict__ y, std::size_t width, std::size_t bw,
                                  std::size_t bh, std::size_t rbx, std::size_t rby, double qgsf,
                                  double lambda, std::int8_t* __restrict__ acs) {
    constexpr double INF{1.0e300};

    const bool fit32{rbx + 4 <= bw && rby + 4 <= bh};
    const double cost32{fit32 ? dev_block_ac_cost(y, width, rbx * 8, rby * 8, 32, qgsf, lambda)
                              : INF};

    double split_cost{0.0};
    int quad_choice[2][2];
    for (int qy{0}; qy < 2; ++qy) {
        for (int qx{0}; qx < 2; ++qx) {
            const std::size_t bx{rbx + 2 * static_cast<std::size_t>(qx)};
            const std::size_t by{rby + 2 * static_cast<std::size_t>(qy)};
            double sum8{0.0};
            bool any8{false};
            for (int dy{0}; dy < 2; ++dy) {
                for (int dx{0}; dx < 2; ++dx) {
                    const std::size_t bbx{bx + dx};
                    const std::size_t bby{by + dy};
                    if (bbx < bw && bby < bh) {
                        sum8 += dev_block_ac_cost(y, width, bbx * 8, bby * 8, 8, qgsf, lambda);
                        any8 = true;
                    }
                }
            }
            if (!any8) {
                quad_choice[qy][qx] = -1;
                continue;
            }
            const bool fit16{bx + 2 <= bw && by + 2 <= bh};
            const double cost16{fit16 ? dev_block_ac_cost(y, width, bx * 8, by * 8, 16, qgsf, lambda)
                                      : INF};
            if (fit16 && cost16 <= sum8 + TRANSFORM_SELECT_SPLIT_EPS) {
                quad_choice[qy][qx] = 16;
                split_cost += cost16;
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

// One thread per 32x32 region. Regions are independent, so this keeps the RD
// reduction sequential (byte-identical) while spreading regions across the grid.
__global__ void select_transforms_kernel(const float* __restrict__ y, std::size_t width,
                                         std::size_t bw, std::size_t bh, std::size_t rbw,
                                         std::size_t rbh, double qgsf, double lambda,
                                         std::int8_t* __restrict__ acs) {
    const std::size_t region{static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x};
    if (region >= rbw * rbh) {
        return;
    }
    const std::size_t rbx{(region % rbw) * 4};
    const std::size_t rby{(region / rbw) * 4};
    decide_region_dev(y, width, bw, bh, rbx, rby, qgsf, lambda, acs);
}

}  // namespace

bool select_transforms(const float* y, std::size_t width, std::size_t height, float distance,
                       std::int8_t* acs) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t rbw{region_grid(bw)};
    const std::size_t rbh{region_grid(bh)};

    const QuantCalibration cal{calibrate_quant(distance)};
    const double qgsf{static_cast<double>(cal.raw_quant_field) * cal.global_scale_float};

    ensure_select_basis();
    const unsigned int threads{64};
    const unsigned int blocks{static_cast<unsigned int>((rbw * rbh + threads - 1) / threads)};
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
    const double qgsf{static_cast<double>(cal.raw_quant_field) * cal.global_scale_float};

    for (std::size_t ry{0}; ry < rbh; ++ry) {
        for (std::size_t rx{0}; rx < rbw; ++rx) {
            decide_region(y, width, bw, bh, rx * 4, ry * 4, qgsf, TRANSFORM_SELECT_LAMBDA, acs);
        }
    }
}

}  // namespace cujpegxl
