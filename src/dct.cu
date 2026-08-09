// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "dct.h"

#include <cmath>
#include <mutex>

#include <cuda_runtime.h>

namespace cujpegxl {
namespace {

// libjxl's 1D DCT8 basis: A[k][n] = g(k) * cos(pi*(n+0.5)*k/8), with
// g(0)=1/8 and g(k>0)=sqrt(2)/8. The forward 8x8 transform libjxl stores is
// coeff[fx*8+fy] = sum_{y,x} A[fy][y] * A[fx][x] * pixel[y][x].
__constant__ float DCT_A[64];

void init_dct_basis() {
    float host[64];
    for (int k{0}; k < 8; ++k) {
        const double g{k == 0 ? 0.125 : 1.4142135623730951 / 8.0};
        for (int n{0}; n < 8; ++n) {
            host[k * 8 + n] = static_cast<float>(g * std::cos(M_PI * (n + 0.5) * k / 8.0));
        }
    }
    cudaMemcpyToSymbol(DCT_A, host, sizeof(host));
}

__global__ void forward_dct8_kernel(const float* __restrict__ xyb, std::size_t width,
                                    std::size_t height, float* __restrict__ coeffs) {
    __shared__ float tile[64];

    const std::size_t bx{blockIdx.x};
    const std::size_t by{blockIdx.y};
    const std::size_t c{blockIdx.z};
    const std::size_t tx{threadIdx.x};
    const std::size_t ty{threadIdx.y};
    const std::size_t plane{width * height};

    tile[ty * 8 + tx] = xyb[c * plane + (by * 8 + ty) * width + (bx * 8 + tx)];
    __syncthreads();

    // Thread (tx, ty) produces coefficient (fx=tx, fy=ty).
    float acc{0.0f};
    for (int y{0}; y < 8; ++y) {
        const float ay{DCT_A[ty * 8 + y]};
        for (int x{0}; x < 8; ++x) {
            acc += ay * DCT_A[tx * 8 + x] * tile[y * 8 + x];
        }
    }

    const std::size_t blocks_x{width / 8};
    const std::size_t block_index{by * blocks_x + bx};
    coeffs[c * plane + block_index * 64 + tx * 8 + ty] = acc;
}

}  // namespace

bool forward_dct8(const float* xyb, std::size_t width, std::size_t height, float* coeffs) {
    static std::once_flag basis_flag;
    std::call_once(basis_flag, init_dct_basis);

    const dim3 block{8, 8};
    const dim3 grid{static_cast<unsigned int>(width / 8), static_cast<unsigned int>(height / 8), 3};
    forward_dct8_kernel<<<grid, block>>>(xyb, width, height, coeffs);

    const cudaError_t launch{cudaGetLastError()};
    const cudaError_t sync{cudaDeviceSynchronize()};
    return launch == cudaSuccess && sync == cudaSuccess;
}

}  // namespace cujpegxl
