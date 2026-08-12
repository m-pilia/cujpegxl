// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "dct.h"

#include <cmath>
#include <mutex>

#include <cuda_runtime.h>

namespace cujpegxl {
namespace {

// libjxl's 1D DCT-N basis: A[k][n] = g(k) * cos(pi*(n+0.5)*k/N), with g(0)=1/N
// and g(k>0)=sqrt(2)/N. The forward NxN transform libjxl stores is
// coeff[fx*N+fy] = sum_{y,x} A[fy][y] * A[fx][x] * pixel[y][x], i.e. the
// orthonormal 2D DCT-II scaled by 1/N, transposed (horizontal frequency major).
__constant__ float DCT_A8[8 * 8];
__constant__ float DCT_A16[16 * 16];
__constant__ float DCT_A32[32 * 32];

void fill_basis(float* host, int n) {
    for (int k{0}; k < n; ++k) {
        const double g{k == 0 ? 1.0 / n : 1.4142135623730951 / n};
        for (int i{0}; i < n; ++i) {
            host[k * n + i] = static_cast<float>(g * std::cos(M_PI * (i + 0.5) * k / n));
        }
    }
}

void init_bases() {
    float b8[8 * 8];
    float b16[16 * 16];
    float b32[32 * 32];
    fill_basis(b8, 8);
    fill_basis(b16, 16);
    fill_basis(b32, 32);
    cudaMemcpyToSymbol(DCT_A8, b8, sizeof(b8));
    cudaMemcpyToSymbol(DCT_A16, b16, sizeof(b16));
    cudaMemcpyToSymbol(DCT_A32, b32, sizeof(b32));
}

template <int N>
__global__ void forward_dctn_kernel(const float* __restrict__ xyb, std::size_t width,
                                    std::size_t height, const float* __restrict__ basis,
                                    float* __restrict__ coeffs) {
    __shared__ float tile[N * N];

    const std::size_t bx{blockIdx.x};
    const std::size_t by{blockIdx.y};
    const std::size_t c{blockIdx.z};
    const int tx{static_cast<int>(threadIdx.x)};
    const int ty{static_cast<int>(threadIdx.y)};
    const std::size_t plane{width * height};

    tile[ty * N + tx] = xyb[c * plane + (by * N + ty) * width + (bx * N + tx)];
    __syncthreads();

    // Thread (tx, ty) produces coefficient (fx=tx, fy=ty).
    float acc{0.0f};
    for (int y{0}; y < N; ++y) {
        const float ay{basis[ty * N + y]};
        for (int x{0}; x < N; ++x) {
            acc += ay * basis[tx * N + x] * tile[y * N + x];
        }
    }

    const std::size_t blocks_x{width / N};
    const std::size_t block_index{by * blocks_x + bx};
    coeffs[c * plane + block_index * (N * N) + tx * N + ty] = acc;
}

const float* basis_symbol(int n) {
    const void* symbol{n == 8 ? static_cast<const void*>(DCT_A8)
                              : (n == 16 ? static_cast<const void*>(DCT_A16)
                                         : static_cast<const void*>(DCT_A32))};
    void* address{nullptr};
    if (cudaGetSymbolAddress(&address, symbol) != cudaSuccess) {
        return nullptr;
    }
    return static_cast<const float*>(address);
}

template <int N>
bool forward_dctn(const float* xyb, std::size_t width, std::size_t height, float* coeffs) {
    static std::once_flag basis_flag;
    std::call_once(basis_flag, init_bases);

    const float* basis{basis_symbol(N)};
    if (basis == nullptr) {
        return false;
    }

    const dim3 block{N, N};
    const dim3 grid{static_cast<unsigned int>(width / N), static_cast<unsigned int>(height / N), 3};
    forward_dctn_kernel<N><<<grid, block>>>(xyb, width, height, basis, coeffs);

    const cudaError_t launch{cudaGetLastError()};
    const cudaError_t sync{cudaDeviceSynchronize()};
    return launch == cudaSuccess && sync == cudaSuccess;
}

}  // namespace

bool forward_dct8(const float* xyb, std::size_t width, std::size_t height, float* coeffs) {
    return forward_dctn<8>(xyb, width, height, coeffs);
}

bool forward_dct16(const float* xyb, std::size_t width, std::size_t height, float* coeffs) {
    return forward_dctn<16>(xyb, width, height, coeffs);
}

bool forward_dct32(const float* xyb, std::size_t width, std::size_t height, float* coeffs) {
    return forward_dctn<32>(xyb, width, height, coeffs);
}

}  // namespace cujpegxl
