// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "quant.h"

#include <mutex>

#include <cuda_runtime.h>

#include "quant_weights_dct8.h"

namespace cujpegxl {
namespace {

__constant__ float QUANT_WEIGHTS[3][64];

void init_quant_weights() {
    cudaMemcpyToSymbol(QUANT_WEIGHTS, DCT8_DEQUANT_WEIGHTS, sizeof(DCT8_DEQUANT_WEIGHTS));
}

__global__ void quantize_dct8_kernel(const float* __restrict__ coeffs, std::size_t plane,
                                     float distance, std::int32_t* __restrict__ q) {
    const std::size_t g{blockIdx.x * blockDim.x + threadIdx.x};
    if (g >= 3 * plane) {
        return;
    }
    const std::size_t c{g / plane};
    const std::size_t k{(g % plane) % 64};
    const float step{QUANT_WEIGHTS[c][k] * distance};
    q[g] = static_cast<std::int32_t>(rintf(coeffs[g] / step));
}

}  // namespace

bool quantize_dct8(const float* coeffs, std::size_t width, std::size_t height, float distance,
                   std::int32_t* q) {
    static std::once_flag weights_flag;
    std::call_once(weights_flag, init_quant_weights);

    const std::size_t plane{width * height};
    const unsigned int threads{256};
    const unsigned int blocks{static_cast<unsigned int>((3 * plane + threads - 1) / threads)};
    quantize_dct8_kernel<<<blocks, threads>>>(coeffs, plane, distance, q);

    const cudaError_t launch{cudaGetLastError()};
    const cudaError_t sync{cudaDeviceSynchronize()};
    return launch == cudaSuccess && sync == cudaSuccess;
}

}  // namespace cujpegxl
