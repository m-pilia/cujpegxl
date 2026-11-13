// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "dct_mma_check.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include <cuda_runtime.h>

#include "dct_mma_impl.cuh"
#include "gpu_arch.h"

namespace cujpegxl {
namespace {

constexpr int TILE{32 * 32};
constexpr int STRIDE{33};

[[maybe_unused]] __device__ Dct32MmaBasis BASIS{DCT32_MMA_BASIS};

__global__ void __launch_bounds__(64) dct32_mma_kernel(const float* __restrict__ tiles,
                                                       float* __restrict__ coeffs) {
    __shared__ float x[32 * STRIDE];
    const std::size_t base{static_cast<std::size_t>(blockIdx.x) * TILE};
    for (int i{static_cast<int>(threadIdx.x)}; i < TILE; i += 64) {
        x[(i / 32) * STRIDE + i % 32] = tiles[base + i];
    }
    __syncthreads();
    if constexpr (COMPILED_ARCH == GpuArch::SM87) {
        const int w{static_cast<int>(threadIdx.x) / 32};
        const int lane{static_cast<int>(threadIdx.x) % 32};
        std::uint32_t u[2][4];
        const float offset{dct32_mma_vertical(BASIS, x, STRIDE, w, lane, u)};
        float y[16];
        dct32_mma_horizontal(BASIS, u, w, lane, offset, y);
#pragma unroll
        for (int i{0}; i < 16; ++i) {
            coeffs[base + dct32_mma_fx(lane, i) * 32 + dct32_mma_fy(w, lane, i)] = y[i];
        }
    }
}

}  // namespace

std::optional<std::vector<float>> dct32_mma_coefficients(const std::vector<float>& tiles) {
    const std::size_t bytes{tiles.size() * sizeof(float)};
    float* device_tiles{};
    float* device_coeffs{};
    bool ok{cudaMalloc(&device_tiles, bytes) == cudaSuccess &&
            cudaMalloc(&device_coeffs, bytes) == cudaSuccess &&
            cudaMemcpy(device_tiles, tiles.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess};
    std::vector<float> coeffs(tiles.size());
    if (ok) {
        dct32_mma_kernel<<<static_cast<unsigned>(tiles.size() / TILE), 64>>>(device_tiles,
                                                                             device_coeffs);
        ok = cudaGetLastError() == cudaSuccess &&
             cudaMemcpy(coeffs.data(), device_coeffs, bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(device_tiles);
    cudaFree(device_coeffs);
    return ok ? std::optional<std::vector<float>>{std::move(coeffs)} : std::nullopt;
}

}  // namespace cujpegxl
