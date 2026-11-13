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
constexpr int REGION{16 * 16};
constexpr int REGION_STRIDE{17};

[[maybe_unused]] __device__ Dct32MmaBasis BASIS{DCT32_MMA_BASIS};
[[maybe_unused]] __device__ RegionMmaBasis REGION_BASES[2]{REGION_MMA_BASIS_DCT16,
                                                           REGION_MMA_BASIS_DCT8};

__global__ void __launch_bounds__(64)
    dct32_mma_kernel(const float* __restrict__ tiles, float* __restrict__ coeffs) {
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

__global__ void __launch_bounds__(32)
    region_mma_kernel(const float* __restrict__ regions, int n, float* __restrict__ coeffs) {
    __shared__ float x[16 * REGION_STRIDE];
    const std::size_t base{static_cast<std::size_t>(blockIdx.x) * REGION};
    const int lane{static_cast<int>(threadIdx.x)};
    for (int i{lane}; i < REGION; i += 32) {
        x[(i / 16) * REGION_STRIDE + i % 16] = regions[base + i];
    }
    __syncwarp();
    if constexpr (COMPILED_ARCH == GpuArch::SM87) {
        float y[8];
        region_mma(REGION_BASES[n == 16 ? 0 : 1], n, x, REGION_STRIDE, lane, y);
#pragma unroll
        for (int i{0}; i < 8; ++i) {
            coeffs[base + region_mma_fx(n, lane, i) * 16 + region_mma_fy(n, lane, i)] = y[i];
        }
    }
}

// Runs `launch(device_inputs, device_outputs)` on a device copy of `inputs`
// and returns the outputs, as many as the inputs.
template <typename Launch>
std::optional<std::vector<float>> run_on_device(const std::vector<float>& inputs, Launch launch) {
    const std::size_t bytes{inputs.size() * sizeof(float)};
    float* device_inputs{};
    float* device_outputs{};
    bool ok{cudaMalloc(&device_inputs, bytes) == cudaSuccess &&
            cudaMalloc(&device_outputs, bytes) == cudaSuccess &&
            cudaMemcpy(device_inputs, inputs.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess};
    std::vector<float> outputs(inputs.size());
    if (ok) {
        launch(device_inputs, device_outputs);
        ok = cudaGetLastError() == cudaSuccess && cudaMemcpy(outputs.data(), device_outputs, bytes,
                                                             cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(device_inputs);
    cudaFree(device_outputs);
    return ok ? std::optional<std::vector<float>>{std::move(outputs)} : std::nullopt;
}

}  // namespace

std::optional<std::vector<float>> dct32_mma_coefficients(const std::vector<float>& tiles) {
    return run_on_device(tiles, [&](const float* device_tiles, float* device_coeffs) {
        dct32_mma_kernel<<<static_cast<unsigned>(tiles.size() / TILE), 64>>>(device_tiles,
                                                                             device_coeffs);
    });
}

std::optional<std::vector<float>> region_mma_coefficients(const std::vector<float>& regions,
                                                          int n) {
    return run_on_device(regions, [&](const float* device_regions, float* device_coeffs) {
        region_mma_kernel<<<static_cast<unsigned>(regions.size() / REGION), 32>>>(device_regions, n,
                                                                                  device_coeffs);
    });
}

}  // namespace cujpegxl
