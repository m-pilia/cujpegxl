// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "toolchains/tests/generated.h"

static_assert(GENERATOR_POINTER_BYTES == 8);

__global__ void increment(std::uint32_t* value) {
    ++*value;
}

int main() {
    std::vector<std::uint32_t> result{41};
    std::uint32_t* device{};
    if (cudaMalloc(&device, sizeof(*device)) != cudaSuccess) {
        return 1;
    }
    cudaError_t status{cudaMemcpy(device, result.data(), sizeof(*device), cudaMemcpyHostToDevice)};
    if (status == cudaSuccess) {
        increment<<<1, 1>>>(device);
        status = cudaGetLastError();
    }
    if (status == cudaSuccess) {
        status = cudaMemcpy(result.data(), device, sizeof(*device), cudaMemcpyDeviceToHost);
    }
    const cudaError_t release_status{cudaFree(device)};
    return status != cudaSuccess || release_status != cudaSuccess || result.front() != 42;
}
