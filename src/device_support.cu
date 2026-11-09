// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "device_support.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace cujpegxl {
namespace {

__global__ void probe_kernel() {}

}  // namespace

DeviceQuery query_device_support(std::int32_t ordinal) {
    if (cudaSetDevice(ordinal) != cudaSuccess) {
        cudaGetLastError();
        return {.support = DeviceSupport::NO_DEVICE};
    }
    cudaFuncAttributes attributes{};
    if (cudaFuncGetAttributes(&attributes, probe_kernel) != cudaSuccess) {
        cudaGetLastError();
        return {.support = DeviceSupport::UNSUPPORTED};
    }
    // Every kernel is built for the same architectures as the probe, so the
    // driver picks the same image for all of them.
    return {.support = DeviceSupport::SUPPORTED, .arch = gpu_arch(attributes.ptxVersion)};
}

}  // namespace cujpegxl
