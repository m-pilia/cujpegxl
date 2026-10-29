// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "device_support.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace cujpegxl {
namespace {

// All the library's kernels are built for the same architectures, so loading
// this kernel succeeds exactly on the devices the encoder can run on.
__global__ void probe_kernel() {}

}  // namespace

// A failed call is also left as the thread's last CUDA error; it is cleared so
// that the caller's own error checks do not report it.
DeviceSupport query_device_support(std::int32_t ordinal) {
    if (cudaSetDevice(ordinal) != cudaSuccess) {
        cudaGetLastError();
        return DeviceSupport::NO_DEVICE;
    }
    cudaFuncAttributes attributes{};
    if (cudaFuncGetAttributes(&attributes, probe_kernel) != cudaSuccess) {
        cudaGetLastError();
        return DeviceSupport::UNSUPPORTED;
    }
    return DeviceSupport::SUPPORTED;
}

}  // namespace cujpegxl
