// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include <cuda_runtime.h>

#include <cstdint>

#include "cujpegxl/capability.h"

namespace cujpegxl {

bool query_device_capability(std::int32_t ordinal, DeviceCapability* out_capability) {
    int device_count{0};
    if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
        return false;
    }
    if (ordinal < 0 || ordinal >= device_count) {
        return false;
    }

    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, ordinal) != cudaSuccess) {
        return false;
    }

    out_capability->compute_major = static_cast<std::int32_t>(properties.major);
    out_capability->compute_minor = static_cast<std::int32_t>(properties.minor);
    return true;
}

}  // namespace cujpegxl
