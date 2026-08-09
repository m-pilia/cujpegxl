/* Copyright (c) 2026 Martino Pilia */
/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef CUJPEGXL_SRC_CAPABILITY_H_
#define CUJPEGXL_SRC_CAPABILITY_H_

#include <cstdint>

namespace cujpegxl {

enum class Backend : std::int32_t {
    UNKNOWN = 0,
    FP32_SIMT = 1,
    FP16_TENSOR = 2,
};

struct DeviceCapability {
    std::int32_t compute_major;
    std::int32_t compute_minor;
};

// Pascal (SM 6.x) and other pre-Ampere SIMT parts run the FP32 functional path;
// Ampere and newer (SM >= 8.0) select the FP16/tensor-core path. Kept as a pure
// constexpr function so backend selection is unit-testable without a device.
constexpr Backend select_backend(DeviceCapability capability) {
    const std::int32_t compute = capability.compute_major * 10 + capability.compute_minor;
    if (compute >= 80) {
        return Backend::FP16_TENSOR;
    }
    if (compute >= 60) {
        return Backend::FP32_SIMT;
    }
    return Backend::UNKNOWN;
}

const char* backend_name(Backend backend);

// Queries the compute capability of `ordinal` via the CUDA runtime. Returns
// false when no such device exists or the query fails. Defined in the CUDA
// translation unit; declared here so host code and tests share the signature.
bool query_device_capability(std::int32_t ordinal, DeviceCapability* out_capability);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_CAPABILITY_H_
