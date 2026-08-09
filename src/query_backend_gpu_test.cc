// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>

#include "cujpegxl/capability.h"
#include "cujpegxl/cujpegxl.h"

// Requires a physical CUDA device; tagged "gpu"/"manual" and excluded from the
// CPU-only CI. On the development host (GTX 1080 Ti, SM 6.1) the encoder must
// report the FP32-SIMT backend.
namespace cujpegxl {
namespace {

TEST(QueryDeviceCapability, DeviceZeroIsQueryable) {
    DeviceCapability capability{};
    ASSERT_TRUE(query_device_capability(0, &capability));
    EXPECT_GE(capability.compute_major, 6);
}

TEST(QueryBackend, HostGpuSelectsFp32Simt) {
    cujpegxl_backend backend{CUJPEGXL_BACKEND_UNKNOWN};
    ASSERT_EQ(cujpegxl_query_backend(0, &backend), CUJPEGXL_OK);
    EXPECT_EQ(backend, CUJPEGXL_BACKEND_FP32_SIMT);
}

TEST(QueryBackend, MissingDeviceReportsNoDevice) {
    cujpegxl_backend backend{CUJPEGXL_BACKEND_UNKNOWN};
    EXPECT_EQ(cujpegxl_query_backend(9999, &backend), CUJPEGXL_NO_DEVICE);
}

}  // namespace
}  // namespace cujpegxl
