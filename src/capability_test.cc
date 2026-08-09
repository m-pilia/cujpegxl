// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "cujpegxl/capability.h"

#include <gtest/gtest.h>

namespace cujpegxl {
namespace {

static_assert(select_backend({6, 1}) == Backend::FP32_SIMT);
static_assert(select_backend({8, 7}) == Backend::FP16_TENSOR);

TEST(SelectBackend, Pascal1080TiSelectsFp32Simt) {
    EXPECT_EQ(select_backend({6, 1}), Backend::FP32_SIMT);
}

TEST(SelectBackend, AmpereOrinNanoSelectsFp16Tensor) {
    EXPECT_EQ(select_backend({8, 7}), Backend::FP16_TENSOR);
}

TEST(SelectBackend, AmpereBoundarySelectsFp16Tensor) {
    EXPECT_EQ(select_backend({8, 0}), Backend::FP16_TENSOR);
}

TEST(SelectBackend, TuringSelectsFp32Simt) {
    EXPECT_EQ(select_backend({7, 5}), Backend::FP32_SIMT);
}

TEST(SelectBackend, PrePascalIsUnknown) {
    EXPECT_EQ(select_backend({3, 5}), Backend::UNKNOWN);
}

TEST(BackendName, MapsEachBackend) {
    EXPECT_STREQ(backend_name(Backend::FP32_SIMT), "fp32-simt");
    EXPECT_STREQ(backend_name(Backend::FP16_TENSOR), "fp16-tensor");
    EXPECT_STREQ(backend_name(Backend::UNKNOWN), "unknown");
}

}  // namespace
}  // namespace cujpegxl
