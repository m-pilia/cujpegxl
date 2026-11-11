// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>
#include <ios>

#include <gtest/gtest.h>

#include "xyb_math_check.h"

// pow_normal and cbrt_normal copy libdevice's __powf and cbrtf without their
// special-value handling, so the opsin transform's output is byte-identical
// for every input, not only the golden corpus. These tests compare them
// bitwise with libdevice over every float of domains that contain the
// transform's operands, and fail if the copies are edited, if libdevice (CUDA)
// or the compiler changes either side, or on a GPU whose flushing and
// non-flushing approximations differ on normal operands. That callers stay
// within the domains is not tested.
namespace cujpegxl {
namespace {

constexpr std::uint32_t float_bits(std::uint32_t biased_exponent) {
    return biased_exponent << 23;
}

TEST(XybImplGpuTest, CbrtNormalMatchesLibdeviceOnEveryPositiveNormal) {
    const auto mismatches{cbrt_normal_mismatches(float_bits(1), float_bits(255) - 1)};
    ASSERT_TRUE(mismatches.has_value());
    EXPECT_EQ(mismatches->count, 0u) << "first at bits 0x" << std::hex << mismatches->first_bits;
}

// Every base in [2^-52, 2^53), where the power and its ex2 operand stay normal.
TEST(XybImplGpuTest, PowNormalMatchesLibdeviceForTheSrgbExponent) {
    const auto mismatches{
        pow_normal_mismatches(2.4f, float_bits(127 - 52), float_bits(127 + 53) - 1)};
    ASSERT_TRUE(mismatches.has_value());
    EXPECT_EQ(mismatches->count, 0u) << "first at bits 0x" << std::hex << mismatches->first_bits;
}

}  // namespace
}  // namespace cujpegxl
