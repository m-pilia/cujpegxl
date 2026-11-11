// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_XYB_MATH_CHECK_H_
#define CUJPEGXL_SRC_XYB_MATH_CHECK_H_

#include <cstdint>
#include <optional>

namespace cujpegxl {

struct Mismatches {
    std::uint32_t count;
    std::uint32_t first_bits;
};

// Bitwise comparisons against libdevice over every float whose representation
// lies in [first_bits, last_bits]; std::nullopt on a CUDA error.
std::optional<Mismatches> cbrt_normal_mismatches(std::uint32_t first_bits, std::uint32_t last_bits);
std::optional<Mismatches> pow_normal_mismatches(float exponent, std::uint32_t first_bits,
                                                std::uint32_t last_bits);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_XYB_MATH_CHECK_H_
