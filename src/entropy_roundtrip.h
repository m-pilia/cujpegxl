// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_ENTROPY_ROUNDTRIP_H_
#define CUJPEGXL_SRC_ENTROPY_ROUNDTRIP_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cujpegxl {

// Test-only host-vector wrapper around the device AC entropy kernels: uploads
// the quantized coefficients and shared prefix code, runs ac_build_histogram and
// ac_encode_groups, and downloads the results. Used by the GPU conformance test
// to diff against the host bitstream reference.
struct AcDeviceResult {
    std::vector<std::uint32_t> histogram{};
    std::vector<std::uint32_t> group_sizes{};
    std::vector<std::uint32_t> group_offsets{};
    std::vector<std::uint8_t> stream{};  // concatenated byte-aligned AcGroups
};

bool ac_encode_device(const std::vector<std::int32_t>& q, std::size_t width, std::size_t height,
                      const std::vector<std::uint8_t>& depth,
                      const std::vector<std::uint16_t>& bits, AcDeviceResult& out);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_ENTROPY_ROUNDTRIP_H_
