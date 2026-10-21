// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_ENTROPY_ROUNDTRIP_H_
#define CUJPEGXL_SRC_ENTROPY_ROUNDTRIP_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cujpegxl {

// Test-only host-vector wrapper around the device AC entropy kernels: uploads
// the quantized coefficients, their non-zero counts (computed on the host) and
// the shared prefix code, runs ac_build_histogram and ac_encode_groups, and
// downloads the results. Used by the GPU conformance test to diff against the
// host bitstream reference.
struct AcDeviceResult {
    std::vector<std::uint32_t> histogram{};
    std::vector<std::uint32_t> group_sizes{};
    std::vector<std::uint32_t> group_offsets{};
    std::vector<std::uint8_t> stream{};  // concatenated byte-aligned AcGroups
};

constexpr std::uint8_t AC_OUT_SENTINEL = 0xA5;

// `q` is the covered-block coefficient layout (three planes of
// (width/8 * height/8) * 64 int32) and `acs` the per-8x8 transform signal.
// The device output buffer of `out_capacity` bytes starts filled with
// AC_OUT_SENTINEL; when the AcGroups exceed it, `stream` receives the whole
// buffer instead.
bool ac_encode_device(const std::vector<std::int32_t>& q, const std::vector<std::int8_t>& acs,
                      std::size_t width, std::size_t height, const std::vector<std::uint8_t>& depth,
                      const std::vector<std::uint16_t>& bits, std::size_t out_capacity,
                      AcDeviceResult& out);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_ENTROPY_ROUNDTRIP_H_
