// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_ENTROPY_ROUNDTRIP_H_
#define CUJPEGXL_SRC_ENTROPY_ROUNDTRIP_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cujpegxl {

// Test-only host-vector wrapper around the device AC token coder: uploads the
// quantized coefficients and their non-zero counts (computed on the host), runs
// ac_build_histogram, ac_token_layout and ac_pack_tokens (over the groups split
// into `batches` batches), and downloads the results. Used by the GPU
// conformance test to diff against the host bitstream reference.
struct AcDeviceTokens {
    std::vector<std::uint32_t> histogram{};
    std::vector<std::uint32_t> layout{};
    std::vector<std::uint16_t> entries{};
    std::vector<std::uint64_t> raw{};
};

// `q` is the covered-block coefficient layout (three planes of
// (width/8 * height/8) * 64 int32) and `acs` the per-8x8 transform signal.
bool ac_tokens_device(const std::vector<std::int32_t>& q, const std::vector<std::int8_t>& acs,
                      std::size_t width, std::size_t height, std::size_t batches,
                      AcDeviceTokens& out);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_ENTROPY_ROUNDTRIP_H_
