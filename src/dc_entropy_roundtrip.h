// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_DC_ENTROPY_ROUNDTRIP_H_
#define CUJPEGXL_SRC_DC_ENTROPY_ROUNDTRIP_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tools/bitstream/vardct_frame.h"

namespace cujpegxl {

// Test-only host-vector wrapper around the device DcGroup kernels: flattens the
// per-group prefix codes and header blobs from a host DcReference, uploads the
// quantized coefficients, runs dc_build_histograms and dc_encode_groups, and
// downloads the per-group DC histograms and the concatenated byte-aligned
// sections. Used by the GPU conformance test to diff against the oracle.
struct DcDeviceResult {
    std::vector<std::vector<std::uint32_t>> histograms{};  // per group, trimmed to alphabet
    std::vector<std::uint32_t> group_sizes{};
    std::vector<std::uint32_t> group_offsets{};
    std::vector<std::uint8_t> stream{};  // concatenated byte-aligned DcGroup sections
};

bool dc_encode_device(const std::vector<std::int32_t>& q, std::size_t width, std::size_t height,
                      std::uint32_t raw_quant_field, const bitstream::DcReference& ref,
                      DcDeviceResult& out);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_DC_ENTROPY_ROUNDTRIP_H_
