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
// quantized coefficients, runs dc_build_histograms, dc_first_blocks and
// dc_encode_groups, and downloads the per-group DC histograms and the
// concatenated byte-aligned sections. Used by the GPU conformance test to diff
// against the oracle.
struct DcDeviceResult {
    std::vector<std::vector<std::uint32_t>> histograms{};         // per group DC, trimmed
    std::vector<std::vector<std::uint32_t>> acmeta_histograms{};  // per group AcMetadata
    std::vector<std::uint32_t> group_sizes{};
    std::vector<std::uint32_t> group_offsets{};
    std::vector<std::uint8_t> stream{};  // concatenated byte-aligned DcGroup sections
};

constexpr std::uint8_t DC_OUT_SENTINEL = 0xA5;

// `dc` is the compact per-block DC (3*blocks, channel-major X, Y, B),
// `acs`/`ytox_map`/`ytob_map` the transform + CfL signals, `quant_field` the
// per-block quant field. Runs dc_build_histograms, acmeta_build_histograms,
// dc_first_blocks and dc_encode_groups. The device output buffer of
// `out_capacity` bytes starts filled with DC_OUT_SENTINEL; when the sections
// exceed it, `stream` receives the whole buffer instead.
bool dc_encode_device(const std::vector<std::int32_t>& dc, const std::vector<std::int8_t>& acs,
                      const std::vector<std::int8_t>& ytox_map,
                      const std::vector<std::int8_t>& ytob_map,
                      const std::vector<std::int32_t>& quant_field, std::size_t width,
                      std::size_t height, const bitstream::DcReference& ref,
                      std::size_t out_capacity, DcDeviceResult& out);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_DC_ENTROPY_ROUNDTRIP_H_
