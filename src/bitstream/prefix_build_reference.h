// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Host builds of the AcGlobal section and the DcGroup header blobs with their
// prefix codes, the byte-exact reference for the on-device prefix build.

#ifndef CUJPEGXL_SRC_BITSTREAM_PREFIX_BUILD_REFERENCE_H_
#define CUJPEGXL_SRC_BITSTREAM_PREFIX_BUILD_REFERENCE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cujpegxl::bitstream {

// `ac_histogram` is the pooled AC cluster histogram
// (AC_NUM_CLUSTERS * HISTOGRAM_STRIDE entries); `depth`/`bits` share its layout.
struct AcGlobalResult {
    std::vector<std::uint8_t> section{};
    std::vector<std::uint8_t> depth{};
    std::vector<std::uint16_t> bits{};
};
AcGlobalResult build_ac_global(const std::uint32_t* ac_histogram, std::size_t num_ac_groups);

// `dc_histograms` and `acmeta_histograms` are per-DcGroup token histograms (each
// num_dc_groups * HISTOGRAM_STRIDE); the depth/bits members share that layout.
// `first_block_counts` (num_dc_groups, nullptr = all DCT8) gives each group's
// number of first-blocks, which the AcMetadata `count` field encodes; its bit
// width stays the total-block upper bound.
struct DcGroupBlobs {
    std::vector<std::uint8_t> dc_depth{};
    std::vector<std::uint16_t> dc_bits{};
    std::vector<std::uint8_t> acmeta_depth{};
    std::vector<std::uint16_t> acmeta_bits{};
    std::vector<std::uint8_t> blob_pre{};
    std::vector<std::uint32_t> blob_pre_off{};
    std::vector<std::uint32_t> blob_pre_bits{};
    std::vector<std::uint8_t> blob_mid{};
    std::vector<std::uint32_t> blob_mid_off{};
    std::vector<std::uint32_t> blob_mid_bits{};
};
DcGroupBlobs build_dc_group_blobs(std::size_t width, std::size_t height,
                                  const std::uint32_t* dc_histograms,
                                  const std::uint32_t* acmeta_histograms,
                                  const std::size_t* first_block_counts = nullptr);

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_SRC_BITSTREAM_PREFIX_BUILD_REFERENCE_H_
