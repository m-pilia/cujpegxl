// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_BITSTREAM_FRAME_ASSEMBLY_H_
#define CUJPEGXL_SRC_BITSTREAM_FRAME_ASSEMBLY_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cujpegxl::bitstream {

// Serialized quantizer state for a VarDCT frame (see libjxl Quantizer). The
// per-block quant field is carried separately (a device buffer through the
// entropy path), not here: only these two values are written into DcGlobal.
struct QuantParams {
    std::uint32_t global_scale{1};
    std::uint32_t quant_dc{1};
};

// Width of one device histogram (AC_HISTOGRAM_SIZE in the device coder): the
// hybrid-uint token alphabet upper bound.
constexpr std::size_t HISTOGRAM_STRIDE = 256;

// AC groups (256px / 32 blocks square) and DC groups (2048px / 256 blocks
// square) tiling a width x height image, including partial edge groups.
std::size_t ac_group_count(std::size_t width, std::size_t height);
std::size_t dc_group_count(std::size_t width, std::size_t height);

// The AcGlobal section and the shared AC prefix code that the device AC token
// emitter consumes. `ac_histogram` is the pooled device AC histogram
// (HISTOGRAM_STRIDE entries).
struct AcGlobalResult {
    std::vector<std::uint8_t> section{};  // byte-aligned AcGlobal section
    std::vector<std::uint8_t> depth{};    // AC prefix code (alphabet entries)
    std::vector<std::uint16_t> bits{};
};
AcGlobalResult build_ac_global(const std::uint32_t* ac_histogram, std::size_t num_ac_groups);
AcGlobalResult build_ac_global(const std::uint32_t* ac_histograms,
                               const std::uint8_t* context_map,
                               std::size_t num_clusters,
                               std::size_t num_ac_groups);

// The byte-aligned DcGlobal section.
std::vector<std::uint8_t> build_dc_global(const QuantParams& qp);

// Per-DcGroup header blobs and prefix codes for the device DcGroup emitter,
// flattened to match dc_encode_groups' inputs. `dc_histograms` and
// `acmeta_histograms` are the device per-DcGroup token histograms (each
// num_dc_groups * HISTOGRAM_STRIDE). The AcMetadata histogram is content-derived
// (per-block quant field), so its prefix code is built from the passed-in
// histogram rather than assumed uniform.
struct DcGroupBlobs {
    std::vector<std::uint8_t> dc_depth{};  // num_groups * HISTOGRAM_STRIDE
    std::vector<std::uint16_t> dc_bits{};
    std::vector<std::uint8_t> acmeta_depth{};  // num_groups * HISTOGRAM_STRIDE
    std::vector<std::uint16_t> acmeta_bits{};
    std::vector<std::uint8_t> blob_pre{};
    std::vector<std::uint32_t> blob_pre_off{};
    std::vector<std::uint32_t> blob_pre_bits{};
    std::vector<std::uint8_t> blob_mid{};
    std::vector<std::uint32_t> blob_mid_off{};
    std::vector<std::uint32_t> blob_mid_bits{};
};
// `first_block_counts` (num_dc_groups, nullptr = all DCT8) gives each group's
// number of first-blocks, which the AcMetadata `count` field encodes (the ACS/QF
// channels have one column per first-block); its bit width stays the total-block
// upper bound.
DcGroupBlobs build_dc_group_blobs(std::size_t width, std::size_t height,
                                  const std::uint32_t* dc_histograms,
                                  const std::uint32_t* acmeta_histograms,
                                  const std::size_t* first_block_counts = nullptr);

// The codestream head: signature + SizeHeader + ImageMetadata + FrameHeader +
// multi-entry TOC for `section_sizes` (in codestream order: DcGlobal, DcGroups,
// AcGlobal, AcGroups). Byte-aligned; the body sections follow directly.
std::vector<std::uint8_t> build_codestream_head(std::uint32_t width, std::uint32_t height,
                                                const std::vector<std::uint32_t>& section_sizes);

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_SRC_BITSTREAM_FRAME_ASSEMBLY_H_
