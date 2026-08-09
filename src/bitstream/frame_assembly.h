// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_BITSTREAM_FRAME_ASSEMBLY_H_
#define CUJPEGXL_SRC_BITSTREAM_FRAME_ASSEMBLY_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cujpegxl::bitstream {

// Serialized quantizer state for a VarDCT frame (see libjxl Quantizer).
struct QuantParams {
    std::uint32_t global_scale{1};
    std::uint32_t quant_dc{1};
    std::uint32_t raw_quant_field{1};
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

// The byte-aligned DcGlobal section.
std::vector<std::uint8_t> build_dc_global(const QuantParams& qp);

// Per-DcGroup header blobs and prefix codes for the device DcGroup emitter,
// flattened to match dc_encode_groups' inputs. `dc_histograms` is the device
// per-DcGroup DC histogram buffer (num_dc_groups * HISTOGRAM_STRIDE). The
// AcMetadata histograms are content-independent and computed here from the
// geometry and quant field.
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
DcGroupBlobs build_dc_group_blobs(std::size_t width, std::size_t height, const QuantParams& qp,
                                  const std::uint32_t* dc_histograms);

// The codestream head: signature + SizeHeader + ImageMetadata + FrameHeader +
// multi-entry TOC for `section_sizes` (in codestream order: DcGlobal, DcGroups,
// AcGlobal, AcGroups). Byte-aligned; the body sections follow directly.
std::vector<std::uint8_t> build_codestream_head(std::uint32_t width, std::uint32_t height,
                                                const std::vector<std::uint32_t>& section_sizes);

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_SRC_BITSTREAM_FRAME_ASSEMBLY_H_
