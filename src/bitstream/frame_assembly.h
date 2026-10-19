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

// AC groups and DC groups tiling a width x height image, including partial edge
// groups.
std::size_t ac_group_count(std::size_t width, std::size_t height);
std::size_t dc_group_count(std::size_t width, std::size_t height);

// The byte-aligned DcGlobal section.
std::vector<std::uint8_t> build_dc_global(const QuantParams& qp);

// The serialized AC context map (write_best_prefix_context_map over the fixed
// ac_cluster map, AC_NUM_CLUSTERS clusters). It is data-independent, so the
// device AcGlobal build blits these bits instead of rebuilding the 7425-entry map
// per frame. `bytes` holds ceil(nbits/8) bytes, LSB-first, with the final byte's
// unused high bits zeroed.
struct AcContextMapBlob {
    std::vector<std::uint8_t> bytes{};
    std::size_t nbits{0};
};
AcContextMapBlob build_ac_context_map_blob();

// The codestream head: signature + SizeHeader + ImageMetadata + FrameHeader +
// multi-entry TOC for `section_sizes` (in codestream order: DcGlobal, DcGroups,
// AcGlobal, AcGroups). Byte-aligned; the body sections follow directly.
std::vector<std::uint8_t> build_codestream_head(std::uint32_t width, std::uint32_t height,
                                                const std::vector<std::uint32_t>& section_sizes);

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_SRC_BITSTREAM_FRAME_ASSEMBLY_H_
