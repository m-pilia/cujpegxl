// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_BITSTREAM_CODESTREAM_H_
#define CUJPEGXL_SRC_BITSTREAM_CODESTREAM_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bit_writer.h"

namespace cujpegxl::bitstream {

// Emits the JXL codestream signature (0xFF 0x0A), SizeHeader and an all-default
// ImageMetadata (8-bit, xyb_encoded, enumerated sRGB), matching libjxl's
// WriteCodestreamHeaders for our fixed configuration. Byte-aligned on exit (the
// decoder jumps to a byte boundary before the frame header).
void write_codestream_headers(BitWriter& w, std::uint32_t xsize, std::uint32_t ysize);

// Emits a FrameHeader for a single-pass, full-frame, last kRegularFrame in
// kVarDCT/kXYB with a neutral quant-matrix scale (x_qm_scale = b_qm_scale = 2)
// and default loop filter. Not byte-aligned on exit.
void write_frame_header(BitWriter& w);

// Emits the single-section TOC (no permutation) whose one entry is the byte
// length of the combined DC+AC section. Byte-aligned on entry and exit.
void write_toc_single_section(BitWriter& w, std::size_t section_size_bytes);

// Emits a multi-entry TOC (no permutation) listing each section's byte length
// in codestream order (DcGlobal, DcGroups, AcGlobal, AcGroups). Byte-aligned on
// entry and exit.
void write_toc_multi_section(BitWriter& w, const std::vector<std::uint32_t>& section_sizes);

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_SRC_BITSTREAM_CODESTREAM_H_
