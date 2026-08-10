// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_TOOLS_BITSTREAM_CONTAINER_H_
#define CUJPEGXL_TOOLS_BITSTREAM_CONTAINER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cujpegxl::bitstream {

// The fixed 32-byte prefix shared by every JXL ISOBMFF file: the 12-byte JXL
// signature box (size 0x0C, type "JXL ", payload 0x0D 0x0A 0x87 0x0A) followed
// by the 20-byte ftyp box whose major and single compatible brand are both
// "jxl ". Matches libjxl's kContainerHeader byte-for-byte.
extern const std::array<std::uint8_t, 32> CONTAINER_HEADER;

// Bytes the jxlc box header occupies for a codestream of the given size: 8 with
// a 32-bit size field, or 16 when the codestream needs the 64-bit extended size
// (a box size >= 2^32, i.e. codestream >= 2^32 - 8 bytes).
std::size_t jxlc_box_header_size(std::size_t codestream_size);

// Total container file size for a codestream of `codestream_size` bytes:
// CONTAINER_HEADER + jxlc box header + codestream.
std::size_t container_size(std::size_t codestream_size);

// Returns the container framing bytes (CONTAINER_HEADER followed by the jxlc box
// header) for a codestream of `codestream_size` bytes. The caller writes the
// (device-produced) codestream immediately after these bytes to form the file,
// so no per-pixel data passes through this call.
std::vector<std::uint8_t> container_framing(std::size_t codestream_size);

// Returns a full container file (framing + codestream) for `codestream`.
std::vector<std::uint8_t> write_container(const std::vector<std::uint8_t>& codestream);

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_TOOLS_BITSTREAM_CONTAINER_H_
