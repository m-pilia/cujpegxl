// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_TOOLS_BITSTREAM_PREFIX_CODE_H_
#define CUJPEGXL_TOOLS_BITSTREAM_PREFIX_CODE_H_

#include <cstddef>
#include <cstdint>

#include "bit_writer.h"

namespace cujpegxl::bitstream {

// Builds a length-limited (<=15 bit) canonical prefix code for `histogram`
// (symbol frequencies, `length` entries) and stores its description into `w`
// using the JXL prefix-code header format (simple form for <=4 used symbols,
// otherwise the code-length-code form). This mirrors libjxl's
// BuildAndStoreHuffmanTree so the emitted header is accepted by
// HuffmanDecodingData::ReadFromBitStream.
//
// On return `depth[i]` holds the code length of symbol i (0 if unused) and
// `bits[i]` its LSB-first code word; both arrays must have `length` entries and
// are consumed by the token writer.
void build_and_store_huffman_tree(const std::uint32_t* histogram, std::size_t length,
                                  std::uint8_t* depth, std::uint16_t* bits, BitWriter& w);

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_TOOLS_BITSTREAM_PREFIX_CODE_H_
