// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_TOOLS_BITSTREAM_MODULAR_H_
#define CUJPEGXL_TOOLS_BITSTREAM_MODULAR_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bit_writer.h"

namespace cujpegxl::bitstream {

// A single Modular channel: `w * h` int32 samples in row-major order.
struct ModularChannel {
    std::size_t w{0};
    std::size_t h{0};
    std::vector<std::int32_t> pixels{};
};

// Emits a Modular sub-bitstream (as consumed by libjxl ModularGenericDecompress)
// for the given channels: a local GroupHeader (no global tree, default weighted
// predictor, no transforms), a single-leaf MA tree (Zero predictor, one
// context), and the channel samples coded raw (PackSigned) through the prefix
// entropy container. Channels are coded in order, each row-major; empty channels
// are skipped, matching the decoder's channel iteration. Not byte-aligned on
// exit.
void write_modular_image(BitWriter& w, const std::vector<ModularChannel>& channels);

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_TOOLS_BITSTREAM_MODULAR_H_
