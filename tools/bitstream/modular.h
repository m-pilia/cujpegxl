// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_TOOLS_BITSTREAM_MODULAR_H_
#define CUJPEGXL_TOOLS_BITSTREAM_MODULAR_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/bitstream/bit_writer.h"

#include "entropy_encoder.h"

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

// The header half of write_modular_image: GroupHeader + single-leaf MA tree +
// the channel data container's histogram description. `data` is populated with
// the channel-sample tokens and its shared prefix code (from write_histograms),
// so write_modular_tokens(w, data) can emit the token bits separately. Splitting
// the image this way lets the device DcGroup coder emit the tokens while the
// header stays a host-built bit blob.
void write_modular_header(BitWriter& w, const std::vector<ModularChannel>& channels,
                          EntropyEncoder& data);

// The token half: the channel samples' prefix-coded bits, using the code built
// by a prior write_modular_header on the same `data`.
void write_modular_tokens(BitWriter& w, const EntropyEncoder& data);

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_TOOLS_BITSTREAM_MODULAR_H_
