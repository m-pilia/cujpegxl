// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_TOOLS_BITSTREAM_MODULAR_H_
#define CUJPEGXL_TOOLS_BITSTREAM_MODULAR_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "src/bitstream/bit_writer.h"
#include "src/dc_predict.h"

namespace cujpegxl::bitstream {

// A single Modular channel: `w * h` int32 samples in row-major order.
struct ModularChannel {
    std::size_t w{0};
    std::size_t h{0};
    std::vector<std::int32_t> pixels{};
};

// The MA trees the encoder writes. Every leaf has `predictor` (DC_PREDICTOR_ZERO
// or DC_PREDICTOR_GRADIENT), offset 0 and multiplier 1. Without `split_channel`
// the tree is a single leaf; with it, the root splits on the channel index and
// sends the channels with index > *split_channel to the first leaf, the others
// to the second.
struct ModularTree {
    int predictor{DC_PREDICTOR_ZERO};
    std::optional<std::size_t> split_channel{};
};

// The channel-sample tokens of a Modular image and their prefix codes, with one
// context and one histogram per tree leaf: row l (HISTOGRAM_STRIDE entries) of
// `histograms`, `depth` and `bits` belongs to leaf l.
struct ModularData {
    struct Token {
        std::size_t leaf;
        std::uint32_t symbol;
        std::uint32_t nbits;
        std::uint32_t bits;
    };

    std::vector<Token> tokens{};
    std::vector<std::uint32_t> histograms{};
    std::vector<std::uint8_t> depth{};
    std::vector<std::uint16_t> bits{};

    void write_tokens(BitWriter& w) const;
};

// Emits a Modular sub-bitstream (as consumed by libjxl ModularGenericDecompress)
// for the given channels: a local GroupHeader (no global tree, default weighted
// predictor, no transforms), `tree`, and the channel samples coded as PackSigned
// residuals against the leaf predictor through the prefix entropy container.
// Channels are coded in order, each row-major (prediction is per-channel); empty
// channels are skipped, matching the decoder's channel iteration. Not
// byte-aligned on exit.
void write_modular_image(BitWriter& w, const std::vector<ModularChannel>& channels,
                         const ModularTree& tree = {});

// The header half of write_modular_image: GroupHeader + MA tree + the data
// histogram description. The returned data's write_tokens() emits the token
// half.
ModularData write_modular_header(BitWriter& w, const std::vector<ModularChannel>& channels,
                                 const ModularTree& tree = {});

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_TOOLS_BITSTREAM_MODULAR_H_
