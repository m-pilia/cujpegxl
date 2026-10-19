// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_TOOLS_BITSTREAM_ENTROPY_ENCODER_H_
#define CUJPEGXL_TOOLS_BITSTREAM_ENTROPY_ENCODER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/bitstream/bit_writer.h"
#include "src/bitstream/hybrid_uint.h"

namespace cujpegxl::bitstream {

// Prefix-code (Huffman) entropy container mirroring libjxl's DecodeHistograms +
// prefix ANSSymbolReader wire format. LZ77 is disabled and all contexts are
// clustered to a single histogram (context map = simple, all zeros), which is
// spec-legal and keeps the context model simple. Tokens are collected across
// contexts, then a length-limited prefix code is built over the pooled symbol
// histogram and the token stream is emitted.
class EntropyEncoder {
public:
    EntropyEncoder(std::size_t num_contexts, HybridUintConfig config)
        : num_contexts_{num_contexts}, config_{config} {}

    explicit EntropyEncoder(std::size_t num_contexts)
        : EntropyEncoder(num_contexts, HybridUintConfig{}) {}

    void add_token(std::size_t context, std::uint32_t value);

    // Emits the histogram description followed by the token stream. Not byte
    // aligned on exit (the container has no trailing padding for prefix codes).
    void write(BitWriter& w) const;

    // The two halves of write(), so the histograms and the token stream can be
    // emitted at different points of a section. write_tokens() requires a prior
    // write_histograms(), which builds the prefix code.
    void write_histograms(BitWriter& w) const;
    void write_tokens(BitWriter& w) const;

    // The pooled symbol histogram (index = symbol).
    const std::vector<std::uint32_t>& histogram() const { return histogram_; }

    // The shared prefix code, valid after write_histograms(): depth[s] is the
    // code length of symbol s and bits[s] its LSB-first code word.
    const std::vector<std::uint8_t>& code_depth() const { return depth_; }
    const std::vector<std::uint16_t>& code_bits() const { return bits_; }

private:
    struct Token {
        std::uint32_t symbol;
        std::uint32_t nbits;
        std::uint32_t bits;
    };

    std::size_t num_contexts_;
    HybridUintConfig config_;
    std::vector<Token> tokens_{};
    std::vector<std::uint32_t> histogram_{};

    // Built by write_histograms(), consumed by write_tokens().
    mutable std::vector<std::uint8_t> depth_{};
    mutable std::vector<std::uint16_t> bits_{};
};

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_TOOLS_BITSTREAM_ENTROPY_ENCODER_H_
