// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "entropy_encoder.h"

#include <cassert>

#include "src/bitstream/histogram_writer.h"

namespace cujpegxl::bitstream {

void EntropyEncoder::add_token(std::size_t context, std::uint32_t value) {
    assert(context < num_contexts_);
    (void)context;
    std::uint32_t symbol{};
    std::uint32_t nbits{};
    std::uint32_t bits{};
    config_.encode(value, symbol, nbits, bits);
    tokens_.push_back({symbol, nbits, bits});
    if (symbol >= histogram_.size()) {
        histogram_.resize(symbol + 1, 0);
    }
    ++histogram_[symbol];
}

void EntropyEncoder::write_histograms(BitWriter& w) const {
    const std::size_t alphabet_size{histogram_.empty() ? 1 : histogram_.size()};
    depth_.assign(alphabet_size, 0);
    bits_.assign(alphabet_size, 0);
    const std::uint32_t one{1};
    const std::uint32_t* hist{histogram_.empty() ? &one : histogram_.data()};
    write_prefix_histograms(w, hist, alphabet_size, num_contexts_, config_, depth_.data(),
                            bits_.data());
}

void EntropyEncoder::write_tokens(BitWriter& w) const {
    write_tokens_range(w, 0, tokens_.size());
}

void EntropyEncoder::write_tokens_range(BitWriter& w, std::size_t begin, std::size_t end) const {
    assert(begin <= end && end <= tokens_.size());
    for (std::size_t i{begin}; i < end; ++i) {
        const Token& t{tokens_[i]};
        w.write(depth_[t.symbol], bits_[t.symbol]);
        if (t.nbits) {
            w.write(t.nbits, t.bits);
        }
    }
}

void EntropyEncoder::write(BitWriter& w) const {
    write_histograms(w);
    write_tokens(w);
}

}  // namespace cujpegxl::bitstream
