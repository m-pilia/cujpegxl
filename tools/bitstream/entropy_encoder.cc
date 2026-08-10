// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "entropy_encoder.h"

#include <cassert>

#include "prefix_code.h"

namespace cujpegxl::bitstream {
namespace {

// Bits needed to hold the values 0..x-1 (libjxl CeilLog2Nonzero semantics).
std::size_t ceil_log2(std::size_t x) {
    if (x <= 1) {
        return 0;
    }
    std::size_t bits{0};
    std::size_t v{x - 1};
    while (v) {
        v >>= 1;
        ++bits;
    }
    return bits;
}

constexpr std::size_t PREFIX_MAX_BITS = 15;

void encode_uint_config(BitWriter& w, const HybridUintConfig& config,
                        std::size_t log_alpha_size) {
    w.write(ceil_log2(log_alpha_size + 1), config.split_exponent);
    if (config.split_exponent == log_alpha_size) {
        return;
    }
    w.write(ceil_log2(config.split_exponent + 1), config.msb_in_token);
    w.write(ceil_log2(config.split_exponent - config.msb_in_token + 1),
            config.lsb_in_token);
}

void store_var_len_uint16(BitWriter& w, std::size_t n) {
    assert(n <= 65535);
    if (n == 0) {
        w.write(1, 0);
        return;
    }
    w.write(1, 1);
    std::size_t nbits{0};
    std::size_t v{n};
    while (v > 1) {
        v >>= 1;
        ++nbits;
    }
    w.write(4, nbits);
    w.write(nbits, n - (1ull << nbits));
}

}  // namespace

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
    w.write(1, 0);  // lz77.enabled = false

    // Context map: only present when there is more than one context. A single
    // clustered histogram is signalled with the simple form (bits_per_entry=0).
    if (num_contexts_ > 1) {
        w.write(1, 1);  // is_simple
        w.write(2, 0);  // bits_per_entry = 0 -> all contexts map to histogram 0
    }

    w.write(1, 1);  // use_prefix_code
    const std::size_t log_alpha_size{PREFIX_MAX_BITS};
    encode_uint_config(w, config_, log_alpha_size);

    const std::size_t alphabet_size{histogram_.empty() ? 1 : histogram_.size()};
    store_var_len_uint16(w, alphabet_size - 1);

    depth_.assign(alphabet_size, 0);
    bits_.assign(alphabet_size, 0);
    if (alphabet_size > 1) {
        build_and_store_huffman_tree(histogram_.data(), alphabet_size, depth_.data(),
                                     bits_.data(), w);
    }
}

void EntropyEncoder::write_tokens(BitWriter& w) const {
    for (const Token& t : tokens_) {
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
