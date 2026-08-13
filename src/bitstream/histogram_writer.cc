// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "histogram_writer.h"

#include <cassert>

#include "prefix_code.h"

namespace cujpegxl::bitstream {
namespace {

constexpr std::size_t PREFIX_MAX_BITS = 15;

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

void encode_uint_config(BitWriter& w, const HybridUintConfig& config, std::size_t log_alpha_size) {
    w.write(ceil_log2(log_alpha_size + 1), config.split_exponent);
    if (config.split_exponent == log_alpha_size) {
        return;
    }
    w.write(ceil_log2(config.split_exponent + 1), config.msb_in_token);
    w.write(ceil_log2(config.split_exponent - config.msb_in_token + 1), config.lsb_in_token);
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

void write_prefix_histograms(BitWriter& w, const std::uint32_t* histogram, std::size_t length,
                             std::size_t num_contexts, const HybridUintConfig& config,
                             std::uint8_t* depth, std::uint16_t* bits) {
    w.write(1, 0);  // lz77.enabled = false

    // Context map: only present when there is more than one context. A single
    // clustered histogram is signalled with the simple form (bits_per_entry=0).
    if (num_contexts > 1) {
        w.write(1, 1);  // is_simple
        w.write(2, 0);  // bits_per_entry = 0 -> all contexts map to histogram 0
    }

    w.write(1, 1);  // use_prefix_code
    encode_uint_config(w, config, PREFIX_MAX_BITS);

    store_var_len_uint16(w, length - 1);

    for (std::size_t i{0}; i < length; ++i) {
        depth[i] = 0;
        bits[i] = 0;
    }
    if (length > 1) {
        build_and_store_huffman_tree(histogram, length, depth, bits, w);
    }
}

void write_clustered_prefix_histograms(BitWriter& w, const std::uint8_t* context_map,
                                       std::size_t num_contexts, std::size_t num_clusters,
                                       const std::uint32_t* cluster_histograms, std::size_t stride,
                                       const HybridUintConfig& config, std::uint8_t* depth,
                                       std::uint16_t* bits) {
    assert(num_clusters >= 1 && num_clusters <= 8);

    w.write(1, 0);  // lz77.enabled = false

    if (num_contexts > 1) {
        const std::size_t bits_per_entry{ceil_log2(num_clusters)};
        w.write(1, 1);  // is_simple
        w.write(2, bits_per_entry);
        if (bits_per_entry != 0) {
            for (std::size_t i{0}; i < num_contexts; ++i) {
                assert(context_map[i] < num_clusters);
                w.write(bits_per_entry, context_map[i]);
            }
        }
    }

    w.write(1, 1);  // use_prefix_code
    for (std::size_t c{0}; c < num_clusters; ++c) {
        encode_uint_config(w, config, PREFIX_MAX_BITS);
    }

    std::size_t alpha[8]{};
    for (std::size_t c{0}; c < num_clusters; ++c) {
        const std::uint32_t* h{cluster_histograms + c * stride};
        std::size_t a{1};
        for (std::size_t s{0}; s < stride; ++s) {
            if (h[s]) {
                a = s + 1;
            }
        }
        alpha[c] = a;
        store_var_len_uint16(w, a - 1);
    }

    for (std::size_t i{0}; i < num_clusters * stride; ++i) {
        depth[i] = 0;
        bits[i] = 0;
    }
    for (std::size_t c{0}; c < num_clusters; ++c) {
        if (alpha[c] > 1) {
            build_and_store_huffman_tree(cluster_histograms + c * stride, alpha[c],
                                         depth + c * stride, bits + c * stride, w);
        }
    }
}

}  // namespace cujpegxl::bitstream
