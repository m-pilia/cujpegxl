// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "entropy_selector.h"

#include <cassert>

#include "histogram_writer.h"

namespace cujpegxl::bitstream {
namespace {

void append_writer_bits(BitWriter& destination, const BitWriter& source) {
    std::size_t remaining{source.bits_written()};
    std::size_t byte{0};
    while (remaining != 0) {
        const std::size_t take{remaining < 8 ? remaining : 8};
        destination.write(take, source.bytes()[byte]);
        remaining -= take;
        ++byte;
    }
}

void write_prefix_tokens(BitWriter& w, const AnsToken* tokens, std::size_t num_tokens,
                         const std::uint8_t* context_map, std::size_t num_contexts,
                         std::size_t num_clusters, std::size_t stride,
                         const HybridUintConfig& config, const std::uint8_t* depth,
                         const std::uint16_t* bits) {
    for (std::size_t i{0}; i < num_tokens; ++i) {
        assert(tokens[i].context < num_contexts);
        const std::size_t cluster{context_map[tokens[i].context]};
        assert(cluster < num_clusters);
        std::uint32_t symbol{};
        std::uint32_t nbits{};
        std::uint32_t extra_bits{};
        config.encode(tokens[i].value, symbol, nbits, extra_bits);
        assert(symbol < stride);
        w.write(depth[cluster * stride + symbol], bits[cluster * stride + symbol]);
        if (nbits != 0) {
            w.write(nbits, extra_bits);
        }
    }
}

}  // namespace

EntropySelectionResult write_best_clustered_entropy(
    BitWriter& w, const AnsToken* tokens, std::size_t num_tokens,
    const std::uint8_t* context_map, std::size_t num_contexts, std::size_t num_clusters,
    const std::uint32_t* cluster_histograms, std::size_t stride,
    const HybridUintConfig& config, std::uint8_t* prefix_depth, std::uint16_t* prefix_bits,
    AnsDistribution* ans_distributions, AnsEncodingTable* ans_tables,
    std::uint32_t* ans_renormalization_scratch) {
    assert(num_tokens == 0 || tokens != nullptr);
    assert(context_map != nullptr && cluster_histograms != nullptr);
    assert(prefix_depth != nullptr && prefix_bits != nullptr);
    assert(ans_distributions != nullptr && ans_tables != nullptr);
    assert(num_tokens == 0 || ans_renormalization_scratch != nullptr);

    BitWriter prefix{};
    write_clustered_prefix_histograms(prefix, context_map, num_contexts, num_clusters,
                                      cluster_histograms, stride, config, prefix_depth,
                                      prefix_bits);
    write_prefix_tokens(prefix, tokens, num_tokens, context_map, num_contexts, num_clusters,
                        stride, config, prefix_depth, prefix_bits);

    BitWriter ans{};
    write_clustered_ans_histograms(ans, context_map, num_contexts, num_clusters,
                                   cluster_histograms, stride, config, ans_distributions);
    for (std::size_t cluster{0}; cluster < num_clusters; ++cluster) {
        build_ans_encoding_table(ans_distributions[cluster], ans_tables[cluster]);
    }
    write_ans_tokens(ans, tokens, num_tokens, context_map, num_contexts, ans_tables,
                     num_clusters, config, ans_renormalization_scratch);

    const EntropySelectionResult result{
        .mode = select_entropy_mode(prefix.bits_written(), ans.bits_written()),
        .prefix_bits = prefix.bits_written(),
        .ans_bits = ans.bits_written(),
    };
    append_writer_bits(w, result.mode == EntropyMode::PREFIX ? prefix : ans);
    return result;
}

}  // namespace cujpegxl::bitstream
