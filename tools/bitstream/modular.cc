// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "modular.h"

#include <cassert>
#include <numeric>

#include "entropy_encoder.h"
#include "src/bitstream/frame_assembly.h"
#include "src/bitstream/histogram_writer.h"
#include "src/bitstream/hybrid_uint.h"

namespace cujpegxl::bitstream {
namespace {

// MATreeContext indices (libjxl modular/encoding/ma_common.h).
enum : std::size_t {
    SPLIT_VAL_CONTEXT = 0,
    PROPERTY_CONTEXT = 1,
    PREDICTOR_CONTEXT = 2,
    OFFSET_CONTEXT = 3,
    MULTIPLIER_LOG_CONTEXT = 4,
    MULTIPLIER_BITS_CONTEXT = 5,
    NUM_TREE_CONTEXTS = 6,
};

constexpr std::uint32_t CHANNEL_PROPERTY{0};

void write_group_header(BitWriter& w) {
    w.write(1, 0);  // use_global_tree = false
    w.write(1, 1);  // weighted::Header all_default = true
    w.write(2, 0);  // num_transforms U32 selector Val(0) -> 0 transforms
}

std::size_t num_leaves(const ModularTree& tree) {
    return tree.split_channel ? 2 : 1;
}

// libjxl DecodeTree numbers the leaves in decode order and takes a split's first
// child, decoded first, when the property exceeds the split value.
std::size_t channel_leaf(const ModularTree& tree, std::size_t channel) {
    if (!tree.split_channel) {
        return 0;
    }
    return channel > *tree.split_channel ? 0 : 1;
}

// A split node is property + 1 and the PackSigned split value; a leaf is
// property 0 (encoded as prop1 = 0), the predictor, offset 0, multiplier 1
// (mul_log 0, mul_bits 0).
void write_tree(BitWriter& w, const ModularTree& tree) {
    EntropyEncoder tokens{NUM_TREE_CONTEXTS};
    if (tree.split_channel) {
        tokens.add_token(PROPERTY_CONTEXT, CHANNEL_PROPERTY + 1);
        tokens.add_token(SPLIT_VAL_CONTEXT,
                         pack_signed(static_cast<std::int32_t>(*tree.split_channel)));
    }
    for (std::size_t l{0}; l < num_leaves(tree); ++l) {
        tokens.add_token(PROPERTY_CONTEXT, 0);
        tokens.add_token(PREDICTOR_CONTEXT, static_cast<std::uint32_t>(tree.predictor));
        tokens.add_token(OFFSET_CONTEXT, 0);
        tokens.add_token(MULTIPLIER_LOG_CONTEXT, 0);
        tokens.add_token(MULTIPLIER_BITS_CONTEXT, 0);
    }
    tokens.write(w);
}

std::int32_t channel_prediction(const ModularChannel& ch, std::size_t x, std::size_t y,
                                int predictor) {
    if (predictor != DC_PREDICTOR_GRADIENT) {
        return 0;
    }
    const bool hl{x > 0};
    const bool ht{y > 0};
    const std::int32_t wv{hl ? ch.pixels[y * ch.w + (x - 1)] : 0};
    const std::int32_t nv{ht ? ch.pixels[(y - 1) * ch.w + x] : 0};
    const std::int32_t nwv{(hl && ht) ? ch.pixels[(y - 1) * ch.w + (x - 1)] : 0};
    return gradient_predict(hl, ht, wv, nv, nwv);
}

}  // namespace

void ModularData::write_tokens(BitWriter& w) const {
    for (const Token& t : tokens) {
        const std::size_t idx{t.leaf * HISTOGRAM_STRIDE + t.symbol};
        w.write(depth[idx], bits[idx]);
        if (t.nbits) {
            w.write(t.nbits, t.bits);
        }
    }
}

ModularData write_modular_header(BitWriter& w, const std::vector<ModularChannel>& channels,
                                 const ModularTree& tree) {
    write_group_header(w);
    write_tree(w, tree);

    const std::size_t leaves{num_leaves(tree)};
    const HybridUintConfig config{};
    ModularData data{};
    data.histograms.assign(leaves * HISTOGRAM_STRIDE, 0);
    for (std::size_t c{0}; c < channels.size(); ++c) {
        const ModularChannel& ch{channels[c]};
        if (ch.w == 0 || ch.h == 0) {
            continue;
        }
        const std::size_t leaf{channel_leaf(tree, c)};
        for (std::size_t y{0}; y < ch.h; ++y) {
            for (std::size_t x{0}; x < ch.w; ++x) {
                const std::int32_t v{ch.pixels[y * ch.w + x]};
                ModularData::Token t{leaf, 0, 0, 0};
                config.encode(pack_signed(v - channel_prediction(ch, x, y, tree.predictor)),
                              t.symbol, t.nbits, t.bits);
                assert(t.symbol < HISTOGRAM_STRIDE);
                data.tokens.push_back(t);
                ++data.histograms[leaf * HISTOGRAM_STRIDE + t.symbol];
            }
        }
    }

    std::vector<std::uint8_t> context_map(leaves);
    std::iota(context_map.begin(), context_map.end(), std::uint8_t{0});
    data.depth.assign(leaves * HISTOGRAM_STRIDE, 0);
    data.bits.assign(leaves * HISTOGRAM_STRIDE, 0);
    write_clustered_prefix_histograms(w, context_map.data(), leaves, leaves, data.histograms.data(),
                                      HISTOGRAM_STRIDE, config, data.depth.data(),
                                      data.bits.data());
    return data;
}

void write_modular_image(BitWriter& w, const std::vector<ModularChannel>& channels,
                         const ModularTree& tree) {
    write_modular_header(w, channels, tree).write_tokens(w);
}

}  // namespace cujpegxl::bitstream
