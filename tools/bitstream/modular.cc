// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "modular.h"

#include "entropy_encoder.h"
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

void write_group_header(BitWriter& w) {
    w.write(1, 0);  // use_global_tree = false
    w.write(1, 1);  // weighted::Header all_default = true
    w.write(2, 0);  // num_transforms U32 selector Val(0) -> 0 transforms
}

// A single leaf: property = -1 (encoded as prop1 = 0), the given predictor,
// offset 0, multiplier 1 (mul_log 0, mul_bits 0).
void write_single_leaf_tree(BitWriter& w, int predictor) {
    EntropyEncoder tree{NUM_TREE_CONTEXTS};
    tree.add_token(PROPERTY_CONTEXT, 0);
    tree.add_token(PREDICTOR_CONTEXT, static_cast<std::uint32_t>(predictor));
    tree.add_token(OFFSET_CONTEXT, 0);
    tree.add_token(MULTIPLIER_LOG_CONTEXT, 0);
    tree.add_token(MULTIPLIER_BITS_CONTEXT, 0);
    tree.write(w);
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

void write_modular_header(BitWriter& w, const std::vector<ModularChannel>& channels,
                          EntropyEncoder& data, int predictor) {
    write_group_header(w);
    write_single_leaf_tree(w, predictor);

    for (const ModularChannel& ch : channels) {
        if (ch.w == 0 || ch.h == 0) {
            continue;
        }
        for (std::size_t y{0}; y < ch.h; ++y) {
            for (std::size_t x{0}; x < ch.w; ++x) {
                const std::int32_t v{ch.pixels[y * ch.w + x]};
                data.add_token(0, pack_signed(v - channel_prediction(ch, x, y, predictor)));
            }
        }
    }
    data.write_histograms(w);
}

void write_modular_tokens(BitWriter& w, const EntropyEncoder& data) {
    data.write_tokens(w);
}

void write_modular_image(BitWriter& w, const std::vector<ModularChannel>& channels, int predictor) {
    // (tree.size() + 1) / 2 == 1 histogram context for the channel samples.
    EntropyEncoder data{1};
    write_modular_header(w, channels, data, predictor);
    write_modular_tokens(w, data);
}

}  // namespace cujpegxl::bitstream
