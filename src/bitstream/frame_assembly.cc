// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frame_assembly.h"

#include <algorithm>

#include "bit_writer.h"
#include "codestream.h"
#include "field_coder.h"
#include "histogram_writer.h"
#include "hybrid_uint.h"
#include "src/dc_predict.h"

namespace cujpegxl::bitstream {
namespace {

constexpr std::size_t AC_GROUP_BLOCKS = 32;
constexpr std::size_t DC_GROUP_BLOCKS = 256;

// libjxl default BlockCtxMap: 15 block contexts * (37 non-zero buckets + 458
// zero-density contexts) clustered to one histogram (see the oracle's
// vardct_frame.cc); the AC container's context map is sized to this count.
constexpr std::size_t NUM_AC_CONTEXTS = std::size_t{15} * (37 + 458);

// Modular MA tree contexts (libjxl modular/encoding/ma_common.h).
constexpr std::size_t NUM_TREE_CONTEXTS = 6;

constexpr U32Enc GLOBAL_SCALE_ENC{
    {BitsOffset(11, 1), BitsOffset(11, 2049), BitsOffset(12, 4097), BitsOffset(16, 8193)}};
constexpr U32Enc QUANT_DC_ENC{{Val(16), BitsOffset(5, 1), BitsOffset(8, 1), BitsOffset(16, 1)}};
constexpr U32Enc ORDER_ENC{{Val(0x5F), Val(0x13), Val(0), Bits(13)}};

std::size_t ceil_div(std::size_t a, std::size_t b) {
    return (a + b - 1) / b;
}

std::size_t ceil_log2_nonzero(std::size_t x) {
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

// Trailing-trimmed alphabet size of a histogram: index of the last non-zero
// symbol plus one, or 1 if empty (matching the host EntropyEncoder, which grows
// the histogram to max_symbol + 1).
std::size_t alphabet_size(const std::uint32_t* histogram, std::size_t stride) {
    std::size_t last{0};
    for (std::size_t s{0}; s < stride; ++s) {
        if (histogram[s]) {
            last = s + 1;
        }
    }
    return last == 0 ? 1 : last;
}

// Writes GroupHeader + single-leaf MA tree (with `predictor`) + the channel data
// container's histogram description for a raw sample histogram, filling
// depth/bits with the data prefix code. Reproduces the oracle modular.cc
// write_modular_header for the device (which emits the data tokens). Not
// byte-aligned on exit.
void write_modular_header(BitWriter& w, const std::uint32_t* data_hist, std::size_t data_len,
                          int predictor, std::vector<std::uint8_t>& depth,
                          std::vector<std::uint16_t>& bits) {
    w.write(1, 0);  // use_global_tree = false
    w.write(1, 1);  // weighted::Header all_default = true
    w.write(2, 0);  // num_transforms = 0

    // Single leaf: property=-1 (token 0), predictor, offset 0, mul_log 0,
    // mul_bits 0. Every token value is below the hybrid split, so symbol == value
    // with no extra bits. For Predictor::Zero all tokens are 0 (a single-symbol
    // zero-bit code, emitting nothing); Gradient adds the symbol 5.
    const std::uint32_t tree_tokens[NUM_TREE_CONTEXTS - 1]{
        0, static_cast<std::uint32_t>(predictor), 0, 0, 0};
    std::uint32_t tree_hist[16]{};
    std::size_t tree_len{1};
    for (std::uint32_t t : tree_tokens) {
        ++tree_hist[t];
        tree_len = std::max(tree_len, static_cast<std::size_t>(t) + 1);
    }
    std::uint8_t tree_depth[16]{};
    std::uint16_t tree_bits[16]{};
    write_prefix_histograms(w, tree_hist, tree_len, NUM_TREE_CONTEXTS, HybridUintConfig{},
                            tree_depth, tree_bits);
    for (std::uint32_t t : tree_tokens) {
        w.write(tree_depth[t], tree_bits[t]);
    }

    depth.assign(data_len, 0);
    bits.assign(data_len, 0);
    write_prefix_histograms(w, data_hist, data_len, 1, HybridUintConfig{}, depth.data(),
                            bits.data());
}

void place_code(std::vector<std::uint8_t>& depth_flat, std::vector<std::uint16_t>& bits_flat,
                std::size_t g, const std::vector<std::uint8_t>& depth,
                const std::vector<std::uint16_t>& bits) {
    for (std::size_t i{0}; i < depth.size(); ++i) {
        depth_flat[g * HISTOGRAM_STRIDE + i] = depth[i];
        bits_flat[g * HISTOGRAM_STRIDE + i] = bits[i];
    }
}

}  // namespace

std::size_t ac_group_count(std::size_t width, std::size_t height) {
    return ceil_div(width / 8, AC_GROUP_BLOCKS) * ceil_div(height / 8, AC_GROUP_BLOCKS);
}

std::size_t dc_group_count(std::size_t width, std::size_t height) {
    return ceil_div(width / 8, DC_GROUP_BLOCKS) * ceil_div(height / 8, DC_GROUP_BLOCKS);
}

AcGlobalResult build_ac_global(const std::uint32_t* ac_histogram, std::size_t num_ac_groups) {
    const std::size_t alpha{alphabet_size(ac_histogram, HISTOGRAM_STRIDE)};
    AcGlobalResult out{};
    out.depth.assign(alpha, 0);
    out.bits.assign(alpha, 0);

    BitWriter w{};
    write_bool(w, true);                                 // DequantMatrices::Decode all_default
    write_bits(w, ceil_log2_nonzero(num_ac_groups), 0);  // num_histograms - 1
    write_u32(w, ORDER_ENC, 0);                          // used_orders = 0 (natural)
    write_prefix_histograms(w, ac_histogram, alpha, NUM_AC_CONTEXTS, HybridUintConfig{},
                            out.depth.data(), out.bits.data());
    w.zero_pad_to_byte();
    out.section = w.bytes();
    return out;
}

std::vector<std::uint8_t> build_dc_global(const QuantParams& qp) {
    BitWriter w{};
    write_bool(w, true);  // DequantMatrices::DecodeDC all_default
    write_u32(w, GLOBAL_SCALE_ENC, qp.global_scale);
    write_u32(w, QUANT_DC_ENC, qp.quant_dc);  // Quantizer::Decode
    write_bool(w, true);                      // DecodeBlockCtxMap is_default
    write_bool(w, true);                      // ColorCorrelation::DecodeDC all_default
    write_bool(w, false);                     // modular global has_tree = 0
    w.zero_pad_to_byte();
    return w.bytes();
}

DcGroupBlobs build_dc_group_blobs(std::size_t width, std::size_t height,
                                  const std::uint32_t* dc_histograms,
                                  const std::uint32_t* acmeta_histograms,
                                  const std::size_t* first_block_counts) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t xdg{ceil_div(bw, DC_GROUP_BLOCKS)};
    const std::size_t ydg{ceil_div(bh, DC_GROUP_BLOCKS)};
    const std::size_t num_groups{xdg * ydg};

    DcGroupBlobs out{};
    out.dc_depth.assign(num_groups * HISTOGRAM_STRIDE, 0);
    out.dc_bits.assign(num_groups * HISTOGRAM_STRIDE, 0);
    out.acmeta_depth.assign(num_groups * HISTOGRAM_STRIDE, 0);
    out.acmeta_bits.assign(num_groups * HISTOGRAM_STRIDE, 0);
    out.blob_pre_off.assign(num_groups, 0);
    out.blob_pre_bits.assign(num_groups, 0);
    out.blob_mid_off.assign(num_groups, 0);
    out.blob_mid_bits.assign(num_groups, 0);

    for (std::size_t g{0}; g < num_groups; ++g) {
        const std::size_t gx{g % xdg};
        const std::size_t gy{g / xdg};
        const std::size_t dgw{std::min(DC_GROUP_BLOCKS, bw - gx * DC_GROUP_BLOCKS)};
        const std::size_t dgh{std::min(DC_GROUP_BLOCKS, bh - gy * DC_GROUP_BLOCKS)};
        const std::size_t total_blocks{dgw * dgh};
        const std::size_t count{first_block_counts == nullptr ? total_blocks
                                                              : first_block_counts[g]};

        // VarDCTDC: extra_precision + modular header over the 3 DC channels. The
        // DC histogram is the device's; the modular header emits its description.
        const std::uint32_t* dc_hist{dc_histograms + g * HISTOGRAM_STRIDE};
        const std::size_t dc_alpha{alphabet_size(dc_hist, HISTOGRAM_STRIDE)};
        std::vector<std::uint8_t> dc_depth{};
        std::vector<std::uint16_t> dc_bits{};
        BitWriter pre{};
        write_bits(pre, 2, 0);  // DecodeVarDCTDC extra_precision = 0
        write_modular_header(pre, dc_hist, dc_alpha, DC_PREDICTOR_GRADIENT, dc_depth, dc_bits);
        out.blob_pre_off[g] = static_cast<std::uint32_t>(out.blob_pre.size());
        out.blob_pre_bits[g] = static_cast<std::uint32_t>(pre.bits_written());
        const std::vector<std::uint8_t>& pre_bytes{pre.bytes()};
        out.blob_pre.insert(out.blob_pre.end(), pre_bytes.begin(), pre_bytes.end());
        place_code(out.dc_depth, out.dc_bits, g, dc_depth, dc_bits);

        // AcMetadata: content-derived from the per-block quant field (the ACS+QF
        // row-1 samples). The full token histogram (structural zeros plus the
        // per-block quant-field tokens) is supplied by the device kernel; build
        // the prefix code from it directly.
        const std::uint32_t* meta_hist{acmeta_histograms + g * HISTOGRAM_STRIDE};
        const std::size_t meta_alpha{alphabet_size(meta_hist, HISTOGRAM_STRIDE)};
        std::vector<std::uint8_t> meta_depth{};
        std::vector<std::uint16_t> meta_bits{};
        BitWriter mid{};
        write_bits(mid, ceil_log2_nonzero(total_blocks), static_cast<std::uint32_t>(count - 1));
        write_modular_header(mid, meta_hist, meta_alpha, DC_PREDICTOR_ZERO, meta_depth, meta_bits);
        out.blob_mid_off[g] = static_cast<std::uint32_t>(out.blob_mid.size());
        out.blob_mid_bits[g] = static_cast<std::uint32_t>(mid.bits_written());
        const std::vector<std::uint8_t>& mid_bytes{mid.bytes()};
        out.blob_mid.insert(out.blob_mid.end(), mid_bytes.begin(), mid_bytes.end());
        place_code(out.acmeta_depth, out.acmeta_bits, g, meta_depth, meta_bits);
    }
    return out;
}

std::vector<std::uint8_t> build_codestream_head(std::uint32_t width, std::uint32_t height,
                                                const std::vector<std::uint32_t>& section_sizes) {
    BitWriter w{};
    write_codestream_headers(w, width, height);
    write_frame_header(w);
    write_toc_multi_section(w, section_sizes);
    return w.bytes();
}

}  // namespace cujpegxl::bitstream
