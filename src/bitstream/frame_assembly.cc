// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frame_assembly.h"

#include <algorithm>

#include "bit_writer.h"
#include "codestream.h"
#include "field_coder.h"
#include "histogram_writer.h"
#include "hybrid_uint.h"
#include "serialize_impl.h"
#include "src/ac_context.h"
#include "src/dc_predict.h"

namespace cujpegxl::bitstream {
namespace {

constexpr std::size_t AC_GROUP_BLOCKS = 32;
constexpr std::size_t DC_GROUP_BLOCKS = 256;

constexpr U32Enc GLOBAL_SCALE_ENC{
    {BitsOffset(11, 1), BitsOffset(11, 2049), BitsOffset(12, 4097), BitsOffset(16, 8193)}};
constexpr U32Enc QUANT_DC_ENC{{Val(16), BitsOffset(5, 1), BitsOffset(8, 1), BitsOffset(16, 1)}};

std::size_t ceil_div(std::size_t a, std::size_t b) {
    return (a + b - 1) / b;
}

}  // namespace

std::size_t ac_group_count(std::size_t width, std::size_t height) {
    return ceil_div(width / 8, AC_GROUP_BLOCKS) * ceil_div(height / 8, AC_GROUP_BLOCKS);
}

std::size_t dc_group_count(std::size_t width, std::size_t height) {
    return ceil_div(width / 8, DC_GROUP_BLOCKS) * ceil_div(height / 8, DC_GROUP_BLOCKS);
}

AcGlobalResult build_ac_global(const std::uint32_t* ac_histogram, std::size_t num_ac_groups) {
    std::vector<std::uint8_t> context_map(AC_NUM_CONTEXTS);
    for (std::size_t i{0}; i < AC_NUM_CONTEXTS; ++i) {
        context_map[i] = static_cast<std::uint8_t>(ac_cluster(static_cast<std::uint32_t>(i)));
    }
    return build_ac_global(ac_histogram, context_map.data(), AC_NUM_CLUSTERS, num_ac_groups);
}

AcGlobalResult build_ac_global(const std::uint32_t* ac_histograms, const std::uint8_t* context_map,
                               std::size_t num_clusters, std::size_t num_ac_groups) {
    AcGlobalResult out{};
    out.depth.assign(num_clusters * HISTOGRAM_STRIDE, 0);
    out.bits.assign(num_clusters * HISTOGRAM_STRIDE, 0);

    BitWriter w{};
    detail::serialize_ac_global(w, ac_histograms, context_map, num_clusters, num_ac_groups,
                                out.depth.data(), out.bits.data());
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

AcContextMapBlob build_ac_context_map_blob() {
    std::vector<std::uint8_t> context_map(AC_NUM_CONTEXTS);
    for (std::size_t i{0}; i < AC_NUM_CONTEXTS; ++i) {
        context_map[i] = static_cast<std::uint8_t>(ac_cluster(static_cast<std::uint32_t>(i)));
    }
    BitWriter w{};
    write_best_prefix_context_map(w, context_map.data(), AC_NUM_CONTEXTS, AC_NUM_CLUSTERS);
    AcContextMapBlob out{};
    out.nbits = w.bits_written();
    out.bytes = w.bytes();
    return out;
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
        BitWriter pre{};
        detail::serialize_dc_pre(pre, dc_hist, out.dc_depth.data() + g * HISTOGRAM_STRIDE,
                                 out.dc_bits.data() + g * HISTOGRAM_STRIDE);
        out.blob_pre_off[g] = static_cast<std::uint32_t>(out.blob_pre.size());
        out.blob_pre_bits[g] = static_cast<std::uint32_t>(pre.bits_written());
        const std::vector<std::uint8_t>& pre_bytes{pre.bytes()};
        out.blob_pre.insert(out.blob_pre.end(), pre_bytes.begin(), pre_bytes.end());

        // AcMetadata: content-derived from the per-block quant field (the ACS+QF
        // row-1 samples). The full token histogram (structural zeros plus the
        // per-block quant-field tokens) is supplied by the device kernel; build
        // the prefix code from it directly.
        const std::uint32_t* meta_hist{acmeta_histograms + g * HISTOGRAM_STRIDE};
        BitWriter mid{};
        detail::serialize_dc_mid(mid, meta_hist, total_blocks, count,
                                 out.acmeta_depth.data() + g * HISTOGRAM_STRIDE,
                                 out.acmeta_bits.data() + g * HISTOGRAM_STRIDE);
        out.blob_mid_off[g] = static_cast<std::uint32_t>(out.blob_mid.size());
        out.blob_mid_bits[g] = static_cast<std::uint32_t>(mid.bits_written());
        const std::vector<std::uint8_t>& mid_bytes{mid.bytes()};
        out.blob_mid.insert(out.blob_mid.end(), mid_bytes.begin(), mid_bytes.end());
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
