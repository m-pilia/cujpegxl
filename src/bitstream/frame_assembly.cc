// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frame_assembly.h"

#include "bit_writer.h"
#include "codestream.h"
#include "field_coder.h"
#include "histogram_writer.h"
#include "hybrid_uint.h"
#include "serialize_impl.h"
#include "src/ac_context.h"
#include "src/dc_predict.h"
#include "src/vardct_layout.h"

namespace cujpegxl::bitstream {
namespace {

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

std::vector<std::uint8_t> build_codestream_head(std::uint32_t width, std::uint32_t height,
                                                const std::vector<std::uint32_t>& section_sizes) {
    BitWriter w{};
    write_codestream_headers(w, width, height);
    write_frame_header(w);
    write_toc_multi_section(w, section_sizes);
    return w.bytes();
}

}  // namespace cujpegxl::bitstream
