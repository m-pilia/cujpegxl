// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frame_assembly.h"

#include <algorithm>
#include <cassert>

#include "ans.h"
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

AcGlobalBuilder::AcGlobalBuilder(const AcClustering& clustering, std::size_t num_ac_groups,
                                 TokenCode& code)
    : clustering_{clustering},
      num_ac_groups_{num_ac_groups},
      code_{code},
      cluster_bits_(clustering.num_clusters) {
    constexpr std::size_t MIN_ANS_TOKENS{100};
    std::uint64_t total{0};
    bool single_symbols{true};
    std::size_t max_alphabet{1};
    for (std::size_t c{0}; c < num_clusters(); ++c) {
        const std::uint32_t* h{clustering.histograms + c * AC_CONTEXT_ALPHABET};
        std::size_t symbols{0};
        for (std::size_t s{0}; s < AC_CONTEXT_ALPHABET; ++s) {
            total += h[s];
            symbols += h[s] != 0 ? 1 : 0;
        }
        single_symbols = single_symbols && symbols <= 1;
        max_alphabet = std::max(max_alphabet, detail::alphabet_size(h, AC_CONTEXT_ALPHABET));
    }
    code.use_prefix_code = total < MIN_ANS_TOKENS || single_symbols;
    log_alpha_size_ = std::max<std::size_t>(5, detail::ceil_log2_nonzero(max_alphabet));
    const std::size_t num_entries{num_clusters() << TOKEN_CLUSTER_SHIFT};
    if (code.use_prefix_code) {
        code.ans.clear();
        code.reverse_map.clear();
        code.prefix.assign(num_entries, PrefixEntryCode{});
    } else {
        code.ans.assign(num_entries, AnsEntryCode{});
        code.reverse_map.resize(num_clusters() * ANS_TAB_SIZE);
        code.prefix.clear();
    }
}

void AcGlobalBuilder::build_cluster(std::size_t c) {
    assert(c < num_clusters());
    if (code_.use_prefix_code) {
        return;
    }
    const HybridUintConfig config{};
    const std::uint32_t* h{clustering_.histograms + c * AC_CONTEXT_ALPHABET};
    AnsHistogram ans{};
    write_ans_histogram(cluster_bits_[c], h, detail::alphabet_size(h, AC_CONTEXT_ALPHABET),
                        log_alpha_size_, ans);
    std::copy(ans.reverse_map.begin(), ans.reverse_map.end(),
              code_.reverse_map.begin() + static_cast<std::ptrdiff_t>(c * ANS_TAB_SIZE));
    constexpr std::uint64_t ONE{std::uint64_t{1} << (32 + ANS_LOG_TAB_SIZE)};
    for (std::size_t s{0}; s < AC_CONTEXT_ALPHABET; ++s) {
        const std::uint64_t freq{ans.freq[s]};
        code_.ans[(c << TOKEN_CLUSTER_SHIFT) | s] = AnsEntryCode{
            freq != 0 ? (ONE + freq - 1) / freq : 0,
            static_cast<std::uint32_t>(c * ANS_TAB_SIZE + ans.offset[s]),
            static_cast<std::uint16_t>(freq),
            static_cast<std::uint16_t>(config.raw_bits(static_cast<std::uint32_t>(s)))};
    }
}

void AcGlobalBuilder::build_context_map() {
    if (code_.use_prefix_code) {
        return;
    }
    write_best_prefix_context_map(context_map_bits_, clustering_.context_map, AC_NUM_CONTEXTS,
                                  num_clusters());
}

std::vector<std::uint8_t> AcGlobalBuilder::finish() {
    const HybridUintConfig config{};
    BitWriter w{};
    detail::serialize_ac_global_head(w, num_ac_groups_);
    if (code_.use_prefix_code) {
        std::vector<std::uint8_t> depth(num_clusters() * AC_CONTEXT_ALPHABET, 0);
        std::vector<std::uint16_t> bits(num_clusters() * AC_CONTEXT_ALPHABET, 0);
        write_clustered_prefix_histograms(w, clustering_.context_map, AC_NUM_CONTEXTS,
                                          num_clusters(), clustering_.histograms,
                                          AC_CONTEXT_ALPHABET, config, depth.data(), bits.data());
        for (std::size_t c{0}; c < num_clusters(); ++c) {
            for (std::size_t s{0}; s < AC_CONTEXT_ALPHABET; ++s) {
                const std::size_t i{c * AC_CONTEXT_ALPHABET + s};
                code_.prefix[(c << TOKEN_CLUSTER_SHIFT) | s] = PrefixEntryCode{
                    bits[i], depth[i],
                    static_cast<std::uint8_t>(config.raw_bits(static_cast<std::uint32_t>(s)))};
            }
        }
        w.zero_pad_to_byte();
        return w.bytes();
    }

    w.write(1, 0);  // lz77.enabled = false
    w.append(context_map_bits_);
    w.write(1, 0);  // use_prefix_code
    w.write(2, log_alpha_size_ - 5);
    for (std::size_t c{0}; c < num_clusters(); ++c) {
        detail::encode_uint_config(w, config, log_alpha_size_);
    }
    for (const BitWriter& cluster : cluster_bits_) {
        w.append(cluster);
    }
    w.zero_pad_to_byte();
    return w.bytes();
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
