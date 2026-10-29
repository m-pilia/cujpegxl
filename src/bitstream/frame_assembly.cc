// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frame_assembly.h"

#include <algorithm>
#include <array>
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
constexpr U32Enc USED_ORDERS_ENC{{Val(0x5F), Val(0x13), Val(0), Bits(13)}};

// libjxl kPermutationContexts. The hybrid-uint tokens of the permutation values
// (below 1024) stay below PERMUTATION_ALPHABET.
constexpr std::size_t PERMUTATION_CONTEXTS{8};
constexpr std::size_t PERMUTATION_ALPHABET{64};

std::size_t ceil_div(std::size_t a, std::size_t b) {
    return (a + b - 1) / b;
}

struct PermutationToken {
    std::uint32_t context;
    std::uint32_t value;
};

// libjxl CoeffOrderContext.
std::uint32_t permutation_context(std::uint32_t value) {
    return value == 0 ? 0
                      : std::min(static_cast<std::uint32_t>(1 + ac_floor_log2(value)),
                                 static_cast<std::uint32_t>(PERMUTATION_CONTEXTS - 1));
}

// libjxl TokenizePermutation of the orders of transform side `side`, in channel
// order X, Y, B: the length of each Lehmer code past the low-frequency corner up
// to its last non-zero entry, then those entries.
void tokenize_orders(const CoeffOrders& orders, int side, std::vector<PermutationToken>& tokens) {
    const auto size{static_cast<std::uint32_t>(side * side)};
    const auto skip{static_cast<std::uint32_t>((side / 8) * (side / 8))};
    for (int c{0}; c < 3; ++c) {
        const std::uint16_t* const lehmer{orders.lehmer + coeff_order_offset(side, c)};
        std::uint32_t end{size};
        while (end > skip && lehmer[end - 1] == 0) {
            --end;
        }
        tokens.push_back({permutation_context(size), end - skip});
        std::uint32_t last{0};
        for (std::uint32_t i{skip}; i < end; ++i) {
            tokens.push_back({permutation_context(last), lehmer[i]});
            last = lehmer[i];
        }
    }
}

// used_orders and the permutations of the orders it marks, those of the sides
// with an order other than the natural one in any channel (libjxl
// EncodeCoeffOrders), prefix-coded with a histogram per context holding tokens.
void write_coeff_orders(BitWriter& w, const CoeffOrders& orders) {
    std::uint32_t used_orders{0};
    std::vector<PermutationToken> tokens{};
    for (int side : {8, 16, 32}) {
        const std::uint16_t* const lehmer{orders.lehmer + coeff_order_offset(side, 0)};
        if (std::any_of(lehmer, lehmer + 3 * side * side, [](std::uint16_t v) { return v != 0; })) {
            used_orders |= 1u << ac_strategy_order(side);
            tokenize_orders(orders, side, tokens);
        }
    }
    write_u32(w, USED_ORDERS_ENC, used_orders);
    if (used_orders == 0) {
        return;
    }

    const HybridUintConfig config{};
    std::array<std::uint32_t, PERMUTATION_CONTEXTS * PERMUTATION_ALPHABET> context_hist{};
    for (const PermutationToken& t : tokens) {
        std::uint32_t symbol{}, nbits{}, bits{};
        config.encode(t.value, symbol, nbits, bits);
        assert(symbol < PERMUTATION_ALPHABET);
        ++context_hist[t.context * PERMUTATION_ALPHABET + symbol];
    }
    std::array<std::uint8_t, PERMUTATION_CONTEXTS> context_map{};
    std::vector<std::uint32_t> cluster_hist{};
    for (std::size_t ctx{0}; ctx < PERMUTATION_CONTEXTS; ++ctx) {
        const auto first{context_hist.begin() +
                         static_cast<std::ptrdiff_t>(ctx * PERMUTATION_ALPHABET)};
        const auto last{first + static_cast<std::ptrdiff_t>(PERMUTATION_ALPHABET)};
        if (std::any_of(first, last, [](std::uint32_t n) { return n != 0; })) {
            context_map[ctx] =
                static_cast<std::uint8_t>(cluster_hist.size() / PERMUTATION_ALPHABET);
            cluster_hist.insert(cluster_hist.end(), first, last);
        }
    }
    const std::size_t num_clusters{cluster_hist.size() / PERMUTATION_ALPHABET};
    std::vector<std::uint8_t> depth(cluster_hist.size());
    std::vector<std::uint16_t> bits(cluster_hist.size());
    write_clustered_prefix_histograms(w, context_map.data(), PERMUTATION_CONTEXTS, num_clusters,
                                      cluster_hist.data(), PERMUTATION_ALPHABET, config,
                                      depth.data(), bits.data());
    for (const PermutationToken& t : tokens) {
        std::uint32_t symbol{}, nbits{}, raw{};
        config.encode(t.value, symbol, nbits, raw);
        const std::size_t i{context_map[t.context] * PERMUTATION_ALPHABET + symbol};
        w.write(depth[i], bits[i]);
        w.write(nbits, raw);
    }
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

AcGlobalBuilder::AcGlobalBuilder(const AcClustering& clustering, const CoeffOrders& orders,
                                 std::size_t num_ac_groups, TokenCode& code)
    : clustering_{clustering},
      orders_{orders},
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

void AcGlobalBuilder::build_coeff_orders() {
    write_coeff_orders(coeff_orders_bits_, orders_);
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
    w.append(coeff_orders_bits_);
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
