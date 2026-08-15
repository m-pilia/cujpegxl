// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "ans_encoder.h"

#include <cassert>

namespace cujpegxl::bitstream {
namespace {

struct AliasSymbol {
    std::size_t value;
    std::size_t offset;
};

AliasSymbol lookup_alias(const AnsDistribution& distribution, std::size_t value) {
    const std::size_t table_index{value >> 4};
    const std::size_t position{value & 15};
    const AnsAliasEntry& entry{distribution.aliases[table_index]};
    const bool use_right{position >= entry.cutoff};
    return {
        .value = use_right ? entry.right_value : table_index,
        .offset = position + (use_right ? entry.right_offset : 0),
    };
}

}  // namespace

void build_ans_encoding_table(const AnsDistribution& distribution, AnsEncodingTable& table) {
    table = {};
    for (std::size_t symbol{0}; symbol < ANS_ALPHABET_SIZE; ++symbol) {
        table.frequencies[symbol] = distribution.counts[symbol];
        table.offsets[symbol + 1] =
            static_cast<std::uint16_t>(table.offsets[symbol] + distribution.counts[symbol]);
    }
    assert(table.offsets.back() == ANS_TABLE_SIZE);
    for (std::size_t value{0}; value < ANS_TABLE_SIZE; ++value) {
        const AliasSymbol symbol{lookup_alias(distribution, value)};
        assert(symbol.value < ANS_ALPHABET_SIZE);
        assert(symbol.offset < table.frequencies[symbol.value]);
        table.reverse_map[table.offsets[symbol.value] + symbol.offset] =
            static_cast<std::uint16_t>(value);
    }
}

AnsStateTransition ans_put_symbol(std::uint32_t state, const AnsEncodingTable& table,
                                  std::size_t symbol) {
    assert(symbol < ANS_ALPHABET_SIZE);
    const std::uint32_t frequency{table.frequencies[symbol]};
    assert(frequency != 0);

    AnsStateTransition result{.state = state};
    if ((result.state >> 20) >= frequency) {
        result.renormalization_bits = static_cast<std::uint16_t>(result.state);
        result.state >>= 16;
        result.renormalized = true;
    }
    const std::uint32_t quotient{result.state / frequency};
    const std::uint32_t remainder{result.state % frequency};
    result.state = (quotient << 12) + table.reverse_map[table.offsets[symbol] + remainder];
    return result;
}

void write_ans_tokens(BitWriter& w, const AnsToken* tokens, std::size_t num_tokens,
                      const std::uint8_t* context_map, std::size_t num_contexts,
                      const AnsEncodingTable* tables, std::size_t num_clusters,
                      const HybridUintConfig& config, std::uint32_t* renormalization_scratch) {
    assert(num_tokens == 0 || (tokens != nullptr && renormalization_scratch != nullptr));
    assert(context_map != nullptr && tables != nullptr);
    assert(num_contexts >= 1);
    assert(num_clusters >= 1 && num_clusters <= 256);

    std::uint32_t state{ANS_INITIAL_STATE};
    for (std::size_t i{num_tokens}; i > 0; --i) {
        const AnsToken& token{tokens[i - 1]};
        assert(token.context < num_contexts);
        const std::size_t cluster{context_map[token.context]};
        assert(cluster < num_clusters);
        std::uint32_t symbol{};
        std::uint32_t nbits{};
        std::uint32_t bits{};
        config.encode(token.value, symbol, nbits, bits);
        assert(symbol < ANS_ALPHABET_SIZE);
        const AnsStateTransition transition{ans_put_symbol(state, tables[cluster], symbol)};
        state = transition.state;
        renormalization_scratch[i - 1] =
            transition.renormalized ? 0x10000u | transition.renormalization_bits : 0;
    }

    w.write(32, state);
    for (std::size_t i{0}; i < num_tokens; ++i) {
        const AnsToken& token{tokens[i]};
        std::uint32_t symbol{};
        std::uint32_t nbits{};
        std::uint32_t bits{};
        config.encode(token.value, symbol, nbits, bits);
        if ((renormalization_scratch[i] & 0x10000u) != 0) {
            w.write(16, renormalization_scratch[i] & 0xffffu);
        }
        if (nbits != 0) {
            w.write(nbits, bits);
        }
    }
}

}  // namespace cujpegxl::bitstream
