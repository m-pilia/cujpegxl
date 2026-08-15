// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "entropy_selector.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <jxl/memory_manager.h>

#include <gtest/gtest.h>

#include "lib/jxl/base/span.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"

namespace cujpegxl::bitstream {
namespace {

void* test_alloc(void*, size_t size) { return std::malloc(size); }
void test_free(void*, void* pointer) { std::free(pointer); }

struct EncodedBundle {
    BitWriter writer{};
    EntropySelectionResult selection{};
};

EncodedBundle encode(const std::vector<AnsToken>& tokens, std::size_t num_contexts,
                     const std::vector<std::uint8_t>& context_map, std::size_t num_clusters,
                     const HybridUintConfig& config) {
    std::vector<std::uint32_t> histograms(num_clusters * ANS_ALPHABET_SIZE, 0);
    for (const AnsToken& token : tokens) {
        std::uint32_t symbol{};
        std::uint32_t nbits{};
        std::uint32_t bits{};
        config.encode(token.value, symbol, nbits, bits);
        ++histograms[context_map[token.context] * ANS_ALPHABET_SIZE + symbol];
    }
    std::vector<std::uint8_t> prefix_depth(num_clusters * ANS_ALPHABET_SIZE);
    std::vector<std::uint16_t> prefix_bits(num_clusters * ANS_ALPHABET_SIZE);
    std::vector<AnsDistribution> distributions(num_clusters);
    std::vector<AnsEncodingTable> tables(num_clusters);
    std::vector<std::uint32_t> renormalization(tokens.size());

    EncodedBundle encoded{};
    encoded.selection = write_best_clustered_entropy(
        encoded.writer, tokens.data(), tokens.size(), context_map.data(), num_contexts,
        num_clusters, histograms.data(), ANS_ALPHABET_SIZE, config, prefix_depth.data(),
        prefix_bits.data(), distributions.data(), tables.data(), renormalization.data());
    return encoded;
}

void expect_decodes(const EncodedBundle& encoded, const std::vector<AnsToken>& tokens,
                    std::size_t num_contexts,
                    const std::vector<std::uint8_t>& expected_context_map) {
    BitWriter padded{encoded.writer};
    padded.zero_pad_to_byte();
    JxlMemoryManager memory_manager{nullptr, &test_alloc, &test_free};
    jxl::BitReader reader{jxl::Bytes(padded.bytes().data(), padded.bytes().size())};
    jxl::ANSCode code{};
    std::vector<std::uint8_t> context_map{};
    ASSERT_TRUE(jxl::DecodeHistograms(&memory_manager, &reader, num_contexts, &code,
                                      &context_map));
    EXPECT_EQ(context_map, expected_context_map);
    EXPECT_EQ(code.use_prefix_code, encoded.selection.mode == EntropyMode::PREFIX);
    auto symbol_reader_or{jxl::ANSSymbolReader::Create(&code, &reader)};
    ASSERT_TRUE(symbol_reader_or.ok());
    jxl::ANSSymbolReader symbol_reader{std::move(symbol_reader_or).value_()};
    for (const AnsToken& token : tokens) {
        const std::uint32_t value{static_cast<std::uint32_t>(
            symbol_reader.ReadHybridUint(token.context, &reader, context_map))};
        EXPECT_EQ(value, token.value);
    }
    EXPECT_TRUE(symbol_reader.CheckANSFinalState());
    EXPECT_TRUE(reader.Close());
}

TEST(EntropySelector, PrefixWinsSingleSymbolBundle) {
    const std::vector<AnsToken> tokens(1000, AnsToken{0, 17});
    const EncodedBundle encoded{encode(tokens, 1, {0}, 1, HybridUintConfig{8, 0, 0})};
    EXPECT_EQ(encoded.selection.mode, EntropyMode::PREFIX);
    EXPECT_EQ(encoded.writer.bits_written(), encoded.selection.prefix_bits);
    expect_decodes(encoded, tokens, 1, {0});
}

TEST(EntropySelector, AnsWinsSkewedLargeBundle) {
    std::vector<AnsToken> tokens{};
    tokens.insert(tokens.end(), 9000, AnsToken{0, 0});
    tokens.insert(tokens.end(), 900, AnsToken{0, 1});
    tokens.insert(tokens.end(), 100, AnsToken{0, 2});
    const EncodedBundle encoded{encode(tokens, 1, {0}, 1, HybridUintConfig{8, 0, 0})};
    EXPECT_EQ(encoded.selection.mode, EntropyMode::ANS);
    EXPECT_EQ(encoded.writer.bits_written(), encoded.selection.ans_bits);
    expect_decodes(encoded, tokens, 1, {0});
}

TEST(EntropySelector, PrefixWinsTies) {
    EXPECT_EQ(select_entropy_mode(100, 100), EntropyMode::PREFIX);
    EXPECT_EQ(select_entropy_mode(101, 100), EntropyMode::ANS);
}

TEST(EntropySelector, NeverLargerAcrossRepresentativeBundles) {
    constexpr std::size_t NUM_CONTEXTS = 23;
    constexpr std::size_t NUM_CLUSTERS = 4;
    std::vector<std::uint8_t> context_map(NUM_CONTEXTS);
    for (std::size_t i{0}; i < context_map.size(); ++i) {
        context_map[i] = static_cast<std::uint8_t>((i * 5) % NUM_CLUSTERS);
    }

    for (std::size_t sample{0}; sample < 8; ++sample) {
        std::vector<AnsToken> tokens{};
        for (std::size_t i{0}; i < 2000 + sample * 137; ++i) {
            const std::size_t context{(i * 17 + sample) % NUM_CONTEXTS};
            const std::uint32_t value{static_cast<std::uint32_t>(
                ((i * (sample + 3)) ^ (context * 29)) % (3 + sample * 19))};
            tokens.push_back({context, value});
        }
        const EncodedBundle encoded{
            encode(tokens, NUM_CONTEXTS, context_map, NUM_CLUSTERS, HybridUintConfig{})};
        EXPECT_EQ(encoded.writer.bits_written(),
                  std::min(encoded.selection.prefix_bits, encoded.selection.ans_bits));
        expect_decodes(encoded, tokens, NUM_CONTEXTS, context_map);
    }
}

}  // namespace
}  // namespace cujpegxl::bitstream
