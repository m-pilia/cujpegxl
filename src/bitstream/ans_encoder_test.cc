// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "ans_encoder.h"

#include <array>
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

void round_trip(const std::vector<AnsToken>& tokens, std::size_t num_contexts,
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

    std::vector<AnsDistribution> distributions(num_clusters);
    BitWriter writer{};
    write_clustered_ans_histograms(writer, context_map.data(), num_contexts, num_clusters,
                                   histograms.data(), ANS_ALPHABET_SIZE, config,
                                   distributions.data());
    std::vector<AnsEncodingTable> tables(num_clusters);
    for (std::size_t cluster{0}; cluster < num_clusters; ++cluster) {
        build_ans_encoding_table(distributions[cluster], tables[cluster]);
    }
    std::vector<std::uint32_t> scratch(tokens.size());
    write_ans_tokens(writer, tokens.data(), tokens.size(), context_map.data(), num_contexts,
                     tables.data(), num_clusters, config, scratch.data());
    writer.zero_pad_to_byte();

    JxlMemoryManager memory_manager{nullptr, &test_alloc, &test_free};
    jxl::BitReader reader{jxl::Bytes(writer.bytes().data(), writer.bytes().size())};
    jxl::ANSCode code{};
    std::vector<std::uint8_t> decoded_context_map{};
    ASSERT_TRUE(jxl::DecodeHistograms(&memory_manager, &reader, num_contexts, &code,
                                      &decoded_context_map));
    EXPECT_EQ(decoded_context_map, context_map);
    auto symbol_reader_or{jxl::ANSSymbolReader::Create(&code, &reader)};
    ASSERT_TRUE(symbol_reader_or.ok());
    jxl::ANSSymbolReader symbol_reader{std::move(symbol_reader_or).value_()};
    for (const AnsToken& token : tokens) {
        const std::uint32_t decoded{static_cast<std::uint32_t>(
            symbol_reader.ReadHybridUint(token.context, &reader, decoded_context_map))};
        EXPECT_EQ(decoded, token.value);
    }
    EXPECT_TRUE(symbol_reader.CheckANSFinalState());
    EXPECT_TRUE(reader.Close());
}

TEST(AnsEncoder, SparseDistributionRoundTrips) {
    std::vector<AnsToken> tokens{};
    for (std::size_t i{0}; i < 2000; ++i) {
        constexpr std::array<std::uint32_t, 5> VALUES{0, 0, 0, 7, 200};
        tokens.push_back({0, VALUES[i % VALUES.size()]});
    }
    round_trip(tokens, 1, {0}, 1, HybridUintConfig{8, 0, 0});
}

TEST(AnsEncoder, HybridUintExtraBitsRoundTrip) {
    const std::vector<AnsToken> tokens{{0, 0},      {0, 15},      {0, 16},
                                       {0, 255},    {0, 4096},    {0, 100000},
                                       {0, 1u << 24}, {0, 99999999}};
    round_trip(tokens, 1, {0}, 1, HybridUintConfig{});
}

TEST(AnsEncoder, RuntimeContextMapRoundTrips) {
    constexpr std::size_t NUM_CONTEXTS = 19;
    constexpr std::size_t NUM_CLUSTERS = 3;
    std::vector<std::uint8_t> context_map(NUM_CONTEXTS);
    for (std::size_t i{0}; i < context_map.size(); ++i) {
        context_map[i] = static_cast<std::uint8_t>((i * 7) % NUM_CLUSTERS);
    }
    std::vector<AnsToken> tokens{};
    for (std::size_t i{0}; i < 4000; ++i) {
        const std::size_t context{(i * 11) % NUM_CONTEXTS};
        const std::uint32_t value{static_cast<std::uint32_t>(
            context_map[context] == 0 ? i % 3 : context_map[context] == 1 ? (i * 13) % 80
                                                                        : (i * 997) % 100000)};
        tokens.push_back({context, value});
    }
    round_trip(tokens, NUM_CONTEXTS, context_map, NUM_CLUSTERS, HybridUintConfig{});
}

TEST(AnsEncoder, EmptyStreamHasInitialState) {
    std::array<std::uint32_t, ANS_ALPHABET_SIZE> histogram{};
    AnsDistribution distribution{};
    build_ans_distribution(histogram.data(), 1, distribution);
    AnsEncodingTable table{};
    build_ans_encoding_table(distribution, table);
    const std::uint8_t context_map{0};
    BitWriter writer{};
    write_ans_tokens(writer, nullptr, 0, &context_map, 1, &table, 1, HybridUintConfig{}, nullptr);
    ASSERT_EQ(writer.bits_written(), 32);
    EXPECT_EQ(writer.bytes(), (std::vector<std::uint8_t>{0x00, 0x00, 0x13, 0x00}));
}

TEST(AnsEncoder, StateTransitionGoldenVectors) {
    AnsEncodingTable single{};
    single.frequencies[0] = ANS_TABLE_SIZE;
    single.offsets[1] = ANS_TABLE_SIZE;
    for (std::size_t i{0}; i < ANS_TABLE_SIZE; ++i) {
        single.reverse_map[i] = static_cast<std::uint16_t>(i);
    }
    const AnsStateTransition stable{ans_put_symbol(ANS_INITIAL_STATE, single, 0)};
    EXPECT_EQ(stable.state, ANS_INITIAL_STATE);
    EXPECT_FALSE(stable.renormalized);

    AnsEncodingTable boundary{};
    boundary.frequencies[0] = 1;
    boundary.offsets[1] = 1;
    boundary.reverse_map[0] = 123;
    const AnsStateTransition renormalized{ans_put_symbol(ANS_INITIAL_STATE, boundary, 0)};
    EXPECT_EQ(renormalized.state, 77947u);
    EXPECT_EQ(renormalized.renormalization_bits, 0);
    EXPECT_TRUE(renormalized.renormalized);
}

}  // namespace
}  // namespace cujpegxl::bitstream
