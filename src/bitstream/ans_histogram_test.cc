// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "ans_histogram.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <vector>

#include <jxl/memory_manager.h>

#include <gtest/gtest.h>

#include "lib/jxl/base/span.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"

namespace cujpegxl::bitstream {
namespace {

void* test_alloc(void*, size_t size) {
    return std::malloc(size);
}
void test_free(void*, void* pointer) {
    std::free(pointer);
}

struct EncodedDistribution {
    BitWriter writer{};
    AnsDistribution distribution{};
};

EncodedDistribution encode_distribution(const std::array<std::uint32_t, ANS_ALPHABET_SIZE>& h) {
    EncodedDistribution encoded{};
    const std::uint8_t context_map{0};
    write_clustered_ans_histograms(encoded.writer, &context_map, 1, 1, h.data(), h.size(),
                                   HybridUintConfig{}, &encoded.distribution);
    return encoded;
}

std::size_t decode_one_symbol(const EncodedDistribution& encoded, std::uint32_t residue) {
    BitWriter writer{encoded.writer};
    writer.write(32, (65536u << 12) | residue);
    writer.zero_pad_to_byte();

    JxlMemoryManager memory_manager{nullptr, &test_alloc, &test_free};
    jxl::BitReader reader{jxl::Bytes(writer.bytes().data(), writer.bytes().size())};
    jxl::ANSCode code{};
    std::vector<std::uint8_t> context_map{};
    EXPECT_TRUE(jxl::DecodeHistograms(&memory_manager, &reader, 1, &code, &context_map));
    EXPECT_FALSE(code.use_prefix_code);
    EXPECT_EQ(context_map, std::vector<std::uint8_t>{0});
    auto symbol_reader_or{jxl::ANSSymbolReader::Create(&code, &reader)};
    EXPECT_TRUE(symbol_reader_or.ok());
    jxl::ANSSymbolReader symbol_reader{std::move(symbol_reader_or).value_()};
    return symbol_reader.ReadSymbol(0, &reader);
}

void expect_round_trip(const std::array<std::uint32_t, ANS_ALPHABET_SIZE>& histogram) {
    const EncodedDistribution encoded{encode_distribution(histogram)};
    EXPECT_EQ(std::accumulate(encoded.distribution.counts.begin(),
                              encoded.distribution.counts.end(), std::size_t{0}),
              ANS_TABLE_SIZE);

    std::array<std::uint16_t, ANS_ALPHABET_SIZE> local_counts{};
    std::array<std::array<bool, ANS_TABLE_SIZE>, ANS_ALPHABET_SIZE> offsets_seen{};
    for (std::size_t value{0}; value < ANS_TABLE_SIZE; ++value) {
        const std::size_t table_index{value >> 4};
        const std::size_t position{value & 15};
        const AnsAliasEntry& entry{encoded.distribution.aliases[table_index]};
        const bool use_right{position >= entry.cutoff};
        const std::size_t symbol{use_right ? entry.right_value : table_index};
        const std::size_t offset{position + (use_right ? entry.right_offset : 0)};
        const std::size_t frequency{static_cast<std::size_t>(
            entry.frequency ^ (use_right ? entry.right_frequency_xor : 0))};
        ASSERT_LT(symbol, local_counts.size());
        ASSERT_EQ(frequency, encoded.distribution.counts[symbol]);
        ASSERT_LT(offset, frequency);
        EXPECT_FALSE(offsets_seen[symbol][offset]);
        offsets_seen[symbol][offset] = true;
        ++local_counts[symbol];
    }
    EXPECT_EQ(local_counts, encoded.distribution.counts);

    std::array<std::uint16_t, ANS_ALPHABET_SIZE> decoded_counts{};
    for (std::uint32_t residue{0}; residue < ANS_TABLE_SIZE; ++residue) {
        const std::size_t symbol{decode_one_symbol(encoded, residue)};
        ASSERT_LT(symbol, decoded_counts.size());
        ++decoded_counts[symbol];
    }
    EXPECT_EQ(decoded_counts, encoded.distribution.counts);
}

TEST(AnsHistogram, SparseDistributionRoundTrips) {
    std::array<std::uint32_t, ANS_ALPHABET_SIZE> histogram{};
    histogram[0] = 31;
    histogram[7] = 5;
    histogram[200] = 11;
    expect_round_trip(histogram);
}

TEST(AnsHistogram, SingleSymbolDistributionRoundTrips) {
    std::array<std::uint32_t, ANS_ALPHABET_SIZE> histogram{};
    histogram[173] = 1000;
    expect_round_trip(histogram);
}

TEST(AnsHistogram, SkewedDistributionRoundTrips) {
    std::array<std::uint32_t, ANS_ALPHABET_SIZE> histogram{};
    histogram[0] = 1000000;
    for (std::size_t i{1}; i < 64; ++i) {
        histogram[i] = 1;
    }
    expect_round_trip(histogram);
}

TEST(AnsHistogram, FullAlphabetDistributionRoundTrips) {
    std::array<std::uint32_t, ANS_ALPHABET_SIZE> histogram{};
    for (std::size_t i{0}; i < histogram.size(); ++i) {
        histogram[i] = static_cast<std::uint32_t>((i * 37) % 101 + 1);
    }
    expect_round_trip(histogram);
}

TEST(AnsHistogram, NormalizationIsDeterministic) {
    std::array<std::uint32_t, ANS_ALPHABET_SIZE> histogram{};
    for (std::size_t i{0}; i < histogram.size(); ++i) {
        histogram[i] = static_cast<std::uint32_t>((i * 13) % 29);
    }
    const EncodedDistribution first{encode_distribution(histogram)};
    const EncodedDistribution second{encode_distribution(histogram)};
    EXPECT_EQ(first.writer.bits_written(), second.writer.bits_written());
    EXPECT_EQ(first.writer.bytes(), second.writer.bytes());
    EXPECT_EQ(first.distribution.counts, second.distribution.counts);
}

}  // namespace
}  // namespace cujpegxl::bitstream
