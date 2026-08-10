// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Validates the prefix entropy container by decoding the emitted bytes with
// libjxl's own DecodeHistograms + prefix ANSSymbolReader and checking that every
// token round-trips to its original value.

#include <cstdint>
#include <cstdlib>
#include <vector>

#include <jxl/memory_manager.h>

#include <gtest/gtest.h>

#include "lib/jxl/base/span.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"

#include "entropy_encoder.h"

namespace cujpegxl::bitstream {
namespace {

void* test_alloc(void*, size_t size) { return std::malloc(size); }
void test_free(void*, void* p) { std::free(p); }

void round_trip(std::size_t num_contexts,
                const std::vector<std::pair<std::size_t, std::uint32_t>>& tokens) {
    EntropyEncoder enc{num_contexts};
    for (const auto& [ctx, value] : tokens) {
        enc.add_token(ctx, value);
    }
    BitWriter w{};
    enc.write(w);
    w.zero_pad_to_byte();

    JxlMemoryManager mm{nullptr, &test_alloc, &test_free};
    jxl::BitReader br{jxl::Bytes(w.bytes().data(), w.bytes().size())};
    jxl::ANSCode code{};
    std::vector<std::uint8_t> context_map{};
    ASSERT_TRUE(jxl::DecodeHistograms(&mm, &br, num_contexts, &code, &context_map));

    auto reader_or = jxl::ANSSymbolReader::Create(&code, &br);
    ASSERT_TRUE(reader_or.ok());
    jxl::ANSSymbolReader reader{std::move(reader_or).value_()};

    for (const auto& [ctx, value] : tokens) {
        const std::uint32_t decoded{static_cast<std::uint32_t>(
            reader.ReadHybridUint(ctx, &br, context_map))};
        EXPECT_EQ(decoded, value) << "context " << ctx;
    }
    EXPECT_TRUE(br.AllReadsWithinBounds());
    EXPECT_TRUE(reader.CheckANSFinalState());
    EXPECT_TRUE(br.Close());
}

TEST(EntropyEncoder, SingleContextSmallValues) {
    round_trip(1, {{0, 0}, {0, 1}, {0, 2}, {0, 3}, {0, 0}, {0, 1}, {0, 7}});
}

TEST(EntropyEncoder, SingleContextLargeValues) {
    round_trip(1, {{0, 0}, {0, 15}, {0, 16}, {0, 255}, {0, 4096}, {0, 100000},
                   {0, 16777215}});
}

TEST(EntropyEncoder, SingleSymbolZeroBitCode) {
    round_trip(1, {{0, 0}, {0, 0}, {0, 0}, {0, 0}});
}

TEST(EntropyEncoder, ManyContextsClusteredToOneHistogram) {
    std::vector<std::pair<std::size_t, std::uint32_t>> tokens{};
    for (std::uint32_t i{0}; i < 200; ++i) {
        tokens.emplace_back(i % 7425, i % 53);
    }
    round_trip(7425, tokens);
}

TEST(EntropyEncoder, SkewedDistributionBuildsComplexTree) {
    std::vector<std::pair<std::size_t, std::uint32_t>> tokens{};
    for (int i{0}; i < 500; ++i) {
        tokens.emplace_back(0, 0);
    }
    for (std::uint32_t v{1}; v < 40; ++v) {
        tokens.emplace_back(0, v);
    }
    round_trip(1, tokens);
}

}  // namespace
}  // namespace cujpegxl::bitstream
