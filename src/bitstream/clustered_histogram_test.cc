// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Validates write_clustered_prefix_histograms by decoding the emitted histogram
// description + token stream with libjxl's own DecodeHistograms + ANSSymbolReader
// and confirming every (context, value) token round-trips.

#include <cstdint>
#include <cstdlib>
#include <vector>

#include <jxl/memory_manager.h>

#include <gtest/gtest.h>

#include "lib/jxl/base/span.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"

#include "bit_writer.h"
#include "histogram_writer.h"
#include "hybrid_uint.h"

namespace cujpegxl::bitstream {
namespace {

void* test_alloc(void*, size_t size) {
    return std::malloc(size);
}
void test_free(void*, void* p) {
    std::free(p);
}

struct Token {
    std::size_t context;
    std::uint32_t value;
};

// Encodes `tokens` with a fixed cluster map, then decodes the whole stream via
// libjxl and checks each recovered value.
void round_trip(std::size_t num_contexts, std::size_t num_clusters,
                const std::vector<std::uint8_t>& context_map, const std::vector<Token>& tokens) {
    constexpr std::size_t STRIDE = 256;
    const HybridUintConfig config{};

    std::vector<std::uint32_t> cluster_hist(num_clusters * STRIDE, 0);
    std::vector<std::uint32_t> tok_symbol(tokens.size());
    std::vector<std::uint32_t> tok_nbits(tokens.size());
    std::vector<std::uint32_t> tok_bits(tokens.size());
    for (std::size_t i{0}; i < tokens.size(); ++i) {
        std::uint32_t symbol{}, nbits{}, bits{};
        config.encode(tokens[i].value, symbol, nbits, bits);
        tok_symbol[i] = symbol;
        tok_nbits[i] = nbits;
        tok_bits[i] = bits;
        ++cluster_hist[context_map[tokens[i].context] * STRIDE + symbol];
    }

    std::vector<std::uint8_t> depth(num_clusters * STRIDE, 0);
    std::vector<std::uint16_t> bits(num_clusters * STRIDE, 0);
    BitWriter w{};
    write_clustered_prefix_histograms(w, context_map.data(), num_contexts, num_clusters,
                                      cluster_hist.data(), STRIDE, config, depth.data(),
                                      bits.data());
    for (std::size_t i{0}; i < tokens.size(); ++i) {
        const std::size_t cl{context_map[tokens[i].context]};
        w.write(depth[cl * STRIDE + tok_symbol[i]], bits[cl * STRIDE + tok_symbol[i]]);
        if (tok_nbits[i]) {
            w.write(tok_nbits[i], tok_bits[i]);
        }
    }
    w.zero_pad_to_byte();

    JxlMemoryManager mm{nullptr, &test_alloc, &test_free};
    jxl::BitReader br{jxl::Bytes(w.bytes().data(), w.bytes().size())};
    jxl::ANSCode code{};
    std::vector<std::uint8_t> decoded_map{};
    ASSERT_TRUE(jxl::DecodeHistograms(&mm, &br, num_contexts, &code, &decoded_map));
    ASSERT_EQ(decoded_map.size(), num_contexts);
    for (std::size_t i{0}; i < num_contexts; ++i) {
        EXPECT_EQ(decoded_map[i], context_map[i]) << "context " << i;
    }

    auto reader_or = jxl::ANSSymbolReader::Create(&code, &br);
    ASSERT_TRUE(reader_or.ok());
    jxl::ANSSymbolReader reader{std::move(reader_or).value_()};
    for (std::size_t i{0}; i < tokens.size(); ++i) {
        const std::uint32_t got{
            static_cast<std::uint32_t>(reader.ReadHybridUint(tokens[i].context, &br, decoded_map))};
        EXPECT_EQ(got, tokens[i].value) << "token " << i;
    }
    EXPECT_TRUE(reader.CheckANSFinalState());
    EXPECT_TRUE(br.Close());
}

// Builds a context map that uses every cluster (id = ctx % num_clusters).
std::vector<std::uint8_t> modulo_map(std::size_t num_contexts, std::size_t num_clusters) {
    std::vector<std::uint8_t> map(num_contexts);
    for (std::size_t i{0}; i < num_contexts; ++i) {
        map[i] = static_cast<std::uint8_t>(i % num_clusters);
    }
    return map;
}

TEST(ClusteredHistogram, SingleClusterMatchesContexts) {
    const std::size_t nctx{7};
    std::vector<Token> tokens{};
    for (std::size_t i{0}; i < 200; ++i) {
        tokens.push_back({i % nctx, static_cast<std::uint32_t>((i * 7) % 40)});
    }
    round_trip(nctx, 1, std::vector<std::uint8_t>(nctx, 0), tokens);
}

TEST(ClusteredHistogram, TwoClusters) {
    const std::size_t nctx{10};
    std::vector<Token> tokens{};
    for (std::size_t i{0}; i < 500; ++i) {
        const std::size_t ctx{i % nctx};
        const std::uint32_t v{ctx < 5 ? static_cast<std::uint32_t>(i % 3)
                                      : static_cast<std::uint32_t>((i * 13) % 300)};
        tokens.push_back({ctx, v});
    }
    round_trip(nctx, 2, modulo_map(nctx, 2), tokens);
}

TEST(ClusteredHistogram, EightClustersLargeValues) {
    const std::size_t nctx{7425};  // the real AC context count
    std::vector<std::uint8_t> map(nctx);
    for (std::size_t i{0}; i < nctx; ++i) {
        map[i] = static_cast<std::uint8_t>((i * 2654435761u) % 8);  // spread across all 8
    }
    // Ensure every cluster id appears.
    for (std::size_t c{0}; c < 8; ++c) {
        map[c] = static_cast<std::uint8_t>(c);
    }
    std::vector<Token> tokens{};
    for (std::size_t i{0}; i < 3000; ++i) {
        const std::size_t ctx{(i * 37) % nctx};
        tokens.push_back({ctx, static_cast<std::uint32_t>((i * 2999) % 5000)});
    }
    round_trip(nctx, 8, map, tokens);
}

TEST(ClusteredHistogram, ClusterWithSingleSymbol) {
    const std::size_t nctx{6};
    std::vector<std::uint8_t> map{0, 1, 2, 0, 1, 2};
    std::vector<Token> tokens{};
    // Cluster 2 only ever sees value 0 -> a zero-bit (single-symbol) code.
    for (std::size_t i{0}; i < 300; ++i) {
        const std::size_t ctx{i % nctx};
        const std::uint32_t v{map[ctx] == 2 ? 0u : static_cast<std::uint32_t>((i * 11) % 60)};
        tokens.push_back({ctx, v});
    }
    round_trip(nctx, 3, map, tokens);
}

}  // namespace
}  // namespace cujpegxl::bitstream
