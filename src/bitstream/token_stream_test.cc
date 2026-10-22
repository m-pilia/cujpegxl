// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Codes random AC token streams with the AcGlobal code of their histograms:
// code_token_streams must reproduce the straightforward reference coder for
// every stream, whatever the number of threads sharing the queue, and libjxl's
// DecodeHistograms + ANSSymbolReader must decode every token back (including the
// rANS final-state check).

#include "token_stream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include <jxl/memory_manager.h>

#include <gtest/gtest.h>

// Before libjxl: its ans_params.h defines macros named like our constants.
#include "bit_sink.h"
#include "frame_assembly.h"
#include "hybrid_uint.h"
#include "serialize_impl.h"
#include "src/ac_clustering.h"
#include "src/ac_context.h"
#include "token_stream_reference.h"

#include "lib/jxl/base/span.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"

namespace cujpegxl::bitstream {
namespace {

void* test_alloc(void*, size_t size) {
    return std::malloc(size);
}
void test_free(void*, void* p) {
    std::free(p);
}

struct Token {
    std::uint32_t context;
    std::uint32_t value;
};

struct Streams {
    std::vector<std::vector<Token>> tokens;
    std::vector<std::vector<std::uint16_t>> entries;
    std::vector<std::vector<std::uint64_t>> raw;
    std::vector<TokenStream> streams;
    AcClustering clustering;
};

std::uint32_t test_cluster(std::uint32_t context) {
    return context % AC_MAX_CLUSTERS;
}

// Splits each token into its entry and raw bits, and accumulates the
// histograms of the test_cluster clustering.
Streams prepare(std::vector<std::vector<Token>> tokens) {
    const HybridUintConfig config{};
    Streams s{};
    s.tokens = std::move(tokens);
    s.clustering.num_clusters = AC_MAX_CLUSTERS;
    for (std::uint32_t context{0}; context < AC_NUM_CONTEXTS; ++context) {
        s.clustering.context_map[context] = static_cast<std::uint8_t>(test_cluster(context));
    }
    for (const std::vector<Token>& stream : s.tokens) {
        std::vector<std::uint16_t> entries{};
        std::vector<std::uint64_t> raw(1, 0);
        std::uint64_t raw_bits{0};
        for (const Token& t : stream) {
            std::uint32_t symbol{}, nbits{}, bits{};
            config.encode(t.value, symbol, nbits, bits);
            const std::uint32_t cluster{test_cluster(t.context)};
            ++s.clustering.histograms[cluster * AC_CONTEXT_ALPHABET + symbol];
            entries.push_back(static_cast<std::uint16_t>(cluster << TOKEN_CLUSTER_SHIFT | symbol));
            for (std::uint32_t b{0}; b < nbits; ++b) {
                if (raw_bits / 64 + 1 >= raw.size()) {
                    raw.push_back(0);
                }
                raw[raw_bits / 64] |= static_cast<std::uint64_t>((bits >> b) & 1)
                                      << (raw_bits % 64);
                ++raw_bits;
            }
        }
        s.entries.push_back(std::move(entries));
        s.raw.push_back(std::move(raw));
        s.streams.push_back(TokenStream{nullptr, 0, nullptr, raw_bits});
    }
    for (std::size_t i{0}; i < s.streams.size(); ++i) {
        s.streams[i].entries = s.entries[i].data();
        s.streams[i].num_tokens = s.entries[i].size();
        s.streams[i].raw = s.raw[i].data();
    }
    return s;
}

std::vector<std::vector<std::uint8_t>> code_with_threads(const TokenCode& code, const Streams& s,
                                                         std::size_t num_threads) {
    std::vector<std::size_t> slot_end{};
    std::size_t words{0};
    for (const TokenStream& stream : s.streams) {
        words += coded_stream_capacity_words(stream);
        slot_end.push_back(words);
    }
    std::vector<std::uint64_t> slots(words, 0);
    std::vector<CodedStream> coded(s.streams.size());
    // Longest first, as the encoder claims them.
    std::vector<std::size_t> order(s.streams.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return s.streams[a].num_tokens > s.streams[b].num_tokens;
    });
    StreamQueue queue{};
    queue.streams = s.streams.data();
    queue.num_streams = s.streams.size();
    queue.order = order.data();
    queue.slot_words = slots.data();
    queue.slot_end = slot_end.data();
    queue.coded = coded.data();
    queue.ready.store(s.streams.size());
    std::vector<std::thread> threads{};
    for (std::size_t t{1}; t < num_threads; ++t) {
        threads.emplace_back([&] { code_token_streams(code, queue); });
    }
    code_token_streams(code, queue);
    for (std::thread& t : threads) {
        t.join();
    }
    std::vector<std::vector<std::uint8_t>> out{};
    for (const CodedStream& c : coded) {
        std::vector<std::uint8_t> bytes(c.size_bytes() + 8, 0xA5);
        place_coded_stream(c, bytes.data());
        EXPECT_EQ(bytes[c.size_bytes()], 0xA5) << "placement wrote past the stream";
        bytes.resize(c.size_bytes());
        out.push_back(std::move(bytes));
    }
    return out;
}

void decode_with_libjxl(const std::vector<std::uint8_t>& ac_global, const Streams& s,
                        const std::vector<std::vector<std::uint8_t>>& coded, bool use_prefix) {
    CountingBitWriter head{};
    detail::serialize_ac_global_head(head, s.streams.size());
    JxlMemoryManager mm{nullptr, &test_alloc, &test_free};
    jxl::BitReader hbr{jxl::Bytes(ac_global.data(), ac_global.size())};
    hbr.SkipBits(head.bits_written());
    jxl::ANSCode ans_code{};
    std::vector<std::uint8_t> map{};
    ASSERT_TRUE(jxl::DecodeHistograms(&mm, &hbr, AC_NUM_CONTEXTS, &ans_code, &map));
    ASSERT_TRUE(hbr.Close());
    ASSERT_EQ(ans_code.use_prefix_code, use_prefix);

    for (std::size_t i{0}; i < coded.size(); ++i) {
        // An all-single-symbol prefix stream codes to no bytes: the reader still
        // needs a valid address.
        std::vector<std::uint8_t> bytes{coded[i]};
        bytes.resize(coded[i].size() + 8, 0);
        jxl::BitReader br{jxl::Bytes(bytes.data(), coded[i].size())};
        auto reader_or{jxl::ANSSymbolReader::Create(&ans_code, &br)};
        ASSERT_TRUE(reader_or.ok());
        jxl::ANSSymbolReader reader{std::move(reader_or).value_()};
        for (const Token& t : s.tokens[i]) {
            ASSERT_EQ(reader.ReadHybridUint(t.context, &br, map), t.value) << "stream " << i;
        }
        EXPECT_TRUE(reader.CheckANSFinalState()) << "stream " << i;
        EXPECT_LE(br.TotalBitsConsumed(), coded[i].size() * 8);
        EXPECT_GT(br.TotalBitsConsumed() + 8, coded[i].size() * 8);
        ASSERT_TRUE(br.Close());
    }
}

void check_streams(std::vector<std::vector<Token>> tokens, bool expect_prefix) {
    const Streams s{prepare(std::move(tokens))};
    TokenCode code{};
    AcGlobalBuilder builder{s.clustering, s.streams.size(), code};
    builder.build_context_map();
    for (std::size_t c{builder.num_clusters()}; c > 0; --c) {
        builder.build_cluster(c - 1);
    }
    const std::vector<std::uint8_t> ac_global{builder.finish()};
    ASSERT_EQ(code.use_prefix_code, expect_prefix);

    std::vector<std::vector<std::uint8_t>> expected{};
    for (const TokenStream& stream : s.streams) {
        expected.push_back(code_token_stream_reference(code, stream));
    }
    for (std::size_t threads : {1, 3, 8}) {
        EXPECT_EQ(code_with_threads(code, s, threads), expected) << threads << " threads";
    }
    decode_with_libjxl(ac_global, s, expected, expect_prefix);
}

std::vector<std::vector<Token>> random_tokens(std::uint32_t seed, std::size_t num_streams,
                                              std::size_t max_tokens) {
    std::mt19937 rng{seed};
    std::vector<std::vector<Token>> tokens(num_streams);
    for (std::vector<Token>& stream : tokens) {
        const std::size_t n{rng() % (max_tokens + 1)};
        for (std::size_t i{0}; i < n; ++i) {
            const std::uint32_t context{static_cast<std::uint32_t>(rng() % AC_NUM_CONTEXTS)};
            // Mostly small values, as AC tokens are, with a tail up to 16 bits.
            const std::uint32_t magnitude{rng() % 16 == 0 ? 16u : 3u};
            stream.push_back(Token{context, static_cast<std::uint32_t>(rng() % (1u << magnitude))});
        }
    }
    return tokens;
}

TEST(TokenStream, AnsStreamsMatchReferenceAndDecode) {
    check_streams(random_tokens(7, 40, 3000), false);
}

TEST(TokenStream, AnsWithEmptyAndTinyStreams) {
    std::vector<std::vector<Token>> tokens{random_tokens(11, 12, 400)};
    tokens[3].clear();
    tokens[5].resize(1);
    check_streams(std::move(tokens), false);
}

TEST(TokenStream, SingleSymbolClustersUsePrefixCodes) {
    // Every cluster sees one symbol: libjxl codes the set with prefix codes.
    std::vector<std::vector<Token>> tokens(5);
    for (std::size_t i{0}; i < tokens.size(); ++i) {
        for (std::uint32_t k{0}; k < 300; ++k) {
            const std::uint32_t context{(k * 37 + static_cast<std::uint32_t>(i)) % AC_NUM_CONTEXTS};
            tokens[i].push_back(Token{context, test_cluster(context) % 4});
        }
    }
    check_streams(std::move(tokens), true);
}

TEST(TokenStream, FewTokensUsePrefixCodes) {
    check_streams(random_tokens(3, 4, 20), true);
}

}  // namespace
}  // namespace cujpegxl::bitstream
