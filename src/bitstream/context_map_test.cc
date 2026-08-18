// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "context_map.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include <jxl/memory_manager.h>

#include <gtest/gtest.h>

#include "lib/jxl/base/span.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/dec_context_map.h"
#include "lib/jxl/inverse_mtf-inl.h"

namespace cujpegxl::bitstream {
namespace {

void* test_alloc(void*, size_t size) {
    return std::malloc(size);
}
void test_free(void*, void* p) {
    std::free(p);
}

void expect_libjxl_inverse(const std::vector<std::uint8_t>& context_map) {
    std::vector<std::uint8_t> transformed(context_map.size());
    move_to_front_transform(context_map.data(), context_map.size(), transformed.data());

    jxl::InverseMoveToFrontTransform(transformed.data(), static_cast<int>(transformed.size()));
    EXPECT_EQ(transformed, context_map);
}

void expect_complex_map_round_trip(const std::vector<std::uint8_t>& context_map, bool use_mtf) {
    BitWriter w{};
    write_complex_prefix_context_map(w, context_map.data(), context_map.size(), use_mtf);
    w.zero_pad_to_byte();

    JxlMemoryManager mm{nullptr, &test_alloc, &test_free};
    jxl::BitReader br{jxl::Bytes(w.bytes().data(), w.bytes().size())};
    std::vector<std::uint8_t> decoded(context_map.size());
    std::size_t num_histograms{0};
    ASSERT_TRUE(jxl::DecodeContextMap(&mm, &decoded, &num_histograms, &br));
    EXPECT_EQ(decoded, context_map);
    const std::uint8_t max_cluster{*std::max_element(context_map.begin(), context_map.end())};
    EXPECT_EQ(num_histograms, static_cast<std::size_t>(max_cluster) + 1);
    EXPECT_TRUE(br.Close());
}

ContextMapEncoding expect_best_map_round_trip(const std::vector<std::uint8_t>& context_map,
                                              std::size_t num_clusters) {
    BitWriter w{};
    w.write(5, 0x15);
    const ContextMapEncoding encoding{
        write_best_prefix_context_map(w, context_map.data(), context_map.size(), num_clusters)};
    w.zero_pad_to_byte();

    JxlMemoryManager mm{nullptr, &test_alloc, &test_free};
    jxl::BitReader br{jxl::Bytes(w.bytes().data(), w.bytes().size())};
    EXPECT_EQ(br.ReadBits(5), 0x15u);
    std::vector<std::uint8_t> decoded(context_map.size());
    std::size_t decoded_clusters{0};
    EXPECT_TRUE(jxl::DecodeContextMap(&mm, &decoded, &decoded_clusters, &br));
    EXPECT_EQ(decoded, context_map);
    EXPECT_EQ(decoded_clusters, num_clusters);
    EXPECT_TRUE(br.Close());
    return encoding;
}

TEST(ContextMap, Empty) {
    move_to_front_transform(nullptr, 0, nullptr);
    expect_libjxl_inverse({});
}

TEST(ContextMap, KnownRanks) {
    const std::vector<std::uint8_t> context_map{0, 1, 1, 0, 2, 1, 2, 0};
    std::vector<std::uint8_t> transformed(context_map.size());
    move_to_front_transform(context_map.data(), context_map.size(), transformed.data());

    EXPECT_EQ(transformed, (std::vector<std::uint8_t>{0, 1, 0, 1, 2, 2, 1, 2}));
    expect_libjxl_inverse(context_map);
}

TEST(ContextMap, RepeatedSymbols) {
    std::vector<std::uint8_t> context_map(7425, 173);
    std::vector<std::uint8_t> transformed(context_map.size());
    move_to_front_transform(context_map.data(), context_map.size(), transformed.data());

    EXPECT_EQ(transformed.front(), 173);
    for (std::size_t i{1}; i < transformed.size(); ++i) {
        EXPECT_EQ(transformed[i], 0);
    }
    expect_libjxl_inverse(context_map);
}

TEST(ContextMap, AllClusterIds) {
    std::vector<std::uint8_t> context_map(256);
    for (std::size_t i{0}; i < context_map.size(); ++i) {
        context_map[i] = static_cast<std::uint8_t>(i);
    }
    expect_libjxl_inverse(context_map);
}

TEST(ContextMap, DenseDeterministicMap) {
    std::vector<std::uint8_t> context_map(7425);
    std::uint32_t state{0x12345678u};
    for (std::uint8_t& value : context_map) {
        state = state * 1664525u + 1013904223u;
        value = static_cast<std::uint8_t>(state >> 24);
    }

    std::vector<std::uint8_t> first(context_map.size());
    std::vector<std::uint8_t> second(context_map.size());
    move_to_front_transform(context_map.data(), context_map.size(), first.data());
    move_to_front_transform(context_map.data(), context_map.size(), second.data());

    EXPECT_EQ(first, second);
    expect_libjxl_inverse(context_map);
}

TEST(ContextMap, ComplexWithoutMtf) {
    std::vector<std::uint8_t> context_map{};
    for (std::size_t i{0}; i < 1024; ++i) {
        context_map.push_back(static_cast<std::uint8_t>(i % 16));
    }
    expect_complex_map_round_trip(context_map, false);
}

TEST(ContextMap, ComplexWithMtf) {
    std::vector<std::uint8_t> context_map{};
    for (std::size_t i{0}; i < 7425; ++i) {
        context_map.push_back(static_cast<std::uint8_t>((i / 31 + i / 509) % 32));
    }
    for (std::size_t i{0}; i < 32; ++i) {
        context_map[i] = static_cast<std::uint8_t>(i);
    }
    expect_complex_map_round_trip(context_map, true);
}

TEST(ContextMap, ComplexSingleCluster) {
    expect_complex_map_round_trip(std::vector<std::uint8_t>(128, 0), true);
}

TEST(ContextMap, BestSingleClusterIsSimple) {
    EXPECT_EQ(expect_best_map_round_trip(std::vector<std::uint8_t>(7425, 0), 1),
              ContextMapEncoding::SIMPLE);
}

TEST(ContextMap, BestEightClusterBoundary) {
    std::vector<std::uint8_t> context_map{};
    for (std::size_t i{0}; i < 8; ++i) {
        context_map.push_back(static_cast<std::uint8_t>(i));
    }
    EXPECT_EQ(expect_best_map_round_trip(context_map, 8), ContextMapEncoding::SIMPLE);
}

TEST(ContextMap, BestChoosesComplexBelowSimpleLimit) {
    std::vector<std::uint8_t> context_map(7425, 0);
    for (std::size_t i{0}; i < 8; ++i) {
        context_map[i] = static_cast<std::uint8_t>(i);
    }
    EXPECT_NE(expect_best_map_round_trip(context_map, 8), ContextMapEncoding::SIMPLE);
}

TEST(ContextMap, BestNineClustersRequiresComplexForm) {
    std::vector<std::uint8_t> context_map{};
    for (std::size_t i{0}; i < 1024; ++i) {
        context_map.push_back(static_cast<std::uint8_t>(i % 9));
    }
    EXPECT_NE(expect_best_map_round_trip(context_map, 9), ContextMapEncoding::SIMPLE);
}

TEST(ContextMap, BestSupportsAllClusters) {
    std::vector<std::uint8_t> context_map(256);
    for (std::size_t i{0}; i < context_map.size(); ++i) {
        context_map[i] = static_cast<std::uint8_t>(i);
    }
    EXPECT_NE(expect_best_map_round_trip(context_map, 256), ContextMapEncoding::SIMPLE);
}

}  // namespace
}  // namespace cujpegxl::bitstream
