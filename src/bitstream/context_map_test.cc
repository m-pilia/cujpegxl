// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "context_map.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "lib/jxl/inverse_mtf-inl.h"

namespace cujpegxl::bitstream {
namespace {

void expect_libjxl_inverse(const std::vector<std::uint8_t>& context_map) {
    std::vector<std::uint8_t> transformed(context_map.size());
    move_to_front_transform(context_map.data(), context_map.size(), transformed.data());

    jxl::InverseMoveToFrontTransform(transformed.data(), static_cast<int>(transformed.size()));
    EXPECT_EQ(transformed, context_map);
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

}  // namespace
}  // namespace cujpegxl::bitstream
