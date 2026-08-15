// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "ac_context.h"

#include <array>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

namespace cujpegxl {
namespace {

TEST(AcContext, FineMapCoversEveryCluster) {
    std::array<std::size_t, AC_NUM_CLUSTERS> cluster_sizes{};
    for (std::uint32_t ctx{0}; ctx < AC_NUM_CONTEXTS; ++ctx) {
        const int cluster{ac_cluster(ctx)};
        ASSERT_GE(cluster, 0);
        ASSERT_LT(cluster, AC_NUM_CLUSTERS);
        ++cluster_sizes[static_cast<std::size_t>(cluster)];
    }
    for (std::size_t size : cluster_sizes) {
        EXPECT_GT(size, 0u);
    }
}

TEST(AcContext, FineMapSeparatesContextFamilies) {
    const std::uint32_t count_contexts{AC_NUM_BLOCK_CTX * AC_NON_ZERO_BUCKETS};
    for (std::uint32_t ctx{0}; ctx < count_contexts; ++ctx) {
        EXPECT_LT(ac_cluster(ctx), 8);
    }
    for (std::uint32_t ctx{count_contexts}; ctx < AC_NUM_CONTEXTS; ++ctx) {
        EXPECT_GE(ac_cluster(ctx), 8);
    }
}

TEST(AcContext, FineMapSeparatesLumaAndChroma) {
    for (std::uint32_t bucket{0}; bucket < AC_NON_ZERO_BUCKETS; ++bucket) {
        const std::uint32_t base{bucket * AC_NUM_BLOCK_CTX};
        EXPECT_LT(ac_cluster(base), 4);
        EXPECT_GE(ac_cluster(base + 7), 4);
        EXPECT_LT(ac_cluster(base + 7), 8);
    }

    const std::uint32_t count_contexts{AC_NUM_BLOCK_CTX * AC_NON_ZERO_BUCKETS};
    for (std::uint32_t zdc{0}; zdc < AC_ZERO_DENSITY_COUNT; ++zdc) {
        EXPECT_LT(ac_cluster(count_contexts + zdc), 20);
        EXPECT_GE(ac_cluster(count_contexts + 7 * AC_ZERO_DENSITY_COUNT + zdc), 20);
    }
}

TEST(AcContext, FineMapBoundariesAreStable) {
    EXPECT_EQ(ac_cluster(0), 0);
    EXPECT_EQ(ac_cluster(4 * AC_NUM_BLOCK_CTX), 1);
    EXPECT_EQ(ac_cluster(12 * AC_NUM_BLOCK_CTX), 2);
    EXPECT_EQ(ac_cluster(24 * AC_NUM_BLOCK_CTX), 3);

    constexpr std::uint32_t count_contexts{AC_NUM_BLOCK_CTX * AC_NON_ZERO_BUCKETS};
    EXPECT_EQ(ac_cluster(count_contexts), 8);
    EXPECT_EQ(ac_cluster(count_contexts + 32), 9);
    EXPECT_EQ(ac_cluster(count_contexts + 424), 19);
    EXPECT_EQ(ac_cluster(count_contexts + 7 * AC_ZERO_DENSITY_COUNT), 20);
}

}  // namespace
}  // namespace cujpegxl
