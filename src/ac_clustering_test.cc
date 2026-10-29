// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "ac_clustering.h"

#include <array>
#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

namespace cujpegxl {
namespace {

constexpr std::uint32_t COUNT_CONTEXTS{AC_NUM_BLOCK_CTX * AC_NON_ZERO_BUCKETS};

TEST(AcClustering, InitialMapCoversEveryCluster) {
    std::array<std::size_t, AC_MAX_CLUSTERS> cluster_sizes{};
    for (std::uint32_t context{0}; context < AC_NUM_CONTEXTS; ++context) {
        const std::uint32_t cluster{ac_initial_cluster(context)};
        ASSERT_LT(cluster, AC_MAX_CLUSTERS);
        ++cluster_sizes[cluster];
    }
    for (std::size_t size : cluster_sizes) {
        EXPECT_GT(size, 0u);
    }
}

TEST(AcClustering, InitialMapSeparatesContextFamilies) {
    for (std::uint32_t context{0}; context < COUNT_CONTEXTS; ++context) {
        EXPECT_LT(ac_initial_cluster(context), 8u);
    }
    for (std::uint32_t context{COUNT_CONTEXTS}; context < AC_NUM_CONTEXTS; ++context) {
        EXPECT_GE(ac_initial_cluster(context), 8u);
    }
}

TEST(AcClustering, InitialMapSeparatesLumaAndChroma) {
    for (std::uint32_t bucket{0}; bucket < AC_NON_ZERO_BUCKETS; ++bucket) {
        const std::uint32_t base{bucket * AC_NUM_BLOCK_CTX};
        EXPECT_LT(ac_initial_cluster(base), 4u);
        EXPECT_GE(ac_initial_cluster(base + 7), 4u);
        EXPECT_LT(ac_initial_cluster(base + 7), 8u);
    }
    for (std::uint32_t zdc{0}; zdc < AC_ZERO_DENSITY_COUNT; ++zdc) {
        EXPECT_LT(ac_initial_cluster(COUNT_CONTEXTS + zdc), 20u);
        EXPECT_GE(ac_initial_cluster(COUNT_CONTEXTS + 7 * AC_ZERO_DENSITY_COUNT + zdc), 20u);
    }
}

TEST(AcClustering, InitialMapBandBoundaries) {
    EXPECT_EQ(ac_initial_cluster(0), 0u);
    EXPECT_EQ(ac_initial_cluster(4 * AC_NUM_BLOCK_CTX), 1u);
    EXPECT_EQ(ac_initial_cluster(12 * AC_NUM_BLOCK_CTX), 2u);
    EXPECT_EQ(ac_initial_cluster(24 * AC_NUM_BLOCK_CTX), 3u);
    EXPECT_EQ(ac_initial_cluster(COUNT_CONTEXTS), 8u);
    EXPECT_EQ(ac_initial_cluster(COUNT_CONTEXTS + 32), 9u);
    EXPECT_EQ(ac_initial_cluster(COUNT_CONTEXTS + 424), 19u);
    EXPECT_EQ(ac_initial_cluster(COUNT_CONTEXTS + 7 * AC_ZERO_DENSITY_COUNT), 20u);
}

}  // namespace
}  // namespace cujpegxl
