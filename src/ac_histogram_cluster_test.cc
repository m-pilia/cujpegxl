// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "ac_histogram_cluster.h"

#include <cstddef>
#include <cstdint>

#include <gtest/gtest.h>

namespace cujpegxl {
namespace {

AcPreclusterResult make_preclusters(std::size_t count) {
    AcPreclusterResult result{};
    result.num_candidates = count;
    for (std::size_t candidate{0}; candidate < count; ++candidate) {
        result.signatures[candidate] = static_cast<std::uint16_t>(candidate);
    }
    for (std::size_t context{0}; context < AC_NUM_CONTEXTS; ++context) {
        result.context_map[context] = static_cast<std::uint16_t>(context % count);
    }
    return result;
}

void set_count(AcPreclusterResult& result, std::size_t candidate, std::size_t symbol,
               std::uint32_t count) {
    result.histograms[candidate * AC_HISTOGRAM_SIZE + symbol] = count;
}

TEST(AcHistogramCluster, MergesKnownBestPair) {
    AcPreclusterResult preclusters{make_preclusters(3)};
    set_count(preclusters, 0, 0, 100);
    set_count(preclusters, 0, 1, 10);
    set_count(preclusters, 1, 0, 200);
    set_count(preclusters, 1, 1, 20);
    set_count(preclusters, 2, 200, 100);
    set_count(preclusters, 2, 201, 100);

    AcClusterResult result{};
    ASSERT_TRUE(ac_cluster_histograms(preclusters, {.max_clusters = 2}, result));
    EXPECT_EQ(result.num_clusters, 2u);
    EXPECT_EQ(result.context_map[0], result.context_map[1]);
    EXPECT_NE(result.context_map[0], result.context_map[2]);
    EXPECT_LE(result.merge_deltas[0], 0);
}

TEST(AcHistogramCluster, TiesUseLowestCandidateIds) {
    AcPreclusterResult preclusters{make_preclusters(3)};
    for (std::size_t candidate{0}; candidate < 3; ++candidate) {
        set_count(preclusters, candidate, 4, 100);
        set_count(preclusters, candidate, 5, 100);
    }

    AcClusterResult result{};
    ASSERT_TRUE(ac_cluster_histograms(preclusters, {.max_clusters = 2}, result));
    EXPECT_EQ(result.context_map[0], result.context_map[1]);
    EXPECT_NE(result.context_map[0], result.context_map[2]);
}

TEST(AcHistogramCluster, HonorsComparisonLimit) {
    AcPreclusterResult preclusters{make_preclusters(5)};
    for (std::size_t candidate{0}; candidate < 5; ++candidate) {
        set_count(preclusters, candidate, candidate, 100);
        set_count(preclusters, candidate, candidate + 1, 10);
    }

    AcClusterResult result{};
    ASSERT_TRUE(ac_cluster_histograms(
        preclusters, {.max_clusters = 2, .max_comparisons = 3}, result));
    EXPECT_EQ(result.num_clusters, 2u);
    EXPECT_EQ(result.comparisons, 3u);
    EXPECT_EQ(result.num_merges, 3u);
}

TEST(AcHistogramCluster, KeepsCandidateSetBelowClusterLimit) {
    AcPreclusterResult preclusters{make_preclusters(2)};
    set_count(preclusters, 0, 0, 100);
    set_count(preclusters, 1, 1, 100);

    AcClusterResult result{};
    ASSERT_TRUE(ac_cluster_histograms(preclusters, {.max_clusters = 4}, result));
    EXPECT_EQ(result.num_clusters, 2u);
    EXPECT_EQ(result.num_merges, 0u);
    EXPECT_EQ(result.comparisons, 0u);
    EXPECT_EQ(result.context_map[0], 0u);
    EXPECT_EQ(result.context_map[1], 1u);
}

TEST(AcHistogramCluster, IdenticalMergesNeverIncreaseCost) {
    AcPreclusterResult preclusters{make_preclusters(6)};
    for (std::size_t candidate{0}; candidate < 6; ++candidate) {
        set_count(preclusters, candidate, 0, 100);
        set_count(preclusters, candidate, 1, 25);
        set_count(preclusters, candidate, 2, 5);
    }

    AcClusterResult result{};
    ASSERT_TRUE(ac_cluster_histograms(preclusters, {.max_clusters = 1}, result));
    for (std::size_t merge{0}; merge < result.num_merges; ++merge) {
        EXPECT_LE(result.merge_deltas[merge], 0) << "merge " << merge;
    }
}

TEST(AcHistogramCluster, PrefixDepthsAndOutputAreDeterministic) {
    AcPreclusterResult preclusters{make_preclusters(8)};
    for (std::size_t candidate{0}; candidate < 8; ++candidate) {
        for (std::size_t symbol{0}; symbol < 16; ++symbol) {
            set_count(preclusters, candidate, symbol,
                      static_cast<std::uint32_t>((candidate + 1) * (symbol + 3)));
        }
    }

    AcClusterResult first{};
    AcClusterResult second{};
    const AcClusterConfig config{.max_clusters = 4, .max_comparisons = 100};
    ASSERT_TRUE(ac_cluster_histograms(preclusters, config, first));
    ASSERT_TRUE(ac_cluster_histograms(preclusters, config, second));
    EXPECT_EQ(first.context_map, second.context_map);
    EXPECT_EQ(first.histograms, second.histograms);
    EXPECT_EQ(first.depths, second.depths);
    EXPECT_EQ(first.merge_deltas, second.merge_deltas);
    for (std::size_t cluster{0}; cluster < first.num_clusters; ++cluster) {
        for (std::size_t symbol{0}; symbol < 16; ++symbol) {
            EXPECT_GT(first.depths[cluster * AC_HISTOGRAM_SIZE + symbol], 0u);
            EXPECT_LE(first.depths[cluster * AC_HISTOGRAM_SIZE + symbol], 15u);
        }
    }
}

}  // namespace
}  // namespace cujpegxl
