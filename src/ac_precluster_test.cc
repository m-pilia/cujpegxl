// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "ac_precluster.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace cujpegxl {
namespace {

std::vector<std::uint32_t> empty_histograms() {
    return std::vector<std::uint32_t>(AC_CONTEXT_HISTOGRAM_ENTRIES, 0);
}

void set_count(std::vector<std::uint32_t>& histograms, std::size_t context,
               std::size_t symbol, std::uint32_t count) {
    histograms[context * AC_HISTOGRAM_SIZE + symbol] = count;
}

void expect_equal(const AcPreclusterResult& a, const AcPreclusterResult& b) {
    EXPECT_EQ(a.num_candidates, b.num_candidates);
    EXPECT_EQ(a.context_map, b.context_map);
    EXPECT_EQ(a.signatures, b.signatures);
    EXPECT_EQ(a.histograms, b.histograms);
}

TEST(AcPrecluster, EmptyContextsShareOneCandidate) {
    const std::vector<std::uint32_t> histograms{empty_histograms()};
    AcPreclusterResult result{};
    ac_precluster(histograms.data(), result);

    EXPECT_EQ(result.num_candidates, 1u);
    EXPECT_EQ(result.signatures[0], 0u);
    for (std::uint16_t candidate : result.context_map) {
        EXPECT_EQ(candidate, 0u);
    }
}

TEST(AcPrecluster, AggregatesMatchingSignatures) {
    std::vector<std::uint32_t> histograms{empty_histograms()};
    set_count(histograms, 0, 0, 90);
    set_count(histograms, 0, 2, 10);
    set_count(histograms, 1, 0, 180);
    set_count(histograms, 1, 2, 20);
    set_count(histograms, 2, 96, 50);
    set_count(histograms, 2, 220, 50);

    AcPreclusterResult result{};
    ac_precluster(histograms.data(), result);

    const std::size_t first{result.context_map[0]};
    EXPECT_EQ(result.context_map[1], first);
    EXPECT_NE(result.context_map[2], first);
    EXPECT_EQ(result.histograms[first * AC_HISTOGRAM_SIZE], 270u);
    EXPECT_EQ(result.histograms[first * AC_HISTOGRAM_SIZE + 2], 30u);
}

TEST(AcPrecluster, PreservesEverySymbolTotal) {
    std::vector<std::uint32_t> histograms{empty_histograms()};
    for (std::size_t context{0}; context < AC_NUM_CONTEXTS; context += 37) {
        set_count(histograms, context, context % AC_HISTOGRAM_SIZE,
                  static_cast<std::uint32_t>(context + 1));
        set_count(histograms, context, (context * 13) % AC_HISTOGRAM_SIZE, 7);
    }

    AcPreclusterResult result{};
    ac_precluster(histograms.data(), result);
    EXPECT_LE(result.num_candidates, AC_MAX_PRECLUSTERS);
    for (std::size_t symbol{0}; symbol < AC_HISTOGRAM_SIZE; ++symbol) {
        std::uint64_t before{0};
        for (std::size_t context{0}; context < AC_NUM_CONTEXTS; ++context) {
            before += histograms[context * AC_HISTOGRAM_SIZE + symbol];
        }
        std::uint64_t after{0};
        for (std::size_t candidate{0}; candidate < result.num_candidates; ++candidate) {
            after += result.histograms[candidate * AC_HISTOGRAM_SIZE + symbol];
        }
        EXPECT_EQ(after, before) << "symbol " << symbol;
    }
}

TEST(AcPrecluster, StableOrderingAndAllocationIndependent) {
    std::vector<std::uint32_t> first{empty_histograms()};
    for (std::size_t context{0}; context < AC_NUM_CONTEXTS; ++context) {
        set_count(first, context, (context * 17) % AC_HISTOGRAM_SIZE,
                  static_cast<std::uint32_t>(context % 101 + 1));
        if (context % 5 == 0) {
            set_count(first, context, 0, 200);
        }
    }
    std::vector<std::uint32_t> second{first.begin(), first.end()};

    AcPreclusterResult a{};
    AcPreclusterResult b{};
    AcPreclusterResult repeated{};
    ac_precluster(first.data(), a);
    ac_precluster(second.data(), b);
    ac_precluster(first.data(), repeated);

    expect_equal(a, b);
    expect_equal(a, repeated);
    EXPECT_TRUE(std::is_sorted(a.signatures.begin(),
                               a.signatures.begin() + a.num_candidates));
}

}  // namespace
}  // namespace cujpegxl
