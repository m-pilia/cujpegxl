// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Validates the device AC token coder against the host bitstream reference: the
// per-context histograms, the context clustering and every group's token
// entries and raw bits must match exactly, whatever the batching of the pack,
// and repeated runs must be identical.

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "ac_clustering_reference.h"
#include "entropy.h"
#include "entropy_roundtrip.h"
#include "src/vardct_layout.h"
#include "tools/bitstream/vardct_frame.h"

namespace cujpegxl {
namespace {

using bitstream::AcReference;
using bitstream::FrameCoefficients;
using bitstream::reference_ac_encode;

// Flattens fc.ac into the device coefficient layout (plane-major over channels
// X, Y, B; blocks in raster order; 64 coefficients each).
void expect_same_clustering(const AcClustering& got, const AcClustering& expected) {
    ASSERT_EQ(got.num_clusters, expected.num_clusters);
    EXPECT_EQ(
        std::vector<std::uint32_t>(std::begin(got.histograms), std::end(got.histograms)),
        std::vector<std::uint32_t>(std::begin(expected.histograms), std::end(expected.histograms)));
    EXPECT_EQ(std::vector<std::uint8_t>(std::begin(got.context_map), std::end(got.context_map)),
              std::vector<std::uint8_t>(std::begin(expected.context_map),
                                        std::end(expected.context_map)));
}

std::vector<std::int32_t> flatten_ac(const FrameCoefficients& fc) {
    std::vector<std::int32_t> q{};
    q.reserve(fc.ac[0].size() * 3);
    for (int c{0}; c < 3; ++c) {
        q.insert(q.end(), fc.ac[c].begin(), fc.ac[c].end());
    }
    return q;
}

FrameCoefficients make_frame(std::size_t w, std::size_t h) {
    FrameCoefficients fc{};
    fc.width = w;
    fc.height = h;
    fc.global_scale = 4096;
    fc.quant_dc = 32;
    fc.raw_quant_field = 32;
    const std::size_t blocks{(w / 8) * (h / 8)};
    for (int c{0}; c < 3; ++c) {
        fc.dc[c].assign(blocks, 0);
        fc.ac[c].assign(blocks * 64, 0);
    }
    return fc;
}

// Deterministic per-coefficient pattern spanning small and large magnitudes so
// the hybrid-uint split (raw trailing bits) and multi-bit prefix codes are
// exercised across groups.
void fill_ac_pattern(FrameCoefficients& fc) {
    const std::size_t bw{fc.width / 8};
    const std::size_t bh{fc.height / 8};
    for (std::size_t by{0}; by < bh; ++by) {
        for (std::size_t bx{0}; bx < bw; ++bx) {
            const std::size_t block{by * bw + bx};
            for (int c{0}; c < 3; ++c) {
                std::int32_t* blk{&fc.ac[c][block * 64]};
                blk[1] = static_cast<std::int32_t>((bx + c) % 7) - 3;
                blk[8] = static_cast<std::int32_t>((by * 2 + c) % 11) - 5;
                blk[9] = static_cast<std::int32_t>((bx + by) % 5) - 2;
                blk[16] = static_cast<std::int32_t>((bx * 3 + by) % 260) - 130;
                blk[2] = static_cast<std::int32_t>((by + c) % 3) - 1;
            }
        }
    }
}

// Marks a first-block of `side` and its covered interior in fc.acs.
void set_first_block(FrameCoefficients& fc, std::size_t bw, int side, std::size_t bx,
                     std::size_t by) {
    const std::size_t s{static_cast<std::size_t>(side) / 8};
    for (std::size_t dy{0}; dy < s; ++dy) {
        for (std::size_t dx{0}; dx < s; ++dx) {
            fc.acs[(by + dy) * bw + (bx + dx)] =
                (dy == 0 && dx == 0) ? static_cast<std::int8_t>(side) : ACS_COVERED;
        }
    }
}

// Fills a mixed DCT32/DCT16/DCT8 frame with a per-coefficient pattern in the
// covered-block layout, so the device mixed-block AC path is diffed against the
// host reference across strategies.
FrameCoefficients make_mixed_frame(std::size_t w, std::size_t h) {
    FrameCoefficients fc{make_frame(w, h)};
    const std::size_t bw{w / 8};
    const std::size_t bh{h / 8};
    fc.acs.assign(bw * bh, 8);
    set_first_block(fc, bw, 32, 0, 0);
    set_first_block(fc, bw, 16, 4, 0);
    set_first_block(fc, bw, 16, 0, 4);
    for (std::size_t by{0}; by < bh; ++by) {
        for (std::size_t bx{0}; bx < bw; ++bx) {
            const int side{fc.acs[by * bw + bx]};
            if (side == ACS_COVERED) {
                continue;
            }
            for (int raw{1}; raw < side * side; raw += 3) {
                const std::size_t slot{
                    covered_plane_slot(side, bx, by, bw, static_cast<std::size_t>(raw))};
                for (int c{0}; c < 3; ++c) {
                    fc.ac[c][slot] = static_cast<std::int32_t>((raw + bx + by + c) % 9) - 4;
                }
            }
        }
    }
    return fc;
}

void check_matches_reference(const FrameCoefficients& fc, std::size_t batches) {
    const AcReference ref{reference_ac_encode(fc)};
    AcDeviceTokens dev{};
    ASSERT_TRUE(ac_tokens_device(flatten_ac(fc), fc.acs, fc.width, fc.height, batches, dev));

    EXPECT_EQ(dev.context_histograms, ref.context_histograms);
    expect_same_clustering(dev.clustering, ref.clustering);
    const std::size_t num_groups{ref.group_entries.size()};
    ASSERT_EQ(dev.layout.size(), 3 * num_groups + 2);
    const std::uint32_t* const token_offset{dev.layout.data()};
    const std::uint32_t* const raw_offset{token_offset + num_groups + 1};
    const std::uint32_t* const raw_bits{raw_offset + num_groups + 1};
    std::uint32_t tokens{0};
    std::uint32_t words{0};
    for (std::size_t g{0}; g < num_groups; ++g) {
        const std::vector<std::uint16_t>& entries{ref.group_entries[g]};
        ASSERT_EQ(token_offset[g], tokens) << "group " << g;
        ASSERT_EQ(raw_offset[g], words) << "group " << g;
        ASSERT_EQ(raw_bits[g], ref.group_raw_bits[g]) << "group " << g;
        EXPECT_EQ(std::vector<std::uint16_t>(dev.entries.begin() + tokens,
                                             dev.entries.begin() + tokens + entries.size()),
                  entries)
            << "group " << g << " entries";
        const std::uint32_t group_words{(raw_bits[g] + 63) / 64};
        EXPECT_EQ(std::vector<std::uint64_t>(dev.raw.begin() + words,
                                             dev.raw.begin() + words + group_words),
                  std::vector<std::uint64_t>(ref.group_raw[g].begin(),
                                             ref.group_raw[g].begin() + group_words))
            << "group " << g << " raw bits";
        tokens += static_cast<std::uint32_t>(entries.size());
        words += group_words;
    }
    EXPECT_EQ(token_offset[num_groups], tokens);
    EXPECT_EQ(raw_offset[num_groups], words);
}

TEST(EntropyGpu, Dct8OnlyMatchesReference) {
    FrameCoefficients fc{make_frame(256, 256)};
    fill_ac_pattern(fc);
    fc.acs.assign((256 / 8) * (256 / 8), 8);  // explicit all-DCT8
    check_matches_reference(fc, 1);
}

TEST(EntropyGpu, MixedSingleGroupMatchesReference) {
    check_matches_reference(make_mixed_frame(256, 256), 1);
}

TEST(EntropyGpu, MixedMultiGroupMatchesReference) {
    check_matches_reference(make_mixed_frame(512, 384), 1);
    check_matches_reference(make_mixed_frame(512, 384), 4);
}

TEST(EntropyGpu, PartialGroupsMatchReference) {
    FrameCoefficients fc{make_frame(1048, 520)};
    fill_ac_pattern(fc);
    // Large magnitudes, so that tokens carry up to the 13 raw bits of an int16.
    const std::size_t blocks{(fc.width / 8) * (fc.height / 8)};
    for (std::size_t block{0}; block < blocks; block += 5) {
        for (int c{0}; c < 3; ++c) {
            fc.ac[c][block * 64 + 40] = (static_cast<std::int32_t>((block + c) % 3) - 1) * 32000 +
                                        static_cast<std::int32_t>(block % 97);
        }
    }
    check_matches_reference(fc, 3);
}

TEST(EntropyGpu, Deterministic) {
    const FrameCoefficients fc{make_mixed_frame(512, 384)};
    const std::vector<std::int32_t> q{flatten_ac(fc)};
    AcDeviceTokens a{};
    AcDeviceTokens b{};
    ASSERT_TRUE(ac_tokens_device(q, fc.acs, fc.width, fc.height, 2, a));
    ASSERT_TRUE(ac_tokens_device(q, fc.acs, fc.width, fc.height, 2, b));
    EXPECT_EQ(a.context_histograms, b.context_histograms);
    expect_same_clustering(a.clustering, b.clustering);
    EXPECT_EQ(a.layout, b.layout);
    EXPECT_EQ(a.entries, b.entries);
    EXPECT_EQ(a.raw, b.raw);
}

// Per-context histograms with `num_contexts` contexts holding tokens, each drawn
// from one of `families` distributions, so that the clustering has structure.
std::vector<std::uint32_t> random_context_histograms(std::uint32_t seed, std::size_t num_contexts,
                                                     std::size_t families) {
    std::mt19937 rng{seed};
    std::vector<std::vector<double>> weights(families, std::vector<double>(AC_CONTEXT_ALPHABET));
    for (std::vector<double>& w : weights) {
        const double decay{0.2 + 0.7 * static_cast<double>(rng() % 1000) / 1000.0};
        double p{1.0};
        for (double& x : w) {
            x = rng() % 7 == 0 ? 0.0 : p;
            p *= decay;
        }
    }
    std::vector<std::uint32_t> hist(AC_NUM_CONTEXTS * AC_CONTEXT_ALPHABET, 0);
    for (std::size_t i{0}; i < num_contexts; ++i) {
        const std::size_t context{rng() % AC_NUM_CONTEXTS};
        const std::vector<double>& w{weights[rng() % families]};
        std::discrete_distribution<std::size_t> symbol{w.begin(), w.end()};
        const std::uint32_t tokens{1 + static_cast<std::uint32_t>(rng() % 5000)};
        for (std::uint32_t t{0}; t < tokens; ++t) {
            ++hist[context * AC_CONTEXT_ALPHABET + symbol(rng)];
        }
    }
    return hist;
}

void check_clustering_matches_reference(const std::vector<std::uint32_t>& hist) {
    AcClustering dev{};
    ASSERT_TRUE(ac_cluster_contexts_device(hist, dev));
    expect_same_clustering(dev, cluster_ac_contexts_reference(hist.data()));
}

TEST(EntropyGpu, ClusteringMatchesReference) {
    check_clustering_matches_reference(random_context_histograms(1, 1500, 48));
    check_clustering_matches_reference(random_context_histograms(2, 300, 5));
    check_clustering_matches_reference(random_context_histograms(3, 7425, 64));
}

TEST(EntropyGpu, ClusteringStopsAtMaxClusters) {
    const std::vector<std::uint32_t> hist{random_context_histograms(4, 3000, 200)};
    AcClustering dev{};
    ASSERT_TRUE(ac_cluster_contexts_device(hist, dev));
    EXPECT_EQ(dev.num_clusters, AC_MAX_CLUSTERS);
    expect_same_clustering(dev, cluster_ac_contexts_reference(hist.data()));
}

TEST(EntropyGpu, IdenticalContextsShareOneCluster) {
    std::vector<std::uint32_t> hist(AC_NUM_CONTEXTS * AC_CONTEXT_ALPHABET, 0);
    for (std::size_t context : {5, 900, 7424}) {
        hist[context * AC_CONTEXT_ALPHABET] = 300;
        hist[context * AC_CONTEXT_ALPHABET + 3] = 100;
    }
    AcClustering dev{};
    ASSERT_TRUE(ac_cluster_contexts_device(hist, dev));
    EXPECT_EQ(dev.num_clusters, 1u);
    EXPECT_EQ(dev.histograms[0], 900u);
    EXPECT_EQ(dev.histograms[3], 300u);
    expect_same_clustering(dev, cluster_ac_contexts_reference(hist.data()));
}

TEST(EntropyGpu, EmptyHistogramsGiveOneCluster) {
    const std::vector<std::uint32_t> hist(AC_NUM_CONTEXTS * AC_CONTEXT_ALPHABET, 0);
    AcClustering dev{};
    ASSERT_TRUE(ac_cluster_contexts_device(hist, dev));
    EXPECT_EQ(dev.num_clusters, 1u);
    expect_same_clustering(dev, cluster_ac_contexts_reference(hist.data()));
}

}  // namespace
}  // namespace cujpegxl
