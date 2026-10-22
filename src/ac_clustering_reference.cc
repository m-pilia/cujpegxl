// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "ac_clustering_reference.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

namespace cujpegxl {
namespace {

constexpr std::size_t NUM_CONTEXTS{AC_NUM_CONTEXTS};

using Histogram = std::vector<std::uint32_t>;

// A histogram with the terms of its AcHistogramRef.
struct Terms {
    std::vector<std::int64_t> bin_xlog2x;
    std::vector<std::int32_t> bin_log2;
    AcHistogramRef ref;
    std::int32_t total_log2;
};

Terms terms_of(const std::uint32_t* bins, const AcLog2Table& table) {
    Terms t{std::vector<std::int64_t>(AC_CONTEXT_ALPHABET, 0),
            std::vector<std::int32_t>(AC_CONTEXT_ALPHABET, 0),
            {},
            0};
    t.ref.bins = bins;
    for (std::size_t s{0}; s < AC_CONTEXT_ALPHABET; ++s) {
        t.ref.total += bins[s];
        if (bins[s] != 0) {
            t.ref.mask |= std::uint64_t{1} << s;
            t.bin_xlog2x[s] = ac_xlog2x(bins[s], table.data());
            t.bin_log2[s] = ac_fixed_log2(bins[s], table.data());
        }
    }
    t.ref.bin_xlog2x = t.bin_xlog2x.data();
    t.ref.total_xlog2x = ac_xlog2x(t.ref.total, table.data());
    t.total_log2 = t.ref.total != 0 ? ac_fixed_log2(t.ref.total, table.data()) : 0;
    return t;
}

std::vector<Histogram> cluster_histograms(const std::uint32_t* context_histograms,
                                          const std::vector<std::size_t>& active,
                                          const std::vector<std::uint32_t>& assignment) {
    std::vector<Histogram> clusters(AC_MAX_CLUSTERS, Histogram(AC_CONTEXT_ALPHABET, 0));
    for (std::size_t context : active) {
        for (std::size_t s{0}; s < AC_CONTEXT_ALPHABET; ++s) {
            clusters[assignment[context]][s] +=
                context_histograms[context * AC_CONTEXT_ALPHABET + s];
        }
    }
    return clusters;
}

}  // namespace

AcClustering cluster_ac_contexts_reference(const std::uint32_t* context_histograms) {
    const AcLog2Table table{ac_log2_table()};
    std::vector<Terms> context_terms{};
    std::vector<AcHistogramRef> contexts{};
    std::vector<std::size_t> active{};
    for (std::size_t i{0}; i < NUM_CONTEXTS; ++i) {
        context_terms.push_back(terms_of(context_histograms + i * AC_CONTEXT_ALPHABET, table));
        contexts.push_back(context_terms.back().ref);
        if (contexts.back().total != 0) {
            active.push_back(i);
        }
    }

    std::vector<std::size_t> seeds{};
    if (!active.empty()) {
        std::size_t largest{0};
        for (std::size_t a{1}; a < active.size(); ++a) {
            if (contexts[active[a]].total > contexts[active[largest]].total) {
                largest = a;
            }
        }
        std::vector<std::int64_t> distances(active.size(), std::numeric_limits<std::int64_t>::max());
        for (;;) {
            seeds.push_back(active[largest]);
            distances[largest] = 0;
            if (seeds.size() == AC_MAX_CLUSTERS) {
                break;
            }
            std::int64_t farthest{-1};
            for (std::size_t a{0}; a < active.size(); ++a) {
                if (distances[a] == 0) {
                    continue;
                }
                const std::int64_t cost{
                    ac_merge_cost(contexts[active[a]], contexts[seeds.back()], table.data())};
                distances[a] = std::min(distances[a], std::max<std::int64_t>(cost, 0));
                if (distances[a] > farthest) {
                    farthest = distances[a];
                    largest = a;
                }
            }
            if (farthest < AC_MIN_SEED_DISTANCE) {
                break;
            }
        }
    }

    std::vector<std::uint32_t> assignment(NUM_CONTEXTS, 0);
    for (std::size_t context : active) {
        std::int64_t best{std::numeric_limits<std::int64_t>::max()};
        for (std::size_t j{0}; j < seeds.size(); ++j) {
            const std::int64_t cost{
                seeds[j] == context
                    ? std::numeric_limits<std::int64_t>::min()
                    : ac_merge_cost(contexts[context], contexts[seeds[j]], table.data())};
            if (cost < best) {
                best = cost;
                assignment[context] = static_cast<std::uint32_t>(j);
            }
        }
    }
    for (int it{0}; it < AC_CLUSTER_LLOYD_ITERATIONS; ++it) {
        const std::vector<Histogram> clusters{
            cluster_histograms(context_histograms, active, assignment)};
        std::vector<Terms> refs{};
        for (const Histogram& c : clusters) {
            refs.push_back(terms_of(c.data(), table));
        }
        for (std::size_t context : active) {
            const AcHistogramRef& a{contexts[context]};
            std::int64_t best{std::numeric_limits<std::int64_t>::max()};
            for (std::size_t j{0}; j < AC_MAX_CLUSTERS; ++j) {
                if (refs[j].ref.total == 0 || (a.mask & ~refs[j].ref.mask) != 0) {
                    continue;
                }
                const std::int64_t cost{
                    ac_coding_cost(a, refs[j].total_log2, refs[j].bin_log2.data())};
                if (cost < best) {
                    best = cost;
                    assignment[context] = static_cast<std::uint32_t>(j);
                }
            }
        }
    }

    const std::vector<Histogram> clusters{
        cluster_histograms(context_histograms, active, assignment)};
    std::vector<std::uint32_t> rank(AC_MAX_CLUSTERS, AC_MAX_CLUSTERS);
    AcClustering out{};
    out.num_clusters = 0;
    for (std::size_t context : active) {
        if (rank[assignment[context]] == AC_MAX_CLUSTERS) {
            const std::uint32_t r{out.num_clusters++};
            rank[assignment[context]] = r;
            std::copy(clusters[assignment[context]].begin(), clusters[assignment[context]].end(),
                      out.histograms + r * AC_CONTEXT_ALPHABET);
        }
    }
    out.num_clusters = std::max<std::uint32_t>(out.num_clusters, 1);
    std::uint8_t previous{0};
    for (std::size_t i{0}; i < NUM_CONTEXTS; ++i) {
        if (contexts[i].total != 0) {
            previous = static_cast<std::uint8_t>(rank[assignment[i]]);
        }
        out.context_map[i] = previous;
    }
    return out;
}

}  // namespace cujpegxl
