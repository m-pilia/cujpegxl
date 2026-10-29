// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "ac_clustering_reference.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

namespace cujpegxl {
namespace {

constexpr std::size_t NUM_CONTEXTS{AC_NUM_CONTEXTS};

using Histogram = std::vector<std::uint32_t>;

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
    std::vector<AcHistogramRef> contexts{};
    std::vector<std::size_t> active{};
    std::vector<std::uint32_t> assignment(NUM_CONTEXTS, 0);
    std::uint32_t alphabet{0};
    for (std::size_t i{0}; i < NUM_CONTEXTS; ++i) {
        AcHistogramRef ref{context_histograms + i * AC_CONTEXT_ALPHABET, 0, 0};
        for (std::size_t s{0}; s < AC_CONTEXT_ALPHABET; ++s) {
            ref.total += ref.bins[s];
            if (ref.bins[s] != 0) {
                ref.mask |= std::uint64_t{1} << s;
                alphabet = std::max(alphabet, static_cast<std::uint32_t>(s + 1));
            }
        }
        contexts.push_back(ref);
        if (ref.total != 0) {
            active.push_back(i);
        }
        assignment[i] = ac_initial_cluster(static_cast<std::uint32_t>(i));
    }

    for (int it{0}; it < AC_CLUSTER_LLOYD_ITERATIONS; ++it) {
        std::vector<std::int32_t> total_log2{};
        std::vector<std::int32_t> bin_log2{};
        const auto cluster_log2 = [&] {
            const std::vector<Histogram> clusters{
                cluster_histograms(context_histograms, active, assignment)};
            total_log2.assign(AC_MAX_CLUSTERS, -1);
            bin_log2.assign(AC_MAX_CLUSTERS * AC_CONTEXT_ALPHABET, 0);
            for (std::size_t j{0}; j < AC_MAX_CLUSTERS; ++j) {
                std::uint32_t total{0};
                for (std::size_t s{0}; s < AC_CONTEXT_ALPHABET; ++s) {
                    total += clusters[j][s];
                    bin_log2[j * AC_CONTEXT_ALPHABET + s] =
                        ac_smoothed_bin_log2(clusters[j][s], table.data());
                }
                if (total != 0) {
                    total_log2[j] = ac_smoothed_total_log2(total, alphabet, table.data());
                }
            }
        };
        cluster_log2();
        if (it == AC_CLUSTER_REFILL_ITERATION) {
            std::vector<std::uint64_t> keys(AC_MAX_CLUSTERS, 0);
            for (std::size_t context : active) {
                const std::uint32_t j{assignment[context]};
                const std::uint64_t excess{
                    ac_refill_excess(contexts[context], total_log2[j],
                                     bin_log2.data() + j * AC_CONTEXT_ALPHABET, table.data())};
                keys[j] =
                    std::max(keys[j], ac_refill_key(excess, static_cast<std::uint32_t>(context)));
            }
            std::vector<std::uint32_t> empty_clusters{};
            for (std::uint32_t j{0}; j < AC_MAX_CLUSTERS; ++j) {
                if (total_log2[j] < 0) {
                    empty_clusters.push_back(j);
                }
            }
            std::sort(keys.begin(), keys.end(), std::greater<>{});
            for (std::size_t m{0}; m < empty_clusters.size() && keys[m] != 0; ++m) {
                assignment[ac_refill_key_context(keys[m])] = empty_clusters[m];
            }
            cluster_log2();
        }
        for (std::size_t context : active) {
            std::int64_t best{std::numeric_limits<std::int64_t>::max()};
            for (std::size_t j{0}; j < AC_MAX_CLUSTERS; ++j) {
                if (total_log2[j] < 0) {
                    continue;
                }
                const std::int64_t cost{ac_coding_cost(contexts[context], total_log2[j],
                                                       bin_log2.data() + j * AC_CONTEXT_ALPHABET)};
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
