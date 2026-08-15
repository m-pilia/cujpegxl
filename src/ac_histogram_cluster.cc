// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "ac_histogram_cluster.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "bitstream/bit_writer.h"
#include "bitstream/prefix_code.h"

namespace cujpegxl {
namespace {

constexpr std::uint64_t HISTOGRAM_BASE_BITS = 32;
constexpr std::uint64_t BITS_PER_USED_SYMBOL = 3;

std::uint32_t ceil_log2(std::uint64_t value) {
    if (value <= 1) {
        return 0;
    }
    std::uint32_t bits{0};
    --value;
    while (value != 0) {
        value >>= 1;
        ++bits;
    }
    return bits;
}

std::uint64_t estimated_cost(const std::uint32_t* a, const std::uint32_t* b = nullptr) {
    std::uint64_t total{0};
    std::size_t used{0};
    for (std::size_t symbol{0}; symbol < AC_HISTOGRAM_SIZE; ++symbol) {
        const std::uint64_t count{static_cast<std::uint64_t>(a[symbol]) +
                                  (b == nullptr ? 0 : b[symbol])};
        total += count;
        used += count != 0 ? 1 : 0;
    }
    if (used == 0) {
        return HISTOGRAM_BASE_BITS;
    }

    std::uint64_t payload{0};
    if (used > 1) {
        for (std::size_t symbol{0}; symbol < AC_HISTOGRAM_SIZE; ++symbol) {
            const std::uint64_t count{static_cast<std::uint64_t>(a[symbol]) +
                                      (b == nullptr ? 0 : b[symbol])};
            if (count == 0) {
                continue;
            }
            const std::uint64_t ratio{(total + count - 1) / count};
            const std::uint32_t depth{std::min(ceil_log2(ratio), 15u)};
            payload += count * depth;
        }
    }
    return payload + HISTOGRAM_BASE_BITS + BITS_PER_USED_SYMBOL * used;
}

std::size_t minimum_comparisons(std::size_t candidates, std::size_t target) {
    return candidates > target ? candidates - target : 0;
}

}  // namespace

bool ac_cluster_histograms(const AcPreclusterResult& preclusters, const AcClusterConfig& config,
                           AcClusterResult& result) {
    if (preclusters.num_candidates == 0 ||
        preclusters.num_candidates > AC_MAX_PRECLUSTERS || config.max_clusters == 0 ||
        config.max_comparisons <
            minimum_comparisons(preclusters.num_candidates,
                                std::min(preclusters.num_candidates,
                                         config.max_clusters))) {
        return false;
    }

    const std::size_t target_clusters{
        std::min(preclusters.num_candidates, config.max_clusters)};

    result.context_map.fill(0);
    result.histograms.fill(0);
    result.depths.fill(0);
    result.merge_deltas.fill(0);
    result.num_clusters = preclusters.num_candidates;
    result.num_merges = 0;
    result.comparisons = 0;
    result.estimated_bits = 0;

    std::array<bool, AC_MAX_PRECLUSTERS> active{};
    std::array<std::size_t, AC_MAX_PRECLUSTERS> parent{};
    std::array<std::uint64_t, AC_MAX_PRECLUSTERS> costs{};
    for (std::size_t candidate{0}; candidate < preclusters.num_candidates; ++candidate) {
        active[candidate] = true;
        parent[candidate] = candidate;
        const std::uint32_t* source{
            preclusters.histograms.data() + candidate * AC_HISTOGRAM_SIZE};
        std::uint32_t* destination{
            result.histograms.data() + candidate * AC_HISTOGRAM_SIZE};
        for (std::size_t symbol{0}; symbol < AC_HISTOGRAM_SIZE; ++symbol) {
            destination[symbol] = source[symbol];
        }
        costs[candidate] = estimated_cost(destination);
        result.estimated_bits += costs[candidate];
    }

    while (result.num_clusters > target_clusters) {
        const std::size_t merges_left{result.num_clusters - target_clusters};
        const std::size_t comparisons_left{config.max_comparisons - result.comparisons};
        const std::size_t scan_limit{comparisons_left - (merges_left - 1)};
        std::size_t best_a{AC_MAX_PRECLUSTERS};
        std::size_t best_b{AC_MAX_PRECLUSTERS};
        std::int64_t best_delta{std::numeric_limits<std::int64_t>::max()};
        std::size_t previous{AC_MAX_PRECLUSTERS};
        std::size_t scanned{0};
        for (std::size_t candidate{0}; candidate < preclusters.num_candidates; ++candidate) {
            if (!active[candidate]) {
                continue;
            }
            if (previous != AC_MAX_PRECLUSTERS) {
                const std::uint32_t* a{
                    result.histograms.data() + previous * AC_HISTOGRAM_SIZE};
                const std::uint32_t* b{
                    result.histograms.data() + candidate * AC_HISTOGRAM_SIZE};
                const std::uint64_t merged{estimated_cost(a, b)};
                const std::int64_t delta{static_cast<std::int64_t>(merged) -
                                         static_cast<std::int64_t>(costs[previous]) -
                                         static_cast<std::int64_t>(costs[candidate])};
                ++result.comparisons;
                ++scanned;
                if (delta < best_delta ||
                    (delta == best_delta &&
                     (previous < best_a || (previous == best_a && candidate < best_b)))) {
                    best_a = previous;
                    best_b = candidate;
                    best_delta = delta;
                }
                if (scanned == scan_limit) {
                    break;
                }
            }
            previous = candidate;
        }
        assert(best_a < AC_MAX_PRECLUSTERS && best_b < AC_MAX_PRECLUSTERS);

        std::uint32_t* a{result.histograms.data() + best_a * AC_HISTOGRAM_SIZE};
        const std::uint32_t* b{result.histograms.data() + best_b * AC_HISTOGRAM_SIZE};
        for (std::size_t symbol{0}; symbol < AC_HISTOGRAM_SIZE; ++symbol) {
            if (b[symbol] > std::numeric_limits<std::uint32_t>::max() - a[symbol]) {
                return false;
            }
            a[symbol] += b[symbol];
        }
        result.estimated_bits = static_cast<std::uint64_t>(
            static_cast<std::int64_t>(result.estimated_bits) + best_delta);
        costs[best_a] = estimated_cost(a);
        active[best_b] = false;
        parent[best_b] = best_a;
        result.merge_deltas[result.num_merges] = best_delta;
        ++result.num_merges;
        --result.num_clusters;
    }

    std::array<std::uint8_t, AC_MAX_PRECLUSTERS> compact_id{};
    std::size_t compact_count{0};
    for (std::size_t candidate{0}; candidate < preclusters.num_candidates; ++candidate) {
        if (!active[candidate]) {
            continue;
        }
        compact_id[candidate] = static_cast<std::uint8_t>(compact_count);
        if (candidate != compact_count) {
            const std::uint32_t* source{
                result.histograms.data() + candidate * AC_HISTOGRAM_SIZE};
            std::uint32_t* destination{
                result.histograms.data() + compact_count * AC_HISTOGRAM_SIZE};
            for (std::size_t symbol{0}; symbol < AC_HISTOGRAM_SIZE; ++symbol) {
                destination[symbol] = source[symbol];
            }
        }
        ++compact_count;
    }
    assert(compact_count == result.num_clusters);

    for (std::size_t context{0}; context < AC_NUM_CONTEXTS; ++context) {
        std::size_t root{preclusters.context_map[context]};
        while (parent[root] != root) {
            root = parent[root];
        }
        result.context_map[context] = compact_id[root];
    }

    for (std::size_t cluster{0}; cluster < result.num_clusters; ++cluster) {
        const std::uint32_t* histogram{
            result.histograms.data() + cluster * AC_HISTOGRAM_SIZE};
        std::size_t alphabet_size{1};
        for (std::size_t symbol{0}; symbol < AC_HISTOGRAM_SIZE; ++symbol) {
            if (histogram[symbol] != 0) {
                alphabet_size = symbol + 1;
            }
        }
        if (alphabet_size > 1) {
            bitstream::BitWriter writer{};
            std::array<std::uint16_t, AC_HISTOGRAM_SIZE> bits{};
            bitstream::build_and_store_huffman_tree(
                histogram, alphabet_size,
                result.depths.data() + cluster * AC_HISTOGRAM_SIZE, bits.data(), writer);
        }
    }
    return true;
}

}  // namespace cujpegxl
