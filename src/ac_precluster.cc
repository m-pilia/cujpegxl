// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "ac_precluster.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cujpegxl {
namespace {

constexpr std::size_t NUM_SIGNATURES = AC_MAX_PRECLUSTERS;

// The signature is scale-independent and captures zero mass, token mean,
// occupied span, and sparsity. Their 3 * 8 * 4 * 2 combinations plus the all-empty
// signature bound the candidate count at 193 before greedy merging.
std::uint16_t histogram_signature(const std::uint32_t* histogram) {
    std::uint64_t total{0};
    std::uint64_t weighted_sum{0};
    std::size_t first{AC_HISTOGRAM_SIZE};
    std::size_t last{0};
    std::size_t populated{0};
    for (std::size_t symbol{0}; symbol < AC_HISTOGRAM_SIZE; ++symbol) {
        const std::uint32_t count{histogram[symbol]};
        if (count == 0) {
            continue;
        }
        total += count;
        weighted_sum += static_cast<std::uint64_t>(count) * symbol;
        if (first == AC_HISTOGRAM_SIZE) {
            first = symbol;
        }
        last = symbol;
        ++populated;
    }
    if (total == 0) {
        return 0;
    }

    const std::uint64_t zero_count{histogram[0]};
    const std::uint32_t zero_band{zero_count * 4 >= total * 3 ? 0u
                                  : zero_count * 4 >= total   ? 1u
                                                             : 2u};
    const std::uint64_t mean{weighted_sum / total};
    const std::uint32_t mean_band{static_cast<std::uint32_t>(mean / 32 < 8 ? mean / 32 : 7)};
    const std::size_t span{last - first};
    const std::uint32_t span_band{static_cast<std::uint32_t>(span / 64 < 4 ? span / 64 : 3)};
    const std::uint32_t populated_band{populated > 4 ? 1u : 0u};
    return static_cast<std::uint16_t>(
        1 + (((zero_band * 8 + mean_band) * 4 + span_band) * 2 + populated_band));
}

}  // namespace

void ac_precluster(const std::uint32_t* context_histograms, AcPreclusterResult& result) {
    assert(context_histograms != nullptr);
    result.context_map.fill(0);
    result.signatures.fill(0);
    result.histograms.fill(0);
    result.num_candidates = 0;

    std::array<bool, NUM_SIGNATURES> used{};
    std::array<std::uint16_t, AC_NUM_CONTEXTS> context_signatures{};
    for (std::size_t context{0}; context < AC_NUM_CONTEXTS; ++context) {
        const std::uint16_t signature{histogram_signature(
            context_histograms + context * AC_HISTOGRAM_SIZE)};
        assert(signature < NUM_SIGNATURES);
        context_signatures[context] = signature;
        used[signature] = true;
    }

    std::array<std::uint16_t, NUM_SIGNATURES> signature_to_candidate{};
    for (std::size_t signature{0}; signature < NUM_SIGNATURES; ++signature) {
        if (!used[signature]) {
            continue;
        }
        signature_to_candidate[signature] = static_cast<std::uint16_t>(result.num_candidates);
        result.signatures[result.num_candidates] = static_cast<std::uint16_t>(signature);
        ++result.num_candidates;
    }

    for (std::size_t context{0}; context < AC_NUM_CONTEXTS; ++context) {
        const std::size_t candidate{signature_to_candidate[context_signatures[context]]};
        result.context_map[context] = static_cast<std::uint16_t>(candidate);
        const std::uint32_t* source{context_histograms + context * AC_HISTOGRAM_SIZE};
        std::uint32_t* destination{
            result.histograms.data() + candidate * AC_HISTOGRAM_SIZE};
        for (std::size_t symbol{0}; symbol < AC_HISTOGRAM_SIZE; ++symbol) {
            assert(source[symbol] <=
                   std::numeric_limits<std::uint32_t>::max() - destination[symbol]);
            destination[symbol] += source[symbol];
        }
    }
}

}  // namespace cujpegxl
