// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_BITSTREAM_ENTROPY_SELECTOR_H_
#define CUJPEGXL_SRC_BITSTREAM_ENTROPY_SELECTOR_H_

#include <cstddef>
#include <cstdint>

#include "ans_encoder.h"
#include "ans_histogram.h"
#include "bit_writer.h"
#include "hybrid_uint.h"

namespace cujpegxl::bitstream {

enum class EntropyMode : std::uint8_t {
    PREFIX,
    ANS,
};

struct EntropySelectionResult {
    EntropyMode mode{};
    std::size_t prefix_bits{};
    std::size_t ans_bits{};
};

constexpr EntropyMode select_entropy_mode(std::size_t prefix_bits, std::size_t ans_bits) {
    return ans_bits < prefix_bits ? EntropyMode::ANS : EntropyMode::PREFIX;
}

EntropySelectionResult write_best_clustered_entropy(
    BitWriter& w, const AnsToken* tokens, std::size_t num_tokens,
    const std::uint8_t* context_map, std::size_t num_contexts, std::size_t num_clusters,
    const std::uint32_t* cluster_histograms, std::size_t stride,
    const HybridUintConfig& config, std::uint8_t* prefix_depth, std::uint16_t* prefix_bits,
    AnsDistribution* ans_distributions, AnsEncodingTable* ans_tables,
    std::uint32_t* ans_renormalization_scratch);

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_SRC_BITSTREAM_ENTROPY_SELECTOR_H_
