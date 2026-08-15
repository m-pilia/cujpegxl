// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_BITSTREAM_ANS_HISTOGRAM_H_
#define CUJPEGXL_SRC_BITSTREAM_ANS_HISTOGRAM_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "bit_writer.h"
#include "hybrid_uint.h"

namespace cujpegxl::bitstream {

constexpr std::size_t ANS_ALPHABET_SIZE = 256;
constexpr std::size_t ANS_TABLE_SIZE = 4096;

struct AnsAliasEntry {
    std::uint8_t cutoff{};
    std::uint8_t right_value{};
    std::uint16_t frequency{};
    std::uint16_t right_offset{};
    std::uint16_t right_frequency_xor{};
};

struct AnsDistribution {
    std::array<std::uint16_t, ANS_ALPHABET_SIZE> counts{};
    std::array<AnsAliasEntry, ANS_ALPHABET_SIZE> aliases{};
};

void build_ans_distribution(const std::uint32_t* histogram, std::size_t alphabet_size,
                            AnsDistribution& distribution);

void write_clustered_ans_histograms(BitWriter& w, const std::uint8_t* context_map,
                                    std::size_t num_contexts, std::size_t num_clusters,
                                    const std::uint32_t* cluster_histograms, std::size_t stride,
                                    const HybridUintConfig& config, AnsDistribution* distributions);

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_SRC_BITSTREAM_ANS_HISTOGRAM_H_
