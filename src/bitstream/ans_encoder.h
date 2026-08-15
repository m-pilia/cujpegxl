// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_BITSTREAM_ANS_ENCODER_H_
#define CUJPEGXL_SRC_BITSTREAM_ANS_ENCODER_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "ans_histogram.h"
#include "bit_writer.h"
#include "hybrid_uint.h"

namespace cujpegxl::bitstream {

constexpr std::uint32_t ANS_INITIAL_STATE = 0x13u << 16;

struct AnsEncodingTable {
    std::array<std::uint16_t, ANS_ALPHABET_SIZE> frequencies{};
    std::array<std::uint16_t, ANS_ALPHABET_SIZE + 1> offsets{};
    std::array<std::uint16_t, ANS_TABLE_SIZE> reverse_map{};
};

struct AnsToken {
    std::size_t context{};
    std::uint32_t value{};
};

struct AnsStateTransition {
    std::uint32_t state{};
    std::uint16_t renormalization_bits{};
    bool renormalized{};
};

void build_ans_encoding_table(const AnsDistribution& distribution, AnsEncodingTable& table);

AnsStateTransition ans_put_symbol(std::uint32_t state, const AnsEncodingTable& table,
                                  std::size_t symbol);

void write_ans_tokens(BitWriter& w, const AnsToken* tokens, std::size_t num_tokens,
                      const std::uint8_t* context_map, std::size_t num_contexts,
                      const AnsEncodingTable* tables, std::size_t num_clusters,
                      const HybridUintConfig& config, std::uint32_t* renormalization_scratch);

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_SRC_BITSTREAM_ANS_ENCODER_H_
