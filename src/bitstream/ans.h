// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_BITSTREAM_ANS_H_
#define CUJPEGXL_SRC_BITSTREAM_ANS_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "bit_writer.h"

namespace cujpegxl::bitstream {

constexpr std::uint32_t ANS_LOG_TAB_SIZE{12};
constexpr std::uint32_t ANS_TAB_SIZE{1u << ANS_LOG_TAB_SIZE};
constexpr std::size_t ANS_MAX_ALPHABET_SIZE{256};

// The decoder starts from this state and checks that it ends there, so the
// encoder, which runs backwards, starts from it as well.
constexpr std::uint32_t ANS_INITIAL_STATE{0x13u << 16};

// A histogram normalized to ANS_TAB_SIZE. Symbol s owns the `freq[s]` decoder
// table slots listed in reverse_map[offset[s], offset[s] + freq[s]), in the
// order the alias table assigns them, which is the rANS encoder's state map.
struct AnsHistogram {
    std::array<std::uint16_t, ANS_MAX_ALPHABET_SIZE> freq{};
    std::array<std::uint16_t, ANS_MAX_ALPHABET_SIZE> offset{};
    std::array<std::uint16_t, ANS_TAB_SIZE> reverse_map{};
};

// Normalizes `histogram` (`alphabet_size` bins, 1..ANS_MAX_ALPHABET_SIZE) and
// writes its description as libjxl's BuildAndStoreANSEncodingData does with the
// kApproximate strategy (cjxl's choice at effort 7), filling `out` for the
// alias table of `log_alpha_size` (5..8), the value signalled with the
// histograms.
void write_ans_histogram(BitWriter& w, const std::uint32_t* histogram, std::size_t alphabet_size,
                         std::size_t log_alpha_size, AnsHistogram& out);

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_SRC_BITSTREAM_ANS_H_
