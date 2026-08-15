// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "context_map.h"

#include <array>
#include <cassert>

namespace cujpegxl::bitstream {

void move_to_front_transform(const std::uint8_t* input, std::size_t size, std::uint8_t* output) {
    assert(size == 0 || (input != nullptr && output != nullptr));

    std::array<std::uint8_t, 256> symbols{};
    for (std::size_t i{0}; i < symbols.size(); ++i) {
        symbols[i] = static_cast<std::uint8_t>(i);
    }

    for (std::size_t i{0}; i < size; ++i) {
        std::size_t rank{0};
        while (symbols[rank] != input[i]) {
            ++rank;
        }
        output[i] = static_cast<std::uint8_t>(rank);

        const std::uint8_t value{symbols[rank]};
        while (rank > 0) {
            symbols[rank] = symbols[rank - 1];
            --rank;
        }
        symbols[0] = value;
    }
}

}  // namespace cujpegxl::bitstream
