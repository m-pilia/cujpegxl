// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "coeff_order.h"

#include <utility>

namespace cujpegxl {

std::vector<std::uint32_t> natural_coeff_order(std::size_t block_dim) {
    // Square specialization of libjxl's CoeffOrderAndLut: for cx == cy the
    // subsampling factor xs collapses to 1, so it degenerates to the diagonal
    // zigzag over the block with the low-frequency (cx*cy) corner reserved for
    // the LLF/DC positions.
    const std::size_t cx{block_dim / 8};
    const std::size_t n{block_dim};
    std::vector<std::uint32_t> out(n * n);

    std::size_t cur{cx * cx};
    for (std::size_t i{0}; i < n; ++i) {
        for (std::size_t j{0}; j <= i; ++j) {
            std::size_t x{j};
            std::size_t y{i - j};
            if (i & 1) {
                std::swap(x, y);
            }
            const std::size_t val{(x < cx && y < cx) ? y * cx + x : cur++};
            out[val] = static_cast<std::uint32_t>(y * n + x);
        }
    }
    for (std::size_t ip{n - 1}; ip > 0; --ip) {
        const std::size_t i{ip - 1};
        for (std::size_t j{0}; j <= i; ++j) {
            std::size_t x{n - 1 - (i - j)};
            std::size_t y{n - 1 - j};
            if (i & 1) {
                std::swap(x, y);
            }
            out[cur++] = static_cast<std::uint32_t>(y * n + x);
        }
    }
    return out;
}

}  // namespace cujpegxl
