// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "vardct_layout.h"

namespace cujpegxl {

void scatter_covered_block(std::size_t block_dim, std::size_t bx, std::size_t by, std::size_t bw,
                           const float* coeffs, float* plane) {
    const std::size_t count{block_dim * block_dim};
    for (std::size_t raw{0}; raw < count; ++raw) {
        plane[covered_plane_slot(block_dim, bx, by, bw, raw)] = coeffs[raw];
    }
}

void gather_covered_block(std::size_t block_dim, std::size_t bx, std::size_t by, std::size_t bw,
                          const float* plane, float* coeffs) {
    const std::size_t count{block_dim * block_dim};
    for (std::size_t raw{0}; raw < count; ++raw) {
        coeffs[raw] = plane[covered_plane_slot(block_dim, bx, by, bw, raw)];
    }
}

}  // namespace cujpegxl
