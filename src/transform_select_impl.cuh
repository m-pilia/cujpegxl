// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_TRANSFORM_SELECT_IMPL_CUH_
#define CUJPEGXL_SRC_TRANSFORM_SELECT_IMPL_CUH_

#include <cstddef>
#include <cstdint>

#include "vardct_layout.h"

namespace cujpegxl {

// Provisional Lagrange weight (pixel-domain SSE per estimated bit) for the
// selection RD proxy. Chosen so smooth regions coalesce into large blocks while
// localized detail splits; it is not yet calibrated against coded size/quality.
inline constexpr float TRANSFORM_SELECT_LAMBDA = 6.0e-4f;

// Minimum proxy-cost reduction required to prefer a split over the larger block.
// A flat block's AC is analytically zero but leaves a small single-precision
// residual; this floor keeps such numerical ties on the larger block while
// staying far below any real split benefit (region costs are O(1) and up).
inline constexpr float TRANSFORM_SELECT_SPLIT_EPS = 1.0e-6f;

#if defined(__CUDACC__)
#define CUJPEGXL_HD __host__ __device__
#else
#define CUJPEGXL_HD
#endif

// Marks the first-block signal for an NxN transform at (bx, by): the top-left 8x8
// block carries the side (8/16/32); the rest of its covered footprint is
// ACS_COVERED.
CUJPEGXL_HD inline void write_first_block(std::int8_t* acs, std::size_t bw, std::size_t bx,
                                          std::size_t by, int n) {
    const int side{n / 8};
    for (int dy{0}; dy < side; ++dy) {
        for (int dx{0}; dx < side; ++dx) {
            acs[(by + dy) * bw + (bx + dx)] =
                (dy == 0 && dx == 0) ? static_cast<std::int8_t>(n) : ACS_COVERED;
        }
    }
}

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_TRANSFORM_SELECT_IMPL_CUH_
