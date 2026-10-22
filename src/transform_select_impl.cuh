// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_TRANSFORM_SELECT_IMPL_CUH_
#define CUJPEGXL_SRC_TRANSFORM_SELECT_IMPL_CUH_

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

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_TRANSFORM_SELECT_IMPL_CUH_
