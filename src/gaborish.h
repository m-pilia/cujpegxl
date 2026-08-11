// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_GABORISH_H_
#define CUJPEGXL_SRC_GABORISH_H_

#include <cstddef>

namespace cujpegxl {

// GaborishInverse (libjxl enc_gaborish.cc) normalized 5x5 symmetric sharpening at
// the encoder's default per-channel strength mul = 1. Applied to the opsin before
// the DCT, it cancels the decoder's default gaborish smoothing (loop_filter
// all_default), which otherwise blurs away high-frequency detail. Weights follow
// the WeightsSymmetric5 quadrant layout {c, r, R, d, L, D}.
inline constexpr float GAB_K0 = -0.09495815671340026f;
inline constexpr float GAB_K1 = -0.041031725066768575f;
inline constexpr float GAB_K2 = 0.013710004822696948f;
inline constexpr float GAB_K3 = 0.006510206083837737f;
inline constexpr float GAB_K4 = -0.0014789063378272242f;
inline constexpr float GAB_NM =
    1.0f / (1.0f + 4.0f * (GAB_K0 + GAB_K1 + GAB_K2 + GAB_K4 + 2.0f * GAB_K3));
inline constexpr float GAB_WC = GAB_NM;            // center (0,0)
inline constexpr float GAB_WR = GAB_NM * GAB_K0;   // r: (1,0),(0,1)
inline constexpr float GAB_WR2 = GAB_NM * GAB_K2;  // R: (2,0),(0,2)
inline constexpr float GAB_WD = GAB_NM * GAB_K1;   // d: (1,1)
inline constexpr float GAB_WL = GAB_NM * GAB_K3;   // L: (2,1),(1,2)
inline constexpr float GAB_WD2 = GAB_NM * GAB_K4;  // D: (2,2)

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_GABORISH_H_
