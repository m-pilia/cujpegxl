// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_EXACT_DIV_CUH_
#define CUJPEGXL_SRC_EXACT_DIV_CUH_

namespace cujpegxl {

// a / b correctly rounded given inv_b = 1 / b correctly rounded, without the
// slow-path branch of the IEEE division. q is faithful after one residual step,
// so the second step yields the correctly rounded quotient (Markstein) unless
// the residual underflows, which needs |a| close to the smallest normal: the
// callers' quotients are then far below 0.5 and only feed a round to integer.
__device__ inline float div_rn(float a, float b, float inv_b) {
    float q{a * inv_b};
    float r{fmaf(-b, q, a)};
    q = fmaf(r, inv_b, q);
    r = fmaf(-b, q, a);
    return fmaf(r, inv_b, q);
}

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_EXACT_DIV_CUH_
