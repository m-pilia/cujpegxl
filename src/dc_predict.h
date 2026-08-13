// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_DC_PREDICT_H_
#define CUJPEGXL_SRC_DC_PREDICT_H_

#include <cstdint>

namespace cujpegxl {

#if defined(__CUDACC__)
#define CUJPEGXL_DCP_HD __host__ __device__
#else
#define CUJPEGXL_DCP_HD
#endif

// libjxl modular Predictor enum values used by the DC coder.
inline constexpr int DC_PREDICTOR_ZERO = 0;
inline constexpr int DC_PREDICTOR_GRADIENT = 5;

// libjxl ClampedGradient(n, w, l) = clamp(n + w - l, [min(n,w), max(n,w)]).
CUJPEGXL_DCP_HD inline std::int32_t clamped_gradient(std::int32_t n, std::int32_t w,
                                                     std::int32_t l) {
    const std::int32_t lo{n < w ? n : w};
    const std::int32_t hi{n < w ? w : n};
    const std::int64_t grad{static_cast<std::int64_t>(n) + w - l};
    if (l < lo) {
        return hi;
    }
    if (l > hi) {
        return lo;
    }
    return static_cast<std::int32_t>(grad);
}

// Gradient prediction for a sample given its within-channel neighbor availability
// and values. Edge rules match libjxl modular Predict(): left = has_left ? W :
// (has_top ? N : 0); top = has_top ? N : left; topleft = (has_left && has_top) ?
// NW : left. Neighbor values are read only where the corresponding flag is set,
// so the caller may pass 0 for an absent neighbor. libjxl calls the Gradient
// predictor as ClampedGradient(left, top, topleft).
CUJPEGXL_DCP_HD inline std::int32_t gradient_predict(bool has_left, bool has_top, std::int32_t w_val,
                                                     std::int32_t n_val, std::int32_t nw_val) {
    const std::int32_t left{has_left ? w_val : (has_top ? n_val : 0)};
    const std::int32_t top{has_top ? n_val : left};
    const std::int32_t topleft{(has_left && has_top) ? nw_val : left};
    return clamped_gradient(left, top, topleft);
}

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_DC_PREDICT_H_
