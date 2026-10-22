// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_DEQUANT_WEIGHT_PAIRS_H_
#define CUJPEGXL_SRC_DEQUANT_WEIGHT_PAIRS_H_

#include <cstddef>

#include "quant_weights_dct16.h"
#include "quant_weights_dct32.h"
#include "quant_weights_dct8.h"

namespace cujpegxl {

// A dequantization weight with its correctly rounded reciprocal, so that a
// division by the weight is exact without the IEEE division (div_rn).
struct alignas(8) DequantWeightPair {
    float weight;
    float inv_weight;
};

template <std::size_t COEFFS>
struct DequantWeightPairs {
    DequantWeightPair w[3][COEFFS];
};

template <std::size_t COEFFS>
constexpr DequantWeightPairs<COEFFS> make_dequant_weight_pairs(const float (&weights)[3][COEFFS]) {
    DequantWeightPairs<COEFFS> pairs{};
    for (std::size_t c{0}; c < 3; ++c) {
        for (std::size_t k{0}; k < COEFFS; ++k) {
            pairs.w[c][k] = {weights[c][k], 1.0f / weights[c][k]};
        }
    }
    return pairs;
}

inline constexpr DequantWeightPairs<64> DCT8_DEQUANT_PAIRS{
    make_dequant_weight_pairs(DCT8_DEQUANT_WEIGHTS)};
inline constexpr DequantWeightPairs<256> DCT16_DEQUANT_PAIRS{
    make_dequant_weight_pairs(DCT16_DEQUANT_WEIGHTS)};
inline constexpr DequantWeightPairs<1024> DCT32_DEQUANT_PAIRS{
    make_dequant_weight_pairs(DCT32_DEQUANT_WEIGHTS)};

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_DEQUANT_WEIGHT_PAIRS_H_
