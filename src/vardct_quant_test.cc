// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "cfl.h"
#include "coeff_order.h"
#include "quant_calibration.h"
#include "quant_weights_dct16.h"
#include "quant_weights_dct32.h"
#include "vardct_layout.h"

namespace cujpegxl {
namespace {

const float* weights_for(int side, int channel) {
    if (side == 16) {
        return DCT16_DEQUANT_WEIGHTS[channel];
    }
    return DCT32_DEQUANT_WEIGHTS[channel];
}

// K3 for large blocks: the CfL residual + quantize + dequantize + re-correlate
// chain round-trips each AC chroma coefficient to within half a quant step,
// using the baked DCT16/DCT32 dequant matrices. Mirrors the DCT8 check in
// cfl_gpu_test for the sizes T4 could not yet exercise.
TEST(VardctQuant, LargeBlockCflResidualWithinHalfStep) {
    const QuantCalibration cal{calibrate_quant(1.0f)};
    const float qgsf{static_cast<float>(cal.raw_quant_field) * cal.global_scale_float};
    const float ytox{cfl_ytox_ratio(-24)};
    const float ytob{cfl_ytob_ratio(18)};

    std::mt19937 rng{17};
    std::normal_distribution<float> nd{0.0f, 0.02f};
    for (int side : {16, 32}) {
        const std::size_t n{static_cast<std::size_t>(side) * side};
        const std::size_t covered{covered_blocks_side(side) * covered_blocks_side(side)};
        const std::vector<std::uint32_t> order{natural_coeff_order(side)};
        const std::set<std::uint32_t> llf(order.begin(), order.begin() + covered);

        for (std::size_t raw{0}; raw < n; ++raw) {
            if (llf.count(static_cast<std::uint32_t>(raw)) != 0) {
                continue;  // low-frequency coefficients are carried as DC
            }
            const float xv{nd(rng)};
            const float yv{nd(rng)};
            const float bv{nd(rng)};
            const float wx{weights_for(side, 0)[raw]};
            const float wy{weights_for(side, 1)[raw]};
            const float wb{weights_for(side, 2)[raw]};

            const int qy{cfl_quantize_coeff(yv, wy, qgsf)};
            const float dqy{cfl_dequantize_coeff(qy, wy, qgsf)};

            const int qx{cfl_quantize_coeff(xv - ytox * dqy, wx, qgsf)};
            const float xrec{cfl_dequantize_coeff(qx, wx, qgsf) + ytox * dqy};
            EXPECT_LE(std::fabs(xrec - xv), 0.5f * wx / qgsf + 1e-6f) << "side " << side;

            const int qb{cfl_quantize_coeff(bv - ytob * dqy, wb, qgsf)};
            const float brec{cfl_dequantize_coeff(qb, wb, qgsf) + ytob * dqy};
            EXPECT_LE(std::fabs(brec - bv), 0.5f * wb / qgsf + 1e-6f) << "side " << side;
        }
    }
}

}  // namespace
}  // namespace cujpegxl
