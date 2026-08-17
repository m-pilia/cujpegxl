// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "cfl.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "quant_calibration.h"
#include "quant_weights_dct8.h"

namespace cujpegxl {
namespace {

struct Maps {
    std::vector<std::int8_t> ytox;
    std::vector<std::int8_t> ytob;
};

Maps estimate_device(const std::vector<float>& x, const std::vector<float>& y,
                     const std::vector<float>& b, std::size_t num_tiles,
                     std::size_t coeffs_per_tile) {
    float* dx{nullptr};
    float* dy{nullptr};
    float* db{nullptr};
    std::int8_t* dmx{nullptr};
    std::int8_t* dmb{nullptr};
    const std::size_t bytes{x.size() * sizeof(float)};
    cudaMalloc(&dx, bytes);
    cudaMalloc(&dy, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dmx, num_tiles);
    cudaMalloc(&dmb, num_tiles);
    cudaMemcpy(dx, x.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice);
    EXPECT_TRUE(estimate_cfl(dx, dy, db, num_tiles, coeffs_per_tile, dmx, dmb));
    Maps m{std::vector<std::int8_t>(num_tiles), std::vector<std::int8_t>(num_tiles)};
    cudaMemcpy(m.ytox.data(), dmx, num_tiles, cudaMemcpyDeviceToHost);
    cudaMemcpy(m.ytob.data(), dmb, num_tiles, cudaMemcpyDeviceToHost);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(db);
    cudaFree(dmx);
    cudaFree(dmb);
    return m;
}

TEST(Cfl, DeviceMatchesHostAndIsDeterministic) {
    const std::size_t num_tiles{40};
    const std::size_t per_tile{200};
    std::vector<float> x(num_tiles * per_tile);
    std::vector<float> y(num_tiles * per_tile);
    std::vector<float> b(num_tiles * per_tile);
    std::mt19937 rng{7};
    std::normal_distribution<float> n{0.0f, 0.05f};
    for (std::size_t t{0}; t < num_tiles; ++t) {
        const float ax{0.02f * (static_cast<float>(t) - 20.0f)};  // per-tile X-from-Y slope
        for (std::size_t i{0}; i < per_tile; ++i) {
            const float yv{n(rng)};
            y[t * per_tile + i] = yv;
            x[t * per_tile + i] = ax * yv + 0.1f * n(rng);
            b[t * per_tile + i] = 1.0f * yv + 0.15f * n(rng);
        }
    }

    const Maps dev{estimate_device(x, y, b, num_tiles, per_tile)};
    std::vector<std::int8_t> hx(num_tiles);
    std::vector<std::int8_t> hb(num_tiles);
    estimate_cfl_host(x.data(), y.data(), b.data(), num_tiles, per_tile, hx.data(), hb.data());
    EXPECT_EQ(dev.ytox, hx);
    EXPECT_EQ(dev.ytob, hb);

    const Maps dev2{estimate_device(x, y, b, num_tiles, per_tile)};
    EXPECT_EQ(dev.ytox, dev2.ytox);
    EXPECT_EQ(dev.ytob, dev2.ytob);
}

// A clean X = s*Y correlation is recovered to the nearest map step, and applying
// the estimated factor drives the residual energy far below the raw X energy.
TEST(Cfl, EstimationRecoversSlopeAndReducesResidual) {
    const std::size_t n{512};
    std::vector<float> x(n);
    std::vector<float> y(n);
    std::vector<float> b(n);
    std::mt19937 rng{3};
    std::normal_distribution<float> nd{0.0f, 0.1f};
    const float slope{0.5f};
    for (std::size_t i{0}; i < n; ++i) {
        const float yv{nd(rng)};
        y[i] = yv;
        x[i] = slope * yv;
        b[i] = CFL_BASE_B * yv;
    }
    int mx{0};
    int mb{0};
    cfl_estimate(x.data(), y.data(), b.data(), n, &mx, &mb);

    EXPECT_NEAR(cfl_ytox_ratio(mx), slope, 1.0f / CFL_COLOR_FACTOR);
    EXPECT_NEAR(cfl_ytob_ratio(mb), CFL_BASE_B, 1.0f / CFL_COLOR_FACTOR);

    double raw{0.0};
    double res{0.0};
    for (std::size_t i{0}; i < n; ++i) {
        raw += static_cast<double>(x[i]) * x[i];
        const float e{x[i] - cfl_ytox_ratio(mx) * y[i]};
        res += static_cast<double>(e) * e;
    }
    EXPECT_LT(res, 0.05 * raw);
}

TEST(Cfl, DegenerateTileMapsToBase) {
    const std::size_t n{64};
    const std::vector<float> zero(n, 0.0f);
    int mx{9};
    int mb{9};
    cfl_estimate(zero.data(), zero.data(), zero.data(), n, &mx, &mb);
    EXPECT_EQ(mx, 0);
    EXPECT_EQ(mb, 0);
}

// The residual/quantize/reconstruct chain round-trips each chroma coefficient to
// within half a quant step of the original, with CfL correctly inverted (using
// the 8x8 dequant weights).
TEST(Cfl, QuantizeReconstructWithinHalfStep) {
    const QuantCalibration cal{calibrate_quant(1.0f)};
    const float qgsf{static_cast<float>(cal.raw_quant_field) * cal.global_scale_float};
    const int mx{-30};
    const int mb{12};
    const float ytox{cfl_ytox_ratio(mx)};
    const float ytob{cfl_ytob_ratio(mb)};

    std::mt19937 rng{5};
    std::normal_distribution<float> nd{0.0f, 0.02f};
    for (int trial{0}; trial < 64; ++trial) {
        for (int k{1}; k < 64; ++k) {
            const float xv{nd(rng)};
            const float yv{nd(rng)};
            const float bv{nd(rng)};
            const float wx{DCT8_DEQUANT_WEIGHTS[0][k]};
            const float wy{DCT8_DEQUANT_WEIGHTS[1][k]};
            const float wb{DCT8_DEQUANT_WEIGHTS[2][k]};

            const int qy{cfl_quantize_coeff(yv, wy, qgsf)};
            const float dqy{cfl_dequantize_coeff(qy, wy, qgsf)};

            const int qx{cfl_quantize_coeff(xv - ytox * dqy, wx, qgsf)};
            const float xrec{cfl_dequantize_coeff(qx, wx, qgsf) + ytox * dqy};
            EXPECT_LE(std::fabs(xrec - xv), 0.5f * wx / qgsf + 1e-6f);

            const int qb{cfl_quantize_coeff(bv - ytob * dqy, wb, qgsf)};
            const float brec{cfl_dequantize_coeff(qb, wb, qgsf) + ytob * dqy};
            EXPECT_LE(std::fabs(brec - bv), 0.5f * wb / qgsf + 1e-6f);
        }
    }
}

}  // namespace
}  // namespace cujpegxl
