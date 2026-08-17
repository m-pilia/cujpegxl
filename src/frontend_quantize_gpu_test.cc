// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frontend_quantize.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "cfl.h"
#include "quant_calibration.h"
#include "quant_weights_dct16.h"
#include "quant_weights_dct32.h"
#include "quant_weights_dct8.h"
#include "vardct_layout.h"

namespace cujpegxl {
namespace {

constexpr std::size_t WIDTH = 128;
constexpr std::size_t HEIGHT = 128;
constexpr std::size_t BW = WIDTH / 8;
constexpr std::size_t BH = HEIGHT / 8;
constexpr std::size_t CPLANE = BW * BH * COEFFS_PER_BLOCK;
constexpr std::size_t NBLK = BW * BH;

float weight(int side, int c, int raw) {
    if (side == 16) {
        return DCT16_DEQUANT_WEIGHTS[c][raw];
    }
    if (side == 32) {
        return DCT32_DEQUANT_WEIGHTS[c][raw];
    }
    return DCT8_DEQUANT_WEIGHTS[c][raw];
}

bool is_llf(int raw, int side) {
    const int cx{side / 8};
    return (raw / side) < cx && (raw % side) < cx;
}

void place(std::vector<std::int8_t>& acs, int side, std::size_t bx, std::size_t by) {
    const std::size_t s{static_cast<std::size_t>(side) / 8};
    for (std::size_t dy{0}; dy < s; ++dy) {
        for (std::size_t dx{0}; dx < s; ++dx) {
            acs[(by + dy) * BW + (bx + dx)] =
                (dy == 0 && dx == 0) ? static_cast<std::int8_t>(side) : ACS_COVERED;
        }
    }
}

// Builds FP16 covered-block coefficients with a correlated chroma signal so CfL
// estimation has something to find, over a mixed DCT32/DCT16/DCT8 tiling.
struct Frame {
    std::vector<__half> coeffs;
    std::vector<std::int8_t> acs;
};

Frame make_frame() {
    Frame f{};
    f.coeffs.assign(3 * CPLANE, __float2half(0.0f));
    f.acs.assign(BW * BH, 8);
    place(f.acs, 32, 0, 0);
    place(f.acs, 16, 4, 0);
    place(f.acs, 16, 0, 4);

    std::mt19937 rng{5};
    std::normal_distribution<float> nd{0.0f, 0.03f};
    for (std::size_t by{0}; by < BH; ++by) {
        for (std::size_t bx{0}; bx < BW; ++bx) {
            const int side{f.acs[by * BW + bx]};
            if (side == ACS_COVERED) {
                continue;
            }
            for (int raw{0}; raw < side * side; ++raw) {
                const std::size_t slot{
                    covered_plane_slot(side, bx, by, BW, static_cast<std::size_t>(raw))};
                const float yv{nd(rng)};
                f.coeffs[CPLANE + slot] = __float2half(yv);                       // Y
                f.coeffs[slot] = __float2half(0.4f * yv + 0.3f * nd(rng));        // X ~ corr
                f.coeffs[2 * CPLANE + slot] = __float2half(yv + 0.2f * nd(rng));  // B ~ base+corr
            }
        }
    }
    return f;
}

std::vector<__half> upload_coeffs(const Frame& f, __half** d_coeffs, std::int8_t** d_acs) {
    cudaMalloc(d_coeffs, f.coeffs.size() * sizeof(__half));
    cudaMalloc(d_acs, f.acs.size());
    cudaMemcpy(*d_coeffs, f.coeffs.data(), f.coeffs.size() * sizeof(__half),
               cudaMemcpyHostToDevice);
    cudaMemcpy(*d_acs, f.acs.data(), f.acs.size(), cudaMemcpyHostToDevice);
    return f.coeffs;
}

TEST(FrontendQuantize, CflEstimateDeviceMatchesHost) {
    const Frame f{make_frame()};
    const std::size_t cmw{(BW + 7) / 8};
    const std::size_t cmh{(BH + 7) / 8};
    __half* d_coeffs{nullptr};
    std::int8_t* d_acs{nullptr};
    upload_coeffs(f, &d_coeffs, &d_acs);
    std::int8_t* d_mx{nullptr};
    std::int8_t* d_mb{nullptr};
    cudaMalloc(&d_mx, cmw * cmh);
    cudaMalloc(&d_mb, cmw * cmh);
    ASSERT_TRUE(estimate_cfl_covered(d_coeffs, d_acs, WIDTH, HEIGHT, d_mx, d_mb));
    std::vector<std::int8_t> dev_mx(cmw * cmh);
    std::vector<std::int8_t> dev_mb(cmw * cmh);
    cudaMemcpy(dev_mx.data(), d_mx, cmw * cmh, cudaMemcpyDeviceToHost);
    cudaMemcpy(dev_mb.data(), d_mb, cmw * cmh, cudaMemcpyDeviceToHost);

    std::vector<std::int8_t> host_mx(cmw * cmh);
    std::vector<std::int8_t> host_mb(cmw * cmh);
    estimate_cfl_covered_host(f.coeffs.data(), f.acs.data(), WIDTH, HEIGHT, host_mx.data(),
                              host_mb.data());
    EXPECT_EQ(dev_mx, host_mx);
    EXPECT_EQ(dev_mb, host_mb);
    // The correlated signal yields a non-trivial X factor somewhere.
    bool any_nonzero{false};
    for (std::int8_t v : dev_mx) {
        any_nonzero = any_nonzero || v != 0;
    }
    EXPECT_TRUE(any_nonzero);

    cudaFree(d_coeffs);
    cudaFree(d_acs);
    cudaFree(d_mx);
    cudaFree(d_mb);
}

// The residual+quantize kernel round-trips each AC chroma coefficient to within
// half a quant step with CfL correctly inverted, and reconstructs the LLF DC
// within half a DC step, across the mixed tiling.
TEST(FrontendQuantize, QuantizeResidualReconstructsWithinHalfStep) {
    const Frame f{make_frame()};
    const std::size_t cmw{(BW + 7) / 8};
    const std::size_t cmh{(BH + 7) / 8};
    __half* d_coeffs{nullptr};
    std::int8_t* d_acs{nullptr};
    upload_coeffs(f, &d_coeffs, &d_acs);

    std::vector<std::int8_t> mx(cmw * cmh, -20);
    std::vector<std::int8_t> mb(cmw * cmh, 10);
    std::vector<std::int32_t> qf(BW * BH, 40);
    std::int8_t* d_mx{nullptr};
    std::int8_t* d_mb{nullptr};
    std::int32_t* d_qf{nullptr};
    std::int16_t* d_ac{nullptr};
    std::int32_t* d_dc{nullptr};
    cudaMalloc(&d_mx, mx.size());
    cudaMalloc(&d_mb, mb.size());
    cudaMalloc(&d_qf, qf.size() * sizeof(std::int32_t));
    cudaMalloc(&d_ac, 3 * CPLANE * sizeof(std::int16_t));
    cudaMalloc(&d_dc, 3 * NBLK * sizeof(std::int32_t));
    cudaMemcpy(d_mx, mx.data(), mx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mb, mb.data(), mb.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qf, qf.data(), qf.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice);

    ASSERT_TRUE(
        quantize_residual(d_coeffs, d_acs, d_mx, d_mb, d_qf, WIDTH, HEIGHT, 1.0f, d_ac, d_dc));
    std::vector<std::int16_t> ac(3 * CPLANE);
    std::vector<std::int32_t> dc(3 * NBLK);
    cudaMemcpy(ac.data(), d_ac, ac.size() * sizeof(std::int16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(dc.data(), d_dc, dc.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost);

    const QuantCalibration cal{calibrate_quant(1.0f)};
    const float gsf{cal.global_scale_float};
    const float dc_scale{gsf * static_cast<float>(cal.quant_dc)};
    for (std::size_t by{0}; by < BH; ++by) {
        for (std::size_t bx{0}; bx < BW; ++bx) {
            const int side{f.acs[by * BW + bx]};
            if (side == ACS_COVERED) {
                continue;
            }
            const std::size_t tile{(by / 8) * cmw + (bx / 8)};
            const float ytox{cfl_ytox_ratio(mx[tile])};
            const float ytob{cfl_ytob_ratio(mb[tile])};
            const float qgsf{static_cast<float>(qf[by * BW + bx]) * gsf};
            for (int raw{0}; raw < side * side; ++raw) {
                if (is_llf(raw, side)) {
                    continue;
                }
                const std::size_t slot{
                    covered_plane_slot(side, bx, by, BW, static_cast<std::size_t>(raw))};
                const float xv{__half2float(f.coeffs[slot])};
                const float yv{__half2float(f.coeffs[CPLANE + slot])};
                const float bv{__half2float(f.coeffs[2 * CPLANE + slot])};
                const float wy{weight(side, 1, raw)};
                const float wx{weight(side, 0, raw)};
                const float wb{weight(side, 2, raw)};
                const float y_round{static_cast<float>(ac[CPLANE + slot]) * wy / qgsf};
                const float x_rec{static_cast<float>(ac[slot]) * wx / qgsf + ytox * y_round};
                const float b_rec{static_cast<float>(ac[2 * CPLANE + slot]) * wb / qgsf +
                                  ytob * y_round};
                EXPECT_LE(std::fabs(x_rec - xv), 0.5f * wx / qgsf + 1e-4f);
                EXPECT_LE(std::fabs(b_rec - bv), 0.5f * wb / qgsf + 1e-4f);
            }
            // DC: dequant + re-add base correlation reconstructs the LLF DC.
            const int cx{side / 8};
            std::vector<float> raw_y(side * side);
            std::vector<float> raw_b(side * side);
            for (int raw{0}; raw < side * side; ++raw) {
                const std::size_t slot{
                    covered_plane_slot(side, bx, by, BW, static_cast<std::size_t>(raw))};
                raw_y[raw] = __half2float(f.coeffs[CPLANE + slot]);
                raw_b[raw] = __half2float(f.coeffs[2 * CPLANE + slot]);
            }
            std::vector<float> want_y(cx * cx);
            std::vector<float> want_b(cx * cx);
            dc_from_llf(side, raw_y.data(), want_y.data());
            dc_from_llf(side, raw_b.data(), want_b.data());
            for (int oy{0}; oy < cx; ++oy) {
                for (int ox{0}; ox < cx; ++ox) {
                    const int p{oy * cx + ox};
                    const std::size_t dblk{(by + oy) * BW + (bx + ox)};
                    const float y_dc_rec{static_cast<float>(dc[NBLK + dblk]) /
                                         (DC_INV_QUANT[1] * dc_scale)};
                    const float b_dc_rec{static_cast<float>(dc[2 * NBLK + dblk]) /
                                             (DC_INV_QUANT[2] * dc_scale) +
                                         CFL_BASE_B * y_dc_rec};
                    EXPECT_LE(std::fabs(y_dc_rec - want_y[p]),
                              0.5f / (DC_INV_QUANT[1] * dc_scale) + 1e-3f);
                    EXPECT_LE(std::fabs(b_dc_rec - want_b[p]),
                              0.5f / (DC_INV_QUANT[2] * dc_scale) + 1e-3f);
                }
            }
        }
    }

    cudaFree(d_coeffs);
    cudaFree(d_acs);
    cudaFree(d_mx);
    cudaFree(d_mb);
    cudaFree(d_qf);
    cudaFree(d_ac);
    cudaFree(d_dc);
}

}  // namespace
}  // namespace cujpegxl
