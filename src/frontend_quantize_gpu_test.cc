// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frontend_quantize.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <random>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "cfl.h"
#include "coeff_order.h"
#include "coeff_order_reference.h"
#include "quant_calibration.h"
#include "quant_weights_dct16.h"
#include "quant_weights_dct32.h"
#include "quant_weights_dct8.h"
#include "vardct_layout.h"

namespace cujpegxl {
namespace {

constexpr std::size_t WIDTH = 136;
constexpr std::size_t HEIGHT = 120;
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
// estimation has something to find, over a mixed DCT32/DCT16/DCT8 tiling whose
// block grid ends in partial 32x32 regions on both axes.
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
    place(f.acs, 32, 8, 8);
    place(f.acs, 16, 14, 12);

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

constexpr std::size_t CMW = (BW + 7) / 8;
constexpr std::size_t CMH = (BH + 7) / 8;
constexpr std::int32_t QUANT_FIELD = 40;

QuantCalibration test_calibration() {
    return calibrate_quant(initial_quant_dc(1.0f), K_AC_QUANT, 0.0f);
}

struct Quantized {
    std::vector<std::int8_t> ytox_map;
    std::vector<std::int8_t> ytob_map;
    std::vector<std::int16_t> ac;
    std::vector<std::int32_t> dc;
    std::vector<std::int32_t> nz_exact;
    std::vector<std::int32_t> nz_grid;
    CoeffOrderStats order_stats;
};

template <typename T>
T* upload(const std::vector<T>& host) {
    T* device{nullptr};
    cudaMalloc(&device, host.size() * sizeof(T));
    cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    return device;
}

template <typename T>
std::vector<T> download(const T* device, std::size_t count) {
    std::vector<T> host(count);
    cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost);
    return host;
}

// Runs quantize_residual on `f` with a uniform quant field under
// test_calibration().
[[nodiscard]] bool quantize(const Frame& f, Quantized& out) {
    __half* d_coeffs{upload(f.coeffs)};
    std::int8_t* d_acs{upload(f.acs)};
    std::int32_t* d_qf{upload(std::vector<std::int32_t>(NBLK, QUANT_FIELD))};
    QuantCalibration* d_calibration{upload(std::vector<QuantCalibration>{test_calibration()})};
    std::int8_t* d_mx{nullptr};
    std::int8_t* d_mb{nullptr};
    std::int16_t* d_ac{nullptr};
    std::int32_t* d_dc{nullptr};
    std::int32_t* d_nz_exact{nullptr};
    std::int32_t* d_nz_grid{nullptr};
    CoeffOrderStats* d_order_stats{nullptr};
    cudaMalloc(&d_mx, CMW * CMH * sizeof(std::int8_t));
    cudaMalloc(&d_mb, CMW * CMH * sizeof(std::int8_t));
    cudaMalloc(&d_ac, 3 * CPLANE * sizeof(std::int16_t));
    cudaMalloc(&d_dc, 3 * NBLK * sizeof(std::int32_t));
    cudaMalloc(&d_nz_exact, 3 * NBLK * sizeof(std::int32_t));
    cudaMalloc(&d_nz_grid, 3 * NBLK * sizeof(std::int32_t));
    cudaMalloc(&d_order_stats, sizeof(CoeffOrderStats));

    const bool ok{quantize_residual(d_coeffs, d_acs, d_qf, d_calibration, WIDTH, HEIGHT, d_mx, d_mb,
                                    d_ac, d_dc, d_nz_exact, d_nz_grid, d_order_stats, nullptr)};
    out.ytox_map = download(d_mx, CMW * CMH);
    out.ytob_map = download(d_mb, CMW * CMH);
    out.ac = download(d_ac, 3 * CPLANE);
    out.dc = download(d_dc, 3 * NBLK);
    out.nz_exact = download(d_nz_exact, 3 * NBLK);
    out.nz_grid = download(d_nz_grid, 3 * NBLK);
    out.order_stats = download(d_order_stats, 1)[0];

    cudaFree(d_coeffs);
    cudaFree(d_acs);
    cudaFree(d_mx);
    cudaFree(d_mb);
    cudaFree(d_qf);
    cudaFree(d_calibration);
    cudaFree(d_ac);
    cudaFree(d_dc);
    cudaFree(d_nz_exact);
    cudaFree(d_nz_grid);
    cudaFree(d_order_stats);
    return ok && cudaDeviceSynchronize() == cudaSuccess;
}

// The residual+quantize kernel round-trips each AC chroma coefficient to within
// half a quant step (the dead zone threshold for zeroed coefficients) with CfL
// correctly inverted, and reconstructs the LLF DC
// within half a DC step, across the mixed tiling.
TEST(FrontendQuantize, QuantizeResidualReconstructsWithinHalfStep) {
    const Frame f{make_frame()};
    Quantized q{};
    ASSERT_TRUE(quantize(f, q));
    const std::vector<std::int16_t>& ac{q.ac};
    const std::vector<std::int32_t>& dc{q.dc};

    const QuantCalibration cal{test_calibration()};
    const float gsf{cal.global_scale_float};
    const float dc_scale{gsf * static_cast<float>(cal.quant_dc)};
    for (std::size_t by{0}; by < BH; ++by) {
        for (std::size_t bx{0}; bx < BW; ++bx) {
            const int side{f.acs[by * BW + bx]};
            if (side == ACS_COVERED) {
                continue;
            }
            const std::size_t tile{(by / 8) * CMW + bx / 8};
            const float ytox{cfl_ytox_ratio(q.ytox_map[tile])};
            const float ytob{cfl_ytob_ratio(q.ytob_map[tile])};
            const float qgsf{static_cast<float>(QUANT_FIELD) * gsf};
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
                const float chroma_deadzone{
                    ac_deadzone_threshold(side, 0, static_cast<std::uint32_t>(raw))};
                const float x_steps{ac[slot] == 0 ? chroma_deadzone : 0.5f};
                const float b_steps{ac[2 * CPLANE + slot] == 0 ? chroma_deadzone : 0.5f};
                EXPECT_LE(std::fabs(x_rec - xv), x_steps * wx / qgsf + 1e-4f);
                EXPECT_LE(std::fabs(b_rec - bv), b_steps * wb / qgsf + 1e-4f);
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
}

// libjxl's per-quadrant thresholds: Y {0.56, 0.62}, X/B {0.58, 0.62} lowered by
// 0.00744 per covered block from 4 covered blocks, floored at 0.5.
TEST(FrontendQuantize, DeadzoneThresholdsMatchLibjxl) {
    EXPECT_FLOAT_EQ(ac_deadzone_threshold(8, 1, 3 * 8 + 3), 0.56f);
    EXPECT_FLOAT_EQ(ac_deadzone_threshold(8, 1, 3 * 8 + 4), 0.62f);
    EXPECT_FLOAT_EQ(ac_deadzone_threshold(8, 1, 4 * 8 + 3), 0.62f);
    EXPECT_FLOAT_EQ(ac_deadzone_threshold(32, 1, 15 * 32 + 15), 0.56f);
    EXPECT_FLOAT_EQ(ac_deadzone_threshold(32, 1, 31 * 32 + 31), 0.62f);
    EXPECT_FLOAT_EQ(ac_deadzone_threshold(8, 0, 1), 0.58f);
    EXPECT_FLOAT_EQ(ac_deadzone_threshold(8, 2, 7 * 8 + 7), 0.62f);
    EXPECT_FLOAT_EQ(ac_deadzone_threshold(16, 0, 7 * 16 + 7), 0.58f - 4 * 0.00744f);
    EXPECT_FLOAT_EQ(ac_deadzone_threshold(16, 2, 8 * 16 + 7), 0.62f - 4 * 0.00744f);
    EXPECT_FLOAT_EQ(ac_deadzone_threshold(32, 0, 15 * 32 + 15), 0.5f);
    EXPECT_FLOAT_EQ(ac_deadzone_threshold(32, 2, 15 * 32 + 16), 0.62f - 16 * 0.00744f);
}

// Every AC coefficient quantizes to its nearest integer number of quant steps,
// or to 0 when its magnitude is below the dead zone threshold, in all three
// channels and transform sizes (X/B as the CfL residual of the Y roundtrip).
// Coefficients within float tolerance of a decision boundary are skipped.
TEST(FrontendQuantize, QuantizeResidualAppliesDeadzone) {
    const Frame f{make_frame()};
    Quantized q{};
    ASSERT_TRUE(quantize(f, q));

    const float qgsf{static_cast<float>(QUANT_FIELD) * test_calibration().global_scale_float};
    constexpr float TOLERANCE = 1e-3f;
    std::size_t zeroed[3]{};
    for (std::size_t by{0}; by < BH; ++by) {
        for (std::size_t bx{0}; bx < BW; ++bx) {
            const int side{f.acs[by * BW + bx]};
            if (side == ACS_COVERED) {
                continue;
            }
            const std::size_t tile{(by / 8) * CMW + bx / 8};
            const float ratio[3]{cfl_ytox_ratio(q.ytox_map[tile]), 0.0f,
                                 cfl_ytob_ratio(q.ytob_map[tile])};
            for (int raw{0}; raw < side * side; ++raw) {
                if (is_llf(raw, side)) {
                    continue;
                }
                const std::size_t slot{
                    covered_plane_slot(side, bx, by, BW, static_cast<std::size_t>(raw))};
                const float y_round{static_cast<float>(q.ac[CPLANE + slot]) * weight(side, 1, raw) /
                                    qgsf};
                for (std::size_t c{0}; c < 3; ++c) {
                    const float residual{__half2float(f.coeffs[c * CPLANE + slot]) -
                                         ratio[c] * y_round};
                    const float steps{residual * qgsf / weight(side, static_cast<int>(c), raw)};
                    const float threshold{ac_deadzone_threshold(static_cast<std::uint32_t>(side),
                                                                static_cast<std::uint32_t>(c),
                                                                static_cast<std::uint32_t>(raw))};
                    const float magnitude{std::fabs(steps)};
                    if (std::fabs(magnitude - threshold) < TOLERANCE ||
                        std::fabs(magnitude - std::floor(magnitude) - 0.5f) < TOLERANCE) {
                        continue;
                    }
                    const bool in_deadzone{magnitude < threshold};
                    zeroed[c] += in_deadzone && magnitude > 0.5f ? 1 : 0;
                    EXPECT_EQ(q.ac[c * CPLANE + slot], in_deadzone ? 0 : std::lrint(steps))
                        << "channel " << c << " side " << side << " raw " << raw;
                }
            }
        }
    }
    for (std::size_t c{0}; c < 3; ++c) {
        EXPECT_GT(zeroed[c], 0u) << "channel " << c;
    }
}

// The per-tile CfL maps match a double-precision host regression over each
// tile's first-block AC coefficients, up to the rounding of the FP32 device sums,
// including the partial tiles at the right and bottom edges.
TEST(FrontendQuantize, QuantizeResidualEstimatesCflMaps) {
    const Frame f{make_frame()};
    Quantized q{};
    ASSERT_TRUE(quantize(f, q));

    std::vector<double> sxy(CMW * CMH, 0.0);
    std::vector<double> syy(CMW * CMH, 0.0);
    std::vector<double> sby(CMW * CMH, 0.0);
    for (std::size_t by{0}; by < BH; ++by) {
        for (std::size_t bx{0}; bx < BW; ++bx) {
            const int side{f.acs[by * BW + bx]};
            if (side == ACS_COVERED) {
                continue;
            }
            const std::size_t tile{(by / 8) * CMW + bx / 8};
            for (int raw{0}; raw < side * side; ++raw) {
                if (is_llf(raw, side)) {
                    continue;
                }
                const std::size_t slot{
                    covered_plane_slot(side, bx, by, BW, static_cast<std::size_t>(raw))};
                const double xv{__half2float(f.coeffs[slot])};
                const double yv{__half2float(f.coeffs[CPLANE + slot])};
                const double bv{__half2float(f.coeffs[2 * CPLANE + slot])};
                sxy[tile] += xv * yv;
                syy[tile] += yv * yv;
                sby[tile] += (bv - CFL_BASE_B * yv) * yv;
            }
        }
    }
    for (std::size_t tile{0}; tile < CMW * CMH; ++tile) {
        int want_x{0};
        int want_b{0};
        cfl_maps_from_sums(sxy[tile], syy[tile], sby[tile], &want_x, &want_b);
        EXPECT_NEAR(q.ytox_map[tile], want_x, 1) << "tile " << tile;
        EXPECT_NEAR(q.ytob_map[tile], want_b, 1) << "tile " << tile;
    }
}

// Each first-block's non-zero count covers its quantized AC (the zeroed LLF
// corner included), and the grid holds the per-covered-block normalized count
// at every position the transform covers.
TEST(FrontendQuantize, QuantizeResidualCountsNonzeros) {
    const Frame f{make_frame()};
    Quantized q{};
    ASSERT_TRUE(quantize(f, q));

    std::size_t nonzero_blocks{0};
    for (std::size_t by{0}; by < BH; ++by) {
        for (std::size_t bx{0}; bx < BW; ++bx) {
            const int side{f.acs[by * BW + bx]};
            if (side == ACS_COVERED) {
                continue;
            }
            const int cx{side / 8};
            const std::int32_t covered{cx * cx};
            for (std::size_t c{0}; c < 3; ++c) {
                std::int32_t count{0};
                for (int raw{0}; raw < side * side; ++raw) {
                    const std::size_t slot{
                        covered_plane_slot(side, bx, by, BW, static_cast<std::size_t>(raw))};
                    count += q.ac[c * CPLANE + slot] != 0 ? 1 : 0;
                }
                nonzero_blocks += count != 0 ? 1 : 0;
                EXPECT_EQ(q.nz_exact[c * NBLK + by * BW + bx], count)
                    << "channel " << c << " block " << bx << "," << by;
                for (int p{0}; p < covered; ++p) {
                    const std::size_t at{(by + p / cx) * BW + (bx + p % cx)};
                    EXPECT_EQ(q.nz_grid[c * NBLK + at], (count + covered - 1) / covered)
                        << "channel " << c << " position " << at;
                }
            }
        }
    }
    EXPECT_GT(nonzero_blocks, 0u);
}

// The order statistics count every first-block per side and every non-zero
// quantized AC coefficient per side, channel and position.
TEST(FrontendQuantize, QuantizeResidualCountsOrderStatistics) {
    const Frame f{make_frame()};
    Quantized q{};
    ASSERT_TRUE(quantize(f, q));
    const std::vector<std::int32_t> ac(q.ac.begin(), q.ac.end());
    const CoeffOrderStats expected{coeff_order_stats_reference(
        {ac.data(), ac.data() + CPLANE, ac.data() + 2 * CPLANE}, f.acs.data(), BW, BH)};
    EXPECT_EQ(std::vector<std::uint32_t>(std::begin(q.order_stats.blocks),
                                         std::end(q.order_stats.blocks)),
              std::vector<std::uint32_t>(std::begin(expected.blocks), std::end(expected.blocks)));
    EXPECT_EQ(
        std::vector<std::uint32_t>(std::begin(q.order_stats.nonzeros),
                                   std::end(q.order_stats.nonzeros)),
        std::vector<std::uint32_t>(std::begin(expected.nonzeros), std::end(expected.nonzeros)));
    for (std::uint32_t blocks : expected.blocks) {
        EXPECT_GT(blocks, 0u);
    }
}

}  // namespace
}  // namespace cujpegxl
