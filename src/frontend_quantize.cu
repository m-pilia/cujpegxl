// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frontend_quantize.h"

#include <cmath>

#include <cuda_runtime.h>

#include "cfl.h"
#include "quant_calibration.h"
#include "quant_weights_dct16.h"
#include "quant_weights_dct32.h"
#include "quant_weights_dct8.h"
#include "vardct_layout.h"

namespace cujpegxl {
namespace {

std::size_t ceil_div(std::size_t a, std::size_t b) {
    return (a + b - 1) / b;
}

// A coefficient raw index (fx*N+fy) belongs to the low-frequency corner (carried
// as DC, excluded from AC) when both frequencies are below the covered side.
__host__ __device__ inline bool raw_is_llf(int raw, int side) {
    const int cx{side / 8};
    return (raw / side) < cx && (raw % side) < cx;
}

__device__ inline float dequant_weight(int side, int channel, int raw) {
    if (side == 16) {
        return DCT16_DEQUANT_WEIGHTS[channel][raw];
    }
    if (side == 32) {
        return DCT32_DEQUANT_WEIGHTS[channel][raw];
    }
    return DCT8_DEQUANT_WEIGHTS[channel][raw];
}

// One thread per 64x64 color tile accumulates its first-blocks' AC-coefficient
// regression sums and quantizes them to the color map. Shared by device and host.
__host__ __device__ inline void estimate_tile(const __half* coeffs, const std::int8_t* acs,
                                              std::size_t bw, std::size_t bh, std::size_t cmw,
                                              std::size_t tile, std::int8_t* ytox_map,
                                              std::int8_t* ytob_map) {
    const std::size_t ctx{tile % cmw};
    const std::size_t cty{tile / cmw};
    const std::size_t cplane{bw * bh * COEFFS_PER_BLOCK};
    double sxy{0.0};
    double syy{0.0};
    double sby{0.0};
    const std::size_t by_end{cty * 8 + 8 < bh ? cty * 8 + 8 : bh};
    const std::size_t bx_end{ctx * 8 + 8 < bw ? ctx * 8 + 8 : bw};
    for (std::size_t by{cty * 8}; by < by_end; ++by) {
        for (std::size_t bx{ctx * 8}; bx < bx_end; ++bx) {
            const int side{acs == nullptr ? 8 : acs[by * bw + bx]};
            if (side == ACS_COVERED) {
                continue;
            }
            for (int raw{0}; raw < side * side; ++raw) {
                if (raw_is_llf(raw, side)) {
                    continue;
                }
                const std::size_t slot{
                    covered_plane_slot(side, bx, by, bw, static_cast<std::size_t>(raw))};
                const double xr{__half2float(coeffs[slot])};
                const double yr{__half2float(coeffs[cplane + slot])};
                const double br{__half2float(coeffs[2 * cplane + slot])};
                sxy += xr * yr;
                syy += yr * yr;
                sby += (br - CFL_BASE_B * yr) * yr;
            }
        }
    }
    int mx{0};
    int mb{0};
    cfl_maps_from_sums(sxy, syy, sby, &mx, &mb);
    ytox_map[tile] = static_cast<std::int8_t>(mx);
    ytob_map[tile] = static_cast<std::int8_t>(mb);
}

__global__ void estimate_cfl_covered_kernel(const __half* __restrict__ coeffs,
                                            const std::int8_t* __restrict__ acs, std::size_t bw,
                                            std::size_t bh, std::size_t cmw, std::size_t cmh,
                                            std::int8_t* __restrict__ ytox_map,
                                            std::int8_t* __restrict__ ytob_map) {
    const std::size_t tile{static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x};
    if (tile >= cmw * cmh) {
        return;
    }
    estimate_tile(coeffs, acs, bw, bh, cmw, tile, ytox_map, ytob_map);
}

// One thread per first-block. Gathers the block's raw coefficients, quantizes
// the AC with per-tile CfL residuals (Y roundtrip drives X/B), and derives the DC
// from the low frequencies with the base correlation.
__global__ void quantize_residual_kernel(
    const __half* __restrict__ coeffs, const std::int8_t* __restrict__ acs,
    const std::int8_t* __restrict__ ytox_map, const std::int8_t* __restrict__ ytob_map,
    const std::int32_t* __restrict__ quant_field, std::size_t bw, std::size_t bh, std::size_t cmw,
    float gsf, float dc_scale, std::int16_t* __restrict__ ac, std::int32_t* __restrict__ dc) {
    const std::size_t blk{static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x};
    if (blk >= bw * bh) {
        return;
    }
    const int side{acs == nullptr ? 8 : acs[blk]};
    if (side == ACS_COVERED) {
        return;
    }
    const std::size_t bx{blk % bw};
    const std::size_t by{blk / bw};
    const int n{side};
    const int cx{n / 8};
    const std::size_t cplane{bw * bh * COEFFS_PER_BLOCK};
    const std::size_t nblk{bw * bh};
    const float qgsf{static_cast<float>(quant_field[blk]) * gsf};

    const std::size_t tile{(by / 8) * cmw + (bx / 8)};
    const float ytox{cfl_ytox_ratio(ytox_map == nullptr ? 0 : ytox_map[tile])};
    const float ytob{cfl_ytob_ratio(ytob_map == nullptr ? 0 : ytob_map[tile])};

    float xs[32 * 32];
    float ys[32 * 32];
    float bs[32 * 32];
    float y_round[32 * 32];
    for (int raw{0}; raw < n * n; ++raw) {
        const std::size_t slot{covered_plane_slot(n, bx, by, bw, static_cast<std::size_t>(raw))};
        xs[raw] = __half2float(coeffs[slot]);
        ys[raw] = __half2float(coeffs[cplane + slot]);
        bs[raw] = __half2float(coeffs[2 * cplane + slot]);
    }

    for (int raw{0}; raw < n * n; ++raw) {
        const std::size_t slot{covered_plane_slot(n, bx, by, bw, static_cast<std::size_t>(raw))};
        if (raw_is_llf(raw, n)) {
            y_round[raw] = 0.0f;
            ac[slot] = 0;
            ac[cplane + slot] = 0;
            ac[2 * cplane + slot] = 0;
            continue;
        }
        const float wy{dequant_weight(n, 1, raw)};
        const int qy{static_cast<int>(lrintf(ys[raw] * qgsf / wy))};
        y_round[raw] = static_cast<float>(qy) * wy / qgsf;
        ac[cplane + slot] = static_cast<std::int16_t>(qy);

        const float wx{dequant_weight(n, 0, raw)};
        ac[slot] = static_cast<std::int16_t>(lrintf((xs[raw] - ytox * y_round[raw]) * qgsf / wx));
        const float wb{dequant_weight(n, 2, raw)};
        ac[2 * cplane + slot] =
            static_cast<std::int16_t>(lrintf((bs[raw] - ytob * y_round[raw]) * qgsf / wb));
    }

    float dc_x[16];
    float dc_y[16];
    float dc_b[16];
    dc_from_llf(n, xs, dc_x);
    dc_from_llf(n, ys, dc_y);
    dc_from_llf(n, bs, dc_b);
    const float dcq_x{DC_INV_QUANT[0] * dc_scale};
    const float dcq_y{DC_INV_QUANT[1] * dc_scale};
    const float dcq_b{DC_INV_QUANT[2] * dc_scale};
    for (int oy{0}; oy < cx; ++oy) {
        for (int ox{0}; ox < cx; ++ox) {
            const int p{oy * cx + ox};
            const std::size_t dblk{(by + oy) * bw + (bx + ox)};
            const int qdc_y{static_cast<int>(lrintf(dc_y[p] * dcq_y))};
            const float y_dc_round{static_cast<float>(qdc_y) / dcq_y};
            dc[nblk + dblk] = qdc_y;
            dc[dblk] = static_cast<int>(lrintf(dc_x[p] * dcq_x));
            dc[2 * nblk + dblk] =
                static_cast<int>(lrintf((dc_b[p] - CFL_BASE_B * y_dc_round) * dcq_b));
        }
    }
}

}  // namespace

bool estimate_cfl_covered(const __half* coeffs, const std::int8_t* acs, std::size_t width,
                          std::size_t height, std::int8_t* ytox_map, std::int8_t* ytob_map) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t cmw{ceil_div(bw, 8)};
    const std::size_t cmh{ceil_div(bh, 8)};
    const unsigned int threads{128};
    const unsigned int blocks{static_cast<unsigned int>((cmw * cmh + threads - 1) / threads)};
    estimate_cfl_covered_kernel<<<blocks, threads>>>(coeffs, acs, bw, bh, cmw, cmh, ytox_map,
                                                     ytob_map);
    const cudaError_t launch{cudaGetLastError()};
    const cudaError_t sync{cudaDeviceSynchronize()};
    return launch == cudaSuccess && sync == cudaSuccess;
}

void estimate_cfl_covered_host(const __half* coeffs, const std::int8_t* acs, std::size_t width,
                               std::size_t height, std::int8_t* ytox_map, std::int8_t* ytob_map) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t cmw{ceil_div(bw, 8)};
    const std::size_t cmh{ceil_div(bh, 8)};
    for (std::size_t tile{0}; tile < cmw * cmh; ++tile) {
        estimate_tile(coeffs, acs, bw, bh, cmw, tile, ytox_map, ytob_map);
    }
}

bool quantize_residual(const __half* coeffs, const std::int8_t* acs, const std::int8_t* ytox_map,
                       const std::int8_t* ytob_map, const std::int32_t* quant_field,
                       std::size_t width, std::size_t height, float distance, std::int16_t* ac,
                       std::int32_t* dc) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t cmw{ceil_div(bw, 8)};
    const QuantCalibration cal{calibrate_quant(distance)};
    const float gsf{cal.global_scale_float};
    const float dc_scale{cal.global_scale_float * static_cast<float>(cal.quant_dc)};
    const unsigned int threads{128};
    const unsigned int blocks{static_cast<unsigned int>((bw * bh + threads - 1) / threads)};
    quantize_residual_kernel<<<blocks, threads>>>(coeffs, acs, ytox_map, ytob_map, quant_field, bw,
                                                  bh, cmw, gsf, dc_scale, ac, dc);
    const cudaError_t launch{cudaGetLastError()};
    const cudaError_t sync{cudaDeviceSynchronize()};
    return launch == cudaSuccess && sync == cudaSuccess;
}

}  // namespace cujpegxl
