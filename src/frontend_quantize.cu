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

struct OwningTransform {
    std::uint32_t bx;
    std::uint32_t by;
    int side;
};

// Transform covering the 8x8 block (bx, by). Relies on the front end's region
// layout: a DCT32 first-block sits on a 4-block-aligned position and a DCT16 on
// a 2-block-aligned one, so only those two candidates need to be inspected.
__device__ inline OwningTransform owning_transform(const std::int8_t* acs, std::uint32_t bw,
                                                   std::uint32_t bx, std::uint32_t by) {
    const std::uint32_t bx32{bx & ~3u};
    const std::uint32_t by32{by & ~3u};
    if (acs[by32 * bw + bx32] == 32) {
        return {bx32, by32, 32};
    }
    const std::uint32_t bx16{bx & ~1u};
    const std::uint32_t by16{by & ~1u};
    if (acs[by16 * bw + bx16] == 16) {
        return {bx16, by16, 16};
    }
    return {bx, by, 8};
}

// Raw index (fx*N+fy) in the owning transform `t` of coefficient `slot` of the
// 8x8 block (bx, by).
__device__ inline int owner_raw_index(const OwningTransform& t, std::uint32_t bx, std::uint32_t by,
                                      std::uint32_t slot) {
    const int covered_ordinal{
        static_cast<int>((by - t.by) * static_cast<std::uint32_t>(t.side / 8) + (bx - t.bx))};
    return covered_ordinal * static_cast<int>(COEFFS_PER_BLOCK) + static_cast<int>(slot);
}

// 8x8 block positions per side of a 64x64 color tile.
constexpr std::uint32_t CFL_TILE_SIDE = 8;

// One CUDA block per 64x64 color tile, one thread per coefficient slot of an 8x8
// block. The block walks the tile's 8x8 positions, so each plane load is one
// coalesced 64-slot run and every thread does the same work whatever the
// transform sizes. A DCT16/DCT32 never straddles a color tile (its alignment
// divides the tile side), so the positions of a tile hold exactly the
// coefficients of the first-blocks whose top-left tile it is. The per-slot FP32
// partials are tree-reduced and thread 0 quantizes the tile's color map.
__global__ void estimate_cfl_covered_kernel(const __half* __restrict__ coeffs,
                                            const std::int8_t* __restrict__ acs, std::uint32_t bw,
                                            std::uint32_t bh, std::uint32_t cmw,
                                            std::int8_t* __restrict__ ytox_map,
                                            std::int8_t* __restrict__ ytob_map) {
    const std::uint32_t tile{blockIdx.x};
    const std::uint32_t slot{threadIdx.x};
    const std::uint32_t cplane{bw * bh * static_cast<std::uint32_t>(COEFFS_PER_BLOCK)};
    const std::uint32_t tile_bx{(tile % cmw) * CFL_TILE_SIDE};
    const std::uint32_t tile_by{(tile / cmw) * CFL_TILE_SIDE};

    float sxy{0.0f};
    float syy{0.0f};
    float sby{0.0f};
    for (std::uint32_t by{tile_by}; by < min(tile_by + CFL_TILE_SIDE, bh); ++by) {
        for (std::uint32_t bx{tile_bx}; bx < min(tile_bx + CFL_TILE_SIDE, bw); ++bx) {
            const OwningTransform t{owning_transform(acs, bw, bx, by)};
            if (raw_is_llf(owner_raw_index(t, bx, by, slot), t.side)) {
                continue;
            }
            const std::uint32_t at{(by * bw + bx) * static_cast<std::uint32_t>(COEFFS_PER_BLOCK) +
                                   slot};
            const float xr{__half2float(coeffs[at])};
            const float yr{__half2float(coeffs[cplane + at])};
            const float br{__half2float(coeffs[2 * cplane + at])};
            sxy += xr * yr;
            syy += yr * yr;
            sby += (br - CFL_BASE_B * yr) * yr;
        }
    }

    __shared__ float rxy[COEFFS_PER_BLOCK];
    __shared__ float ryy[COEFFS_PER_BLOCK];
    __shared__ float rby[COEFFS_PER_BLOCK];
    rxy[slot] = sxy;
    ryy[slot] = syy;
    rby[slot] = sby;
    __syncthreads();
    for (std::uint32_t s{COEFFS_PER_BLOCK / 2}; s > 0; s >>= 1) {
        if (slot < s) {
            rxy[slot] += rxy[slot + s];
            ryy[slot] += ryy[slot + s];
            rby[slot] += rby[slot + s];
        }
        __syncthreads();
    }
    if (slot == 0) {
        int mx{0};
        int mb{0};
        cfl_maps_from_sums(rxy[0], ryy[0], rby[0], &mx, &mb);
        ytox_map[tile] = static_cast<std::int8_t>(mx);
        ytob_map[tile] = static_cast<std::int8_t>(mb);
    }
}

// One thread per covered-block coefficient slot: the slot index is the thread
// index, so loads and stores are coalesced and a DCT32 spreads over 1024 threads.
// Each coefficient's Y quantization and X/B CfL residuals are independent of all
// other coefficients. LLF-corner slots are zeroed; their DC is derived by
// quantize_dc_kernel.
__global__ void quantize_ac_kernel(const __half* __restrict__ coeffs,
                                   const std::int8_t* __restrict__ acs,
                                   const std::int8_t* __restrict__ ytox_map,
                                   const std::int8_t* __restrict__ ytob_map,
                                   const std::int32_t* __restrict__ quant_field, std::uint32_t bw,
                                   std::uint32_t bh, std::uint32_t cmw, float gsf,
                                   std::int16_t* __restrict__ ac) {
    const std::uint32_t cplane{bw * bh * static_cast<std::uint32_t>(COEFFS_PER_BLOCK)};
    const std::uint32_t slot{blockIdx.x * blockDim.x + threadIdx.x};
    if (slot >= cplane) {
        return;
    }
    const std::uint32_t blk{slot / static_cast<std::uint32_t>(COEFFS_PER_BLOCK)};
    const std::uint32_t bx{blk % bw};
    const std::uint32_t by{blk / bw};
    const OwningTransform t{owning_transform(acs, bw, bx, by)};
    const int n{t.side};
    const int raw{owner_raw_index(t, bx, by, slot % static_cast<std::uint32_t>(COEFFS_PER_BLOCK))};
    if (raw_is_llf(raw, n)) {
        ac[slot] = 0;
        ac[cplane + slot] = 0;
        ac[2 * cplane + slot] = 0;
        return;
    }
    const float qgsf{static_cast<float>(quant_field[t.by * bw + t.bx]) * gsf};
    const std::uint32_t tile{(t.by / 8) * cmw + (t.bx / 8)};
    const float ytox{cfl_ytox_ratio(ytox_map[tile])};
    const float ytob{cfl_ytob_ratio(ytob_map[tile])};

    const float xr{__half2float(coeffs[slot])};
    const float yr{__half2float(coeffs[cplane + slot])};
    const float br{__half2float(coeffs[2 * cplane + slot])};
    const float wy{dequant_weight(n, 1, raw)};
    const int qy{static_cast<int>(lrintf(yr * qgsf / wy))};
    const float y_round{static_cast<float>(qy) * wy / qgsf};
    ac[cplane + slot] = static_cast<std::int16_t>(qy);

    const float wx{dequant_weight(n, 0, raw)};
    ac[slot] = static_cast<std::int16_t>(lrintf((xr - ytox * y_round) * qgsf / wx));
    const float wb{dequant_weight(n, 2, raw)};
    ac[2 * cplane + slot] = static_cast<std::int16_t>(lrintf((br - ytob * y_round) * qgsf / wb));
}

// One thread per 8x8 block: derives that block's DC from its transform's
// compacted cx*cx low-frequency corner and quantizes it, with the base
// correlation removed from B.
__global__ void quantize_dc_kernel(const __half* __restrict__ coeffs,
                                   const std::int8_t* __restrict__ acs, std::uint32_t bw,
                                   std::uint32_t bh, float dc_scale,
                                   std::int32_t* __restrict__ dc) {
    const std::uint32_t nblk{bw * bh};
    const std::uint32_t blk{blockIdx.x * blockDim.x + threadIdx.x};
    if (blk >= nblk) {
        return;
    }
    const std::uint32_t bx{blk % bw};
    const std::uint32_t by{blk / bw};
    const OwningTransform t{owning_transform(acs, bw, bx, by)};
    const int n{t.side};
    const int cx{n / 8};
    const std::size_t cplane{static_cast<std::size_t>(nblk) * COEFFS_PER_BLOCK};

    float llf_x[16];
    float llf_y[16];
    float llf_b[16];
    for (int fx{0}; fx < cx; ++fx) {
        for (int fy{0}; fy < cx; ++fy) {
            const std::size_t slot{
                covered_plane_slot(n, t.bx, t.by, bw, static_cast<std::size_t>(fx * n + fy))};
            const int lp{fx * cx + fy};
            llf_x[lp] = __half2float(coeffs[slot]);
            llf_y[lp] = __half2float(coeffs[cplane + slot]);
            llf_b[lp] = __half2float(coeffs[2 * cplane + slot]);
        }
    }
    const std::size_t out_x{bx - t.bx};
    const std::size_t out_y{by - t.by};
    const std::size_t m{static_cast<std::size_t>(cx)};
    const LlfDcWeights w{llf_dc_weights(m, out_x, out_y)};
    const float dc_x{dc_from_llf_at(m, llf_x, m, w)};
    const float dc_y{dc_from_llf_at(m, llf_y, m, w)};
    const float dc_b{dc_from_llf_at(m, llf_b, m, w)};

    const float dcq_x{DC_INV_QUANT[0] * dc_scale};
    const float dcq_y{DC_INV_QUANT[1] * dc_scale};
    const float dcq_b{DC_INV_QUANT[2] * dc_scale};
    const int qdc_y{static_cast<int>(lrintf(dc_y * dcq_y))};
    const float y_dc_round{static_cast<float>(qdc_y) / dcq_y};
    dc[nblk + blk] = qdc_y;
    dc[blk] = static_cast<int>(lrintf(dc_x * dcq_x));
    dc[2 * nblk + blk] = static_cast<int>(lrintf((dc_b - CFL_BASE_B * y_dc_round) * dcq_b));
}

}  // namespace

bool estimate_cfl_covered(const __half* coeffs, const std::int8_t* acs, std::size_t width,
                          std::size_t height, std::int8_t* ytox_map, std::int8_t* ytob_map) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t cmw{ceil_div(bw, 8)};
    const std::size_t cmh{ceil_div(bh, 8)};
    const unsigned int threads{static_cast<unsigned int>(COEFFS_PER_BLOCK)};
    const unsigned int blocks{static_cast<unsigned int>(cmw * cmh)};
    estimate_cfl_covered_kernel<<<blocks, threads>>>(
        coeffs, acs, static_cast<std::uint32_t>(bw), static_cast<std::uint32_t>(bh),
        static_cast<std::uint32_t>(cmw), ytox_map, ytob_map);
    return cudaGetLastError() == cudaSuccess;
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
    const std::uint32_t bw32{static_cast<std::uint32_t>(bw)};
    const std::uint32_t bh32{static_cast<std::uint32_t>(bh)};

    const unsigned int ac_threads{256};
    const unsigned int ac_blocks{
        static_cast<unsigned int>(ceil_div(bw * bh * COEFFS_PER_BLOCK, ac_threads))};
    quantize_ac_kernel<<<ac_blocks, ac_threads>>>(coeffs, acs, ytox_map, ytob_map, quant_field,
                                                  bw32, bh32, static_cast<std::uint32_t>(cmw), gsf,
                                                  ac);
    const unsigned int dc_threads{128};
    const unsigned int dc_blocks{static_cast<unsigned int>(ceil_div(bw * bh, dc_threads))};
    quantize_dc_kernel<<<dc_blocks, dc_threads>>>(coeffs, acs, bw32, bh32, dc_scale, dc);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace cujpegxl
