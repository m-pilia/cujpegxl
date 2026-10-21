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
// quantize_dc_kernel. A warp spans 32 slots of one 8x8 position, so it adds its
// per-channel non-zero counts to its first-block's `nz_exact` entries (zeroed by
// the caller) with one atomic per channel.
__global__ void quantize_ac_kernel(const __half* __restrict__ coeffs,
                                   const std::int8_t* __restrict__ acs,
                                   const std::int8_t* __restrict__ ytox_map,
                                   const std::int8_t* __restrict__ ytob_map,
                                   const std::int32_t* __restrict__ quant_field, std::uint32_t bw,
                                   std::uint32_t bh, std::uint32_t cmw, float gsf,
                                   std::int16_t* __restrict__ ac,
                                   std::int32_t* __restrict__ nz_exact) {
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
    std::int16_t qx{0};
    std::int16_t qy{0};
    std::int16_t qb{0};
    if (!raw_is_llf(raw, n)) {
        const float qgsf{static_cast<float>(quant_field[t.by * bw + t.bx]) * gsf};
        const std::uint32_t tile{(t.by / 8) * cmw + (t.bx / 8)};
        const float ytox{cfl_ytox_ratio(ytox_map[tile])};
        const float ytob{cfl_ytob_ratio(ytob_map[tile])};

        const float xr{__half2float(coeffs[slot])};
        const float yr{__half2float(coeffs[cplane + slot])};
        const float br{__half2float(coeffs[2 * cplane + slot])};
        const float wy{dequant_weight(n, 1, raw)};
        const int qy_int{static_cast<int>(lrintf(yr * qgsf / wy))};
        const float y_round{static_cast<float>(qy_int) * wy / qgsf};
        qy = static_cast<std::int16_t>(qy_int);
        const float wx{dequant_weight(n, 0, raw)};
        qx = static_cast<std::int16_t>(lrintf((xr - ytox * y_round) * qgsf / wx));
        const float wb{dequant_weight(n, 2, raw)};
        qb = static_cast<std::int16_t>(lrintf((br - ytob * y_round) * qgsf / wb));
    }
    ac[slot] = qx;
    ac[cplane + slot] = qy;
    ac[2 * cplane + slot] = qb;

    const int nz_x{__popc(__ballot_sync(~0u, qx != 0))};
    const int nz_y{__popc(__ballot_sync(~0u, qy != 0))};
    const int nz_b{__popc(__ballot_sync(~0u, qb != 0))};
    const std::uint32_t lane{threadIdx.x % 32};
    if (lane < 3) {
        const int nz{lane == 0 ? nz_x : (lane == 1 ? nz_y : nz_b)};
        if (nz != 0) {
            atomicAdd(&nz_exact[lane * bw * bh + t.by * bw + t.bx], nz);
        }
    }
}

// llf_dc_weights of every covered block of a DCT16 (entries 0-3) and a DCT32
// (entries 4-19), row-major in (out_y, out_x).
constexpr std::uint32_t LLF_DC_WEIGHT_ENTRIES = 20;

__host__ __device__ constexpr std::uint32_t llf_dc_weight_entry(std::uint32_t m,
                                                                std::uint32_t out_x,
                                                                std::uint32_t out_y) {
    return (m == 2 ? 0 : 4) + out_y * m + out_x;
}

__global__ void build_llf_dc_weights_kernel(LlfDcWeights* __restrict__ table) {
    const std::uint32_t entry{threadIdx.x};
    const std::uint32_t m{entry < 4 ? 2u : 4u};
    const std::uint32_t local{entry < 4 ? entry : entry - 4};
    table[entry] = llf_dc_weights(m, local % m, local / m);
}

// Built by a device kernel so that the weights carry the device math library's
// FP64 cos/sqrt, which may round differently from the host's. Lazily
// initialized on the device active at first use (the single target device).
const LlfDcWeights* llf_dc_weights_device() {
    static const LlfDcWeights* const table{[]() -> const LlfDcWeights* {
        LlfDcWeights* t{nullptr};
        if (cudaMalloc(&t, LLF_DC_WEIGHT_ENTRIES * sizeof(LlfDcWeights)) != cudaSuccess) {
            return nullptr;
        }
        build_llf_dc_weights_kernel<<<1, LLF_DC_WEIGHT_ENTRIES>>>(t);
        if (cudaGetLastError() != cudaSuccess) {
            cudaFree(t);
            return nullptr;
        }
        return t;
    }()};
    return table;
}

// 8x8 block positions per side of a quantize_dc_kernel strip. The height and
// the width are multiples of the DCT32 alignment, so a transform never straddles
// two strips.
constexpr std::uint32_t DC_STRIP_WIDTH = 32;
constexpr std::uint32_t DC_STRIP_HEIGHT = 4;

// One CUDA block per strip of 8x8 positions, one thread per position. The m*m
// low-frequency corner of a transform and its m*m covered blocks correspond one
// to one, so each thread first stages one corner coefficient per channel in
// shared memory, at its own position: the corner then lies in the strip's
// shared plane with the strip width as row pitch. Each thread then derives its
// block's DC from the staged corner and quantizes it, with the base correlation
// removed from B. Each thread also broadcasts its transform's normalized
// non-zero counts (from quantize_ac_kernel's `nz_exact`) into `nz_grid`.
__global__ void quantize_dc_kernel(const __half* __restrict__ coeffs,
                                   const std::int8_t* __restrict__ acs,
                                   const LlfDcWeights* __restrict__ weights,
                                   const std::int32_t* __restrict__ nz_exact, std::uint32_t bw,
                                   std::uint32_t bh, float dc_scale, std::int32_t* __restrict__ dc,
                                   std::int32_t* __restrict__ nz_grid) {
    __shared__ float llf[3][DC_STRIP_HEIGHT * DC_STRIP_WIDTH];
    const std::uint32_t nblk{bw * bh};
    const std::uint32_t strip_bx{blockIdx.x * DC_STRIP_WIDTH};
    const std::uint32_t strip_by{blockIdx.y * DC_STRIP_HEIGHT};
    const std::uint32_t bx{strip_bx + threadIdx.x};
    const std::uint32_t by{strip_by + threadIdx.y};
    const bool in_frame{bx < bw && by < bh};
    const std::uint32_t local{threadIdx.y * DC_STRIP_WIDTH + threadIdx.x};

    OwningTransform t{};
    if (in_frame) {
        t = owning_transform(acs, bw, bx, by);
        const std::size_t cplane{static_cast<std::size_t>(nblk) * COEFFS_PER_BLOCK};
        const std::size_t raw{(by - t.by) * static_cast<std::size_t>(t.side) + (bx - t.bx)};
        const std::size_t slot{covered_plane_slot(t.side, t.bx, t.by, bw, raw)};
        llf[0][local] = __half2float(coeffs[slot]);
        llf[1][local] = __half2float(coeffs[cplane + slot]);
        llf[2][local] = __half2float(coeffs[2 * cplane + slot]);
    }
    __syncthreads();
    if (!in_frame) {
        return;
    }

    const std::uint32_t m{static_cast<std::uint32_t>(t.side) / 8};
    float dc_x{llf[0][local]};
    float dc_y{llf[1][local]};
    float dc_b{llf[2][local]};
    if (m > 1) {
        const LlfDcWeights& w{weights[llf_dc_weight_entry(m, bx - t.bx, by - t.by)]};
        const std::uint32_t corner{(t.by - strip_by) * DC_STRIP_WIDTH + (t.bx - strip_bx)};
        dc_x = dc_from_llf_at(m, &llf[0][corner], DC_STRIP_WIDTH, w);
        dc_y = dc_from_llf_at(m, &llf[1][corner], DC_STRIP_WIDTH, w);
        dc_b = dc_from_llf_at(m, &llf[2][corner], DC_STRIP_WIDTH, w);
    }

    const std::uint32_t blk{by * bw + bx};
    const float dcq_x{DC_INV_QUANT[0] * dc_scale};
    const float dcq_y{DC_INV_QUANT[1] * dc_scale};
    const float dcq_b{DC_INV_QUANT[2] * dc_scale};
    const int qdc_y{static_cast<int>(lrintf(dc_y * dcq_y))};
    const float y_dc_round{static_cast<float>(qdc_y) / dcq_y};
    dc[nblk + blk] = qdc_y;
    dc[blk] = static_cast<int>(lrintf(dc_x * dcq_x));
    dc[2 * nblk + blk] = static_cast<int>(lrintf((dc_b - CFL_BASE_B * y_dc_round) * dcq_b));

    const std::uint32_t owner{t.by * bw + t.bx};
    const std::uint32_t covered{m * m};
    const std::uint32_t log2_covered{m == 4 ? 4u : (m == 2 ? 2u : 0u)};
    for (std::uint32_t c{0}; c < 3; ++c) {
        const std::uint32_t nz{static_cast<std::uint32_t>(nz_exact[c * nblk + owner])};
        nz_grid[c * nblk + blk] = static_cast<std::int32_t>((nz + covered - 1) >> log2_covered);
    }
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
                       std::int32_t* dc, std::int32_t* nz_exact, std::int32_t* nz_grid) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t cmw{ceil_div(bw, 8)};
    const QuantCalibration cal{calibrate_quant(distance)};
    const float gsf{cal.global_scale_float};
    const float dc_scale{cal.global_scale_float * static_cast<float>(cal.quant_dc)};
    const LlfDcWeights* dc_weights{llf_dc_weights_device()};
    if (dc_weights == nullptr) {
        return false;
    }
    const std::uint32_t bw32{static_cast<std::uint32_t>(bw)};
    const std::uint32_t bh32{static_cast<std::uint32_t>(bh)};
    if (cudaMemsetAsync(nz_exact, 0, 3 * bw * bh * sizeof(std::int32_t)) != cudaSuccess) {
        return false;
    }

    const unsigned int ac_threads{256};
    const unsigned int ac_blocks{
        static_cast<unsigned int>(ceil_div(bw * bh * COEFFS_PER_BLOCK, ac_threads))};
    quantize_ac_kernel<<<ac_blocks, ac_threads>>>(coeffs, acs, ytox_map, ytob_map, quant_field,
                                                  bw32, bh32, static_cast<std::uint32_t>(cmw), gsf,
                                                  ac, nz_exact);
    const dim3 dc_threads{DC_STRIP_WIDTH, DC_STRIP_HEIGHT};
    const dim3 dc_blocks{static_cast<unsigned int>(ceil_div(bw, DC_STRIP_WIDTH)),
                         static_cast<unsigned int>(ceil_div(bh, DC_STRIP_HEIGHT))};
    quantize_dc_kernel<<<dc_blocks, dc_threads>>>(coeffs, acs, dc_weights, nz_exact, bw32, bh32,
                                                  dc_scale, dc, nz_grid);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace cujpegxl
