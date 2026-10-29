// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frontend_quantize.h"

#include <cmath>
#include <cstddef>

#include <cuda_runtime.h>

#include "cfl.h"
#include "coeff_order.h"
#include "dequant_weight_pairs.h"
#include "exact_div.cuh"
#include "quant_calibration.h"
#include "vardct_layout.h"

namespace cujpegxl {
namespace {

std::size_t ceil_div(std::size_t a, std::size_t b) {
    return (a + b - 1) / b;
}

// In global memory: the lanes of a warp read different weights, which
// constant memory would serialize.
__device__ const DequantWeightPairs<64> DCT8_PAIRS{DCT8_DEQUANT_PAIRS};
__device__ const DequantWeightPairs<256> DCT16_PAIRS{DCT16_DEQUANT_PAIRS};
__device__ const DequantWeightPairs<1024> DCT32_PAIRS{DCT32_DEQUANT_PAIRS};

__device__ inline DequantWeightPair dequant_weight(int side, int channel, int raw) {
    const DequantWeightPair* pair{side == 16   ? &DCT16_PAIRS.w[channel][raw]
                                  : side == 32 ? &DCT32_PAIRS.w[channel][raw]
                                               : &DCT8_PAIRS.w[channel][raw]};
    const float2 w{__ldg(reinterpret_cast<const float2*>(pair))};
    return {w.x, w.y};
}

__device__ inline int quantize_deadzone(float steps, float threshold) {
    return fabsf(steps) < threshold ? 0 : static_cast<int>(lrintf(steps));
}

// LlfDcTerms of every covered block of a DCT16 (entries 0-3) and a DCT32
// (entries 4-19), row-major in (out_y, out_x).
constexpr std::uint32_t LLF_DC_TERM_ENTRIES = 20;

__host__ __device__ constexpr std::uint32_t llf_dc_term_entry(std::uint32_t m, std::uint32_t out_x,
                                                              std::uint32_t out_y) {
    return (m == 2 ? 0 : 4) + out_y * m + out_x;
}

// The FP64 weights of dc_from_llf_at combined into one FP32 weight per corner
// coefficient, w[fx * m + fy] = m * x[fx] * y[fy] * scale[fx] * scale[fy], so
// that a DC is m*m FP32 multiply-adds.
struct LlfDcTerms {
    float w[16];
};

__global__ void build_llf_dc_terms_kernel(LlfDcTerms* __restrict__ table) {
    const std::uint32_t entry{threadIdx.x};
    const std::uint32_t m{entry < 4 ? 2u : 4u};
    const std::uint32_t local{entry < 4 ? entry : entry - 4};
    const LlfDcWeights w{llf_dc_weights(m, local % m, local / m)};
    for (std::uint32_t fx{0}; fx < m; ++fx) {
        for (std::uint32_t fy{0}; fy < m; ++fy) {
            table[entry].w[fx * m + fy] = static_cast<float>(static_cast<double>(m) * w.x[fx] *
                                                             w.y[fy] * w.scale[fx] * w.scale[fy]);
        }
    }
}

// Built by a device kernel so that the weights carry the device math library's
// FP64 cos/sqrt, which may round differently from the host's. Lazily
// initialized on the device active at first use (the single target device). The
// kernel runs on the legacy default stream; the wait orders it before the
// kernels of the non-blocking streams.
const LlfDcTerms* llf_dc_terms_device() {
    static const LlfDcTerms* const table{[]() -> const LlfDcTerms* {
        LlfDcTerms* t{nullptr};
        if (cudaMalloc(&t, LLF_DC_TERM_ENTRIES * sizeof(LlfDcTerms)) != cudaSuccess) {
            return nullptr;
        }
        build_llf_dc_terms_kernel<<<1, LLF_DC_TERM_ENTRIES>>>(t);
        if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(nullptr) != cudaSuccess) {
            cudaFree(t);
            return nullptr;
        }
        return t;
    }()};
    return table;
}

// 8x8 block positions per side of a 64x64 color tile.
constexpr std::uint32_t TILE_SIDE = 8;
constexpr std::uint32_t BLOCK_SLOTS = COEFFS_PER_BLOCK;
constexpr std::uint32_t TILE_POSITIONS = TILE_SIDE * TILE_SIDE;
constexpr std::uint32_t TILE_SLOTS = TILE_POSITIONS * BLOCK_SLOTS;
constexpr std::uint32_t HALVES_PER_VECTOR = sizeof(uint4) / sizeof(__half);
constexpr std::uint32_t ROW_VECTORS = TILE_SIDE * BLOCK_SLOTS / HALVES_PER_VECTOR;
constexpr std::uint32_t TILE_VECTORS = 3 * TILE_SIDE * ROW_VECTORS;
constexpr std::uint32_t QUANT_THREADS = 512;
// Bounded by the shared-memory tile; sets the register budget.
constexpr std::uint32_t QUANT_BLOCKS_PER_SM = 3;
constexpr std::uint32_t QUANT_WARPS = QUANT_THREADS / 32;
constexpr std::uint32_t QUANT_STEPS = TILE_POSITIONS / QUANT_WARPS;
constexpr std::uint32_t FETCH_VECTORS = TILE_VECTORS / QUANT_THREADS;
static_assert(BLOCK_SLOTS == 64 && TILE_POSITIONS == 64);
static_assert(TILE_POSITIONS % QUANT_WARPS == 0 && TILE_VECTORS % QUANT_THREADS == 0);
// One CfL partial-sum thread per slot and tile row.
static_assert(QUANT_THREADS == TILE_SIDE * BLOCK_SLOTS);

// A thread's quantized coefficients are two slots of QUANT_STEPS positions per
// channel; bit nonzero_bit(c, k, j) of its mask flags slot j of step k non-zero.
__device__ inline std::uint32_t nonzero_bit(std::uint32_t c, std::uint32_t k, std::uint32_t j) {
    return std::uint32_t{1} << ((c * QUANT_STEPS + k) * 2 + j);
}

constexpr std::uint32_t ORDER_STATS_WORDS = sizeof(CoeffOrderStats) / sizeof(std::uint32_t);
constexpr std::uint32_t ORDER_STATS_VECTORS = (ORDER_STATS_WORDS + 3) / 4;
static_assert(3 * QUANT_STEPS * 2 <= 32);
static_assert(ORDER_STATS_VECTORS * sizeof(uint4) <= 3 * TILE_SLOTS * sizeof(__half),
              "the tile's order statistics reuse its coefficient buffer");

// A thread's share of a tile's coefficients, loaded into registers in one batch
// so that all global loads are in flight before the shared-memory store.
// Threads below TILE_POSITIONS also carry the signals of their position.
struct TileFetch {
    uint4 coeffs[FETCH_VECTORS];
    std::int32_t acs;
    std::int32_t quant_field;
};

__device__ inline TileFetch fetch_tile(const __half* __restrict__ coeffs,
                                       const std::int8_t* __restrict__ acs,
                                       const std::int32_t* __restrict__ quant_field,
                                       std::uint32_t bw, std::uint32_t bh, std::uint32_t cplane,
                                       std::uint32_t tile_bx, std::uint32_t tile_by) {
    TileFetch f{};
    const std::uint32_t cols{min(TILE_SIDE, bw - tile_bx)};
    const std::uint32_t rows{min(TILE_SIDE, bh - tile_by)};
#pragma unroll
    for (std::uint32_t i{0}; i < FETCH_VECTORS; ++i) {
        const std::uint32_t v{i * QUANT_THREADS + threadIdx.x};
        const std::uint32_t c{v / (TILE_SIDE * ROW_VECTORS)};
        const std::uint32_t row{(v / ROW_VECTORS) % TILE_SIDE};
        const std::uint32_t in_row{v % ROW_VECTORS};
        if (row < rows && in_row * HALVES_PER_VECTOR / BLOCK_SLOTS < cols) {
            const std::uint32_t src{c * cplane + ((tile_by + row) * bw + tile_bx) * BLOCK_SLOTS +
                                    in_row * HALVES_PER_VECTOR};
            f.coeffs[i] = *reinterpret_cast<const uint4*>(&coeffs[src]);
        }
    }
    const std::uint32_t lx{threadIdx.x % TILE_SIDE};
    const std::uint32_t ly{threadIdx.x / TILE_SIDE};
    if (threadIdx.x < TILE_POSITIONS && lx < cols && ly < rows) {
        const std::uint32_t blk{(tile_by + ly) * bw + tile_bx + lx};
        f.acs = acs[blk];
        f.quant_field = quant_field[blk];
    }
    return f;
}

// Slots of covered block `covered_ordinal` of a transform of side `side` that
// lie in its low-frequency corner (raw index fx*N+fy with fx, fy < N/8),
// carried as DC and excluded from AC.
__device__ inline std::uint64_t llf_slot_mask(std::uint32_t side, std::uint32_t covered_ordinal) {
    const std::uint32_t m{side / 8};
    std::uint64_t mask{0};
    for (std::uint32_t fx{0}; fx < m; ++fx) {
        for (std::uint32_t fy{0}; fy < m; ++fy) {
            const std::uint32_t raw{fx * side + fy};
            if (raw / BLOCK_SLOTS == covered_ordinal) {
                mask |= std::uint64_t{1} << (raw % BLOCK_SLOTS);
            }
        }
    }
    return mask;
}

// The transform owning a tile position, in tile-local position indices.
struct TilePosition {
    std::uint8_t owner;
    std::uint8_t side;
    std::uint8_t covered_ordinal;
};

// One CUDA block per 64x64 color tile. A DCT16/DCT32 never straddles a color
// tile (its alignment divides the tile side), so the tile's positions hold
// exactly the coefficients of the first-blocks whose CfL tile it is, and the
// whole tile is staged in shared memory from one coalesced read. Then:
// - Each position's owner is resolved from the transform signals by warp
//   shuffles (a 4x4-aligned DCT32 or 2x2-aligned DCT16 owner lies in the same
//   4 tile rows, i.e. the same warp), with its quantizer strength and
//   low-frequency slot mask.
// - Y is quantized first, since it does not depend on the color map; each
//   thread keeps its Y roundtrip values in registers for the X/B residuals.
// - The CfL regression sums are reduced per slot and tile row, then over the
//   block, in a fixed order.
// - Each position derives its DC from its transform's LLF corner in the staged
//   tile and quantizes it.
// - X and B are quantized against the tile's color map.
// - The non-zero counts per transform position (CoeffOrderStats) are summed in
//   the coefficient buffer, free by then, and added to `order_stats`.
// A warp quantizes the 64 slots of one position per step (two per lane), so
// the non-zero counts of a step all go to one first-block.
__global__ void __launch_bounds__(QUANT_THREADS, QUANT_BLOCKS_PER_SM)
    quantize_tile_kernel(const __half* __restrict__ coeffs, const std::int8_t* __restrict__ acs,
                         const std::int32_t* __restrict__ quant_field,
                         const LlfDcTerms* __restrict__ dc_terms, std::uint32_t bw,
                         std::uint32_t bh, std::uint32_t tiles_x,
                         const QuantCalibration* __restrict__ calibration,
                         std::int8_t* __restrict__ ytox_map, std::int8_t* __restrict__ ytob_map,
                         std::int16_t* __restrict__ ac, std::int32_t* __restrict__ dc,
                         std::int32_t* __restrict__ nz_exact, std::int32_t* __restrict__ nz_grid,
                         CoeffOrderStats* __restrict__ order_stats) {
    __shared__ __align__(16) __half tile_coeffs[3][TILE_SLOTS];
    __shared__ std::uint64_t tile_llf_mask[TILE_POSITIONS];
    // The owner's quantizer strength and its reciprocal.
    __shared__ float2 tile_qgsf[TILE_POSITIONS];
    __shared__ TilePosition tile_pos[TILE_POSITIONS];
    __shared__ std::int32_t tile_nz[3][TILE_POSITIONS];
    __shared__ float cfl_warp[3][QUANT_WARPS];
    __shared__ float cfl_ratio[2];

    const std::uint32_t nblk{bw * bh};
    const std::uint32_t cplane{nblk * BLOCK_SLOTS};
    const float gsf{calibration->global_scale_float};
    const float dc_scale{gsf * static_cast<float>(calibration->quant_dc)};
    const std::uint32_t tid{threadIdx.x};
    const std::uint32_t warp{tid / 32};
    const std::uint32_t lane{tid % 32};

    const std::uint32_t tile{blockIdx.x};
    const std::uint32_t tile_bx{(tile % tiles_x) * TILE_SIDE};
    const std::uint32_t tile_by{(tile / tiles_x) * TILE_SIDE};
    const TileFetch fetched{fetch_tile(coeffs, acs, quant_field, bw, bh, cplane, tile_bx, tile_by)};
    const std::uint32_t cols{min(TILE_SIDE, bw - tile_bx)};
    const std::uint32_t rows{min(TILE_SIDE, bh - tile_by)};

#pragma unroll
    for (std::uint32_t i{0}; i < FETCH_VECTORS; ++i) {
        const std::uint32_t v{i * QUANT_THREADS + tid};
        const std::uint32_t c{v / (TILE_SIDE * ROW_VECTORS)};
        const std::uint32_t at{v % (TILE_SIDE * ROW_VECTORS) * HALVES_PER_VECTOR};
        *reinterpret_cast<uint4*>(&tile_coeffs[c][at]) = fetched.coeffs[i];
    }
    TilePosition pos{};
    if (tid < TILE_POSITIONS) {
        const std::uint32_t lx{tid % TILE_SIDE};
        const std::uint32_t ly{tid / TILE_SIDE};
        const std::uint32_t lane32{(ly & ~3u) % 4 * TILE_SIDE + (lx & ~3u)};
        const std::uint32_t lane16{(ly & ~1u) % 4 * TILE_SIDE + (lx & ~1u)};
        const std::int32_t acs32{__shfl_sync(~0u, fetched.acs, lane32)};
        const std::int32_t acs16{__shfl_sync(~0u, fetched.acs, lane16)};
        const std::uint32_t side{acs32 == 32 ? 32u : (acs16 == 16 ? 16u : 8u)};
        const std::uint32_t owner_lane{side == 32 ? lane32 : (side == 16 ? lane16 : lane)};
        const std::int32_t owner_qf{__shfl_sync(~0u, fetched.quant_field, owner_lane)};
        const std::uint32_t owner{warp * 32 + owner_lane};
        const std::uint32_t m{side / 8};
        const std::uint32_t covered_ordinal{(ly - owner / TILE_SIDE) * m +
                                            (lx - owner % TILE_SIDE)};
        pos = TilePosition{static_cast<std::uint8_t>(owner), static_cast<std::uint8_t>(side),
                           static_cast<std::uint8_t>(covered_ordinal)};
        const bool in_frame{lx < cols && ly < rows};
        tile_pos[tid] = pos;
        tile_llf_mask[tid] = in_frame ? llf_slot_mask(side, covered_ordinal) : ~std::uint64_t{0};
        const float qgsf{static_cast<float>(owner_qf) * gsf};
        tile_qgsf[tid] = make_float2(qgsf, 1.0f / qgsf);
        for (std::uint32_t c{0}; c < 3; ++c) {
            tile_nz[c][tid] = 0;
        }
    }
    __syncthreads();

    {
        const std::uint32_t slot{tid % BLOCK_SLOTS};
        const std::uint32_t row{tid / BLOCK_SLOTS};
        float sxy{0.0f};
        float syy{0.0f};
        float sby{0.0f};
#pragma unroll
        for (std::uint32_t p{row * TILE_SIDE}; p < (row + 1) * TILE_SIDE; ++p) {
            if (!((tile_llf_mask[p] >> slot) & 1)) {
                const std::uint32_t at{p * BLOCK_SLOTS + slot};
                const float xr{__half2float(tile_coeffs[0][at])};
                const float yr{__half2float(tile_coeffs[1][at])};
                const float br{__half2float(tile_coeffs[2][at])};
                sxy += xr * yr;
                syy += yr * yr;
                sby += (br - CFL_BASE_B * yr) * yr;
            }
        }
        for (std::uint32_t s{16}; s > 0; s >>= 1) {
            sxy += __shfl_xor_sync(~0u, sxy, s);
            syy += __shfl_xor_sync(~0u, syy, s);
            sby += __shfl_xor_sync(~0u, sby, s);
        }
        if (lane == 0) {
            cfl_warp[0][warp] = sxy;
            cfl_warp[1][warp] = syy;
            cfl_warp[2][warp] = sby;
        }
    }

    if (tid < TILE_POSITIONS && tid % TILE_SIDE < cols && tid / TILE_SIDE < rows) {
        const std::uint32_t m{static_cast<std::uint32_t>(pos.side) / 8};
        const std::uint32_t olx{pos.owner % TILE_SIDE};
        const std::uint32_t oly{pos.owner / TILE_SIDE};
        float dc_value[3];
        for (std::uint32_t c{0}; c < 3; ++c) {
            dc_value[c] = __half2float(tile_coeffs[c][tid * BLOCK_SLOTS]);
        }
        if (m > 1) {
            const LlfDcTerms& w{
                dc_terms[llf_dc_term_entry(m, tid % TILE_SIDE - olx, tid / TILE_SIDE - oly)]};
            for (std::uint32_t c{0}; c < 3; ++c) {
                float value{0.0f};
                for (std::uint32_t fx{0}; fx < m; ++fx) {
                    for (std::uint32_t fy{0}; fy < m; ++fy) {
                        const std::size_t at{
                            covered_plane_slot(pos.side, olx, oly, TILE_SIDE, fx * pos.side + fy)};
                        value += w.w[fx * m + fy] * __half2float(tile_coeffs[c][at]);
                    }
                }
                dc_value[c] = value;
            }
        }

        const std::uint32_t blk{(tile_by + tid / TILE_SIDE) * bw + tile_bx + tid % TILE_SIDE};
        const float dcq_x{DC_INV_QUANT[0] * dc_scale};
        const float dcq_y{DC_INV_QUANT[1] * dc_scale};
        const float dcq_b{DC_INV_QUANT[2] * dc_scale};
        const int qdc_y{static_cast<int>(lrintf(dc_value[1] * dcq_y))};
        const float y_dc_round{static_cast<float>(qdc_y) / dcq_y};
        dc[nblk + blk] = qdc_y;
        dc[blk] = static_cast<int>(lrintf(dc_value[0] * dcq_x));
        dc[2 * nblk + blk] =
            static_cast<int>(lrintf((dc_value[2] - CFL_BASE_B * y_dc_round) * dcq_b));
    }

    float y_round[QUANT_STEPS][2];
    std::uint32_t nonzero_mask{0};
#pragma unroll
    for (std::uint32_t k{0}; k < QUANT_STEPS; ++k) {
        const std::uint32_t p{k * QUANT_WARPS + warp};
        if (p % TILE_SIDE >= cols || p / TILE_SIDE >= rows) {
            continue;
        }
        const TilePosition at_pos{tile_pos[p]};
        const std::uint64_t llf_mask{tile_llf_mask[p]};
        const float2 qgsf{tile_qgsf[p]};
        std::int16_t qy[2]{};
#pragma unroll
        for (std::uint32_t j{0}; j < 2; ++j) {
            const std::uint32_t slot{2 * lane + j};
            y_round[k][j] = 0.0f;
            if (!((llf_mask >> slot) & 1)) {
                const std::uint32_t raw{at_pos.covered_ordinal * BLOCK_SLOTS + slot};
                const float yr{__half2float(tile_coeffs[1][p * BLOCK_SLOTS + slot])};
                const DequantWeightPair wy{dequant_weight(at_pos.side, 1, static_cast<int>(raw))};
                const int qy_int{quantize_deadzone(div_rn(yr * qgsf.x, wy.weight, wy.inv_weight),
                                                   ac_deadzone_threshold(at_pos.side, 1, raw))};
                y_round[k][j] = div_rn(static_cast<float>(qy_int) * wy.weight, qgsf.x, qgsf.y);
                qy[j] = static_cast<std::int16_t>(qy_int);
                nonzero_mask |= qy_int != 0 ? nonzero_bit(1, k, j) : 0u;
            }
        }
        const std::uint32_t blk{(tile_by + p / TILE_SIDE) * bw + tile_bx + p % TILE_SIDE};
        *reinterpret_cast<short2*>(&ac[cplane + blk * BLOCK_SLOTS + 2 * lane]) =
            make_short2(qy[0], qy[1]);
        const int nz{__popc(__ballot_sync(~0u, qy[0] != 0)) +
                     __popc(__ballot_sync(~0u, qy[1] != 0))};
        if (lane == 0 && nz != 0) {
            atomicAdd(&tile_nz[1][at_pos.owner], nz);
        }
    }
    __syncthreads();

    if (warp == 0) {
        float sxy{lane < QUANT_WARPS ? cfl_warp[0][lane] : 0.0f};
        float syy{lane < QUANT_WARPS ? cfl_warp[1][lane] : 0.0f};
        float sby{lane < QUANT_WARPS ? cfl_warp[2][lane] : 0.0f};
        for (std::uint32_t s{QUANT_WARPS / 2}; s > 0; s >>= 1) {
            sxy += __shfl_xor_sync(~0u, sxy, s);
            syy += __shfl_xor_sync(~0u, syy, s);
            sby += __shfl_xor_sync(~0u, sby, s);
        }
        if (lane == 0) {
            int mx{0};
            int mb{0};
            cfl_maps_from_sums(sxy, syy, sby, &mx, &mb);
            ytox_map[tile] = static_cast<std::int8_t>(mx);
            ytob_map[tile] = static_cast<std::int8_t>(mb);
            cfl_ratio[0] = cfl_ytox_ratio(mx);
            cfl_ratio[1] = cfl_ytob_ratio(mb);
        }
    }
    __syncthreads();

    const float ytox{cfl_ratio[0]};
    const float ytob{cfl_ratio[1]};
#pragma unroll
    for (std::uint32_t k{0}; k < QUANT_STEPS; ++k) {
        const std::uint32_t p{k * QUANT_WARPS + warp};
        if (p % TILE_SIDE >= cols || p / TILE_SIDE >= rows) {
            continue;
        }
        const TilePosition at_pos{tile_pos[p]};
        const std::uint64_t llf_mask{tile_llf_mask[p]};
        const float qgsf{tile_qgsf[p].x};
        std::int16_t qx[2]{};
        std::int16_t qb[2]{};
#pragma unroll
        for (std::uint32_t j{0}; j < 2; ++j) {
            const std::uint32_t slot{2 * lane + j};
            if (!((llf_mask >> slot) & 1)) {
                const std::uint32_t raw{at_pos.covered_ordinal * BLOCK_SLOTS + slot};
                const float xr{__half2float(tile_coeffs[0][p * BLOCK_SLOTS + slot])};
                const float br{__half2float(tile_coeffs[2][p * BLOCK_SLOTS + slot])};
                // X and B share their dead zone.
                const float threshold{ac_deadzone_threshold(at_pos.side, 0, raw)};
                const DequantWeightPair wx{dequant_weight(at_pos.side, 0, static_cast<int>(raw))};
                qx[j] = static_cast<std::int16_t>(quantize_deadzone(
                    div_rn((xr - ytox * y_round[k][j]) * qgsf, wx.weight, wx.inv_weight),
                    threshold));
                const DequantWeightPair wb{dequant_weight(at_pos.side, 2, static_cast<int>(raw))};
                qb[j] = static_cast<std::int16_t>(quantize_deadzone(
                    div_rn((br - ytob * y_round[k][j]) * qgsf, wb.weight, wb.inv_weight),
                    threshold));
                nonzero_mask |= (qx[j] != 0 ? nonzero_bit(0, k, j) : 0u) |
                                (qb[j] != 0 ? nonzero_bit(2, k, j) : 0u);
            }
        }
        const std::uint32_t blk{(tile_by + p / TILE_SIDE) * bw + tile_bx + p % TILE_SIDE};
        *reinterpret_cast<short2*>(&ac[blk * BLOCK_SLOTS + 2 * lane]) = make_short2(qx[0], qx[1]);
        *reinterpret_cast<short2*>(&ac[2 * cplane + blk * BLOCK_SLOTS + 2 * lane]) =
            make_short2(qb[0], qb[1]);
        const int nz_x{__popc(__ballot_sync(~0u, qx[0] != 0)) +
                       __popc(__ballot_sync(~0u, qx[1] != 0))};
        const int nz_b{__popc(__ballot_sync(~0u, qb[0] != 0)) +
                       __popc(__ballot_sync(~0u, qb[1] != 0))};
        if (lane == 0 && nz_x != 0) {
            atomicAdd(&tile_nz[0][at_pos.owner], nz_x);
        }
        if (lane == 1 && nz_b != 0) {
            atomicAdd(&tile_nz[2][at_pos.owner], nz_b);
        }
    }
    __syncthreads();

    std::uint32_t* const stats{reinterpret_cast<std::uint32_t*>(&tile_coeffs[0][0])};
    for (std::uint32_t v{tid}; v < ORDER_STATS_VECTORS; v += QUANT_THREADS) {
        reinterpret_cast<uint4*>(stats)[v] = make_uint4(0, 0, 0, 0);
    }
    const bool first_block{tid < TILE_POSITIONS && tid % TILE_SIDE < cols &&
                           tid / TILE_SIDE < rows && pos.owner == tid};
    if (tid < TILE_POSITIONS && tid % TILE_SIDE < cols && tid / TILE_SIDE < rows) {
        const std::uint32_t m{static_cast<std::uint32_t>(pos.side) / 8};
        const std::uint32_t covered{m * m};
        const std::uint32_t log2_covered{m == 4 ? 4u : (m == 2 ? 2u : 0u)};
        const std::uint32_t blk{(tile_by + tid / TILE_SIDE) * bw + tile_bx + tid % TILE_SIDE};
        for (std::uint32_t c{0}; c < 3; ++c) {
            const std::uint32_t nz{static_cast<std::uint32_t>(tile_nz[c][pos.owner])};
            nz_exact[c * nblk + blk] = first_block ? static_cast<std::int32_t>(nz) : 0;
            nz_grid[c * nblk + blk] = static_cast<std::int32_t>((nz + covered - 1) >> log2_covered);
        }
    }
    __syncthreads();

    if (first_block) {
        atomicAdd(&stats[offsetof(CoeffOrderStats, blocks) / sizeof(std::uint32_t) +
                         coeff_order_side_index(pos.side)],
                  1u);
    }
    for (std::uint32_t bits{nonzero_mask}; bits != 0; bits &= bits - 1) {
        const auto b{static_cast<std::uint32_t>(__ffs(static_cast<int>(bits)) - 1)};
        const std::uint32_t c{b / (2 * QUANT_STEPS)};
        const TilePosition at_pos{tile_pos[(b / 2) % QUANT_STEPS * QUANT_WARPS + warp]};
        atomicAdd(&stats[coeff_order_offset(at_pos.side, static_cast<int>(c)) +
                         at_pos.covered_ordinal * BLOCK_SLOTS + 2 * lane + b % 2],
                  1u);
    }
    __syncthreads();

    auto* const order_stats_words{reinterpret_cast<std::uint32_t*>(order_stats)};
    for (std::uint32_t w{tid}; w < ORDER_STATS_WORDS; w += QUANT_THREADS) {
        const std::uint32_t count{stats[w]};
        if (count != 0) {
            atomicAdd(&order_stats_words[w], count);
        }
    }
}

}  // namespace

bool quantize_residual(const __half* coeffs, const std::int8_t* acs,
                       const std::int32_t* quant_field, const QuantCalibration* calibration,
                       std::size_t width, std::size_t height, std::int8_t* ytox_map,
                       std::int8_t* ytob_map, std::int16_t* ac, std::int32_t* dc,
                       std::int32_t* nz_exact, std::int32_t* nz_grid, CoeffOrderStats* order_stats,
                       cudaStream_t stream) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const LlfDcTerms* dc_terms{llf_dc_terms_device()};
    if (dc_terms == nullptr) {
        return false;
    }
    const std::uint32_t tiles_x{static_cast<std::uint32_t>(ceil_div(bw, TILE_SIDE))};
    const std::uint32_t tiles{tiles_x * static_cast<std::uint32_t>(ceil_div(bh, TILE_SIDE))};
    if (cudaMemsetAsync(order_stats, 0, sizeof(CoeffOrderStats), stream) != cudaSuccess) {
        return false;
    }
    quantize_tile_kernel<<<tiles, QUANT_THREADS, 0, stream>>>(
        coeffs, acs, quant_field, dc_terms, static_cast<std::uint32_t>(bw),
        static_cast<std::uint32_t>(bh), tiles_x, calibration, ytox_map, ytob_map, ac, dc, nz_exact,
        nz_grid, order_stats);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace cujpegxl
