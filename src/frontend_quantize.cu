// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frontend_quantize.h"

#include <cmath>
#include <cstddef>
#include <limits>

#include <cuda_runtime.h>

#include "cfl.h"
#include "coeff_order.h"
#include "dequant_weight_pairs.h"
#include "exact_div.cuh"
#include "quant_calibration.h"
#include "round_to_int.h"
#include "vardct_layout.h"

namespace cujpegxl {
namespace {

std::size_t ceil_div(std::size_t a, std::size_t b) {
    return (a + b - 1) / b;
}

// The per-coefficient quantization tables hold the coefficients of the DCT8,
// DCT16 and DCT32 transforms back to back.
constexpr std::uint32_t DCT16_ENTRY_OFFSET = 64;
constexpr std::uint32_t DCT32_ENTRY_OFFSET = DCT16_ENTRY_OFFSET + 256;
constexpr std::uint32_t TABLE_ENTRIES = DCT32_ENTRY_OFFSET + 1024;

__host__ __device__ constexpr std::uint32_t table_entry(std::uint32_t side, std::uint32_t raw) {
    return (side == 32 ? DCT32_ENTRY_OFFSET : (side == 16 ? DCT16_ENTRY_OFFSET : 0u)) + raw;
}

struct alignas(sizeof(float4)) WeightPairTable {
    DequantWeightPair w[3][TABLE_ENTRIES];
};

template <std::size_t COEFFS>
constexpr void copy_weight_pairs(const DequantWeightPairs<COEFFS>& pairs, std::uint32_t side,
                                 WeightPairTable* table) {
    for (std::size_t c{0}; c < 3; ++c) {
        for (std::uint32_t raw{0}; raw < COEFFS; ++raw) {
            table->w[c][table_entry(side, raw)] = pairs.w[c][raw];
        }
    }
}

constexpr WeightPairTable make_weight_pair_table() {
    WeightPairTable table{};
    copy_weight_pairs(DCT8_DEQUANT_PAIRS, 8, &table);
    copy_weight_pairs(DCT16_DEQUANT_PAIRS, 16, &table);
    copy_weight_pairs(DCT32_DEQUANT_PAIRS, 32, &table);
    return table;
}

// The dead-zone threshold of each coefficient, [0] for X and B (which share
// it) and [1] for Y. A coefficient of the low-frequency corner (fx, fy < N/8),
// carried as DC, has an infinite threshold: it quantizes to 0, with a zero Y
// roundtrip.
struct alignas(sizeof(float2)) ThresholdTable {
    float t[2][TABLE_ENTRIES];
};

constexpr ThresholdTable make_threshold_table() {
    ThresholdTable table{};
    for (const std::uint32_t side : {8u, 16u, 32u}) {
        for (std::uint32_t raw{0}; raw < side * side; ++raw) {
            const bool llf{raw / side < side / 8 && raw % side < side / 8};
            for (std::uint32_t luma{0}; luma < 2; ++luma) {
                table.t[luma][table_entry(side, raw)] =
                    llf ? std::numeric_limits<float>::infinity()
                        : ac_deadzone(side, luma).threshold(raw);
            }
        }
    }
    return table;
}

// In global memory: the lanes of a warp read different entries, which
// constant memory would serialize.
__device__ const WeightPairTable WEIGHT_PAIRS{make_weight_pair_table()};
__device__ const ThresholdTable THRESHOLDS{make_threshold_table()};

// The quantization parameters of a lane's two adjacent slots, from table
// entries `entry` and `entry + 1` (`entry` even).
struct LaneWeightPairs {
    DequantWeightPair slot[2];
};

__device__ inline LaneWeightPairs lane_weight_pairs(std::uint32_t channel, std::uint32_t entry) {
    const float4 w{__ldg(reinterpret_cast<const float4*>(&WEIGHT_PAIRS.w[channel][entry]))};
    return {{{w.x, w.y}, {w.z, w.w}}};
}

struct LaneThresholds {
    float slot[2];
};

__device__ inline LaneThresholds lane_thresholds(std::uint32_t luma, std::uint32_t entry) {
    const float2 t{__ldg(reinterpret_cast<const float2*>(&THRESHOLDS.t[luma][entry]))};
    return {{t.x, t.y}};
}

__device__ inline int quantize_deadzone(float steps, float threshold) {
    return fabsf(steps) < threshold ? 0 : round_to_int(steps);
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

// A tile's order statistics are summed in shared memory as one 8-bit counter
// per CoeffOrderStats word, four per packed word: a tile adds at most
// TILE_POSITIONS to any of them.
constexpr std::uint32_t ORDER_STATS_WORDS = sizeof(CoeffOrderStats) / sizeof(std::uint32_t);
constexpr std::uint32_t PACKED_ORDER_STATS_VECTORS = (ORDER_STATS_WORDS + 15) / 16;
constexpr std::uint32_t PACKED_ORDER_STATS_WORDS = 4 * PACKED_ORDER_STATS_VECTORS;
static_assert(TILE_POSITIONS <= 0xff && PACKED_ORDER_STATS_VECTORS <= QUANT_THREADS);

__device__ inline void count_order_stat(std::uint32_t* packed_stats, std::uint32_t word) {
    atomicAdd(&packed_stats[word / 4], std::uint32_t{1} << (word % 4 * 8));
}

// Counts the non-zero flags of a lane's two slots, words `base + 2 * lane` and
// `base + 2 * lane + 1`, in one atomic; `base` is a multiple of 4, so the
// lane's byte shifts depend on the lane only.
__device__ inline void count_lane_nonzeros(std::uint32_t* packed_stats, std::uint32_t base,
                                           std::uint32_t lane, bool nonzero0, bool nonzero1) {
    const std::uint32_t shift{lane % 2 * 16};
    const std::uint32_t increment{(nonzero0 ? std::uint32_t{1} << shift : 0u) |
                                  (nonzero1 ? std::uint32_t{1} << (shift + 8) : 0u)};
    if (increment != 0) {
        atomicAdd(&packed_stats[base / 4 + lane / 2], increment);
    }
}

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

// First CoeffOrderStats word of the non-zero counts of a position's slots.
__device__ inline std::uint32_t position_stats_base(const TilePosition& p, std::uint32_t c) {
    return static_cast<std::uint32_t>(coeff_order_offset(p.side, static_cast<int>(c))) +
           p.covered_ordinal * BLOCK_SLOTS;
}

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
// - The non-zero counts per coefficient (CoeffOrderStats) are summed per step
//   in the tile's packed counters, then added to `order_stats`.
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
    __shared__ __align__(16) std::uint32_t tile_order_stats[PACKED_ORDER_STATS_WORDS];
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

    // Indexed as a uint4 array: through __half element addresses, clang splits
    // each vector into two repacked 8-byte stores.
#pragma unroll
    for (std::uint32_t i{0}; i < FETCH_VECTORS; ++i) {
        reinterpret_cast<uint4*>(&tile_coeffs[0][0])[i * QUANT_THREADS + tid] = fetched.coeffs[i];
    }
    if (tid < PACKED_ORDER_STATS_VECTORS) {
        reinterpret_cast<uint4*>(tile_order_stats)[tid] = make_uint4(0, 0, 0, 0);
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

    // One DC per thread: warps 0-3 take Y in lanes 0-15 and B in lanes 16-31
    // of 16 positions each (B is predicted from the Y DC of its position),
    // warps 4-5 take X of 32 positions each.
    if (tid < 3 * TILE_POSITIONS) {
        const bool y_or_b{tid < 2 * TILE_POSITIONS};
        const std::uint32_t p{y_or_b ? warp * 16 + lane % 16 : tid - 2 * TILE_POSITIONS};
        const std::uint32_t c{y_or_b ? (lane < 16 ? 1u : 2u) : 0u};
        const bool in_frame{p % TILE_SIDE < cols && p / TILE_SIDE < rows};
        const TilePosition dc_pos{tile_pos[p]};
        float value{__half2float(tile_coeffs[c][p * BLOCK_SLOTS])};
        const std::uint32_t m{static_cast<std::uint32_t>(dc_pos.side) / 8};
        if (in_frame && m > 1) {
            const std::uint32_t olx{dc_pos.owner % TILE_SIDE};
            const std::uint32_t oly{dc_pos.owner / TILE_SIDE};
            const LlfDcTerms& w{
                dc_terms[llf_dc_term_entry(m, p % TILE_SIDE - olx, p / TILE_SIDE - oly)]};
            value = 0.0f;
            for (std::uint32_t fx{0}; fx < m; ++fx) {
                for (std::uint32_t fy{0}; fy < m; ++fy) {
                    const std::size_t at{covered_plane_slot(dc_pos.side, olx, oly, TILE_SIDE,
                                                            fx * dc_pos.side + fy)};
                    value += w.w[fx * m + fy] * __half2float(tile_coeffs[c][at]);
                }
            }
        }

        const float dcq{DC_INV_QUANT[c] * dc_scale};
        int qdc{round_to_int(value * dcq)};
        if (y_or_b) {
            const int qdc_y{__shfl_xor_sync(~0u, qdc, 16)};
            if (c == 2) {
                const float y_dc_round{static_cast<float>(qdc_y) / (DC_INV_QUANT[1] * dc_scale)};
                qdc = round_to_int((value - CFL_BASE_B * y_dc_round) * dcq);
            }
        }
        if (in_frame) {
            const std::uint32_t blk{(tile_by + p / TILE_SIDE) * bw + tile_bx + p % TILE_SIDE};
            dc[c * nblk + blk] = qdc;
        }
    }

    float y_round[QUANT_STEPS][2];
#pragma unroll
    for (std::uint32_t k{0}; k < QUANT_STEPS; ++k) {
        const std::uint32_t p{k * QUANT_WARPS + warp};
        if (p % TILE_SIDE >= cols || p / TILE_SIDE >= rows) {
            continue;
        }
        const TilePosition at_pos{tile_pos[p]};
        const float2 qgsf{tile_qgsf[p]};
        const std::uint32_t entry{
            table_entry(at_pos.side, at_pos.covered_ordinal * BLOCK_SLOTS + 2 * lane)};
        const LaneWeightPairs wy{lane_weight_pairs(1, entry)};
        const LaneThresholds thresholds{lane_thresholds(1, entry)};
        std::int16_t qy[2];
        bool nonzero[2];
#pragma unroll
        for (std::uint32_t j{0}; j < 2; ++j) {
            const float yr{__half2float(tile_coeffs[1][p * BLOCK_SLOTS + 2 * lane + j])};
            const int qy_int{quantize_deadzone(
                div_rn(yr * qgsf.x, wy.slot[j].weight, wy.slot[j].inv_weight), thresholds.slot[j])};
            y_round[k][j] = div_rn(static_cast<float>(qy_int) * wy.slot[j].weight, qgsf.x, qgsf.y);
            qy[j] = static_cast<std::int16_t>(qy_int);
            nonzero[j] = qy_int != 0;
        }
        const std::uint32_t blk{(tile_by + p / TILE_SIDE) * bw + tile_bx + p % TILE_SIDE};
        *reinterpret_cast<short2*>(&ac[cplane + blk * BLOCK_SLOTS + 2 * lane]) =
            make_short2(qy[0], qy[1]);
        const int nz{__popc(__ballot_sync(~0u, nonzero[0])) +
                     __popc(__ballot_sync(~0u, nonzero[1]))};
        if (lane == 0 && nz != 0) {
            atomicAdd(&tile_nz[1][at_pos.owner], nz);
        }
        count_lane_nonzeros(tile_order_stats, position_stats_base(at_pos, 1), lane, nonzero[0],
                            nonzero[1]);
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
        const float qgsf{tile_qgsf[p].x};
        const std::uint32_t entry{
            table_entry(at_pos.side, at_pos.covered_ordinal * BLOCK_SLOTS + 2 * lane)};
        const LaneWeightPairs wx{lane_weight_pairs(0, entry)};
        const LaneWeightPairs wb{lane_weight_pairs(2, entry)};
        const LaneThresholds thresholds{lane_thresholds(0, entry)};
        std::int16_t qx[2];
        std::int16_t qb[2];
        bool nonzero_x[2];
        bool nonzero_b[2];
#pragma unroll
        for (std::uint32_t j{0}; j < 2; ++j) {
            const std::uint32_t at{p * BLOCK_SLOTS + 2 * lane + j};
            const float xr{__half2float(tile_coeffs[0][at])};
            const float br{__half2float(tile_coeffs[2][at])};
            const int qx_int{quantize_deadzone(div_rn((xr - ytox * y_round[k][j]) * qgsf,
                                                      wx.slot[j].weight, wx.slot[j].inv_weight),
                                               thresholds.slot[j])};
            const int qb_int{quantize_deadzone(div_rn((br - ytob * y_round[k][j]) * qgsf,
                                                      wb.slot[j].weight, wb.slot[j].inv_weight),
                                               thresholds.slot[j])};
            qx[j] = static_cast<std::int16_t>(qx_int);
            qb[j] = static_cast<std::int16_t>(qb_int);
            nonzero_x[j] = qx_int != 0;
            nonzero_b[j] = qb_int != 0;
        }
        const std::uint32_t blk{(tile_by + p / TILE_SIDE) * bw + tile_bx + p % TILE_SIDE};
        *reinterpret_cast<short2*>(&ac[blk * BLOCK_SLOTS + 2 * lane]) = make_short2(qx[0], qx[1]);
        *reinterpret_cast<short2*>(&ac[2 * cplane + blk * BLOCK_SLOTS + 2 * lane]) =
            make_short2(qb[0], qb[1]);
        const int nz_x{__popc(__ballot_sync(~0u, nonzero_x[0])) +
                       __popc(__ballot_sync(~0u, nonzero_x[1]))};
        const int nz_b{__popc(__ballot_sync(~0u, nonzero_b[0])) +
                       __popc(__ballot_sync(~0u, nonzero_b[1]))};
        if (lane == 0 && nz_x != 0) {
            atomicAdd(&tile_nz[0][at_pos.owner], nz_x);
        }
        if (lane == 1 && nz_b != 0) {
            atomicAdd(&tile_nz[2][at_pos.owner], nz_b);
        }
        count_lane_nonzeros(tile_order_stats, position_stats_base(at_pos, 0), lane, nonzero_x[0],
                            nonzero_x[1]);
        count_lane_nonzeros(tile_order_stats, position_stats_base(at_pos, 2), lane, nonzero_b[0],
                            nonzero_b[1]);
    }
    __syncthreads();

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
    if (first_block) {
        count_order_stat(
            tile_order_stats,
            static_cast<std::uint32_t>(offsetof(CoeffOrderStats, blocks) / sizeof(std::uint32_t) +
                                       coeff_order_side_index(pos.side)));
    }
    __syncthreads();

    // The counters past ORDER_STATS_WORDS stay 0.
    auto* const order_stats_words{reinterpret_cast<std::uint32_t*>(order_stats)};
    for (std::uint32_t w{tid}; w < PACKED_ORDER_STATS_WORDS; w += QUANT_THREADS) {
        const std::uint32_t packed{tile_order_stats[w]};
        for (std::uint32_t b{0}; b < 4; ++b) {
            const std::uint32_t count{(packed >> (b * 8)) & 0xff};
            if (count != 0) {
                atomicAdd(&order_stats_words[4 * w + b], count);
            }
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
