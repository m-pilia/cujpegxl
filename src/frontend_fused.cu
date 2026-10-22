// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frontend_fused.h"

#include <algorithm>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "adaptive_quant_impl.cuh"
#include "dct_impl.cuh"
#include "dequant_weight_pairs.h"
#include "exact_div.cuh"
#include "gaborish.h"
#include "quant_calibration.h"
#include "transform_select_impl.cuh"
#include "vardct_layout.h"
#include "xyb_impl.cuh"

namespace cujpegxl {
namespace {

// Tile geometry: one CUDA block processes TB x TB image blocks. The shared XYB
// region carries the two-pixel halo required by gaborish. The pre-erosion map is
// at 4x-subsampled resolution (CELLS x CELLS cells).
constexpr int HALO = 2;
constexpr int TB = 4;
constexpr int TILE_PX = TB * 8;
constexpr int PAD = TILE_PX + 2 * HALO;
constexpr int CELLS = 2 * TB;
constexpr int TILE_SAMPLES = TILE_PX * TILE_PX;
// The DCT horizontal pass has the lanes of a warp read different tile rows at
// the same column; the odd row stride maps them to distinct shared-memory banks.
constexpr int TILE_STRIDE = TILE_PX + 1;
constexpr int TILE_FLOATS = TILE_PX * TILE_STRIDE;
constexpr int PAD_SAMPLES = PAD * PAD;
constexpr int FRONTEND_THREADS = 256;
// The pre-erosion takes one cell row per warp and one tile column per lane.
static_assert(FRONTEND_THREADS == 32 * CELLS && TILE_PX == 32);
// The erosion takes four lanes per cell.
static_assert(FRONTEND_THREADS == 4 * CELLS * CELLS);
// The raw X and B planes stay resident from the load to the X/B gaborish at
// RESIDENT_X_B_OFFSET; the buffers local to each phase lie below them.
constexpr std::size_t LOAD_AQ_LOCAL_PEAK =
    TILE_FLOATS + PAD_SAMPLES + 2 * CELLS * CELLS + 6 * TB * TB;
// The selection's horizontal-pass rows of each candidate size N start at sharp
// Y, which they overlap, with vector v at v * (N + 1): the lanes of a warp
// reading consecutive vectors hit distinct banks. The selection words follow:
// the 8x8 and 16x16 costs, the 32x32 partial distortions and rates, and the
// region layout. Then the rate terms of the quantized magnitudes below
// RATE_TERMS (see accumulate_coeff_cost).
constexpr int SELECT_ROWS16_OFFSET = 32 * (32 + 1);
constexpr int SELECT_ROWS8_OFFSET = SELECT_ROWS16_OFFSET + 64 * (16 + 1);
constexpr int SELECT_WORDS_OFFSET = SELECT_ROWS8_OFFSET + 128 * (8 + 1);
constexpr std::size_t SELECT_WORDS = 16 + 4 + 2 * 2 + 1;
constexpr int RATE_TERMS = 128;
constexpr std::size_t RATE_TERMS_OFFSET = SELECT_WORDS_OFFSET + SELECT_WORDS;
constexpr std::size_t REGION_LOCAL_PEAK = RATE_TERMS_OFFSET + RATE_TERMS;
// The X/B horizontal-pass rows and coefficients, padded by region_row_index.
constexpr int REGION_ROWS_FLOATS = TILE_SAMPLES + TILE_SAMPLES / 32;
constexpr std::size_t RESIDENT_X_B_OFFSET = std::max(LOAD_AQ_LOCAL_PEAK, REGION_LOCAL_PEAK);
// The X/B pass holds sharp X and B below the raw planes, its horizontal-pass
// rows in the raw planes once gaborish has consumed them, and its coefficients
// in the sharp planes once the horizontal pass has consumed those.
// Warps that pair X and B lanes read distinct banks: the horizontal pass reads
// 16 consecutive rows per channel, so sharp B starts 16 banks after sharp X;
// the DCT16 vertical pass reads and writes 16 vectors per channel on 16 banks
// with a gap of 8 after each 8, so the B rows and coefficients start 8 (mod 16)
// banks after the X ones.
constexpr std::size_t SHARP_B_OFFSET = TILE_FLOATS + 16;
static_assert(SHARP_B_OFFSET % 32 == 16 && SHARP_B_OFFSET + TILE_FLOATS <= RESIDENT_X_B_OFFSET);
constexpr int ROWS_B_SKEW = 8;
static_assert((PAD_SAMPLES + ROWS_B_SKEW) % 16 == 8);
static_assert(REGION_ROWS_FLOATS + ROWS_B_SKEW <= PAD_SAMPLES);
constexpr std::size_t COEFFS_B_OFFSET = SHARP_B_OFFSET + 8;
static_assert(COEFFS_B_OFFSET % 16 == 8 && REGION_ROWS_FLOATS <= SHARP_B_OFFSET &&
              COEFFS_B_OFFSET + REGION_ROWS_FLOATS <= RESIDENT_X_B_OFFSET);
constexpr std::size_t WORKSPACE_FLOATS = RESIDENT_X_B_OFFSET + 2 * PAD_SAMPLES;
constexpr std::size_t WORKSPACE_BYTES = WORKSPACE_FLOATS * sizeof(float);
// Four resident blocks per SM: sm_61 has 96 KB of shared memory per SM, and
// FRONTEND_BLOCKS_PER_SM * FRONTEND_THREADS threads leave 64 registers each.
constexpr int FRONTEND_BLOCKS_PER_SM = 4;
static_assert(FRONTEND_BLOCKS_PER_SM * WORKSPACE_BYTES <= 96 * 1024);

// Loads the N floats at `src` and computes F of their 1D DCT frequencies (N
// times A, see dct_impl.cuh) into out[i], frequency i * N / F + part: all of
// them, or for F = N / 2 the even (part 0) or odd (part 1) ones.
template <int N, int F, int SIZE>
__device__ __forceinline__ void dct_vector(const float* __restrict__ src, int part,
                                           float (&out)[SIZE]) {
    static_assert((F == N || 2 * F == N) && F <= SIZE);
    float in[N];
#pragma unroll
    for (int t{0}; t < N; ++t) {
        in[t] = src[t];
    }
    float freqs[F];
    if constexpr (F == N) {
        dct1d(in);
#pragma unroll
        for (int i{0}; i < F; ++i) {
            freqs[i] = in[i];
        }
    } else if (part == 0) {
        dct1d_even(in, freqs);
    } else {
        dct1d_odd(in, freqs);
    }
#pragma unroll
    for (int i{0}; i < F; ++i) {
        out[i] = freqs[i];
    }
}

// Sum over aligned groups of G lanes, valid on every lane.
template <int G>
__device__ inline float lane_group_sum(float v) {
#pragma unroll
    for (int off{1}; off < G; off <<= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

// Full-mask warp sum, valid on lane 0. Pairs lanes like a halving tree
// v[i] += v[i + s] for s = 16..1; the adds are non-contractible, so callers can
// reproduce a shared-memory tree reduction bit for bit.
__device__ inline float warp_reduce_sum(float v) {
    for (int off{16}; off > 0; off >>= 1) {
        v = __fadd_rn(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    return v;
}

// Neighbour k (row-major in the 3x3 neighbourhood) of pre-erosion cell (cx, cy),
// clamped to the tile (the seam approximation).
__device__ inline float erosion_neighbour(const float* __restrict__ pre, int cx, int cy, int k) {
    const int x{min(max(cx + k % 3 - 1, 0), CELLS - 1)};
    const int y{min(max(cy + k / 3 - 1, 0), CELLS - 1)};
    return pre[y * CELLS + x];
}

// FuzzyErosion of all pre-erosion cells: the weighted sum of the four smallest
// values in each cell's 3x3 neighbourhood. Four lanes share a cell, each sorting
// up to three of the nine neighbours, and two butterfly merges give all four
// lanes the cell's four smallest values. These do not depend on how the
// neighbours are split, and the weighted sum keeps its order.
__device__ void erode_cells(const float* __restrict__ pre, float* __restrict__ eroded, int tid,
                            float km0, float km1, float km2, float km3) {
    const int cell{tid / 4};
    const int part{tid % 4};
    const int cx{cell % CELLS};
    const int cy{cell / CELLS};
    float m0{erosion_neighbour(pre, cx, cy, part)};
    float m1{erosion_neighbour(pre, cx, cy, part + 4)};
    float m2{part == 0 ? erosion_neighbour(pre, cx, cy, 8) : INFINITY};
    float m3{INFINITY};
    compare_swap(m0, m1);
    compare_swap(m1, m2);
    compare_swap(m0, m1);
    merge_min4_xor(m0, m1, m2, m3, 1);
    merge_min4_xor(m0, m1, m2, m3, 2);
    if (part == 0) {
        eroded[cell] = km0 * m0 + km1 * m1 + km2 * m2 + km3 * m3;
    }
}

// Each thread filters a strip of GAB_STRIP_ROWS outputs in one column, sliding
// a window of 5-wide input rows held in registers, so each input row is read
// from shared memory once rather than once per output it contributes to.
constexpr int GAB_STRIP_ROWS{TILE_SAMPLES / FRONTEND_THREADS};
static_assert(GAB_STRIP_ROWS * FRONTEND_THREADS == TILE_SAMPLES);

__device__ void gaborish_tile(const float* __restrict__ in, float* __restrict__ out, int tid) {
    constexpr int WIDTH{2 * HALO + 1};
    const int x0{tid % TILE_PX};
    const int y0{tid / TILE_PX * GAB_STRIP_ROWS};
    float w[GAB_STRIP_ROWS + 2 * HALO][WIDTH];
#pragma unroll
    for (int r{0}; r < GAB_STRIP_ROWS + 2 * HALO; ++r) {
#pragma unroll
        for (int c{0}; c < WIDTH; ++c) {
            w[r][c] = in[(y0 + r) * PAD + x0 + c];
        }
    }
#pragma unroll
    for (int k{0}; k < GAB_STRIP_ROWS; ++k) {
        const auto at = [&](int dy, int dx) { return w[k + HALO + dy][HALO + dx]; };
        const float e{at(0, -1) + at(0, 1) + at(-1, 0) + at(1, 0)};
        const float r2{at(0, -2) + at(0, 2) + at(-2, 0) + at(2, 0)};
        const float dg{at(-1, -1) + at(-1, 1) + at(1, -1) + at(1, 1)};
        const float l8{at(-1, -2) + at(-1, 2) + at(1, -2) + at(1, 2) + at(-2, -1) + at(-2, 1) +
                       at(2, -1) + at(2, 1)};
        const float d4{at(-2, -2) + at(-2, 2) + at(2, -2) + at(2, 2)};
        out[(y0 + k) * TILE_STRIDE + x0] = GAB_WC * at(0, 0) + GAB_WR * e + GAB_WR2 * r2 +
                                           GAB_WD * dg + GAB_WL * l8 + GAB_WD2 * d4;
    }
}

// rate_terms[i] = 1 + log2f(1 + i) for 0 < i < RATE_TERMS, and 0 for i = 0.
__device__ inline void init_rate_terms(float* __restrict__ rate_terms, int tid) {
    if (tid < RATE_TERMS) {
        rate_terms[tid] = tid == 0 ? 0.0f : 1.0f + log2f(1.0f + static_cast<float>(tid));
    }
}

// Rate-distortion contribution of one Y candidate coefficient quantized with the
// nearest DCT8 weight; the LLF corner is excluded.
__device__ inline void accumulate_coeff_cost(float coeff, int fx, int fy, int n, float qgsf,
                                             float inv_qgsf, const float* __restrict__ rate_terms,
                                             float& dist, float& rate) {
    const int m{n / 8};
    if (fx < m && fy < m) {
        return;
    }
    // Nearest DCT8 position, ties away from zero.
    int sx{(fx * 8 + n / 2) / n};
    int sy{(fy * 8 + n / 2) / n};
    sx = sx > 7 ? 7 : sx;
    sy = sy > 7 ? 7 : sy;
    const DequantWeightPair w{DCT8_DEQUANT_PAIRS.w[1][sx * 8 + sy]};
    const float q{rintf(div_rn(coeff * qgsf, w.weight, w.inv_weight))};
    const float err{div_rn(q * w.weight, qgsf, inv_qgsf) - coeff};
    dist += err * err;
    const float mag{fabsf(q)};
    if (mag < static_cast<float>(RATE_TERMS)) {
        rate += rate_terms[static_cast<int>(mag)];
    } else {
        rate += 1.0f + log2f(1.0f + mag);
    }
}

__device__ inline float candidate_cost(int n, float dist, float rate, float lambda) {
    return static_cast<float>(n * n) * dist + lambda * rate;
}

constexpr std::uint32_t WHOLE32_LAYOUT{0x20202020u};

__device__ inline int quadrant_side(std::uint32_t layout, int q) {
    return static_cast<int>((layout >> (8 * q)) & 0xffu);
}

struct alignas(16) Half8 {
    __half2 h[4];
};

struct alignas(8) Half4 {
    __half2 h[2];
};

__device__ inline void store_half8(__half* dst, const float* v) {
    *reinterpret_cast<Half8*>(dst) =
        Half8{{__floats2half2_rn(v[0], v[1]), __floats2half2_rn(v[2], v[3]),
               __floats2half2_rn(v[4], v[5]), __floats2half2_rn(v[6], v[7])}};
}

__device__ inline void store_half4(__half* dst, const float* v) {
    *reinterpret_cast<Half4*>(dst) =
        Half4{{__floats2half2_rn(v[0], v[1]), __floats2half2_rn(v[2], v[3])}};
}

// Each warp takes one candidate size: warps 0-3 the 8x8 candidates of tile
// column block w, warps 4-5 the 16x16 candidates of tile column half w - 4,
// warps 6-7 the even and odd frequencies of the 32x32 candidate. A lane
// transforms one N-vector per pass: a tile row, then the rows of one fx.
//
// Returns the region layout: byte q holds the side (8, 16 or 32) of the
// transforms covering quadrant q, or 0 if the quadrant lies outside the image.
// Only the chosen transforms' Y coefficients are stored to y_plane.
__device__ std::uint32_t select_transforms(float* __restrict__ select, std::size_t rbx,
                                           std::size_t rby, std::size_t bw, std::size_t bh,
                                           float qgsf, float lambda, std::int8_t* __restrict__ acs,
                                           __half* __restrict__ y_plane, int tid) {
    const float* sharp_y{select};
    float* rows32{select};
    float* rows16{select + SELECT_ROWS16_OFFSET};
    float* rows8{select + SELECT_ROWS8_OFFSET};
    float* cost8{select + SELECT_WORDS_OFFSET};
    float* cost16{cost8 + 16};
    float* dist32{cost16 + 4};
    float* rate32{dist32 + 2};
    std::uint32_t* layout_word{reinterpret_cast<std::uint32_t*>(rate32 + 2)};
    float* rate_terms{select + RATE_TERMS_OFFSET};
    init_rate_terms(rate_terms, tid);
    const float inv_qgsf{1.0f / qgsf};
    constexpr float INF{1.0e30f};
    const int lane{tid % 32};
    const int warp{tid / 32};
    const int part32{warp - 6};

    // Horizontal pass, lane r on tile row r. Rows32 overlaps sharp Y, so the
    // rows are stored once every warp has loaded its pixels.
    float h[16];
    const float* pixel_row{sharp_y + lane * TILE_STRIDE};
    if (warp < 4) {
        dct_vector<8, 8>(pixel_row + 8 * warp, 0, h);
    } else if (warp < 6) {
        dct_vector<16, 16>(pixel_row + 16 * (warp - 4), 0, h);
    } else {
        dct_vector<32, 16>(pixel_row, part32, h);
    }
    __syncthreads();
    // Vector (4 cb + rb) * 8 + fx of rows8 holds frequency fx of the rows of
    // 8x8 block (cb, rb), vector (2 ch + rb) * 16 + fx of rows16 that of 16x16
    // block (ch, rb), and vector fx of rows32 that of the tile.
    if (warp < 4) {
        float* dst{rows8 + (4 * warp + lane / 8) * 8 * 9 + lane % 8};
#pragma unroll
        for (int fx{0}; fx < 8; ++fx) {
            dst[fx * 9] = h[fx];
        }
    } else if (warp < 6) {
        float* dst{rows16 + (2 * (warp - 4) + lane / 16) * 16 * 17 + lane % 16};
#pragma unroll
        for (int fx{0}; fx < 16; ++fx) {
            dst[fx * 17] = h[fx];
        }
    } else {
#pragma unroll
        for (int m{0}; m < 16; ++m) {
            rows32[(2 * m + part32) * 33 + lane] = h[m];
        }
    }
    __syncthreads();

    // Vertical pass, lane v on vector 32 w + v of its size (w relative to the
    // first warp of that size), and the candidates' costs.
    float c[16];
    if (warp < 4) {
        const int rb{lane / 8};
        const int fx{lane % 8};
        dct_vector<8, 8>(rows8 + (32 * warp + lane) * 9, 0, c);
        float dist{0.0f};
        float rate{0.0f};
#pragma unroll
        for (int fy{0}; fy < 8; ++fy) {
            c[fy] *= DCT2D_SCALE<8>;
            accumulate_coeff_cost(c[fy], fx, fy, 8, qgsf, inv_qgsf, rate_terms, dist, rate);
        }
        dist = lane_group_sum<8>(dist);
        rate = lane_group_sum<8>(rate);
        const bool inside{rbx + warp < bw && rby + rb < bh};
        if (fx == 0) {
            cost8[8 * (rb / 2) + 4 * (warp / 2) + 2 * (rb % 2) + warp % 2] =
                inside ? candidate_cost(8, dist, rate, lambda) : INF;
        }
    } else if (warp < 6) {
        const int ch{warp - 4};
        const int rb{lane / 16};
        const int fx{lane % 16};
        dct_vector<16, 16>(rows16 + (32 * ch + lane) * 17, 0, c);
        float dist{0.0f};
        float rate{0.0f};
#pragma unroll
        for (int fy{0}; fy < 16; ++fy) {
            c[fy] *= DCT2D_SCALE<16>;
            accumulate_coeff_cost(c[fy], fx, fy, 16, qgsf, inv_qgsf, rate_terms, dist, rate);
        }
        dist = lane_group_sum<16>(dist);
        rate = lane_group_sum<16>(rate);
        const bool fit16{rbx + 2 * ch + 2 <= bw && rby + 2 * rb + 2 <= bh};
        if (fx == 0) {
            cost16[2 * rb + ch] = fit16 ? candidate_cost(16, dist, rate, lambda) : INF;
        }
    } else {
        dct_vector<32, 16>(rows32 + lane * 33, part32, c);
        float dist{0.0f};
        float rate{0.0f};
#pragma unroll
        for (int m{0}; m < 16; ++m) {
            c[m] *= DCT2D_SCALE<32>;
            accumulate_coeff_cost(c[m], lane, 2 * m + part32, 32, qgsf, inv_qgsf, rate_terms, dist,
                                  rate);
        }
        dist = lane_group_sum<32>(dist);
        rate = lane_group_sum<32>(rate);
        if (lane == 0) {
            dist32[part32] = dist;
            rate32[part32] = rate;
        }
    }
    __syncthreads();

    // Lane q < 4 of warp 0 decides quadrant q, whose 8x8 costs are cost8[4 q +
    // 2 dy + dx]; lane l < 16 then writes the signal of block (l % 4, l / 4).
    if (warp == 0 && lane < 16) {
        const int q{lane % 4};
        const std::size_t qbx{rbx + 2 * static_cast<std::size_t>(q % 2)};
        const std::size_t qby{rby + 2 * static_cast<std::size_t>(q / 2)};
        float sum8{0.0f};
        bool any8{false};
        for (int dy{0}; dy < 2; ++dy) {
            for (int dx{0}; dx < 2; ++dx) {
                if (qbx + dx < bw && qby + dy < bh) {
                    sum8 += cost8[4 * q + 2 * dy + dx];
                    any8 = true;
                }
            }
        }
        const bool fit16{qbx + 2 <= bw && qby + 2 <= bh};
        const float c16{cost16[q]};
        int choice{0};
        float quadrant_cost{0.0f};
        if (any8) {
            if (fit16 && c16 <= sum8 + TRANSFORM_SELECT_SPLIT_EPS) {
                choice = 16;
                quadrant_cost = c16;
            } else {
                choice = 8;
                quadrant_cost = sum8;
            }
        }
        float split_cost{0.0f};
        std::uint32_t split_layout{0};
#pragma unroll
        for (int i{0}; i < 4; ++i) {
            split_cost += __shfl_sync(0xffffu, quadrant_cost, i);
            split_layout |= static_cast<std::uint32_t>(__shfl_sync(0xffffu, choice, i)) << (8 * i);
        }
        const bool fit32{rbx + 4 <= bw && rby + 4 <= bh};
        const float cost32{
            candidate_cost(32, dist32[0] + dist32[1], rate32[0] + rate32[1], lambda)};
        const bool whole32{fit32 && cost32 <= split_cost + TRANSFORM_SELECT_SPLIT_EPS};
        const int dx{lane % 4};
        const int dy{lane / 4};
        const std::size_t bx{rbx + static_cast<std::size_t>(dx)};
        const std::size_t by{rby + static_cast<std::size_t>(dy)};
        const int side{whole32 ? 32 : quadrant_side(split_layout, 2 * (dy / 2) + dx / 2)};
        // The top-left block of a transform carries its side, the rest of its
        // footprint ACS_COVERED.
        if (side != 0 && bx < bw && by < bh) {
            const int footprint_mask{side / 8 - 1};
            acs[by * bw + bx] = (dx & footprint_mask) == 0 && (dy & footprint_mask) == 0
                                    ? static_cast<std::int8_t>(side)
                                    : ACS_COVERED;
        }
        if (lane == 0) {
            *layout_word = whole32 ? WHOLE32_LAYOUT : split_layout;
        }
    }
    __syncthreads();
    const std::uint32_t layout{*layout_word};

    if (layout == WHOLE32_LAYOUT) {
        // Gathered in rows32 for coalesced stores, four per thread.
        if (warp >= 6) {
#pragma unroll
            for (int m{0}; m < 16; ++m) {
                rows32[lane * 33 + 2 * m + part32] = c[m];
            }
        }
        __syncthreads();
        const int e{4 * tid};
        store_half4(y_plane + covered_plane_slot(32, rbx, rby, bw, static_cast<std::size_t>(e)),
                    rows32 + (e / 32) * 33 + e % 32);
    } else if (warp < 4) {
        const int rb{lane / 8};
        const int fx{lane % 8};
        const std::size_t bbx{rbx + static_cast<std::size_t>(warp)};
        const std::size_t bby{rby + static_cast<std::size_t>(rb)};
        if (quadrant_side(layout, 2 * (rb / 2) + warp / 2) == 8 && bbx < bw && bby < bh) {
            store_half8(
                y_plane + covered_plane_slot(8, bbx, bby, bw, static_cast<std::size_t>(8 * fx)), c);
        }
    } else if (warp < 6) {
        const int ch{warp - 4};
        const int rb{lane / 16};
        const int fx{lane % 16};
        if (quadrant_side(layout, 2 * rb + ch) == 16) {
            __half* dst{y_plane + covered_plane_slot(16, rbx + 2 * static_cast<std::size_t>(ch),
                                                     rby + 2 * static_cast<std::size_t>(rb), bw,
                                                     static_cast<std::size_t>(16 * fx))};
            store_half8(dst, c);
            store_half8(dst + 8, c + 8);
        }
    }
    return layout;
}

// Shared-memory index of region entry k (a horizontal-pass row entry or a
// coefficient): one padding float after every 32 entries. A pass's N-vectors
// stay contiguous, and the lanes of a warp reading 32 consecutive vectors at
// the same position hit distinct banks.
__device__ inline int region_row_index(int k) {
    return k + k / 32;
}

// Entry k of a region's X or B rows and coefficients: the transforms in
// quadrant order, so quadrant q holds k in [256 q, 256 q + 256) unless the
// region is one DCT32, DCT8 sub-block (dx, dy) of a quadrant holds the 64
// entries from 64 (2 dy + dx), and entry fx * N + i of a transform is its
// frequency fx on row i (rows) or its coefficient (fx, i).
//
// Both passes give each warp one task. A whole DCT32 gives warp w < 4 channel
// w / 2 and the even or odd frequencies (w % 2), one vector per lane. Otherwise
// warp w takes quadrant w / 2: a DCT16 puts X on lanes 0-15 and B on 16-31
// with the even or odd frequencies (w % 2); the DCT8s put column w % 2 (rows)
// or channel w % 2 (coefficients) on all lanes.

// Horizontal pass of the X and B transforms of the region into rows.
__device__ __forceinline__ void region_rows(const float* const (&sharp)[2], float* const (&rows)[2],
                                            std::uint32_t layout, std::size_t rbx, std::size_t rby,
                                            std::size_t bw, std::size_t bh, int warp, int lane) {
    if (layout == WHOLE32_LAYOUT) {
        if (warp < 4) {
            const int c{warp / 2};
            const int part{warp % 2};
            float out[16];
            dct_vector<32, 16>(sharp[c] + lane * TILE_STRIDE, part, out);
#pragma unroll
            for (int m{0}; m < 16; ++m) {
                rows[c][region_row_index((2 * m + part) * 32 + lane)] = out[m];
            }
        }
        return;
    }
    const int q{warp / 2};
    const int sub{warp % 2};
    const int n{quadrant_side(layout, q)};
    const int c{lane / 16};
    const int r{lane % 16};
    const float* src{sharp[c] + (16 * (q / 2) + r) * TILE_STRIDE + 16 * (q % 2)};
    if (n == 16) {
        float out[8];
        dct_vector<16, 8>(src, sub, out);
#pragma unroll
        for (int m{0}; m < 8; ++m) {
            rows[c][region_row_index(256 * q + (2 * m + sub) * 16 + r)] = out[m];
        }
    } else if (n == 8) {
        const std::size_t bx{rbx + static_cast<std::size_t>(2 * (q % 2) + sub)};
        const std::size_t by{rby + static_cast<std::size_t>(2 * (q / 2) + r / 8)};
        if (bx < bw && by < bh) {
            float out[8];
            dct_vector<8, 8>(src + 8 * sub, 0, out);
            const int k0{256 * q + 64 * (2 * (r / 8) + sub) + r % 8};
#pragma unroll
            for (int f{0}; f < 8; ++f) {
                rows[c][region_row_index(k0 + 8 * f)] = out[f];
            }
        }
    }
}

// Vertical pass of the X and B transforms of the region from rows into
// coefficients.
__device__ __forceinline__ void region_coeffs(const float* const (&rows)[2], std::uint32_t layout,
                                              std::size_t rbx, std::size_t rby, std::size_t bw,
                                              std::size_t bh, float* const (&coeffs)[2], int warp,
                                              int lane) {
    if (layout == WHOLE32_LAYOUT) {
        if (warp < 4) {
            const int c{warp / 2};
            const int part{warp % 2};
            float out[16];
            dct_vector<32, 16>(rows[c] + region_row_index(32 * lane), part, out);
#pragma unroll
            for (int m{0}; m < 16; ++m) {
                coeffs[c][region_row_index(32 * lane + 2 * m + part)] = out[m] * DCT2D_SCALE<32>;
            }
        }
        return;
    }
    const int q{warp / 2};
    const int sub{warp % 2};
    const int n{quadrant_side(layout, q)};
    if (n == 16) {
        const int c{lane / 16};
        const int k0{256 * q + 16 * (lane % 16)};
        float out[8];
        dct_vector<16, 8>(rows[c] + region_row_index(k0), sub, out);
#pragma unroll
        for (int m{0}; m < 8; ++m) {
            coeffs[c][region_row_index(k0 + 2 * m + sub)] = out[m] * DCT2D_SCALE<16>;
        }
    } else if (n == 8) {
        const int block{lane / 8};
        const std::size_t bx{rbx + static_cast<std::size_t>(2 * (q % 2) + block % 2)};
        const std::size_t by{rby + static_cast<std::size_t>(2 * (q / 2) + block / 2)};
        if (bx < bw && by < bh) {
            const int k0{256 * q + 8 * lane};
            float out[8];
            dct_vector<8, 8>(rows[sub] + region_row_index(k0), 0, out);
#pragma unroll
            for (int f{0}; f < 8; ++f) {
                coeffs[sub][region_row_index(k0 + f)] = out[f] * DCT2D_SCALE<8>;
            }
        }
    }
}

// Stores the region's X and B coefficients to the covered-block planes, eight
// consecutive entries per thread: channel tid / 128, entries from
// 8 (tid % 128).
__device__ __forceinline__ void store_region_coeffs(const float* const (&coeffs)[2],
                                                    std::uint32_t layout, std::size_t rbx,
                                                    std::size_t rby, std::size_t bw, std::size_t bh,
                                                    __half* const (&dst)[2], int tid) {
    const int c{tid / 128};
    const int k{8 * (tid % 128)};
    const float* src{coeffs[c] + region_row_index(k)};
    if (layout == WHOLE32_LAYOUT) {
        store_half8(dst[c] + covered_plane_slot(32, rbx, rby, bw, static_cast<std::size_t>(k)),
                    src);
        return;
    }
    const int q{k / 256};
    const int n{quadrant_side(layout, q)};
    const std::size_t qbx{rbx + 2 * static_cast<std::size_t>(q % 2)};
    const std::size_t qby{rby + 2 * static_cast<std::size_t>(q / 2)};
    if (n == 16) {
        store_half8(
            dst[c] + covered_plane_slot(16, qbx, qby, bw, static_cast<std::size_t>(k % 256)), src);
    } else if (n == 8) {
        const int block{(k % 256) / 64};
        const std::size_t bx{qbx + static_cast<std::size_t>(block % 2)};
        const std::size_t by{qby + static_cast<std::size_t>(block / 2)};
        if (bx < bw && by < bh) {
            store_half8(
                dst[c] + covered_plane_slot(8, bx, by, bw, static_cast<std::size_t>(k % 64)), src);
        }
    }
}

__global__ void __launch_bounds__(FRONTEND_THREADS, FRONTEND_BLOCKS_PER_SM)
    frontend_fused_kernel(const std::uint8_t* __restrict__ luma, std::size_t luma_pitch,
                          cudaTextureObject_t chroma_tex, std::size_t width, std::size_t height,
                          std::size_t bw, std::size_t bh, float km0, float km1, float km2,
                          float km3, float mul_pb, float add_pb, float inv_global_scale, float qgsf,
                          __half* __restrict__ coeffs, std::int8_t* __restrict__ acs,
                          std::int32_t* __restrict__ quant_field) {
    extern __shared__ __align__(16) unsigned char workspace_bytes[];
    float* workspace{reinterpret_cast<float*>(workspace_bytes)};
    float* sharp_y{workspace};
    float* sh_y{sharp_y + TILE_FLOATS};
    float* sh_pre{sh_y + PAD_SAMPLES};
    float* sh_eroded{sh_pre + CELLS * CELLS};
    float* sh_red{sh_eroded + CELLS * CELLS};
    float* raw_x{workspace + RESIDENT_X_B_OFFSET};
    float* raw_b{raw_x + PAD_SAMPLES};

    const std::size_t tbx0{static_cast<std::size_t>(blockIdx.x) * TB};
    const std::size_t tby0{static_cast<std::size_t>(blockIdx.y) * TB};
    const std::size_t px0{tbx0 * 8};
    const std::size_t py0{tby0 * 8};
    const int tid{static_cast<int>(threadIdx.x)};

    // Load the padded tile and compute XYB into shared memory.
    for (int i{tid}; i < PAD_SAMPLES; i += FRONTEND_THREADS) {
        const int lx{i % PAD};
        const int ly{i / PAD};
        long gx{static_cast<long>(px0) + lx - HALO};
        long gy{static_cast<long>(py0) + ly - HALO};
        gx = gx < 0 ? 0 : (gx >= static_cast<long>(width) ? static_cast<long>(width) - 1 : gx);
        gy = gy < 0 ? 0 : (gy >= static_cast<long>(height) ? static_cast<long>(height) - 1 : gy);
        const float yv{luma[static_cast<std::size_t>(gy) * luma_pitch + gx] * (1.0f / 255.0f)};
        const float2 c{tex2D<float2>(chroma_tex, (gx + 0.5f) * 0.5f, (gy + 0.5f) * 0.5f)};
        float xo{};
        float yo{};
        float bo{};
        nv12_pixel_to_xyb(yv, c.x, c.y, xo, yo, bo);
        sh_y[i] = yo;
        raw_x[i] = xo;
        raw_b[i] = bo;
    }
    __syncthreads();

    // Pre-erosion cells: 0.25 * sum of the 4x4 pixel diffs (Laplacian of Y).
    // Warp w takes cell row w, and lane x sums the four terms of tile column x;
    // the four lanes of a cell then combine their column sums.
    const int lane{tid % 32};
    const int warp{tid / 32};
    {
        const int lx{lane + HALO};
        float sum{0.0f};
        for (int dy{0}; dy < 4; ++dy) {
            const int ly{4 * warp + dy + HALO};
            const float yv{sh_y[ly * PAD + lx]};
            const float base{0.25f * (sh_y[(ly + 1) * PAD + lx] + sh_y[(ly - 1) * PAD + lx] +
                                      sh_y[ly * PAD + lx - 1] + sh_y[ly * PAD + lx + 1])};
            float diff{ratio_cbrt_gamma<false>(yv + MATCH_GAMMA_OFFSET) * (yv - base)};
            diff *= diff;
            diff = fminf(diff, DIFF_LIMIT);
            sum += masking_sqrt(diff);
        }
        sum += __shfl_xor_sync(0xffffffffu, sum, 1);
        sum += __shfl_xor_sync(0xffffffffu, sum, 2);
        if (lane % 4 == 0) {
            sh_pre[warp * CELLS + lane / 4] = 0.25f * sum;
        }
    }
    __syncthreads();

    // The AQ loop below does not read the eroded cells, so the barrier after it
    // also covers this step.
    erode_cells(sh_pre, sh_eroded, tid, km0, km1, km2, km3);

    // Adaptive quant integer. Every pixel modulation term is an additive
    // reduction over the block's 64 pixels, reduced as two half-blocks of four
    // pixel rows with one warp each; the eight warps take the 32 half-blocks of
    // the tile in turn. Thread lb then combines block lb's half-block sums and
    // its four eroded cells.
    for (int hb{tid / 32}; hb < 2 * TB * TB; hb += FRONTEND_THREADS / 32) {
        const int lb{hb / 2};
        const int half{hb % 2};
        const int lbx{lb % TB};
        const int lby{lb / TB};
        if (tbx0 + lbx >= bw || tby0 + lby >= bh) {
            continue;
        }
        const int fx{lane % 8};
        const int fy{4 * half + lane / 8};
        const int idx{(lby * 8 + fy + HALO) * PAD + lbx * 8 + fx + HALO};
        const float yv{sh_y[idx]};
        const float xv{raw_x[idx]};
        const float bv{raw_b[idx]};

        constexpr float valmin{0.0206f};
        float hf_part{0.0f};
        if (fx < 7) {
            hf_part += fminf(valmin, fabsf(yv - sh_y[idx + 1]));
        }
        if (fy < 7) {
            hf_part += fminf(valmin, fabsf(yv - sh_y[idx + PAD]));
        }

        constexpr float kBias{0.16f};
        const float iny{yv + kBias};
        const float gamma_part{ratio_cbrt_gamma<true>(iny - xv) + ratio_cbrt_gamma<true>(iny + xv)};

        constexpr float kLimit{0.027121074570634722f};
        constexpr float kOffset{0.084381641171960495f};
        const float pye{yv + kOffset + fabsf(xv)};
        const float blue_part{bv > pye ? fminf(bv - pye, kLimit) : 0.0f};

        const float hf_red{warp_reduce_sum(hf_part)};
        const float gamma_red{warp_reduce_sum(gamma_part)};
        const float blue_red{warp_reduce_sum(blue_part)};
        if (lane == 0) {
            float* red{sh_red + hb * 3};
            red[0] = hf_red;
            red[1] = gamma_red;
            red[2] = blue_red;
        }
    }
    __syncthreads();
    if (tid < TB * TB) {
        const int lbx{tid % TB};
        const int lby{tid / TB};
        const std::size_t gbx{tbx0 + static_cast<std::size_t>(lbx)};
        const std::size_t gby{tby0 + static_cast<std::size_t>(lby)};
        if (gbx < bw && gby < bh) {
            const float* red{sh_red + tid * 6};
            const float hf_sum{red[0] + red[3]};
            const float gamma_overall{(red[1] + red[4]) * (0.5f / 64.0f)};
            float blue_sum{red[2] + red[5]};
            const float* eroded{sh_eroded + 2 * lby * CELLS + 2 * lbx};
            const float aq{(eroded[0] + eroded[CELLS]) + (eroded[1] + eroded[CELLS + 1])};

            constexpr float kLimit{0.027121074570634722f};
            if (blue_sum >= 32.0f * kLimit) {
                blue_sum = 64.0f * kLimit - blue_sum;
            }
            constexpr float kMaxLimit{15.398788439047934f};
            if (blue_sum >= kMaxLimit * kLimit) {
                blue_sum = kMaxLimit * kLimit;
            }

            float ov{compute_mask(aq)};
            ov += hf_sum * -0.38f + 0.42f;
            ov += 0.1005613337192697f * log2f(gamma_overall);
            ov += blue_sum * 0.14207000358439159f;

            const float qval{expf(ov) * mul_pb + add_pb};
            int qi{static_cast<int>(qval * inv_global_scale + 0.5f)};
            qi = qi < 1 ? 1 : (qi > K_QUANT_MAX ? K_QUANT_MAX : qi);
            quant_field[gby * bw + gbx] = qi;
        }
    }

    gaborish_tile(sh_y, sharp_y, tid);
    __syncthreads();

    const std::size_t cplane{bw * bh * COEFFS_PER_BLOCK};
    const std::uint32_t layout{select_transforms(
        workspace, tbx0, tby0, bw, bh, qgsf, TRANSFORM_SELECT_LAMBDA, acs, coeffs + cplane, tid)};
    // Sharp B overlaps the selection rows.
    __syncthreads();

    float* sharp_x{workspace};
    float* sharp_b{workspace + SHARP_B_OFFSET};
    float* rows_b{raw_b + ROWS_B_SKEW};
    float* coeffs_b{workspace + COEFFS_B_OFFSET};
    gaborish_tile(raw_x, sharp_x, tid);
    gaborish_tile(raw_b, sharp_b, tid);
    __syncthreads();
    region_rows({sharp_x, sharp_b}, {raw_x, rows_b}, layout, tbx0, tby0, bw, bh, warp, lane);
    __syncthreads();
    region_coeffs({raw_x, rows_b}, layout, tbx0, tby0, bw, bh, {sharp_x, coeffs_b}, warp, lane);
    __syncthreads();
    store_region_coeffs({sharp_x, coeffs_b}, layout, tbx0, tby0, bw, bh,
                        {coeffs, coeffs + 2 * cplane}, tid);
}

}  // namespace

ChromaTexture::ChromaTexture(const std::uint8_t* chroma, std::size_t chroma_pitch,
                             std::size_t width, std::size_t height, cudaStream_t stream)
    : stream_{stream} {
    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypePitch2D;
    resource.res.pitch2D.devPtr = const_cast<std::uint8_t*>(chroma);
    resource.res.pitch2D.desc = cudaCreateChannelDesc<uchar2>();
    resource.res.pitch2D.width = width / 2;
    resource.res.pitch2D.height = height / 2;
    resource.res.pitch2D.pitchInBytes = chroma_pitch;

    cudaTextureDesc texture{};
    texture.addressMode[0] = cudaAddressModeClamp;
    texture.addressMode[1] = cudaAddressModeClamp;
    texture.filterMode = cudaFilterModeLinear;
    texture.readMode = cudaReadModeNormalizedFloat;
    texture.normalizedCoords = 0;

    valid_ = cudaCreateTextureObject(&texture_, &resource, &texture, nullptr) == cudaSuccess;
}

ChromaTexture::~ChromaTexture() {
    if (valid_) {
        cudaStreamSynchronize(stream_);
        cudaDestroyTextureObject(texture_);
    }
}

bool encode_frontend(const std::uint8_t* luma, std::size_t luma_pitch, const ChromaTexture& chroma,
                     std::size_t width, std::size_t height, float distance, __half* coeffs,
                     std::int8_t* acs, std::int32_t* quant_field, cudaStream_t stream) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};

    const QuantCalibration cal{calibrate_quant(distance)};
    const float scale{K_AC_QUANT / distance};
    const float qgsf{static_cast<float>(cal.raw_quant_field) * cal.global_scale_float};

    // PerBlockModulations exponent -> multiplicative field mapping.
    const float base_level{0.48f * scale};
    float dampen{1.0f};
    if (distance >= 2.0f) {
        dampen = 1.0f - (distance - 2.0f) / (14.0f - 2.0f);
        if (dampen < 0.0f) {
            dampen = 0.0f;
        }
    }
    const float mul_pb{scale * dampen};
    const float add_pb{(1.0f - dampen) * base_level};

    // FuzzyErosion weights (butteraugli-target dependent), normalized to kTotal.
    const float efm{distance < 2.0f ? (2.0f - distance) * 0.5f : 0.0f};
    float km0{0.125f};
    float km1{0.10f - efm * 0.10f};
    float km2{0.09f - efm * 0.09f};
    float km3{0.06f - efm * 0.06f};
    const float norm{0.29959705784054957f / (km0 + km1 + km2 + km3)};
    km0 *= norm;
    km1 *= norm;
    km2 *= norm;
    km3 *= norm;

    if (!chroma.valid()) {
        return false;
    }

    const dim3 block{FRONTEND_THREADS};
    const dim3 grid{static_cast<unsigned int>((bw + TB - 1) / TB),
                    static_cast<unsigned int>((bh + TB - 1) / TB)};
    frontend_fused_kernel<<<grid, block, WORKSPACE_BYTES, stream>>>(
        luma, luma_pitch, chroma.get(), width, height, bw, bh, km0, km1, km2, km3, mul_pb, add_pb,
        cal.inv_global_scale, qgsf, coeffs, acs, quant_field);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace cujpegxl
