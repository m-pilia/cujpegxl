// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frontend_fused.h"

#include <algorithm>
#include <mutex>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "adaptive_quant_impl.cuh"
#include "dct_impl.cuh"
#include "gaborish.h"
#include "quant_calibration.h"
#include "quant_weights_dct8.h"
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
// The raw X and B planes stay resident from the load to the X/B gaborish at
// RESIDENT_X_B_OFFSET; the buffers local to each phase lie below them.
constexpr std::size_t LOAD_AQ_LOCAL_PEAK = TILE_FLOATS + PAD_SAMPLES + CELLS * CELLS + 8 * TB * TB;
// From the selection to the X/B DCT, the 8-, 16- and 32-point bases stay
// resident above two tile planes. The selection and the Y pass read sharp Y
// from the first and keep their horizontal-pass rows in the second, followed by
// the selection costs; the X/B pass holds sharp X and B there.
constexpr int BASIS16_OFFSET = 8 * (8 + 1);
constexpr int BASIS32_OFFSET = BASIS16_OFFSET + 16 * (16 + 1);
constexpr std::size_t BASES_OFFSET = 2 * TILE_FLOATS;
constexpr std::size_t REGION_LOCAL_PEAK = BASES_OFFSET + BASIS32_OFFSET + 32 * (32 + 1);
// The selection costs are followed by one word for the region layout.
constexpr std::size_t SELECT_COST_FLOATS = 16 + 4 + 1;
static_assert(TILE_SAMPLES + SELECT_COST_FLOATS + 1 <= TILE_FLOATS);
// The selection's partial cost sums reuse its horizontal-pass rows.
static_assert(2 * FRONTEND_THREADS <= TILE_SAMPLES);
// The X/B horizontal-pass rows reuse the raw X and B planes once gaborish has
// consumed them.
static_assert(TILE_SAMPLES <= PAD_SAMPLES);
constexpr std::size_t RESIDENT_X_B_OFFSET = std::max(LOAD_AQ_LOCAL_PEAK, REGION_LOCAL_PEAK);
constexpr std::size_t WORKSPACE_FLOATS = RESIDENT_X_B_OFFSET + 2 * PAD_SAMPLES;
constexpr std::size_t WORKSPACE_BYTES = WORKSPACE_FLOATS * sizeof(float);
// Four resident blocks per SM: sm_61 has 96 KB of shared memory per SM, and
// FRONTEND_BLOCKS_PER_SM * FRONTEND_THREADS threads leave 64 registers each.
constexpr int FRONTEND_BLOCKS_PER_SM = 4;
static_assert(FRONTEND_BLOCKS_PER_SM * WORKSPACE_BYTES <= 96 * 1024);

__device__ float g_basis8[8 * 8];
__device__ float g_basis16[16 * 16];
__device__ float g_basis32[32 * 32];

__device__ void fill_basis(float* basis, int n, int i) {
    if (i < n * n) {
        basis[i] = dct_basis_entry(n, i / n, i % n);
    }
}

__global__ void init_frontend_basis_kernel() {
    const int i{static_cast<int>(threadIdx.x)};
    fill_basis(g_basis8, 8, i);
    fill_basis(g_basis16, 16, i);
    fill_basis(g_basis32, 32, i);
}

void ensure_frontend_basis() {
    static std::once_flag flag;
    std::call_once(flag, [] {
        init_frontend_basis_kernel<<<1, 32 * 32>>>();
        cudaDeviceSynchronize();
    });
}

__device__ void load_bases(float* bases, int tid) {
    load_dct_basis(g_basis8, bases, 8, tid, FRONTEND_THREADS);
    load_dct_basis(g_basis16, bases + BASIS16_OFFSET, 16, tid, FRONTEND_THREADS);
    load_dct_basis(g_basis32, bases + BASIS32_OFFSET, 32, tid, FRONTEND_THREADS);
}

__device__ inline const float* region_basis(const float* bases, int n) {
    return bases + (n == 32 ? BASIS32_OFFSET : (n == 16 ? BASIS16_OFFSET : 0));
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

// Sum of the 256 partials v[0..255] in the pairing of the halving tree
// v[i] += v[i + s] for s = 128..1, evaluated by one warp; valid on lane 0.
// Lane l folds strides 128, 64 and 32 over v[l + 32 * k] locally.
__device__ float warp_tree_sum_256(const float* __restrict__ v, int lane) {
    const float w0{__fadd_rn(v[lane], v[lane + 128])};
    const float w1{__fadd_rn(v[lane + 32], v[lane + 160])};
    const float w2{__fadd_rn(v[lane + 64], v[lane + 192])};
    const float w3{__fadd_rn(v[lane + 96], v[lane + 224])};
    return warp_reduce_sum(__fadd_rn(__fadd_rn(w0, w2), __fadd_rn(w1, w3)));
}

// FuzzyErosion at pre-erosion cell (cx, cy): weighted sum of the four smallest
// values in its 3x3 neighbourhood, clamped to the tile (the seam approximation).
__device__ float erosion_cell(const float* __restrict__ pre, int cx, int cy, float km0, float km1,
                              float km2, float km3) {
    const int xm1{cx > 0 ? cx - 1 : cx};
    const int xp1{cx + 1 < CELLS ? cx + 1 : cx};
    const int ym1{cy > 0 ? cy - 1 : cy};
    const int yp1{cy + 1 < CELLS ? cy + 1 : cy};
    float m0{pre[cy * CELLS + cx]};
    float m1{pre[cy * CELLS + xm1]};
    float m2{pre[cy * CELLS + xp1]};
    float m3{pre[ym1 * CELLS + xm1]};
    sort4(m0, m1, m2, m3);
    store_min4(pre[ym1 * CELLS + cx], m0, m1, m2, m3);
    store_min4(pre[ym1 * CELLS + xp1], m0, m1, m2, m3);
    store_min4(pre[yp1 * CELLS + xm1], m0, m1, m2, m3);
    store_min4(pre[yp1 * CELLS + cx], m0, m1, m2, m3);
    store_min4(pre[yp1 * CELLS + xp1], m0, m1, m2, m3);
    return km0 * m0 + km1 * m1 + km2 * m2 + km3 * m3;
}

__device__ void gaborish_tile(const float* __restrict__ in, float* __restrict__ out, int tid) {
    for (int i{tid}; i < TILE_SAMPLES; i += FRONTEND_THREADS) {
        const int x{i % TILE_PX + HALO};
        const int y{i / TILE_PX + HALO};
        const float e{in[y * PAD + x - 1] + in[y * PAD + x + 1] + in[(y - 1) * PAD + x] +
                      in[(y + 1) * PAD + x]};
        const float r2{in[y * PAD + x - 2] + in[y * PAD + x + 2] + in[(y - 2) * PAD + x] +
                       in[(y + 2) * PAD + x]};
        const float dg{in[(y - 1) * PAD + x - 1] + in[(y - 1) * PAD + x + 1] +
                       in[(y + 1) * PAD + x - 1] + in[(y + 1) * PAD + x + 1]};
        const float l8{in[(y - 1) * PAD + x - 2] + in[(y - 1) * PAD + x + 2] +
                       in[(y + 1) * PAD + x - 2] + in[(y + 1) * PAD + x + 2] +
                       in[(y - 2) * PAD + x - 1] + in[(y - 2) * PAD + x + 1] +
                       in[(y + 2) * PAD + x - 1] + in[(y + 2) * PAD + x + 1]};
        const float d4{in[(y - 2) * PAD + x - 2] + in[(y - 2) * PAD + x + 2] +
                       in[(y + 2) * PAD + x - 2] + in[(y + 2) * PAD + x + 2]};
        out[(y - HALO) * TILE_STRIDE + x - HALO] = GAB_WC * in[y * PAD + x] + GAB_WR * e +
                                                   GAB_WR2 * r2 + GAB_WD * dg + GAB_WL * l8 +
                                                   GAB_WD2 * d4;
    }
}

// Rate-distortion contribution of one Y candidate coefficient quantized with the
// nearest DCT8 weight; the LLF corner is excluded.
__device__ inline void accumulate_coeff_cost(float coeff, int fx, int fy, int n, float qgsf,
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
    const float w{static_cast<float>(DCT8_DEQUANT_WEIGHTS[1][sx * 8 + sy])};
    const float q{rintf(coeff * qgsf / w)};
    const float err{q * w / qgsf - coeff};
    dist += err * err;
    if (q != 0.0f) {
        rate += 1.0f + log2f(1.0f + fabsf(q));
    }
}

__device__ inline float candidate_cost(int n, float dist, float rate, float lambda) {
    return static_cast<float>(n * n) * dist + lambda * rate;
}

// Tile pixel origin of candidate idx of size N. The sixteen 8x8 candidates are
// ordered by 16x16 quadrant, then by position within the quadrant.
template <int N>
__device__ inline void candidate_origin(int idx, int& px0, int& py0) {
    if (N == 8) {
        px0 = 16 * ((idx / 4) % 2) + 8 * (idx % 2);
        py0 = 16 * (idx / 8) + 8 * ((idx / 2) % 2);
    } else if (N == 16) {
        px0 = 16 * (idx % 2);
        py0 = 16 * (idx / 2);
    } else {
        px0 = 0;
        py0 = 0;
    }
}

constexpr int SELECT_ENTRIES_PER_THREAD{32 * 32 / FRONTEND_THREADS};

// Horizontal pass of all 1024 / (N * N) candidates of size N into
// rows[idx * N * N + fx * N + r]. Thread t computes the entries
// k = t + 256 j: for N = 32 they share the pixel row, otherwise they share fx
// across four candidates.
template <int N>
__device__ void candidate_rows(const float* __restrict__ y, const float* basis, float* rows,
                               int tid) {
    const float* basis_rows[SELECT_ENTRIES_PER_THREAD];
    const float* pixel_rows[SELECT_ENTRIES_PER_THREAD];
#pragma unroll
    for (int j{0}; j < SELECT_ENTRIES_PER_THREAD; ++j) {
        int fx{};
        int r{};
        int idx{};
        if (N == 32) {
            fx = tid / 32 + 8 * j;
            r = tid % 32;
        } else {
            const int e{tid % (N * N)};
            fx = e / N;
            r = e % N;
            idx = tid / (N * N) + j * (FRONTEND_THREADS / (N * N));
        }
        int px0{};
        int py0{};
        candidate_origin<N>(idx, px0, py0);
        basis_rows[j] = dct_basis_row(basis, N, fx);
        pixel_rows[j] = y + (py0 + r) * TILE_STRIDE + px0;
    }
    float entries[SELECT_ENTRIES_PER_THREAD];
    dct_dots<SELECT_ENTRIES_PER_THREAD>(N, basis_rows, pixel_rows, entries);
#pragma unroll
    for (int j{0}; j < SELECT_ENTRIES_PER_THREAD; ++j) {
        rows[tid + FRONTEND_THREADS * j] = entries[j];
    }
    __syncthreads();
}

// Vertical pass of the coefficients (fx + h * rows_step / N, fy) of one thread,
// h = 0..3, from the horizontal-pass rows rows_fx of frequency fx. They share
// the basis row of fy.
template <int N>
__device__ void candidate_coeffs(const float* basis, const float* rows_fx, int rows_step, int fy,
                                 float (&coeffs)[SELECT_ENTRIES_PER_THREAD]) {
    const float* basis_rows[SELECT_ENTRIES_PER_THREAD];
    const float* vectors[SELECT_ENTRIES_PER_THREAD];
#pragma unroll
    for (int h{0}; h < SELECT_ENTRIES_PER_THREAD; ++h) {
        basis_rows[h] = dct_basis_row(basis, N, fy);
        vectors[h] = rows_fx + h * rows_step;
    }
    dct_dots<SELECT_ENTRIES_PER_THREAD>(N, basis_rows, vectors, coeffs);
}

constexpr std::uint32_t WHOLE32_LAYOUT{0x20202020u};

// A candidate's distortion and rate are sums over 256 slots in the pairing of
// the halving tree slot[i] += slot[i + s], s = 128..1, where coefficient e
// accumulates into slot e % 256 in increasing e. Each candidate size spreads its
// coefficients over the warps so that the leading strides fold locally.
//
// Returns the region layout: byte q holds the side (8, 16 or 32) of the
// transforms covering quadrant q, or 0 if the quadrant lies outside the image.
// A region that fits a DCT32 gets its DCT32 Y coefficients written to y_plane
// speculatively; if it splits, the Y pass overwrites the same slots.
__device__ std::uint32_t select_transforms(const float* __restrict__ y, std::size_t rbx,
                                           std::size_t rby, std::size_t bw, std::size_t bh,
                                           float qgsf, float lambda, const float* bases,
                                           float* rows, float* costs, std::int8_t* __restrict__ acs,
                                           __half* __restrict__ y_plane, int tid) {
    const float* basis8{region_basis(bases, 8)};
    const float* basis16{region_basis(bases, 16)};
    const float* basis32{region_basis(bases, 32)};
    float* pd{rows};
    float* pr{rows + FRONTEND_THREADS};
    float* cost8{costs};
    float* cost16{cost8 + 16};
    float* cost32{cost16 + 4};
    std::uint32_t* layout_word{reinterpret_cast<std::uint32_t*>(cost32 + 1)};
    constexpr float INF{1.0e30f};
    const int lane{tid % 32};
    const int warp{tid / 32};

    // 8x8: each warp owns two candidates; lane l holds coefficients l and l + 32
    // of each (slots 64..255 are empty).
    candidate_rows<8>(y, basis8, rows, tid);
    {
        const int fx{lane / 8};
        const int fy{lane % 8};
        float coeffs[SELECT_ENTRIES_PER_THREAD];
        candidate_coeffs<8>(basis8, rows + 2 * warp * 64 + fx * 8, 32, fy, coeffs);
#pragma unroll
        for (int j{0}; j < 2; ++j) {
            const int idx{2 * warp + j};
            float d0{0.0f};
            float r0{0.0f};
            float d1{0.0f};
            float r1{0.0f};
            accumulate_coeff_cost(coeffs[2 * j], fx, fy, 8, qgsf, d0, r0);
            accumulate_coeff_cost(coeffs[2 * j + 1], fx + 4, fy, 8, qgsf, d1, r1);
            const float dist{warp_reduce_sum(__fadd_rn(d0, d1))};
            const float rate{warp_reduce_sum(__fadd_rn(r0, r1))};
            if (lane == 0) {
                const std::size_t bbx{rbx +
                                      static_cast<std::size_t>(2 * ((idx / 4) % 2) + idx % 2)};
                const std::size_t bby{rby +
                                      static_cast<std::size_t>(2 * (idx / 8) + (idx / 2) % 2)};
                cost8[idx] = bbx < bw && bby < bh ? candidate_cost(8, dist, rate, lambda) : INF;
            }
        }
    }
    __syncthreads();

    // 16x16: warps 2q and 2q + 1 own candidate q; local lane j holds
    // coefficients j + 64 * h and folds strides 128 and 64.
    candidate_rows<16>(y, basis16, rows, tid);
    {
        const int q{warp / 2};
        const int j{tid % 64};
        const int fx{j / 16};
        const int fy{j % 16};
        float coeffs[SELECT_ENTRIES_PER_THREAD];
        candidate_coeffs<16>(basis16, rows + q * 256 + fx * 16, 64, fy, coeffs);
        float d[4]{};
        float r[4]{};
#pragma unroll
        for (int h{0}; h < 4; ++h) {
            accumulate_coeff_cost(coeffs[h], fx + 4 * h, fy, 16, qgsf, d[h], r[h]);
        }
        const float dist{__fadd_rn(__fadd_rn(d[0], d[2]), __fadd_rn(d[1], d[3]))};
        const float rate{__fadd_rn(__fadd_rn(r[0], r[2]), __fadd_rn(r[1], r[3]))};
        __syncthreads();
        pd[tid] = dist;
        pr[tid] = rate;
    }
    __syncthreads();
    if (warp % 2 == 0) {
        const int q{warp / 2};
        const float dist{warp_reduce_sum(__fadd_rn(pd[tid], pd[tid + 32]))};
        const float rate{warp_reduce_sum(__fadd_rn(pr[tid], pr[tid + 32]))};
        if (lane == 0) {
            const std::size_t bx{rbx + 2 * static_cast<std::size_t>(q % 2)};
            const std::size_t by{rby + 2 * static_cast<std::size_t>(q / 2)};
            const bool fit16{bx + 2 <= bw && by + 2 <= bh};
            cost16[q] = fit16 ? candidate_cost(16, dist, rate, lambda) : INF;
        }
    }

    // 32x32: thread t holds coefficients t + 256 * h; warp 0 reduces. The
    // barrier keeps the 16x16 reduction's reads of pd/pr ahead of the rows.
    __syncthreads();
    candidate_rows<32>(y, basis32, rows, tid);
    const bool fit32{rbx + 4 <= bw && rby + 4 <= bh};
    float dist32{0.0f};
    float rate32{0.0f};
    {
        const int fx{tid / 32};
        const int fy{tid % 32};
        float coeffs[SELECT_ENTRIES_PER_THREAD];
        candidate_coeffs<32>(basis32, rows + fx * 32, 256, fy, coeffs);
#pragma unroll
        for (int h{0}; h < SELECT_ENTRIES_PER_THREAD; ++h) {
            accumulate_coeff_cost(coeffs[h], fx + 8 * h, fy, 32, qgsf, dist32, rate32);
            if (fit32) {
                const int e{tid + FRONTEND_THREADS * h};
                y_plane[covered_plane_slot(32, rbx, rby, bw, static_cast<std::size_t>(e))] =
                    __float2half_rn(coeffs[h]);
            }
        }
    }
    __syncthreads();
    pd[tid] = dist32;
    pr[tid] = rate32;
    __syncthreads();
    if (warp == 0) {
        const float dist{warp_tree_sum_256(pd, lane)};
        const float rate{warp_tree_sum_256(pr, lane)};
        if (lane == 0) {
            *cost32 = fit32 ? candidate_cost(32, dist, rate, lambda) : INF;
        }
    }

    if (tid == 0) {
        float split_cost{0.0f};
        std::uint32_t split_layout{0};
        int quad_choice[2][2];
        for (int qy{0}; qy < 2; ++qy) {
            for (int qx{0}; qx < 2; ++qx) {
                const std::size_t bx{rbx + 2 * static_cast<std::size_t>(qx)};
                const std::size_t by{rby + 2 * static_cast<std::size_t>(qy)};
                float sum8{0.0f};
                bool any8{false};
                for (int dy{0}; dy < 2; ++dy) {
                    for (int dx{0}; dx < 2; ++dx) {
                        const std::size_t bbx{bx + dx};
                        const std::size_t bby{by + dy};
                        if (bbx < bw && bby < bh) {
                            sum8 += cost8[qy * 8 + qx * 4 + dy * 2 + dx];
                            any8 = true;
                        }
                    }
                }
                if (!any8) {
                    quad_choice[qy][qx] = -1;
                    continue;
                }
                const bool fit16{bx + 2 <= bw && by + 2 <= bh};
                const float c16{cost16[qy * 2 + qx]};
                if (fit16 && c16 <= sum8 + TRANSFORM_SELECT_SPLIT_EPS) {
                    quad_choice[qy][qx] = 16;
                    split_cost += c16;
                } else {
                    quad_choice[qy][qx] = 8;
                    split_cost += sum8;
                }
                split_layout |= static_cast<std::uint32_t>(quad_choice[qy][qx])
                                << (8 * (qy * 2 + qx));
            }
        }

        const bool whole32{fit32 && *cost32 <= split_cost + TRANSFORM_SELECT_SPLIT_EPS};
        if (whole32) {
            write_first_block(acs, bw, rbx, rby, 32);
        } else {
            for (int qy{0}; qy < 2; ++qy) {
                for (int qx{0}; qx < 2; ++qx) {
                    const int choice{quad_choice[qy][qx]};
                    if (choice < 0) {
                        continue;
                    }
                    const std::size_t bx{rbx + 2 * static_cast<std::size_t>(qx)};
                    const std::size_t by{rby + 2 * static_cast<std::size_t>(qy)};
                    if (choice == 16) {
                        write_first_block(acs, bw, bx, by, 16);
                        continue;
                    }
                    for (int dy{0}; dy < 2; ++dy) {
                        for (int dx{0}; dx < 2; ++dx) {
                            const std::size_t bbx{bx + dx};
                            const std::size_t bby{by + dy};
                            if (bbx < bw && bby < bh) {
                                acs[bby * bw + bbx] = 8;
                            }
                        }
                    }
                }
            }
        }
        *layout_word = whole32 ? WHOLE32_LAYOUT : split_layout;
    }
    __syncthreads();
    return *layout_word;
}

// Transform of a region (one CUDA block's TB x TB blocks) that owns region
// coefficient k. The region coefficients concatenate the transforms in quadrant
// order, so quadrant q holds k in [256 q, 256 q + 256) unless the region is one
// DCT32; within a quadrant split into 8x8 transforms, sub-block (dx, dy) holds
// the 64 entries from 64 * (2 dy + dx). n == 0 marks a k outside the image.
// `layout` is the region layout returned by select_transforms.
struct RegionTransform {
    int n;
    int first_k;
    int px0;
    int py0;
    std::size_t bx;
    std::size_t by;
};

__device__ RegionTransform region_transform(int k, std::uint32_t layout, std::size_t rbx,
                                            std::size_t rby, std::size_t bw, std::size_t bh) {
    const int q{k / 256};
    const int n{static_cast<int>((layout >> (8 * q)) & 0xffu)};
    if (n == 32) {
        return {32, 0, 0, 0, rbx, rby};
    }
    if (n == 0) {
        return {};
    }
    const int qpx{16 * (q % 2)};
    const int qpy{16 * (q / 2)};
    const std::size_t qbx{rbx + static_cast<std::size_t>(qpx / 8)};
    const std::size_t qby{rby + static_cast<std::size_t>(qpy / 8)};
    if (n == 16) {
        return {16, 256 * q, qpx, qpy, qbx, qby};
    }
    const int sub{(k % 256) / 64};
    const std::size_t bx{qbx + static_cast<std::size_t>(sub % 2)};
    const std::size_t by{qby + static_cast<std::size_t>(sub / 2)};
    if (bx >= bw || by >= bh) {
        return {};
    }
    return {8, k - k % 64, qpx + 8 * (sub % 2), qpy + 8 * (sub / 2), bx, by};
}

// DCT of all transforms of the region for CHANNELS planes at once. Each warp's
// consecutive k fall within one transform. The vertical pass gives each thread
// the pair k, k + 1 (k even): both share fx and land in adjacent plane slots.
template <int CHANNELS>
__device__ void forward_region(const float* const (&sharp)[CHANNELS], const float* bases,
                               float* const (&rows)[CHANNELS], std::uint32_t layout,
                               std::size_t rbx, std::size_t rby, std::size_t bw, std::size_t bh,
                               __half* const (&dst)[CHANNELS], int tid) {
    for (int k{tid}; k < TILE_SAMPLES; k += FRONTEND_THREADS) {
        const RegionTransform t{region_transform(k, layout, rbx, rby, bw, bh)};
        if (t.n == 0) {
            continue;
        }
        const int e{k - t.first_k};
        const float* basis_row{dct_basis_row(region_basis(bases, t.n), t.n, e / t.n)};
        const int pixel_row{(t.py0 + e % t.n) * TILE_STRIDE + t.px0};
#pragma unroll
        for (int c{0}; c < CHANNELS; ++c) {
            float row[1];
            dct_dots<1>(t.n, {basis_row}, {sharp[c] + pixel_row}, row);
            rows[c][k] = row[0];
        }
    }
    __syncthreads();
    for (int k{2 * tid}; k < TILE_SAMPLES; k += 2 * FRONTEND_THREADS) {
        const RegionTransform t{region_transform(k, layout, rbx, rby, bw, bh)};
        if (t.n == 0) {
            continue;
        }
        const float* basis{region_basis(bases, t.n)};
        const int e{k - t.first_k};
        const int rows_fx{t.first_k + (e / t.n) * t.n};
        const float* basis_row0{dct_basis_row(basis, t.n, e % t.n)};
        const float* basis_row1{dct_basis_row(basis, t.n, e % t.n + 1)};
        const std::size_t slot{
            covered_plane_slot(t.n, t.bx, t.by, bw, static_cast<std::size_t>(e))};
#pragma unroll
        for (int c{0}; c < CHANNELS; ++c) {
            const float* rows_c{rows[c] + rows_fx};
            float coeff[2];
            dct_dots<2>(t.n, {basis_row0, basis_row1}, {rows_c, rows_c}, coeff);
            *reinterpret_cast<__half2*>(dst[c] + slot) = __floats2half2_rn(coeff[0], coeff[1]);
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
    float* sh_red{sh_pre + CELLS * CELLS};
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
    for (int ci{tid}; ci < CELLS * CELLS; ci += FRONTEND_THREADS) {
        const int ccx{ci % CELLS};
        const int ccy{ci / CELLS};
        float sum{0.0f};
        for (int dy{0}; dy < 4; ++dy) {
            for (int dx{0}; dx < 4; ++dx) {
                const int lx{ccx * 4 + dx + HALO};
                const int ly{ccy * 4 + dy + HALO};
                const float yv{sh_y[ly * PAD + lx]};
                const float base{0.25f * (sh_y[(ly + 1) * PAD + lx] + sh_y[(ly - 1) * PAD + lx] +
                                          sh_y[ly * PAD + lx - 1] + sh_y[ly * PAD + lx + 1])};
                float diff{ratio_cbrt_gamma<false>(yv + MATCH_GAMMA_OFFSET) * (yv - base)};
                diff *= diff;
                diff = fminf(diff, DIFF_LIMIT);
                sum += masking_sqrt(diff);
            }
        }
        sh_pre[ci] = 0.25f * sum;
    }
    __syncthreads();

    // Adaptive quant integer. Every modulation term is an additive reduction
    // over the block's 64 pixels, reduced as two half-blocks of four pixel rows
    // with one warp each; the eight warps take the 32 half-blocks of the tile in
    // turn. Thread lb then combines block lb's half-block sums.
    const int lane{tid % 32};
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

        float er_part{0.0f};
        if (half == 0 && lane < 4) {
            er_part = erosion_cell(sh_pre, 2 * lbx + (lane & 1), 2 * lby + (lane >> 1), km0, km1,
                                   km2, km3);
        }

        const float hf_red{warp_reduce_sum(hf_part)};
        const float gamma_red{warp_reduce_sum(gamma_part)};
        const float blue_red{warp_reduce_sum(blue_part)};
        const float er_red{warp_reduce_sum(er_part)};
        if (lane == 0) {
            float* red{sh_red + hb * 4};
            red[0] = hf_red;
            red[1] = gamma_red;
            red[2] = blue_red;
            red[3] = er_red;
        }
    }
    __syncthreads();
    if (tid < TB * TB) {
        const std::size_t gbx{tbx0 + static_cast<std::size_t>(tid % TB)};
        const std::size_t gby{tby0 + static_cast<std::size_t>(tid / TB)};
        if (gbx < bw && gby < bh) {
            const float* red{sh_red + tid * 8};
            const float hf_sum{red[0] + red[4]};
            const float gamma_overall{(red[1] + red[5]) * (0.5f / 64.0f)};
            float blue_sum{red[2] + red[6]};
            const float aq{red[3] + red[7]};

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

    float* rows{sharp_y + TILE_FLOATS};
    float* costs{rows + TILE_SAMPLES};
    float* bases{workspace + BASES_OFFSET};
    load_bases(bases, tid);
    __syncthreads();
    const std::size_t cplane{bw * bh * COEFFS_PER_BLOCK};
    const std::uint32_t layout{select_transforms(sharp_y, tbx0, tby0, bw, bh, qgsf,
                                                 TRANSFORM_SELECT_LAMBDA, bases, rows, costs, acs,
                                                 coeffs + cplane, tid)};
    if (layout != WHOLE32_LAYOUT) {
        forward_region<1>({sharp_y}, bases, {rows}, layout, tbx0, tby0, bw, bh, {coeffs + cplane},
                          tid);
    }
    __syncthreads();

    float* sharp_x{workspace};
    float* sharp_b{sharp_x + TILE_FLOATS};
    gaborish_tile(raw_x, sharp_x, tid);
    gaborish_tile(raw_b, sharp_b, tid);
    __syncthreads();
    forward_region<2>({sharp_x, sharp_b}, bases, {raw_x, raw_b}, layout, tbx0, tby0, bw, bh,
                      {coeffs, coeffs + 2 * cplane}, tid);
}

}  // namespace

bool encode_frontend(const std::uint8_t* luma, std::size_t luma_pitch, const std::uint8_t* chroma,
                     std::size_t chroma_pitch, std::size_t width, std::size_t height,
                     float distance, __half* coeffs, std::int8_t* acs, std::int32_t* quant_field) {
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

    cudaTextureObject_t chroma_tex{};
    if (cudaCreateTextureObject(&chroma_tex, &resource, &texture, nullptr) != cudaSuccess) {
        return false;
    }

    ensure_frontend_basis();
    const dim3 block{FRONTEND_THREADS};
    const dim3 grid{static_cast<unsigned int>((bw + TB - 1) / TB),
                    static_cast<unsigned int>((bh + TB - 1) / TB)};
    frontend_fused_kernel<<<grid, block, WORKSPACE_BYTES>>>(
        luma, luma_pitch, chroma_tex, width, height, bw, bh, km0, km1, km2, km3, mul_pb, add_pb,
        cal.inv_global_scale, qgsf, coeffs, acs, quant_field);

    const cudaError_t launch{cudaGetLastError()};
    const cudaError_t sync{cudaDeviceSynchronize()};
    cudaDestroyTextureObject(chroma_tex);
    return launch == cudaSuccess && sync == cudaSuccess;
}

}  // namespace cujpegxl
