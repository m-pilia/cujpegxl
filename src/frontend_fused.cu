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
// BASIS_FLOATS holds the 32-point basis padded the same way (dct_impl.cuh).
constexpr int TILE_STRIDE = TILE_PX + 1;
constexpr int TILE_FLOATS = TILE_PX * TILE_STRIDE;
constexpr int BASIS_FLOATS = 32 * (32 + 1);
constexpr int PAD_SAMPLES = PAD * PAD;
constexpr int FRONTEND_THREADS = 256;
constexpr std::size_t SELECT_COST_FLOATS = 16 + 4 + 1;
constexpr std::size_t SELECT_YSTORE_HALFS = 16 * 8 * 8 + 4 * 16 * 16 + 32 * 32;
constexpr std::size_t SELECT_YSTORE_FLOATS =
    (SELECT_YSTORE_HALFS * sizeof(__half) + sizeof(float) - 1) / sizeof(float);
constexpr std::size_t SELECT_WORKSPACE_PEAK = TILE_FLOATS + BASIS_FLOATS + 32 * 32 +
                                              2 * FRONTEND_THREADS + SELECT_COST_FLOATS +
                                              SELECT_YSTORE_FLOATS;
constexpr std::size_t RESIDENT_X_B_OFFSET = SELECT_WORKSPACE_PEAK;
constexpr std::size_t RESIDENT_X_B_PEAK = RESIDENT_X_B_OFFSET + 2 * PAD_SAMPLES;
constexpr std::size_t LOAD_AQ_LOCAL_PEAK = TILE_FLOATS + PAD_SAMPLES + CELLS * CELLS + 8;
constexpr std::size_t LOAD_AQ_WORKSPACE_PEAK = std::max(LOAD_AQ_LOCAL_PEAK, RESIDENT_X_B_PEAK);
constexpr std::size_t SELECT_X_B_WORKSPACE_PEAK = RESIDENT_X_B_PEAK;
constexpr std::size_t X_B_DCT_LOCAL_PEAK = TILE_FLOATS + BASIS_FLOATS + 32 * 32;
constexpr std::size_t X_B_DCT_WORKSPACE_PEAK = std::max(X_B_DCT_LOCAL_PEAK, RESIDENT_X_B_PEAK);
constexpr std::size_t WORKSPACE_FLOATS = RESIDENT_X_B_PEAK;
constexpr std::size_t WORKSPACE_BYTES = WORKSPACE_FLOATS * sizeof(float);
static_assert(LOAD_AQ_WORKSPACE_PEAK <= WORKSPACE_FLOATS);
static_assert(SELECT_X_B_WORKSPACE_PEAK <= WORKSPACE_FLOATS);
static_assert(X_B_DCT_WORKSPACE_PEAK <= WORKSPACE_FLOATS);
static_assert(WORKSPACE_FLOATS == std::max({LOAD_AQ_WORKSPACE_PEAK, SELECT_X_B_WORKSPACE_PEAK,
                                            X_B_DCT_WORKSPACE_PEAK}));

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

__device__ const float* basis_for(int n) {
    return n == 32 ? g_basis32 : (n == 16 ? g_basis16 : g_basis8);
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

// Loads the N-point basis and runs the horizontal pass of all 1024 / (N * N)
// candidates of size N into rows[idx * N * N + fx * N + r].
template <int N>
__device__ void candidate_rows(const float* __restrict__ y, float* basis, float* rows, int tid) {
    load_dct_basis(basis_for(N), basis, N, tid, FRONTEND_THREADS);
    __syncthreads();
    for (int k{tid}; k < 32 * 32; k += FRONTEND_THREADS) {
        const int e{k % (N * N)};
        int px0{};
        int py0{};
        candidate_origin<N>(k / (N * N), px0, py0);
        rows[k] = dct_rows_entry(basis, N, e / N, y + (py0 + e % N) * TILE_STRIDE + px0);
    }
    __syncthreads();
}

template <int N>
__device__ void candidate_coeff(const float* basis, const float* rows, int idx, int e, float qgsf,
                                __half* ystore, float& dist, float& rate) {
    const int fx{e / N};
    const int fy{e % N};
    const float coeff{dct_coeff_entry(basis, N, fy, rows + idx * N * N + fx * N)};
    ystore[idx * N * N + e] = __float2half(coeff);
    accumulate_coeff_cost(coeff, fx, fy, N, qgsf, dist, rate);
}

// A candidate's distortion and rate are sums over 256 slots in the pairing of
// the halving tree slot[i] += slot[i + s], s = 128..1, where coefficient e
// accumulates into slot e % 256 in increasing e. Each candidate size spreads its
// coefficients over the warps so that the leading strides fold locally.
__device__ void select_and_store_y(const float* __restrict__ y, std::size_t rbx, std::size_t rby,
                                   std::size_t bw, std::size_t bh, float qgsf, float lambda,
                                   float* basis, float* rows, float* pd, float* pr, float* costs,
                                   __half* ystore, __half* ycoeffs, std::int8_t* __restrict__ acs,
                                   int tid) {
    float* cost8{costs};
    float* cost16{cost8 + 16};
    float* cost32{cost16 + 4};
    __half* ystore8{ystore};
    __half* ystore16{ystore8 + 16 * 64};
    __half* ystore32{ystore16 + 4 * 256};
    constexpr float INF{1.0e30f};
    const int lane{tid % 32};
    const int warp{tid / 32};

    // 8x8: each warp owns two candidates; lane l holds coefficients l and l + 32
    // (slots 64..255 are empty).
    candidate_rows<8>(y, basis, rows, tid);
    for (int j{0}; j < 2; ++j) {
        const int idx{2 * warp + j};
        float d0{0.0f};
        float r0{0.0f};
        float d1{0.0f};
        float r1{0.0f};
        candidate_coeff<8>(basis, rows, idx, lane, qgsf, ystore8, d0, r0);
        candidate_coeff<8>(basis, rows, idx, lane + 32, qgsf, ystore8, d1, r1);
        const float dist{warp_reduce_sum(__fadd_rn(d0, d1))};
        const float rate{warp_reduce_sum(__fadd_rn(r0, r1))};
        if (lane == 0) {
            const std::size_t bbx{rbx + static_cast<std::size_t>(2 * ((idx / 4) % 2) + idx % 2)};
            const std::size_t bby{rby + static_cast<std::size_t>(2 * (idx / 8) + (idx / 2) % 2)};
            cost8[idx] = bbx < bw && bby < bh ? candidate_cost(8, dist, rate, lambda) : INF;
        }
    }
    __syncthreads();

    // 16x16: warps 2q and 2q + 1 own candidate q; local lane j holds
    // coefficients j + 64 * h and folds strides 128 and 64.
    candidate_rows<16>(y, basis, rows, tid);
    {
        const int q{warp / 2};
        const int j{tid % 64};
        float d[4]{};
        float r[4]{};
#pragma unroll
        for (int h{0}; h < 4; ++h) {
            candidate_coeff<16>(basis, rows, q, j + 64 * h, qgsf, ystore16, d[h], r[h]);
        }
        pd[tid] = __fadd_rn(__fadd_rn(d[0], d[2]), __fadd_rn(d[1], d[3]));
        pr[tid] = __fadd_rn(__fadd_rn(r[0], r[2]), __fadd_rn(r[1], r[3]));
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

    // 32x32: thread t holds coefficients t + 256 * h; warp 0 reduces.
    candidate_rows<32>(y, basis, rows, tid);
    float dist32{0.0f};
    float rate32{0.0f};
    for (int e{tid}; e < 32 * 32; e += FRONTEND_THREADS) {
        candidate_coeff<32>(basis, rows, 0, e, qgsf, ystore32, dist32, rate32);
    }
    pd[tid] = dist32;
    pr[tid] = rate32;
    __syncthreads();
    const bool fit32{rbx + 4 <= bw && rby + 4 <= bh};
    if (warp == 0) {
        const float dist{warp_tree_sum_256(pd, lane)};
        const float rate{warp_tree_sum_256(pr, lane)};
        if (lane == 0) {
            *cost32 = fit32 ? candidate_cost(32, dist, rate, lambda) : INF;
        }
    }

    if (tid == 0) {
        float split_cost{0.0f};
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
            }
        }

        if (fit32 && *cost32 <= split_cost + TRANSFORM_SELECT_SPLIT_EPS) {
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
    }
    __syncthreads();

    for (int p{0}; p < 16; ++p) {
        const std::size_t bx{rbx + static_cast<std::size_t>(p % 4)};
        const std::size_t by{rby + static_cast<std::size_t>(p / 4)};
        if (bx >= bw || by >= bh) {
            continue;
        }
        const int n{acs[by * bw + bx]};
        if (n == ACS_COVERED) {
            continue;
        }

        const std::size_t lx{bx - rbx};
        const std::size_t ly{by - rby};
        const __half* src{nullptr};
        if (n == 32) {
            src = ystore32;
        } else if (n == 16) {
            const std::size_t q{(ly / 2) * 2 + lx / 2};
            src = ystore16 + q * 256;
        } else {
            const std::size_t idx{((ly / 2) * 2 + lx / 2) * 4 + (ly % 2) * 2 + lx % 2};
            src = ystore8 + idx * 64;
        }
        for (int raw{tid}; raw < n * n; raw += FRONTEND_THREADS) {
            ycoeffs[covered_plane_slot(n, bx, by, bw, static_cast<std::size_t>(raw))] = src[raw];
        }
    }
    __syncthreads();
}

__device__ void forward_channel(const float* __restrict__ src, std::size_t rbx, std::size_t rby,
                                std::size_t bw, std::size_t bh, const std::int8_t* __restrict__ acs,
                                float* basis, float* rows, __half* dst, int tid) {
    int cur_basis_n{0};
    for (int p{0}; p < 16; ++p) {
        const std::size_t bx{rbx + static_cast<std::size_t>(p % 4)};
        const std::size_t by{rby + static_cast<std::size_t>(p / 4)};
        if (bx >= bw || by >= bh) {
            continue;
        }
        const int n{acs[by * bw + bx]};
        if (n == ACS_COVERED) {
            continue;
        }

        __syncthreads();
        if (n != cur_basis_n) {
            load_dct_basis(basis_for(n), basis, n, tid, FRONTEND_THREADS);
            cur_basis_n = n;
            __syncthreads();
        }

        const int px0{static_cast<int>(bx - rbx) * 8};
        const int py0{static_cast<int>(by - rby) * 8};
        for (int e{tid}; e < n * n; e += FRONTEND_THREADS) {
            rows[e] = dct_rows_entry(basis, n, e / n, src + (py0 + e % n) * TILE_STRIDE + px0);
        }
        __syncthreads();
        for (int e{tid}; e < n * n; e += FRONTEND_THREADS) {
            const float coeff{dct_coeff_entry(basis, n, e % n, rows + (e / n) * n)};
            dst[covered_plane_slot(n, bx, by, bw, static_cast<std::size_t>(e))] =
                __float2half(coeff);
        }
        __syncthreads();
    }
}

__global__ void frontend_fused_kernel(const std::uint8_t* __restrict__ luma, std::size_t luma_pitch,
                                      cudaTextureObject_t chroma_tex, std::size_t width,
                                      std::size_t height, std::size_t bw, std::size_t bh, float km0,
                                      float km1, float km2, float km3, float mul_pb, float add_pb,
                                      float inv_global_scale, float qgsf,
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

    for (int lb{0}; lb < TB * TB; ++lb) {
        const int lbx{lb % TB};
        const int lby{lb / TB};
        const std::size_t gbx{tbx0 + lbx};
        const std::size_t gby{tby0 + lby};
        if (gbx >= bw || gby >= bh) {
            continue;
        }
        const int x0{lbx * 8 + HALO};  // shared interior origin (+halo)
        const int y0{lby * 8 + HALO};

        // Adaptive quant integer: each thread reduces its own pixel's modulation
        // contributions across the 64-thread block; thread 0 combines the
        // reduced sums and broadcasts the quant integer via shared memory. Every
        // modulation term is an additive reduction over the block's 64 pixels,
        // so the per-pixel partials can be summed in parallel.
        if (tid < 64) {
            const int fx{tid % 8};
            const int fy{tid / 8};
            const int idx{(y0 + fy) * PAD + x0 + fx};
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
            const float gamma_part{ratio_cbrt_gamma<true>(iny - xv) +
                                   ratio_cbrt_gamma<true>(iny + xv)};

            constexpr float kLimit{0.027121074570634722f};
            constexpr float kOffset{0.084381641171960495f};
            const float pye{yv + kOffset + fabsf(xv)};
            const float blue_part{bv > pye ? fminf(bv - pye, kLimit) : 0.0f};

            float er_part{0.0f};
            if (tid < 4) {
                er_part = erosion_cell(sh_pre, 2 * lbx + (tid & 1), 2 * lby + (tid >> 1), km0, km1,
                                       km2, km3);
            }

            const float hf_red{warp_reduce_sum(hf_part)};
            const float gamma_red{warp_reduce_sum(gamma_part)};
            const float blue_red{warp_reduce_sum(blue_part)};
            const float er_red{warp_reduce_sum(er_part)};
            if ((tid & 31) == 0) {
                const int w{tid >> 5};
                sh_red[w * 4] = hf_red;
                sh_red[w * 4 + 1] = gamma_red;
                sh_red[w * 4 + 2] = blue_red;
                sh_red[w * 4 + 3] = er_red;
            }
        }
        __syncthreads();
        if (tid == 0) {
            const float hf_sum{sh_red[0] + sh_red[4]};
            const float gamma_overall{(sh_red[1] + sh_red[5]) * (0.5f / 64.0f)};
            float blue_sum{sh_red[2] + sh_red[6]};
            const float aq{sh_red[3] + sh_red[7]};

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
        __syncthreads();
    }

    gaborish_tile(sh_y, sharp_y, tid);
    __syncthreads();

    float* basis{sharp_y + TILE_FLOATS};
    float* rows{basis + BASIS_FLOATS};
    float* pd{rows + 32 * 32};
    float* pr{pd + FRONTEND_THREADS};
    float* costs{pr + FRONTEND_THREADS};
    __half* ystore{reinterpret_cast<__half*>(costs + SELECT_COST_FLOATS)};
    const std::size_t cplane{bw * bh * COEFFS_PER_BLOCK};
    select_and_store_y(sharp_y, tbx0, tby0, bw, bh, qgsf, TRANSFORM_SELECT_LAMBDA, basis, rows, pd,
                       pr, costs, ystore, coeffs + cplane, acs, tid);

    float* sharp{workspace};
    basis = sharp + TILE_FLOATS;
    rows = basis + BASIS_FLOATS;
    gaborish_tile(raw_x, sharp, tid);
    __syncthreads();
    forward_channel(sharp, tbx0, tby0, bw, bh, acs, basis, rows, coeffs, tid);
    gaborish_tile(raw_b, sharp, tid);
    __syncthreads();
    forward_channel(sharp, tbx0, tby0, bw, bh, acs, basis, rows, coeffs + 2 * cplane, tid);
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
