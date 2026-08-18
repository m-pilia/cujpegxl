// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_TRANSFORM_SELECT_IMPL_CUH_
#define CUJPEGXL_SRC_TRANSFORM_SELECT_IMPL_CUH_

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "quant_weights_dct8.h"
#include "vardct_layout.h"

namespace cujpegxl {

// Provisional Lagrange weight (pixel-domain SSE per estimated bit) for the
// selection RD proxy. Chosen so smooth regions coalesce into large blocks while
// localized detail splits; it is not yet calibrated against coded size/quality.
inline constexpr float TRANSFORM_SELECT_LAMBDA = 6.0e-4f;

// Minimum proxy-cost reduction required to prefer a split over the larger block.
// A flat block's AC is analytically zero but leaves a small single-precision
// residual; this floor keeps such numerical ties on the larger block while
// staying far below any real split benefit (region costs are O(1) and up).
inline constexpr float TRANSFORM_SELECT_SPLIT_EPS = 1.0e-6f;

// Threads per cooperative selection block. The device kernel and the host
// reference both accumulate the per-block dist/rate over this many partials and
// reduce them with the same power-of-two tree, so their float costs stay
// bit-identical regardless of which coefficient lands on which thread.
inline constexpr int TRANSFORM_SELECT_THREADS = 256;

#if defined(__CUDACC__)
#define CUJPEGXL_HD __host__ __device__
#else
#define CUJPEGXL_HD
#endif

// AC rate/distortion proxy cost of encoding one NxN Y block at (px0, py0) with a
// uniform-field quantizer of strength `qgsf` (= raw_quant_field * global_scale).
// Candidate coefficients are quantized with the DCT8 Y perceptual weights sampled
// at the block's normalized frequency (exact for N=8); the low-frequency MxM
// coefficients (M=N/8, carried as DC) are excluded. Distortion is normalized to
// the pixel domain (forward_dctN scales coefficients by 1/N per axis).
// The DCT basis is derived in double (matching the device-precomputed table) and
// narrowed to float; the row/column matmuls and the dist/rate accumulation then
// run in single precision so this stays byte-identical to the cooperative device
// kernel, which sums coefficients in the same order.
CUJPEGXL_HD inline float block_ac_cost(const float* y, std::size_t width, std::size_t px0,
                                       std::size_t py0, int n, float qgsf, float lambda) {
    float basis[32 * 32];
    for (int k{0}; k < n; ++k) {
        const double g{k == 0 ? 1.0 / n : 1.4142135623730950488 / n};
        for (int t{0}; t < n; ++t) {
            basis[k * n + t] =
                static_cast<float>(g * cos(3.14159265358979323846 * (t + 0.5) * k / n));
        }
    }

    // Separable row pass: rows[fx*n + r] = sum_c basis[fx][c] * pixel[r][c].
    float rows[32 * 32];
    for (int fx{0}; fx < n; ++fx) {
        for (int r{0}; r < n; ++r) {
            float s{0.0f};
            const float* prow{y + (py0 + r) * width + px0};
            for (int c{0}; c < n; ++c) {
                s += basis[fx * n + c] * prow[c];
            }
            rows[fx * n + r] = s;
        }
    }

    // Column pass fused with quantization, accumulated into per-thread partials in
    // the same strided coefficient assignment (e -> e % THREADS) and reduced with
    // the same tree as the cooperative device kernel, so the float cost matches
    // bit for bit.
    const int m{n / 8};
    float pd[TRANSFORM_SELECT_THREADS]{};
    float pr[TRANSFORM_SELECT_THREADS]{};
    for (int t{0}; t < TRANSFORM_SELECT_THREADS; ++t) {
        float dist{0.0f};
        float rate{0.0f};
        for (int e{t}; e < n * n; e += TRANSFORM_SELECT_THREADS) {
            const int fx{e / n};
            const int fy{e % n};
            if (fx < m && fy < m) {
                continue;  // low-frequency block carried as DC
            }
            float coeff{0.0f};
            for (int r{0}; r < n; ++r) {
                coeff += basis[fy * n + r] * rows[fx * n + r];
            }
            int sx{static_cast<int>(lround(fx * 8.0 / n))};
            int sy{static_cast<int>(lround(fy * 8.0 / n))};
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
        pd[t] = dist;
        pr[t] = rate;
    }
    for (int s{TRANSFORM_SELECT_THREADS / 2}; s > 0; s >>= 1) {
        for (int t{0}; t < s; ++t) {
            pd[t] += pd[t + s];
            pr[t] += pr[t + s];
        }
    }
    return static_cast<float>(n * n) * pd[0] + lambda * pr[0];
}

CUJPEGXL_HD inline void write_first_block(std::int8_t* acs, std::size_t bw, std::size_t bx,
                                          std::size_t by, int n) {
    const int side{n / 8};
    for (int dy{0}; dy < side; ++dy) {
        for (int dx{0}; dx < side; ++dx) {
            acs[(by + dy) * bw + (bx + dx)] =
                (dy == 0 && dx == 0) ? static_cast<std::int8_t>(n) : ACS_COVERED;
        }
    }
}

// Decides one 32x32 region (block origin rbx, rby) and writes its ACS entries.
// Blocks outside [0,bw)x[0,bh) do not exist and are left untouched.
CUJPEGXL_HD inline void decide_region(const float* y, std::size_t width, std::size_t bw,
                                      std::size_t bh, std::size_t rbx, std::size_t rby, float qgsf,
                                      float lambda, std::int8_t* acs) {
    constexpr float INF{1.0e30f};

    const bool fit32{rbx + 4 <= bw && rby + 4 <= bh};
    const float cost32{fit32 ? block_ac_cost(y, width, rbx * 8, rby * 8, 32, qgsf, lambda) : INF};

    float split_cost{0.0f};
    int quad_choice[2][2];  // 16, 8, or -1 for an empty (fully out-of-image) quad
    for (int qy{0}; qy < 2; ++qy) {
        for (int qx{0}; qx < 2; ++qx) {
            const std::size_t bx{rbx + 2 * qx};
            const std::size_t by{rby + 2 * qy};
            float sum8{0.0f};
            bool any8{false};
            for (int dy{0}; dy < 2; ++dy) {
                for (int dx{0}; dx < 2; ++dx) {
                    const std::size_t bbx{bx + dx};
                    const std::size_t bby{by + dy};
                    if (bbx < bw && bby < bh) {
                        sum8 += block_ac_cost(y, width, bbx * 8, bby * 8, 8, qgsf, lambda);
                        any8 = true;
                    }
                }
            }
            if (!any8) {
                quad_choice[qy][qx] = -1;
                continue;
            }
            const bool fit16{bx + 2 <= bw && by + 2 <= bh};
            const float cost16{fit16 ? block_ac_cost(y, width, bx * 8, by * 8, 16, qgsf, lambda)
                                     : INF};
            // Keep the larger block unless the split strictly reduces cost.
            if (fit16 && cost16 <= sum8 + TRANSFORM_SELECT_SPLIT_EPS) {
                quad_choice[qy][qx] = 16;
                split_cost += cost16;
            } else {
                quad_choice[qy][qx] = 8;
                split_cost += sum8;
            }
        }
    }

    if (fit32 && cost32 <= split_cost + TRANSFORM_SELECT_SPLIT_EPS) {
        write_first_block(acs, bw, rbx, rby, 32);
        return;
    }
    for (int qy{0}; qy < 2; ++qy) {
        for (int qx{0}; qx < 2; ++qx) {
            const int choice{quad_choice[qy][qx]};
            if (choice < 0) {
                continue;
            }
            const std::size_t bx{rbx + 2 * qx};
            const std::size_t by{rby + 2 * qy};
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

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_TRANSFORM_SELECT_IMPL_CUH_
