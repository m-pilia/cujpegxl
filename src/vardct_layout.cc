// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "vardct_layout.h"

#include <cmath>

namespace cujpegxl {
namespace {

// libjxl DCTResampleScales<N, N/8>: the per-frequency scale ReinterpretingIDCT
// applies to the low-frequency coefficients before the covered-side IDCT.
constexpr float RESAMPLE_16[2] = {1.0f, 0.901764195028874394f};
constexpr float RESAMPLE_32[4] = {1.0f, 0.974886821136879522f, 0.901764195028874394f,
                                  0.787054918159101335f};

const float* resample_scales(std::size_t side) {
    return side == 2 ? RESAMPLE_16 : RESAMPLE_32;
}

// Orthonormal 1D DCT-II basis entry O[k][n] for an M-point transform.
double ortho_basis(std::size_t m, std::size_t k, std::size_t n) {
    const auto md{static_cast<double>(m)};
    const double norm{k == 0 ? std::sqrt(1.0 / md) : std::sqrt(2.0 / md)};
    return norm * std::cos(M_PI * (static_cast<double>(n) + 0.5) * static_cast<double>(k) / md);
}

}  // namespace

std::size_t covered_plane_slot(std::size_t block_dim, std::size_t bx, std::size_t by,
                               std::size_t bw, std::size_t raw_index) {
    const std::size_t side{covered_blocks_side(block_dim)};
    const std::size_t q{raw_index / COEFFS_PER_BLOCK};
    const std::size_t slot{raw_index % COEFFS_PER_BLOCK};
    const std::size_t block{(by + q / side) * bw + (bx + q % side)};
    return block * COEFFS_PER_BLOCK + slot;
}

void scatter_covered_block(std::size_t block_dim, std::size_t bx, std::size_t by, std::size_t bw,
                           const float* coeffs, float* plane) {
    const std::size_t count{block_dim * block_dim};
    for (std::size_t raw{0}; raw < count; ++raw) {
        plane[covered_plane_slot(block_dim, bx, by, bw, raw)] = coeffs[raw];
    }
}

void gather_covered_block(std::size_t block_dim, std::size_t bx, std::size_t by, std::size_t bw,
                          const float* plane, float* coeffs) {
    const std::size_t count{block_dim * block_dim};
    for (std::size_t raw{0}; raw < count; ++raw) {
        coeffs[raw] = plane[covered_plane_slot(block_dim, bx, by, bw, raw)];
    }
}

void dc_from_llf(std::size_t block_dim, const float* coeffs, float* dc) {
    const std::size_t m{covered_blocks_side(block_dim)};
    if (m == 1) {
        dc[0] = coeffs[0];
        return;
    }

    // ReinterpretingIDCT for the square DCTNxN strategy: scale the top-left MxM
    // low-frequency coefficients by the resample factors, then apply the MxN
    // covered-side inverse DCT. forward_dctN stores coeff[fx*N+fy] = (1/N) times
    // the orthonormal 2D DCT, so the covered-side IDCT is the orthonormal MxM
    // inverse scaled by M; horizontal frequency fx pairs with column x, vertical
    // frequency fy with row y.
    const float* scale{resample_scales(m)};
    for (std::size_t out_y{0}; out_y < m; ++out_y) {
        for (std::size_t out_x{0}; out_x < m; ++out_x) {
            double sum{0.0};
            for (std::size_t fx{0}; fx < m; ++fx) {
                for (std::size_t fy{0}; fy < m; ++fy) {
                    const double c{coeffs[fx * block_dim + fy]};
                    sum += ortho_basis(m, fx, out_x) * ortho_basis(m, fy, out_y) * c * scale[fx] *
                           scale[fy];
                }
            }
            dc[out_y * m + out_x] = static_cast<float>(static_cast<double>(m) * sum);
        }
    }
}

}  // namespace cujpegxl
