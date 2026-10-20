// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_DCT_IMPL_CUH_
#define CUJPEGXL_SRC_DCT_IMPL_CUH_

// Separable forward square DCT used by the fused front end, matching libjxl's
// VarDCT square transforms. The 1D DCT-N basis is
// A[k][t] = g(k) * cos(pi * (t + 0.5) * k / N), with g(0) = 1/N and
// g(k > 0) = sqrt(2)/N. The stored coefficient is
// coeff[fx * N + fy] = sum_{r,c} A[fy][r] * A[fx][c] * pixel[r][c], i.e. the
// orthonormal 2D DCT-II scaled by 1/N in libjxl's transposed raster order
// (horizontal frequency major).
//
// The transform runs as a horizontal pass producing rows[fx * N + r] followed
// by a vertical pass over those rows. In shared memory the basis rows are
// padded to N + 1 floats: in both passes the lanes of a warp read different
// basis rows at the same column, and the odd stride maps them to distinct
// banks.

namespace cujpegxl {

__device__ inline float dct_basis_entry(int n, int k, int t) {
    const double g{k == 0 ? 1.0 / n : 1.4142135623730950488 / n};
    return static_cast<float>(g * cos(3.14159265358979323846 * (t + 0.5) * k / n));
}

__device__ inline void load_dct_basis(const float* __restrict__ src, float* __restrict__ dst, int n,
                                      int tid, int threads) {
    for (int e{tid}; e < n * n; e += threads) {
        dst[(e / n) * (n + 1) + e % n] = src[e];
    }
}

// Horizontal pass: frequency fx of one pixel row.
__device__ inline float dct_rows_entry(const float* __restrict__ basis, int n, int fx,
                                       const float* __restrict__ pixel_row) {
    float s{0.0f};
    for (int c{0}; c < n; ++c) {
        s += basis[fx * (n + 1) + c] * pixel_row[c];
    }
    return s;
}

// Vertical pass: coefficient (fx, fy) from the n horizontal-pass rows of
// frequency fx.
__device__ inline float dct_coeff_entry(const float* __restrict__ basis, int n, int fy,
                                        const float* __restrict__ rows_fx) {
    float coeff{0.0f};
    for (int r{0}; r < n; ++r) {
        coeff += basis[fy * (n + 1) + r] * rows_fx[r];
    }
    return coeff;
}

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_DCT_IMPL_CUH_
