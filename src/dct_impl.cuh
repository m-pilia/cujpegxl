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

__device__ inline const float* dct_basis_row(const float* basis, int n, int k) {
    return basis + k * (n + 1);
}

// W dot products out[w] = sum_i basis_rows[w][i] * vectors[w][i], accumulated
// in increasing i. Both passes are such dot products: frequency fx with a
// pixel row (horizontal), frequency fy with the horizontal-pass rows of
// frequency fx (vertical). Outputs that share a basis row or a vector are
// passed the same pointer, so the shared operand is loaded once per i.
template <int W>
__device__ inline void dct_dots(int n, const float* const (&basis_rows)[W],
                                const float* const (&vectors)[W], float (&out)[W]) {
#pragma unroll
    for (int w{0}; w < W; ++w) {
        out[w] = 0.0f;
    }
    for (int i{0}; i < n; ++i) {
#pragma unroll
        for (int w{0}; w < W; ++w) {
            out[w] += basis_rows[w][i] * vectors[w][i];
        }
    }
}

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_DCT_IMPL_CUH_
