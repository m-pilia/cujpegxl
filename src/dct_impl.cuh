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
// by a vertical pass over those rows. Both passes are 1D DCTs of N-vectors:
// pixel row r, then the rows of frequency fx. Every output accumulates its N
// products in increasing t, whichever of the two evaluations below computes
// it, so both give bit-identical results.

namespace cujpegxl {

__device__ inline float dct_basis_entry(int n, int k, int t) {
    const double g{k == 0 ? 1.0 / n : 1.4142135623730950488 / n};
    return static_cast<float>(g * cos(3.14159265358979323846 * (t + 0.5) * k / n));
}

// A[k][t] = +-g(k) cos(pi m / (2N)) with m = (2t + 1) k mod 4N folded into
// [0, N), so the basis has only N distinct magnitudes
// A[m][0] = dct_basis_entry(N, m, 0), m = 0..N-1. This is the index of A[k][t]
// among them and whether A[k][t] is the negated magnitude.
struct DctBasisRef {
    int index;
    bool negate;
};

__host__ __device__ constexpr DctBasisRef dct_basis_ref(int n, int k, int t) {
    int m{((2 * t + 1) * k) % (4 * n)};
    if (m > 2 * n) {
        m = 4 * n - m;
    }
    if (m > n) {
        return {2 * n - m, true};
    }
    return {m, false};
}

// Shared-memory evaluation: the basis is expanded from its magnitudes with rows
// padded to N + 1 floats. When the lanes of a warp read different basis rows at
// the same column, the odd stride maps them to distinct banks.
__device__ inline void expand_dct_basis(const float* __restrict__ magnitudes,
                                        float* __restrict__ dst, int n, int tid, int threads) {
    for (int e{tid}; e < n * n; e += threads) {
        const DctBasisRef ref{dct_basis_ref(n, e / n, e % n)};
        dst[(e / n) * (n + 1) + e % n] =
            ref.negate ? -magnitudes[ref.index] : magnitudes[ref.index];
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
//
// Of the two evaluations, this one has the smaller code: a short loop over i
// that every warp of every resident block runs, so it stays in the
// instruction cache. Its cost is about one shared-memory load per
// multiply-add (more when the W outputs share few operands). It is the faster
// one where that load rate is moderate and each warp's work per pass is small,
// as in the front end's transform selection, which evaluates every candidate
// size of Y; there, the unrolled code of dct_outputs costs more in
// instruction fetch than it saves in loads.
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

// Register evaluation: frequencies [FIRST, FIRST + F) of the 1D DCT of `in`;
// `magnitudes` holds the N entries A[m][0]. The loops unroll fully, so with
// `magnitudes` in constant memory every multiply-add reads its basis entry at a
// compile-time offset straight from the constant cache, with the sign folded
// into the operand.
//
// Of the two evaluations, this one needs the fewest shared-memory loads: N per
// vector for F * N multiply-adds, against about one per multiply-add for
// dct_dots. The price is code size: the unrolled code runs once per vector and
// pays its instruction fetch each time, and warps of one scheduler running
// different unrolled parts thrash the instruction cache. It is the faster one
// where dct_dots would be strongly load-bound, as in the front end's X and B
// region DCT, which pairs two channels per thread.
template <int N, int F, int FIRST>
__device__ __forceinline__ void dct_outputs(const float* __restrict__ magnitudes,
                                            const float (&in)[N], float (&out)[F]) {
#pragma unroll
    for (int f{0}; f < F; ++f) {
        out[f] = 0.0f;
    }
#pragma unroll
    for (int t{0}; t < N; ++t) {
#pragma unroll
        for (int f{0}; f < F; ++f) {
            const DctBasisRef ref{dct_basis_ref(N, FIRST + f, t)};
            const float a{ref.negate ? -magnitudes[ref.index] : magnitudes[ref.index]};
            out[f] += a * in[t];
        }
    }
}

// dct_outputs for frequency part `part`, i.e. FIRST = part * F. Each part is a
// separate unrolled branch; `part` should be warp-uniform.
template <int N, int F, int PART = 0>
__device__ __forceinline__ void dct_part(int part, const float* __restrict__ magnitudes,
                                         const float (&in)[N], float (&out)[F]) {
    static_assert(N % F == 0);
    if constexpr (PART + 1 < N / F) {
        if (part != PART) {
            dct_part<N, F, PART + 1>(part, magnitudes, in, out);
            return;
        }
    }
    dct_outputs<N, F, PART * F>(magnitudes, in, out);
    // Distinct per part, so that LLVM cannot merge the parts' branches into
    // one with register-indexed basis entries.
    asm volatile("// dct part %0" ::"n"(PART));
}

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_DCT_IMPL_CUH_
