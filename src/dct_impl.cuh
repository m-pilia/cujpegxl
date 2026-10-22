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
// pixel row r, then the rows of frequency fx. The 1D DCT is libjxl's recursive
// butterfly (DCT1DImpl in dct-inl.h) without its 1/N factor: the even
// frequencies of an N-point DCT are the N/2-point DCT of v[i] + v[N-1-i], the
// odd ones the N/2-point DCT of (v[i] - v[N-1-i]) / (2 cos(pi (i + 0.5) / N))
// with adjacent outputs added. The two halves are independent, so they can be
// computed by different threads with the same result.

namespace cujpegxl {

constexpr float DCT_SQRT2{1.41421356237309504880f};

// 1 / (2 cos(pi (i + 0.5) / N)), i < N/2, for N = 4, 8, 16, 32 at offset N/2 - 2.
constexpr float DCT_ODD_MULTIPLIERS[30]{
    0.54119610014619701f, 1.3065629648763764f,  0.50979557910415918f, 0.60134488693504529f,
    0.89997622313641557f, 2.5629154477415055f,  0.50241928618815568f, 0.52249861493968885f,
    0.56694403481635769f, 0.64682178335999008f, 0.7881546234512502f,  1.0606776859903471f,
    1.7224470982383342f,  5.1011486186891553f,  0.50060299823519627f, 0.50547095989754365f,
    0.51544730992262455f, 0.53104259108978413f, 0.55310389603444454f, 0.58293496820613389f,
    0.62250412303566482f, 0.67480834145500568f, 0.74453627100229858f, 0.83934964541552681f,
    0.97256823786196078f, 1.1694399334328847f,  1.4841646163141662f,  2.0577810099534108f,
    3.407608418468719f,   10.190008123548033f,
};

// dct1d computes N times A, so the 2D transform is scaled by 1/N^2, a power of
// two.
template <int N>
constexpr float DCT2D_SCALE{1.0f / static_cast<float>(N * N)};

template <int N>
__device__ __forceinline__ void dct1d(float (&v)[N]);

// even[m] = N * (A v)[2 m].
template <int N>
__device__ __forceinline__ void dct1d_even(const float (&v)[N], float (&even)[N / 2]) {
#pragma unroll
    for (int i{0}; i < N / 2; ++i) {
        even[i] = v[i] + v[N - 1 - i];
    }
    dct1d(even);
}

// odd[m] = N * (A v)[2 m + 1].
template <int N>
__device__ __forceinline__ void dct1d_odd(const float (&v)[N], float (&odd)[N / 2]) {
    if constexpr (N == 2) {
        odd[0] = v[0] - v[1];
    } else {
#pragma unroll
        for (int i{0}; i < N / 2; ++i) {
            odd[i] = (v[i] - v[N - 1 - i]) * DCT_ODD_MULTIPLIERS[N / 2 - 2 + i];
        }
        dct1d(odd);
        odd[0] = fmaf(odd[0], DCT_SQRT2, odd[1]);
#pragma unroll
        for (int i{1}; i + 1 < N / 2; ++i) {
            odd[i] += odd[i + 1];
        }
    }
}

// In place: v <- N * A v.
template <int N>
__device__ __forceinline__ void dct1d(float (&v)[N]) {
    if constexpr (N > 1) {
        float even[N / 2];
        float odd[N / 2];
        dct1d_even(v, even);
        dct1d_odd(v, odd);
#pragma unroll
        for (int i{0}; i < N / 2; ++i) {
            v[2 * i] = even[i];
            v[2 * i + 1] = odd[i];
        }
    }
}

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_DCT_IMPL_CUH_
