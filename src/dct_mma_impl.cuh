// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_DCT_MMA_IMPL_CUH_
#define CUJPEGXL_SRC_DCT_MMA_IMPL_CUH_

// Tensor-core 2D DCTs (sm_80 and later): the transforms of dct_impl.cuh as the
// matrix products Y = A X A^T with FP16 operands and FP32 accumulation, A the
// basis including its 1/N factor, so Y[fy][fx] is coefficient fx * N + fy.
//
// Both products are mma.sync.m16n8k16. The vertical pass U = A X takes rows of
// A as the A operand and X as the B operand; its accumulators hold U in the C
// layout, which for two adjacent n-tiles is the A operand layout of a k-step,
// so the horizontal pass Y = U A^T takes U straight from registers (rounded to
// FP16) with A^T as the B operand.
//
// X is offset by a constant before its rounding to FP16, so that the rounding
// error scales with the tile's deviation from it rather than with its values;
// since A 1 1^T A^T = e_0 e_0^T, the offset is added back to coefficient (0, 0).
//
// DCT32: two warps split Y by its rows, warp half w computing rows fy in
// [16 w, 16 w + 16). The sum over k of the vertical pass runs over X's rows in
// a permuted order: k-step s, k index 2t + 8h + i (i = 0, 1) takes row
// 8t + 4s + 2h + i, so with a row stride of 1 modulo 32 banks each B operand
// load of the 32 lanes reads 32 distinct banks.
//
// Region transform: one warp transforms a 16x16 region, as one DCT16 or as the
// four DCT8s of its 8x8 blocks with the block-diagonal basis
// B = blockdiag(A8, A8), whose Y = B X B^T holds the DCT8 of block (dx, dy) at
// fx in [8 dx, 8 dx + 8) and fy in [8 dy, 8 dy + 8); the offset is added back
// to each block's (0, 0). Each pass is a single k-step over two n-tiles. Rows
// and columns of both products are permuted at compile time: k index
// 8 r + 2 t + i of the vertical pass takes X row region_mma_row(k), and C
// element e of n-tile j of the horizontal pass holds frequency
// (region_mma_freq_x(n, j, 2 t + e % 2), region_mma_freq_y(n, g + 8 (e / 2))).
// The frequency orders put the 32 lanes' stores of the coefficients into the
// region layout of frontend_fused.cu on 32 distinct banks.
//
// Fragment layouts, with g = lane / 4 and t = lane % 4: an A operand register
// r holds (row g + 8 (r % 2), columns 2t + 8 (r / 2) and the next one), a B
// operand register r holds (rows 2t + 8 r and the next one, column g), and C
// element e holds (row g + 8 (e / 2), column 2t + e % 2). The element at the
// lower row or column is in the lower half of a register.

#include <cstdint>
#include <cstring>

#include <cuda_fp16.h>

namespace cujpegxl {

namespace dct_mma_detail {

constexpr double PI{3.14159265358979323846};

constexpr std::uint32_t half2_bits(std::uint16_t lo, std::uint16_t hi) {
    return static_cast<std::uint32_t>(lo) | static_cast<std::uint32_t>(hi) << 16;
}

// sin and cos on [0, pi/4], to double precision.
constexpr double taylor_sin(double x) {
    double term{x};
    double sum{x};
    for (int i{1}; i < 12; ++i) {
        term *= -x * x / static_cast<double>((2 * i) * (2 * i + 1));
        sum += term;
    }
    return sum;
}

constexpr double taylor_cos(double x) {
    double term{1.0};
    double sum{1.0};
    for (int i{1}; i < 12; ++i) {
        term *= -x * x / static_cast<double>((2 * i - 1) * (2 * i));
        sum += term;
    }
    return sum;
}

// cos(pi m / 64).
constexpr double cos_pi_64ths(int m) {
    m %= 128;
    if (m > 64) {
        m = 128 - m;
    }
    double sign{1.0};
    if (m > 32) {
        m = 64 - m;
        sign = -1.0;
    }
    return m > 16 ? sign * taylor_sin(PI * (32 - m) / 64.0) : sign * taylor_cos(PI * m / 64.0);
}

// A[k][t] = g(k) cos(pi (t + 0.5) k / n), g(0) = 1/n, g(k > 0) = sqrt(2)/n, for
// n = 8, 16, 32.
constexpr double dct_basis(int n, int k, int t) {
    constexpr double SQRT2{1.41421356237309504880};
    return (k == 0 ? 1.0 : SQRT2) / n * cos_pi_64ths((2 * t + 1) * k * (32 / n));
}

// FP16 bits of v rounded to nearest even, for zero and normal FP16 magnitudes.
constexpr std::uint16_t half_bits(double v) {
    if (v == 0.0) {
        return 0;
    }
    const std::uint16_t sign{static_cast<std::uint16_t>(v < 0.0 ? 0x8000u : 0u)};
    double mag{v < 0.0 ? -v : v};
    int exponent{0};
    while (mag >= 2.0) {
        mag *= 0.5;
        ++exponent;
    }
    while (mag < 1.0) {
        mag *= 2.0;
        --exponent;
    }
    const double scaled{mag * 1024.0};
    auto mantissa{static_cast<std::uint32_t>(scaled)};
    const double frac{scaled - static_cast<double>(mantissa)};
    if (frac > 0.5 || (frac == 0.5 && (mantissa & 1u) != 0)) {
        ++mantissa;
    }
    if (mantissa == 2048) {
        mantissa = 1024;
        ++exponent;
    }
    return static_cast<std::uint16_t>(sign | static_cast<std::uint32_t>(exponent + 15) << 10 |
                                      (mantissa - 1024));
}

// FP16 bits of an N x N basis, [k][t].
template <int N>
struct HalfBasis {
    std::uint16_t a[N][N];
};

template <int N, typename Basis>
constexpr HalfBasis<N> make_half_basis(Basis basis) {
    HalfBasis<N> half{};
    for (int k{0}; k < N; ++k) {
        for (int t{0}; t < N; ++t) {
            half.a[k][t] = half_bits(basis(k, t));
        }
    }
    return half;
}

// The register holding a[k][t] and a[k][t + 1].
template <int N>
constexpr std::uint32_t half2_pair(const HalfBasis<N>& half, int k, int t) {
    return half2_bits(half.a[k][t], half.a[k][t + 1]);
}

}  // namespace dct_mma_detail

// The basis operands in per-lane fragment order, so each lane loads its
// registers with 16-byte loads: rows[w][s][lane] are the A operand registers
// of A rows [16 w, 16 w + 16) and the columns of k-step s; cols[j][lane]
// are the B operand registers of A^T rows (k-steps) 0-15 and 16-31 and columns
// [8 j, 8 j + 8).
struct alignas(16) Dct32MmaBasis {
    std::uint32_t rows[2][2][32][4];
    std::uint32_t cols[4][32][4];
};

constexpr Dct32MmaBasis make_dct32_mma_basis() {
    using dct_mma_detail::half2_pair;
    const dct_mma_detail::HalfBasis<32> half{dct_mma_detail::make_half_basis<32>(
        [](int k, int t) { return dct_mma_detail::dct_basis(32, k, t); })};
    Dct32MmaBasis basis{};
    for (int lane{0}; lane < 32; ++lane) {
        const int g{lane / 4};
        const int t{lane % 4};
        for (int w{0}; w < 2; ++w) {
            for (int s{0}; s < 2; ++s) {
                for (int r{0}; r < 4; ++r) {
                    basis.rows[w][s][lane][r] =
                        half2_pair(half, 16 * w + g + 8 * (r % 2), 8 * t + 4 * s + 2 * (r / 2));
                }
            }
        }
        for (int j{0}; j < 4; ++j) {
            for (int r{0}; r < 4; ++r) {
                basis.cols[j][lane][r] = half2_pair(half, 8 * j + g, 2 * t + 8 * r);
            }
        }
    }
    return basis;
}

inline constexpr Dct32MmaBasis DCT32_MMA_BASIS{make_dct32_mma_basis()};

__device__ __forceinline__ std::uint32_t pack_half2(float lo, float hi) {
    const __half2 h{__floats2half2_rn(lo, hi)};
    std::uint32_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// d += a b.
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const std::uint32_t (&a)[4],
                                             std::uint32_t b0, std::uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
        "{%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// `src` is in global memory.
__device__ __forceinline__ void load_fragment(const std::uint32_t* src, std::uint32_t (&r)[4]) {
    const uint4 v{__ldg(reinterpret_cast<const uint4*>(src))};
    r[0] = v.x;
    r[1] = v.y;
    r[2] = v.z;
    r[3] = v.w;
}

// Vertical pass of warp half w over the 32x32 tile X (row-major with `stride`
// floats per row): u holds the A operand registers of U = A (X - offset) for
// its horizontal pass, k-step s in u[s]. Returns the offset, the mean of X's
// rows 0, 4, ..., 28.
__device__ __forceinline__ float dct32_mma_vertical(const Dct32MmaBasis& basis,
                                                    const float* __restrict__ x, int stride, int w,
                                                    int lane, std::uint32_t (&u)[2][4]) {
    const int g{lane / 4};
    const int t{lane % 4};
    std::uint32_t a[2][4];
    load_fragment(basis.rows[w][0][lane], a[0]);
    load_fragment(basis.rows[w][1][lane], a[1]);
    float first[4][2];
    float sum{0.0f};
#pragma unroll
    for (int j{0}; j < 4; ++j) {
#pragma unroll
        for (int s{0}; s < 2; ++s) {
            first[j][s] = x[(8 * t + 4 * s) * stride + 8 * j + g];
            sum += first[j][s];
        }
    }
#pragma unroll
    for (int mask{16}; mask > 0; mask /= 2) {
        sum += __shfl_xor_sync(0xffffffffu, sum, mask);
    }
    const float offset{sum * (1.0f / 256.0f)};
    float acc[4][4]{};
#pragma unroll
    for (int j{0}; j < 4; ++j) {
#pragma unroll
        for (int s{0}; s < 2; ++s) {
            const float* col{x + (8 * t + 4 * s) * stride + 8 * j + g};
            mma_m16n8k16(acc[j], a[s], pack_half2(first[j][s] - offset, col[stride] - offset),
                         pack_half2(col[2 * stride] - offset, col[3 * stride] - offset));
        }
    }
#pragma unroll
    for (int s{0}; s < 2; ++s) {
        u[s][0] = pack_half2(acc[2 * s][0], acc[2 * s][1]);
        u[s][1] = pack_half2(acc[2 * s][2], acc[2 * s][3]);
        u[s][2] = pack_half2(acc[2 * s + 1][0], acc[2 * s + 1][1]);
        u[s][3] = pack_half2(acc[2 * s + 1][2], acc[2 * s + 1][3]);
    }
    return offset;
}

// Horizontal pass: y[4 j + e] is C element e of n-tile j of Y = U A^T plus the
// vertical pass's offset at (0, 0), i.e. coefficient
// (dct32_mma_fx(lane, 4 j + e), dct32_mma_fy(w, lane, 4 j + e)).
__device__ __forceinline__ void dct32_mma_horizontal(const Dct32MmaBasis& basis,
                                                     const std::uint32_t (&u)[2][4], int w,
                                                     int lane, float offset, float (&y)[16]) {
#pragma unroll
    for (int j{0}; j < 4; ++j) {
        std::uint32_t b[4];
        load_fragment(basis.cols[j][lane], b);
        float acc[4]{};
        mma_m16n8k16(acc, u[0], b[0], b[1]);
        mma_m16n8k16(acc, u[1], b[2], b[3]);
#pragma unroll
        for (int e{0}; e < 4; ++e) {
            y[4 * j + e] = acc[e];
        }
    }
    if (w == 0 && lane == 0) {
        y[0] += offset;
    }
}

constexpr int dct32_mma_fx(int lane, int i) {
    return 8 * (i / 4) + 2 * (lane % 4) + i % 2;
}

constexpr int dct32_mma_fy(int w, int lane, int i) {
    return 16 * w + lane / 4 + 8 * ((i % 4) / 2);
}

// X row of k index k of the region transform's vertical pass.
constexpr int region_mma_row(int k) {
    return 4 * ((k % 8) / 2) + 2 * (k / 8) + k % 2;
}

// Frequency fx of column c of n-tile j, and frequency fy of row m, of the
// region transform's result for side n.
constexpr int region_mma_freq_x(int n, int j, int c) {
    const int t{c / 2};
    return 8 * j + 4 * (c % 2) + (n == 16 ? 2 * (t % 2) + t / 2 : t);
}

constexpr int region_mma_freq_y(int n, int m) {
    return n == 16 ? 2 * (m % 8) + m / 8 : m;
}

// The basis operands of the region transform of side n in per-lane fragment
// order: rows[lane] are the A operand registers of the vertical pass, and
// cols[lane][2 j], cols[lane][2 j + 1] the B operand registers of n-tile j of
// the horizontal pass.
struct alignas(16) RegionMmaBasis {
    std::uint32_t rows[32][4];
    std::uint32_t cols[32][4];
};

constexpr RegionMmaBasis make_region_mma_basis(int n) {
    using dct_mma_detail::dct_basis;
    const dct_mma_detail::HalfBasis<16> half{
        dct_mma_detail::make_half_basis<16>([n](int k, int t) {
            if (n == 16) {
                return dct_basis(16, k, t);
            }
            return k / 8 == t / 8 ? dct_basis(8, k % 8, t % 8) : 0.0;
        })};
    RegionMmaBasis basis{};
    for (int lane{0}; lane < 32; ++lane) {
        const int g{lane / 4};
        const int t{lane % 4};
        for (int r{0}; r < 4; ++r) {
            const int fy{region_mma_freq_y(n, g + 8 * (r % 2))};
            const int k{2 * t + 8 * (r / 2)};
            basis.rows[lane][r] = dct_mma_detail::half2_bits(half.a[fy][region_mma_row(k)],
                                                             half.a[fy][region_mma_row(k + 1)]);
        }
        for (int j{0}; j < 2; ++j) {
            for (int r{0}; r < 2; ++r) {
                basis.cols[lane][2 * j + r] =
                    dct_mma_detail::half2_pair(half, region_mma_freq_x(n, j, g), 2 * t + 8 * r);
            }
        }
    }
    return basis;
}

inline constexpr RegionMmaBasis REGION_MMA_BASIS_DCT16{make_region_mma_basis(16)};
inline constexpr RegionMmaBasis REGION_MMA_BASIS_DCT8{make_region_mma_basis(8)};

// Region transform of side n (warp-uniform, with `basis` its basis) of the
// 16x16 region X (row-major with `stride` floats per row): y[4 j + e] is C
// element e of n-tile j of Y = B (X - offset) B^T plus the offset at each
// transform's (0, 0), i.e. coefficient
// (region_mma_fx(n, lane, 4 j + e), region_mma_fy(n, lane, 4 j + e)). The
// offset is the mean of X.
__device__ __forceinline__ void region_mma(const RegionMmaBasis& basis, int n,
                                           const float* __restrict__ x, int stride, int lane,
                                           float (&y)[8]) {
    const int g{lane / 4};
    const int t{lane % 4};
    std::uint32_t a[4];
    load_fragment(basis.rows[lane], a);
    // v[j][r][i]: half i of B operand register r of n-tile j, X column 8 j + g.
    float v[2][2][2];
    float sum{0.0f};
#pragma unroll
    for (int j{0}; j < 2; ++j) {
#pragma unroll
        for (int r{0}; r < 2; ++r) {
#pragma unroll
            for (int i{0}; i < 2; ++i) {
                v[j][r][i] = x[region_mma_row(2 * t + 8 * r + i) * stride + 8 * j + g];
                sum += v[j][r][i];
            }
        }
    }
#pragma unroll
    for (int mask{16}; mask > 0; mask /= 2) {
        sum += __shfl_xor_sync(0xffffffffu, sum, mask);
    }
    const float offset{sum * (1.0f / 256.0f)};
    float acc[2][4]{};
#pragma unroll
    for (int j{0}; j < 2; ++j) {
        mma_m16n8k16(acc[j], a, pack_half2(v[j][0][0] - offset, v[j][0][1] - offset),
                     pack_half2(v[j][1][0] - offset, v[j][1][1] - offset));
    }
    const std::uint32_t u[4]{pack_half2(acc[0][0], acc[0][1]), pack_half2(acc[0][2], acc[0][3]),
                             pack_half2(acc[1][0], acc[1][1]), pack_half2(acc[1][2], acc[1][3])};
    std::uint32_t b[4];
    load_fragment(basis.cols[lane], b);
#pragma unroll
    for (int j{0}; j < 2; ++j) {
        float c[4]{};
        mma_m16n8k16(c, u, b[2 * j], b[2 * j + 1]);
#pragma unroll
        for (int e{0}; e < 4; ++e) {
            y[4 * j + e] = c[e];
        }
    }
    // The (0, 0) of every transform is on lane 0: element 0 of n-tile 0, and
    // for the DCT8s elements 0 and 2 of both n-tiles.
    if (lane == 0) {
        y[0] += offset;
        if (n == 8) {
            y[2] += offset;
            y[4] += offset;
            y[6] += offset;
        }
    }
}

constexpr int region_mma_fx(int n, int lane, int i) {
    return region_mma_freq_x(n, i / 4, 2 * (lane % 4) + i % 2);
}

constexpr int region_mma_fy(int n, int lane, int i) {
    return region_mma_freq_y(n, lane / 4 + 8 * ((i % 4) / 2));
}

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_DCT_MMA_IMPL_CUH_
