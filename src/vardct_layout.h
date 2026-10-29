// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_VARDCT_LAYOUT_H_
#define CUJPEGXL_SRC_VARDCT_LAYOUT_H_

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace cujpegxl {

using std::cos;
using std::sqrt;

// Data model for the variable-block front end: transform-type signaling, the
// covered-block coefficient storage, and the low-frequency -> DC-image
// derivation. Host reference used to validate the device kernels and exercised
// against the libjxl oracle.

#if defined(__CUDACC__)
#define CUJPEGXL_VL_HD __host__ __device__
#else
#define CUJPEGXL_VL_HD
#endif

// Covered 8x8 blocks per side of a square DCT of side `block_dim` (8/16/32).
CUJPEGXL_VL_HD inline std::size_t covered_blocks_side(std::size_t block_dim) {
    return block_dim / 8;
}

// Per-8x8-block transform-type signal. A block that is the top-left of a
// transform holds the transform side (8, 16, or 32); every other 8x8 block the
// transform covers holds ACS_COVERED. So a first-block is any block with a
// non-zero signal; covered blocks are skipped by selection and entropy.
constexpr std::int8_t ACS_COVERED = 0;

inline bool acs_is_first_block(std::int8_t signal) {
    return signal != ACS_COVERED;
}

// Coefficients stored per 8x8 block position in the coefficient buffer. Every
// position reserves a full 8x8 block's worth of slots (64); a first-block of side
// N packs its N*N coefficients across the slots of its covered blocks.
constexpr std::size_t COEFFS_PER_BLOCK = 64;

// An image extent padded to whole 8x8 blocks. The block pipeline covers the
// padded frame, whose pixels past the image edge replicate the edge pixels.
CUJPEGXL_VL_HD inline std::size_t padded_to_blocks(std::size_t pixels) {
    return (pixels + 7) / 8 * 8;
}

// Side lengths, in 8x8 blocks, of the square AC groups (256px) and DC groups
// (2048px) tiling a frame; edge groups are partial.
constexpr std::size_t AC_GROUP_BLOCKS = 32;
constexpr std::size_t DC_GROUP_BLOCKS = 256;

// Maps a first-block's raw coefficient index (forward_dctN transposed-raster
// layout: raw = fx*block_dim + fy, in [0, block_dim*block_dim)) to the flat slot
// index within one channel plane of the covered-block coefficient buffer. The
// coefficient buffer has COEFFS_PER_BLOCK slots per 8x8 block position in block
// raster order (bw blocks per row). The N*N coefficients fill the covered blocks
// row-major: covered ordinal q = raw / COEFFS_PER_BLOCK selects covered block
// (by + q/side, bx + q%side); the low bits raw % COEFFS_PER_BLOCK are the slot.
CUJPEGXL_VL_HD inline std::size_t covered_plane_slot(std::size_t block_dim, std::size_t bx,
                                                     std::size_t by, std::size_t bw,
                                                     std::size_t raw_index) {
    const std::size_t side{covered_blocks_side(block_dim)};
    const std::size_t q{raw_index / COEFFS_PER_BLOCK};
    const std::size_t slot{raw_index % COEFFS_PER_BLOCK};
    const std::size_t block{(by + q / side) * bw + (bx + q % side)};
    return block * COEFFS_PER_BLOCK + slot;
}

// Orthonormal 1D DCT-II basis entry O[k][n] for an M-point transform.
CUJPEGXL_VL_HD inline double vl_ortho_basis(std::size_t m, std::size_t k, std::size_t n) {
    const double md{static_cast<double>(m)};
    const double norm{k == 0 ? sqrt(1.0 / md) : sqrt(2.0 / md)};
    return norm * cos(3.14159265358979323846 * (static_cast<double>(n) + 0.5) *
                      static_cast<double>(k) / md);
}

// Separable reinterpreting-IDCT weights of the m*m low-frequency corner (m =
// covered_blocks_side) for the covered block at column `out_x`, row `out_y`:
// the orthonormal basis times libjxl's DCTResampleScales<N, N/8>. Shared by all
// channels of that block.
struct LlfDcWeights {
    double x[4];
    double y[4];
    double scale[4];
};

CUJPEGXL_VL_HD inline LlfDcWeights llf_dc_weights(std::size_t m, std::size_t out_x,
                                                  std::size_t out_y) {
    LlfDcWeights w{};
    // A DCT8's DC is its single LLF coefficient (see dc_from_llf_at).
    if (m == 1) {
        return w;
    }
    const double scale16[2]{1.0, 0.901764195028874394};
    const double scale32[4]{1.0, 0.974886821136879522, 0.901764195028874394, 0.787054918159101335};
    for (std::size_t f{0}; f < m; ++f) {
        w.x[f] = vl_ortho_basis(m, f, out_x);
        w.y[f] = vl_ortho_basis(m, f, out_y);
        w.scale[f] = m == 2 ? scale16[f] : scale32[f];
    }
    return w;
}

// Low-frequency -> DC derivation from the m*m low-frequency corner, read as
// `llf[fx*stride + fy]`, for the covered block whose weights are `w`, matching
// libjxl's DCFromLowestFrequencies for the square DCTNxN strategy. `stride` is
// the source block's row pitch: block_dim for a full transposed-raster block, or
// m for a compacted corner.
CUJPEGXL_VL_HD inline float dc_from_llf_at(std::size_t m, const float* llf, std::size_t stride,
                                           const LlfDcWeights& w) {
    if (m == 1) {
        return llf[0];
    }
    double sum{0.0};
    for (std::size_t fx{0}; fx < m; ++fx) {
        for (std::size_t fy{0}; fy < m; ++fy) {
            const double c{llf[fx * stride + fy]};
            sum += w.x[fx] * w.y[fy] * c * w.scale[fx] * w.scale[fy];
        }
    }
    return static_cast<float>(static_cast<double>(m) * sum);
}

// All m*m DC values of the corner (row-major, dc[Y*m + X] for covered block row
// Y, column X); see dc_from_llf_at.
CUJPEGXL_VL_HD inline void dc_from_llf_strided(std::size_t m, const float* llf, std::size_t stride,
                                               float* dc) {
    for (std::size_t out_y{0}; out_y < m; ++out_y) {
        for (std::size_t out_x{0}; out_x < m; ++out_x) {
            dc[out_y * m + out_x] = dc_from_llf_at(m, llf, stride, llf_dc_weights(m, out_x, out_y));
        }
    }
}

// Convenience overload reading the corner in-place from a full block_dim*block_dim
// transposed-raster block.
CUJPEGXL_VL_HD inline void dc_from_llf(std::size_t block_dim, const float* coeffs, float* dc) {
    dc_from_llf_strided(covered_blocks_side(block_dim), coeffs, block_dim, dc);
}

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_VARDCT_LAYOUT_H_
