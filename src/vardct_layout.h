// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_VARDCT_LAYOUT_H_
#define CUJPEGXL_SRC_VARDCT_LAYOUT_H_

#include <cstddef>
#include <cstdint>

namespace cujpegxl {

// Data model for the M3 variable-block front-end: transform-type signaling, the
// covered-block coefficient storage, and the low-frequency -> DC-image
// derivation. Host reference used to validate the device kernels (T3/T4) and
// exercised against the libjxl oracle.

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

inline bool acs_is_first_block(std::int8_t signal) { return signal != ACS_COVERED; }

// Coefficients stored per 8x8 block position in the M3 coefficient buffer. Every
// position reserves a full DCT8 block's worth of slots; a first-block of side N
// packs its N*N coefficients across the slots of its covered blocks.
constexpr std::size_t COEFFS_PER_BLOCK = 64;

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

// Scatters a first-block's `block_dim*block_dim` coefficients (raw layout) into
// `plane` (a full channel plane, bw*bh*COEFFS_PER_BLOCK slots) at block (bx, by).
void scatter_covered_block(std::size_t block_dim, std::size_t bx, std::size_t by, std::size_t bw,
                           const float* coeffs, float* plane);

// Inverse of scatter_covered_block: gathers the first-block's coefficients back
// into `coeffs` (raw layout) from `plane`.
void gather_covered_block(std::size_t block_dim, std::size_t bx, std::size_t by, std::size_t bw,
                          const float* plane, float* coeffs);

// Low-frequency -> DC derivation: given the `block_dim*block_dim` transposed-
// raster DCT coefficients of one square block (forward_dctN layout), writes the
// covered_blocks_side^2 DC values (row-major, dc[Y*side + X] for covered block
// row Y, column X), matching libjxl's DCFromLowestFrequencies for the square
// DCTNxN strategy. These are the DC-image entries the covered 8x8 positions take.
void dc_from_llf(std::size_t block_dim, const float* coeffs, float* dc);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_VARDCT_LAYOUT_H_
