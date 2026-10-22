// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_ENTROPY_H_
#define CUJPEGXL_SRC_ENTROPY_H_

#include <cstddef>
#include <cstdint>

#include <cuda_runtime_api.h>

#include "ac_clustering.h"
#include "ac_context.h"
#include "vardct_layout.h"

namespace cujpegxl {

// The device wrappers below enqueue all their work on `stream` and do not wait
// for it, so that the DC path runs concurrently with the AC path and frames run
// concurrently with each other.

// Width of the DC and AcMetadata histograms: hybrid-uint tokens of 32-bit values
// stay below this, so a fixed histogram of this size covers every symbol.
constexpr std::size_t AC_HISTOGRAM_SIZE = 256;

// AC coefficients stored per block (the 63 non-DC libjxl-raster slots). The DC
// slot lives in the separate int32 DC buffer, so it is elided from AC storage.
constexpr std::size_t AC_COEFFS_PER_BLOCK = 63;

// Number of 256x256 AC groups (32x32 blocks each) tiling a width x height image,
// including partial edge groups. width/height must be multiples of 8.
std::size_t ac_num_groups(std::size_t width, std::size_t height);

// Phase 1 of the AC token coder: accumulate the per-context AC symbol
// histograms over every group. `ac` is the covered-block coefficient buffer
// (device; three channel-major planes X, Y, B of (width/8 * height/8) *
// COEFFS_PER_BLOCK int16) and `acs` the per-8x8 transform signal (nullptr = all
// 8x8 blocks). Each first-block tokenizes order[covered..size) of its size
// gathered via covered_plane_slot; covered blocks contribute nothing. `nz_exact`
// holds each first-block's non-zero count over order[covered..size), and
// `nz_grid` that count normalized per covered 8x8 and broadcast to every covered
// position (the neighbor-prediction input); both are device buffers of 3 *
// (width/8 * height/8) int32 in the planes' channel order, as produced by
// quantize_residual. `context_histograms` is a device buffer of AC_NUM_CONTEXTS *
// AC_CONTEXT_ALPHABET uint32 (context * AC_CONTEXT_ALPHABET + symbol; zeroed by
// the call). `item_sizes` (device, 3 * width/8 * height/8) receives each
// block/channel item's token count (low 32 bits) and raw bit count (high 32
// bits) at block * 3 + slot, slot 0..2 being the channels Y, X, B. Deterministic
// (integer atomics). Returns false on a CUDA error.
bool ac_build_histogram(const std::int16_t* ac, const std::int8_t* acs, const std::int32_t* nz_grid,
                        const std::int32_t* nz_exact, std::size_t width, std::size_t height,
                        std::uint32_t* context_histograms, std::uint64_t* item_sizes,
                        cudaStream_t stream);

std::size_t ac_cluster_scratch_bytes();

// Phase 2: clusters the AC contexts from the phase-1 `context_histograms` (see
// ac_clustering.h) into `clustering` (device). `scratch` (device) holds
// ac_cluster_scratch_bytes() bytes. Deterministic.
bool ac_cluster_contexts(const std::uint32_t* context_histograms, std::uint8_t* scratch,
                         AcClustering* clustering, cudaStream_t stream);

// Upper bounds of a frame's AC tokens and of the 64-bit words of their raw bits
// in the ac_token_layout.
std::size_t ac_max_tokens(std::size_t width, std::size_t height);
std::size_t ac_max_raw_words(std::size_t width, std::size_t height);

// Entries of the ac_token_layout `chunk_base` buffer.
std::size_t ac_token_chunks(std::size_t width, std::size_t height);

std::size_t ac_token_layout_scratch_bytes(std::size_t width, std::size_t height);

// Phase 3: the AC groups' token layout from the phase-1 `item_sizes`. Each AC
// group (one AcGroup section, raster order) is one token stream: its tokens in
// decoder order, each an entry cluster << bitstream::TOKEN_CLUSTER_SHIFT |
// symbol, and their raw bits packed LSB-first from a 64-bit word boundary.
// `layout` (device, 3 * num_groups + 2 uint32, num_groups = ac_num_groups)
// receives token_offset[num_groups + 1] (each group's first token, then the
// total), raw_offset[num_groups + 1] (each group's first raw word, then the
// total) and raw_bits[num_groups]. `chunk_base` (device, ac_token_chunks)
// receives the per-chunk offsets ac_pack_tokens reads. `scratch` (device) holds
// ac_token_layout_scratch_bytes(width, height) bytes. Deterministic.
bool ac_token_layout(const std::uint64_t* item_sizes, std::size_t width, std::size_t height,
                     std::uint8_t* scratch, std::uint64_t* chunk_base, std::uint32_t* layout,
                     cudaStream_t stream);

// Phase 4: writes the token streams of AC groups [group_begin, group_end) at
// their `layout` positions: entries into `entries` (device uint16) and raw bits
// into `raw` (device 64-bit words; the groups' words are zeroed by the call).
// `context_map` (device, AC_NUM_CONTEXTS) is the AcClustering's. The other
// inputs are those of phases 1 and 3. Deterministic.
bool ac_pack_tokens(const std::int16_t* ac, const std::int8_t* acs, const std::int32_t* nz_grid,
                    const std::int32_t* nz_exact, std::size_t width, std::size_t height,
                    const std::uint8_t* context_map, const std::uint64_t* item_sizes,
                    const std::uint64_t* chunk_base, const std::uint32_t* layout,
                    std::size_t group_begin, std::size_t group_end, std::uint16_t* entries,
                    std::uint64_t* raw, cudaStream_t stream);

// Number of 2048x2048 DC groups (256x256 blocks each) tiling a width x height
// image, including partial edge groups. width/height must be multiples of 8.
std::size_t dc_num_groups(std::size_t width, std::size_t height);

// Per-group slot stride of the dc_first_blocks position lists.
constexpr std::size_t DC_FIRST_BLOCK_STRIDE = DC_GROUP_BLOCKS * DC_GROUP_BLOCKS;

// Per-DcGroup first-blocks (blocks whose transform is not ACS_COVERED). `acs` is
// the per-8x8 transform signal (nullptr = all 8x8, so every block is one).
// `positions` (device, dc_num_groups * DC_FIRST_BLOCK_STRIDE std::size_t)
// receives group g's first-block indices in group raster order from
// g * DC_FIRST_BLOCK_STRIDE, and `counts` (device, dc_num_groups std::size_t)
// their number, which is the AcMetadata `count` field. dc_encode_groups maps an
// ACS/QF column (a first-block ordinal) back to its block through these lists.
// `scratch` (device) holds dc_first_blocks_scratch_bytes(width, height) bytes.
// Deterministic. Returns false on a CUDA error.
bool dc_first_blocks(const std::int8_t* acs, std::size_t width, std::size_t height,
                     std::uint8_t* scratch, std::size_t* positions, std::size_t* counts,
                     cudaStream_t stream);

std::size_t dc_first_blocks_scratch_bytes(std::size_t width, std::size_t height);

// Per-DcGroup DC symbol histogram, pooled over the 3 DC channels (physical order
// Y, X, B) of that group at block resolution. `dc` is the compact int32 DC buffer
// (device; three channel-major planes X, Y, B; one DC per block; blocks in raster
// order). `histograms` is a device buffer of dc_num_groups * AC_HISTOGRAM_SIZE
// uint32 (zeroed by the call). Deterministic (integer atomics). Returns false on
// a CUDA error.
bool dc_build_histograms(const std::int32_t* dc, std::size_t width, std::size_t height,
                         std::uint32_t* histograms, cudaStream_t stream);

// Per-DcGroup AcMetadata token histogram, pooled over the samples of that
// group's YtoX, YtoB and ACS+QF channels: the per-tile CfL tokens, the
// per-first-block ACS strategy tokens (row 0) and quant-field tokens (row 1).
// The all-zero EPF channel has a leaf of its own in the AcMetadata MA tree and
// is not counted. `quant_field` is the per-block quant integer buffer (device;
// bw*bh, block raster order), `acs` the per-8x8 transform signal (nullptr = all
// DCT8), `ytox_map`/`ytob_map` the per-64x64-tile CfL factors (nullptr = base).
// `histograms` is a device buffer of dc_num_groups * AC_HISTOGRAM_SIZE uint32
// (zeroed by the call). Deterministic (integer atomics). Returns false on a CUDA
// error.
bool acmeta_build_histograms(const std::int32_t* quant_field, const std::int8_t* acs,
                             const std::int8_t* ytox_map, const std::int8_t* ytob_map,
                             std::size_t width, std::size_t height, std::uint32_t* histograms,
                             cudaStream_t stream);

// Emit each DcGroup's byte-aligned section (extra_precision + VarDCTDC modular +
// AcMetadata count + AcMetadata modular) concatenated into `out`, mirroring the
// host DcGroup reference byte-for-byte. Per group the host supplies the two
// header bit blobs bracketing the token runs and the two shared prefix codes;
// the device emits the DC tokens (from `dc`) and the AcMetadata tokens (from the
// group geometry and the per-block `quant_field`). `group_sizes` and
// `group_offsets` receive each group's byte size and offset in `out`. A
// scan of the group sizes on the device fixes the offsets, so the call never
// blocks the host; a section larger than `out_capacity` leaves `out` untouched,
// and the caller detects it from the sum of `group_sizes`. Deterministic. All pointers are device
// pointers. `scratch` holds dc_encode_scratch_bytes(width, height) bytes.
//
// `quant_field` is the per-block quant integer buffer (device; bw*bh, block
// raster order); its value at each block is written as the block's ACS+QF row-1
// sample. `first_block_pos` / `first_block_counts` are the dc_first_blocks
// output for `acs`. Flattened per-group inputs (num_groups == dc_num_groups): `dc_depth` /
// `acmeta_depth` are num_groups * AC_HISTOGRAM_SIZE uint8, `dc_bits` /
// `acmeta_bits` likewise uint16. `blob_pre` / `blob_mid` are concatenated bytes
// indexed by `blob_pre_off` / `blob_mid_off` (byte offsets) with bit lengths
// `blob_pre_bits` / `blob_mid_bits`.
bool dc_encode_groups(const std::int32_t* dc, std::size_t width, std::size_t height,
                      const std::int32_t* quant_field, const std::int8_t* acs,
                      const std::int8_t* ytox_map, const std::int8_t* ytob_map,
                      const std::size_t* first_block_pos, const std::size_t* first_block_counts,
                      const std::uint8_t* dc_depth, const std::uint16_t* dc_bits,
                      const std::uint8_t* acmeta_depth, const std::uint16_t* acmeta_bits,
                      const std::uint8_t* blob_pre, const std::uint32_t* blob_pre_off,
                      const std::uint32_t* blob_pre_bits, const std::uint8_t* blob_mid,
                      const std::uint32_t* blob_mid_off, const std::uint32_t* blob_mid_bits,
                      std::uint8_t* scratch, std::uint8_t* out, std::size_t out_capacity,
                      std::uint32_t* group_sizes, std::uint32_t* group_offsets,
                      cudaStream_t stream);

std::size_t dc_encode_scratch_bytes(std::size_t width, std::size_t height);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_ENTROPY_H_
