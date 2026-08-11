// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_ENTROPY_H_
#define CUJPEGXL_SRC_ENTROPY_H_

#include <cstddef>
#include <cstdint>

namespace cujpegxl {

// Upper bound on the AC token alphabet: hybrid-uint tokens of 32-bit values stay
// below this, so a fixed histogram of this size covers every symbol.
constexpr std::size_t AC_HISTOGRAM_SIZE = 256;

// AC coefficients stored per block (the 63 non-DC libjxl-raster slots). The DC
// slot lives in the separate int32 DC buffer, so it is elided from AC storage.
constexpr std::size_t AC_COEFFS_PER_BLOCK = 63;

// Number of 256x256 AC groups (32x32 blocks each) tiling a width x height image,
// including partial edge groups. width/height must be multiples of 8.
std::size_t ac_num_groups(std::size_t width, std::size_t height);

// Phase 1 of the device entropy coder: accumulate the pooled AC symbol histogram
// over every group. `ac` is the packed int16 AC coefficient buffer (device;
// three channel-major planes X, Y, B; blocks in raster order; AC_COEFFS_PER_BLOCK
// libjxl-raster coefficients per block, DC slot elided so coefficient index k in
// [1, 63] lives at slot k-1). `histogram` is a device buffer of AC_HISTOGRAM_SIZE
// uint32 (zeroed by the call). Deterministic (integer atomics). Returns false on
// a CUDA error.
bool ac_build_histogram(const std::int16_t* ac, std::size_t width, std::size_t height,
                        std::uint32_t* histogram);

// Phase 2: emit each AC group's token bitstream (byte-aligned, one AcGroup TOC
// section per group) concatenated into `out`, mirroring the host bitstream
// writer byte-for-byte. `ac` is the packed int16 AC buffer (see
// ac_build_histogram). `depth`/`bits` are the shared prefix code (device,
// alphabet_size entries) built from the phase-1 histogram. A CUB exclusive scan
// of the per-group byte sizes fixes each group's offset; `group_sizes`,
// `group_offsets` (num_groups entries, device) and `total_bytes` (host) receive
// the layout. Fails if `out_capacity` is exceeded. Deterministic. All device
// pointers except `total_bytes`.
bool ac_encode_groups(const std::int16_t* ac, std::size_t width, std::size_t height,
                      const std::uint8_t* depth, const std::uint16_t* bits,
                      std::size_t alphabet_size, std::uint8_t* out, std::size_t out_capacity,
                      std::uint32_t* group_sizes, std::uint32_t* group_offsets,
                      std::size_t* total_bytes);

// Mixed-block (M3) AC histogram: as ac_build_histogram, but `ac` is the
// covered-block layout (three channel planes of (width/8 * height/8) *
// COEFFS_PER_BLOCK int16) and `acs` the per-8x8 transform signal (nullptr = all
// DCT8). Each first-block tokenizes order[covered..size) of its size gathered via
// covered_plane_slot; covered blocks contribute nothing. Byte-exact with the host
// mixed-block reference. Deterministic. Returns false on a CUDA error.
bool ac_build_histogram_m3(const std::int16_t* ac, const std::int8_t* acs, std::size_t width,
                           std::size_t height, std::uint32_t* histogram);

// Mixed-block (M3) AC group emit: as ac_encode_groups, over the covered-block
// layout under `acs`. `depth`/`bits` are the shared prefix code from the phase-1
// histogram. Byte-exact with the host reference. Deterministic. Returns false on
// a CUDA error or if `out_capacity` is exceeded.
bool ac_encode_groups_m3(const std::int16_t* ac, const std::int8_t* acs, std::size_t width,
                         std::size_t height, const std::uint8_t* depth, const std::uint16_t* bits,
                         std::uint8_t* out, std::size_t out_capacity, std::uint32_t* group_sizes,
                         std::uint32_t* group_offsets, std::size_t* total_bytes);

// Number of 2048x2048 DC groups (256x256 blocks each) tiling a width x height
// image, including partial edge groups. width/height must be multiples of 8.
std::size_t dc_num_groups(std::size_t width, std::size_t height);

// Per-DcGroup DC symbol histogram, pooled over the 3 DC channels (physical order
// Y, X, B) of that group at block resolution. `dc` is the compact int32 DC buffer
// (device; three channel-major planes X, Y, B; one DC per block; blocks in raster
// order). `histograms` is a device buffer of dc_num_groups * AC_HISTOGRAM_SIZE
// uint32 (zeroed by the call). Deterministic (integer atomics). Returns false on
// a CUDA error.
bool dc_build_histograms(const std::int32_t* dc, std::size_t width, std::size_t height,
                         std::uint32_t* histograms);

// Per-DcGroup AcMetadata token histogram, pooled over the metadata samples of
// that group: the structural zero samples (YtoX, YtoB, ACS row 0, EPF) plus the
// per-block quant-field tokens (ACS+QF row 1). `quant_field` is the per-block
// quant integer buffer (device; bw*bh, block raster order). `histograms` is a
// device buffer of dc_num_groups * AC_HISTOGRAM_SIZE uint32 (zeroed by the call).
// Deterministic (integer atomics). Returns false on a CUDA error.
bool acmeta_build_histograms(const std::int32_t* quant_field, std::size_t width, std::size_t height,
                             std::uint32_t* histograms);

// Emit each DcGroup's byte-aligned section (extra_precision + VarDCTDC modular +
// AcMetadata count + AcMetadata modular) concatenated into `out`, mirroring the
// host DcGroup reference byte-for-byte. Per group the host supplies the two
// header bit blobs bracketing the token runs and the two shared prefix codes;
// the device emits the DC tokens (from `dc`) and the AcMetadata tokens (from the
// group geometry and the per-block `quant_field`). A CUB exclusive scan of the
// per-group byte sizes fixes each group's offset. Fails if `out_capacity` is
// exceeded. Deterministic. All device pointers except `total_bytes`.
//
// `quant_field` is the per-block quant integer buffer (device; bw*bh, block
// raster order); its value at each block is written as the block's ACS+QF row-1
// sample. Flattened per-group inputs (num_groups == dc_num_groups): `dc_depth` /
// `acmeta_depth` are num_groups * AC_HISTOGRAM_SIZE uint8, `dc_bits` /
// `acmeta_bits` likewise uint16. `blob_pre` / `blob_mid` are concatenated bytes
// indexed by `blob_pre_off` / `blob_mid_off` (byte offsets) with bit lengths
// `blob_pre_bits` / `blob_mid_bits`.
bool dc_encode_groups(const std::int32_t* dc, std::size_t width, std::size_t height,
                      const std::int32_t* quant_field, const std::uint8_t* dc_depth,
                      const std::uint16_t* dc_bits, const std::uint8_t* acmeta_depth,
                      const std::uint16_t* acmeta_bits, const std::uint8_t* blob_pre,
                      const std::uint32_t* blob_pre_off, const std::uint32_t* blob_pre_bits,
                      const std::uint8_t* blob_mid, const std::uint32_t* blob_mid_off,
                      const std::uint32_t* blob_mid_bits, std::uint8_t* out,
                      std::size_t out_capacity, std::uint32_t* group_sizes,
                      std::uint32_t* group_offsets, std::size_t* total_bytes);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_ENTROPY_H_
