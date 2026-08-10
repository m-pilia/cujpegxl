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

// Number of 256x256 AC groups (32x32 blocks each) tiling a width x height image,
// including partial edge groups. width/height must be multiples of 8.
std::size_t ac_num_groups(std::size_t width, std::size_t height);

// Phase 1 of the device entropy coder: accumulate the pooled AC symbol histogram
// over every group. `q` is the quantized DCT8 coefficient buffer (device; three
// planes X, Y, B; blocks in raster order; 64 libjxl-raster coefficients per
// block). `histogram` is a device buffer of AC_HISTOGRAM_SIZE uint32 (zeroed by
// the call). Deterministic (integer atomics). Returns false on a CUDA error.
bool ac_build_histogram(const std::int32_t* q, std::size_t width, std::size_t height,
                        std::uint32_t* histogram);

// Phase 2: emit each AC group's token bitstream (byte-aligned, one AcGroup TOC
// section per group) concatenated into `out`, mirroring the host bitstream
// writer byte-for-byte. `depth`/`bits` are the shared prefix code (device,
// alphabet_size entries) built from the phase-1 histogram. A CUB exclusive scan
// of the per-group byte sizes fixes each group's offset; `group_sizes`,
// `group_offsets` (num_groups entries, device) and `total_bytes` (host) receive
// the layout. Fails if `out_capacity` is exceeded. Deterministic. All device
// pointers except `total_bytes`.
bool ac_encode_groups(const std::int32_t* q, std::size_t width, std::size_t height,
                      const std::uint8_t* depth, const std::uint16_t* bits,
                      std::size_t alphabet_size, std::uint8_t* out,
                      std::size_t out_capacity, std::uint32_t* group_sizes,
                      std::uint32_t* group_offsets, std::size_t* total_bytes);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_ENTROPY_H_
