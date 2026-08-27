// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_PREFIX_BUILD_H_
#define CUJPEGXL_SRC_PREFIX_BUILD_H_

#include <cstddef>
#include <cstdint>

namespace cujpegxl {

// Fixed byte stride reserved for each DcGroup header blob in the device blob
// buffers. Both header blobs (VarDCTDC modular header, AcMetadata modular header)
// are only a handful of bytes; this bounds them with wide margin so the parallel
// per-group emitters can write at a static offset without a prefix sum. The
// consuming dc_encode_groups reads exactly blob_pre_bits/blob_mid_bits, so the
// padding between blobs is never emitted.
constexpr std::size_t DC_BLOB_STRIDE = 256;

// On-device AcGlobal build (replaces the host build_ac_global + histogram
// round-trip). From the pooled AC cluster histogram `d_ac_hist`
// (AC_NUM_CLUSTERS * HISTOGRAM_STRIDE uint32, device), emits the byte-aligned
// AcGlobal section into `d_section` (its byte length into `d_section_len`) and
// the shared AC prefix code into `d_depth` / `d_bits`
// (AC_NUM_CLUSTERS * HISTOGRAM_STRIDE). The context map is data-independent, so
// its serialization (`d_ctxmap_blob`, `ctxmap_nbits`) is precomputed once on the
// host and reused every frame rather than rebuilt on-device. Byte-identical to
// bitstream::build_ac_global. Deterministic. Returns false on a CUDA error.
bool build_ac_global_device(const std::uint32_t* d_ac_hist, std::size_t num_ac_groups,
                            const std::uint8_t* d_ctxmap_blob, std::size_t ctxmap_nbits,
                            std::uint8_t* d_depth, std::uint16_t* d_bits, std::uint8_t* d_section,
                            std::size_t section_cap, std::uint32_t* d_section_len);

// On-device DcGroup header-blob build (replaces the host build_dc_group_blobs +
// histogram round-trip). Per Dc group, from the group's DC histogram
// `d_dc_hist`, AcMetadata histogram `d_am_hist` (each num_dc_groups *
// HISTOGRAM_STRIDE uint32, device) and first-block count `d_fbc`, emits the two
// header blobs into `d_blob_pre` / `d_blob_mid` at offset g * DC_BLOB_STRIDE
// (offsets into `d_blob_pre_off` / `d_blob_mid_off`, bit lengths into
// `d_blob_pre_bits` / `d_blob_mid_bits`) and the DC and AcMetadata prefix codes
// into `d_dc_depth` / `d_dc_bits` / `d_am_depth` / `d_am_bits`
// (num_dc_groups * HISTOGRAM_STRIDE). Byte-identical to
// bitstream::build_dc_group_blobs. Deterministic. Returns false on a CUDA error.
bool build_dc_group_blobs_device(std::size_t width, std::size_t height,
                                 const std::uint32_t* d_dc_hist, const std::uint32_t* d_am_hist,
                                 const std::size_t* d_fbc, std::uint8_t* d_dc_depth,
                                 std::uint16_t* d_dc_bits, std::uint8_t* d_am_depth,
                                 std::uint16_t* d_am_bits, std::uint8_t* d_blob_pre,
                                 std::uint32_t* d_blob_pre_off, std::uint32_t* d_blob_pre_bits,
                                 std::uint8_t* d_blob_mid, std::uint32_t* d_blob_mid_off,
                                 std::uint32_t* d_blob_mid_bits);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_PREFIX_BUILD_H_
