// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_TOOLS_BITSTREAM_VARDCT_FRAME_H_
#define CUJPEGXL_TOOLS_BITSTREAM_VARDCT_FRAME_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cujpegxl::bitstream {

// Quantized coefficients and quantizer parameters for a VarDCT frame with mixed
// square {8,16,32} blocks and chroma-from-luma. XYB channel order is 0=X, 1=Y,
// 2=B (the encoder pipeline's layout). `width`/`height` are pixel dimensions,
// multiples of 8.
struct FrameCoefficients {
    std::size_t width{0};
    std::size_t height{0};

    // Serialized quantizer state (see libjxl Quantizer). global_scale in
    // [1, 2^16], quant_dc in [1, 2^16], raw_quant_field (uniform AC quantizer)
    // in [1, 256].
    std::uint32_t global_scale{1};
    std::uint32_t quant_dc{1};
    std::uint32_t raw_quant_field{1};

    // DC per XYB channel at block resolution (width/8 * height/8), row-major.
    // For a block covered by a larger transform, the entry holds that block's
    // low-frequency (LLF) DC contribution (see src/vardct_layout dc_from_llf).
    std::array<std::vector<std::int32_t>, 3> dc{};

    // AC per XYB channel in the covered-block layout (src/vardct_layout): 64
    // coefficient slots per 8x8 block position (width/8 * height/8 positions), a
    // block of side N packing its N*N coefficients across its covered positions'
    // slots (covered_plane_slot). Slots below a block's LLF are ignored (DC is in
    // `dc`). The coefficients are the CfL residuals the decoder re-correlates.
    std::array<std::vector<std::int32_t>, 3> ac{};

    // Per-8x8-block transform signal (width/8 * height/8, block raster): a
    // first-block holds its side 8/16/32, a covered block holds ACS_COVERED (0).
    // Empty means an all-DCT8 frame (every block is a first-block).
    std::vector<std::int8_t> acs{};

    // Per-64x64-color-tile CfL factors (ceil(width/64) * ceil(height/64), row-
    // major) written into the AcMetadata YtoX/YtoB channels. Empty means the base
    // correlation everywhere (factor 0).
    std::vector<std::int8_t> ytox_map{};
    std::vector<std::int8_t> ytob_map{};
};

// Emits a complete JXL codestream (signature + headers + frame) for `fc` into a
// freshly allocated byte buffer. The output is decodable by libjxl/djxl. With
// `clustered_ac`, the AC coefficients are coded with libjxl's real per-token
// contexts split across 8 entropy clusters instead of a single shared histogram.
std::vector<std::uint8_t> write_vardct_codestream(const FrameCoefficients& fc,
                                                  bool clustered_ac = false);

// The host reference for the device AC entropy encoder: the pooled symbol
// histogram, the shared prefix code (depth/bits), and each AC group's
// byte-aligned token stream (one AcGroup TOC section per group, raster order).
// This is exactly what write_vardct_codestream emits for the AC groups; the
// device port must reproduce `histogram` and `group_streams` byte-for-byte.
struct AcReference {
    std::vector<std::uint32_t> histogram{};
    std::vector<std::uint8_t> depth{};
    std::vector<std::uint16_t> bits{};
    std::vector<std::vector<std::uint8_t>> group_streams{};
};
AcReference reference_ac_encode(const FrameCoefficients& fc);

// The natural (scan) coefficient order for an 8x8 DCT block: order[k] is the
// libjxl-raster index of the k-th coefficient in scan order (order[0] == 0, the
// DC). Exposed for tests.
const std::array<std::uint32_t, 64>& dct8_natural_order();

// The host reference for the device DcGroup coder. Per DcGroup it carries the
// full byte-aligned section (the truth to diff against) plus the pieces that
// drive the device: the two header bit blobs that bracket the device-emitted
// token runs, the two shared prefix codes, the per-group DC histogram, and the
// group's block geometry. The section is exactly what write_vardct_codestream
// emits for that DcGroup; the device must reproduce it byte-for-byte.
//
// Section layout (bit-contiguous, then zero-padded):
//   blob_pre  = extra_precision(2) + VarDCTDC modular header (GroupHeader + tree
//               + DC data histograms)
//   DC tokens = the 3 DC channels (physical order Y, X, B), row-major,
//               PackSigned, prefix-coded with dc_depth/dc_bits  [device]
//   blob_mid  = AcMetadata count-1 + AcMetadata modular header
//   AcMetadata tokens = YtoX, YtoB, ACS+QF, EPF channels, row-major, PackSigned,
//               prefix-coded with acmeta_depth/acmeta_bits      [device]
struct DcGroupReference {
    std::vector<std::uint8_t> section{};

    std::vector<std::uint8_t> blob_pre{};
    std::size_t blob_pre_bits{0};
    std::vector<std::uint8_t> blob_mid{};
    std::size_t blob_mid_bits{0};

    std::vector<std::uint8_t> dc_depth{};
    std::vector<std::uint16_t> dc_bits{};
    std::vector<std::uint8_t> acmeta_depth{};
    std::vector<std::uint16_t> acmeta_bits{};
    std::vector<std::uint32_t> dc_histogram{};
    std::vector<std::uint32_t> acmeta_histogram{};

    std::size_t bx0{0};
    std::size_t by0{0};
    std::size_t dgw{0};
    std::size_t dgh{0};
};
struct DcReference {
    std::vector<DcGroupReference> groups{};
};
DcReference reference_dc_encode(const FrameCoefficients& fc);

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_TOOLS_BITSTREAM_VARDCT_FRAME_H_
