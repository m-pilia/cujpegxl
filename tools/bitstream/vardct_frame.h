// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_TOOLS_BITSTREAM_VARDCT_FRAME_H_
#define CUJPEGXL_TOOLS_BITSTREAM_VARDCT_FRAME_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cujpegxl::bitstream {

// Quantized coefficients and quantizer parameters for a single-group, all-DCT8
// VarDCT frame. XYB channel order is 0=X, 1=Y, 2=B (the encoder pipeline's
// layout). `width`/`height` are pixel dimensions, multiples of 8 and (for M1's
// single-group path) at most 256 each.
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
    std::array<std::vector<std::int32_t>, 3> dc{};

    // AC per XYB channel: one 8x8 block per image block in raster order, 64
    // coefficients each in libjxl-raster (fx*8+fy) layout. Slot 0 (DC) is
    // ignored here; DC is carried by `dc`.
    std::array<std::vector<std::int32_t>, 3> ac{};
};

// Emits a complete JXL codestream (signature + headers + frame) for `fc` into a
// freshly allocated byte buffer. The output is decodable by libjxl/djxl.
std::vector<std::uint8_t> write_vardct_codestream(const FrameCoefficients& fc);

// The natural (scan) coefficient order for an 8x8 DCT block: order[k] is the
// libjxl-raster index of the k-th coefficient in scan order (order[0] == 0, the
// DC). Exposed for tests.
const std::array<std::uint32_t, 64>& dct8_natural_order();

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_TOOLS_BITSTREAM_VARDCT_FRAME_H_
