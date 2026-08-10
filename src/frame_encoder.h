// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_FRAME_ENCODER_H_
#define CUJPEGXL_SRC_FRAME_ENCODER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/bitstream/frame_assembly.h"

namespace cujpegxl {

// Encodes a device-resident quantized DCT8 coefficient buffer into a complete
// ISOBMFF `.jxl` file, returned in `out_file` (host bytes).
//
// `q_device` is the quantizer output: three tightly packed planes (X, Y, B) of
// `width/8 * height/8` blocks in raster order, 64 coefficients each, DC in slot
// 0. width/height are multiples of 8 and must span more than one AC group (the
// M1 ladder; single 256x256 frames use the combined-section layout and are not
// yet handled here).
//
// The AC groups and DcGroups are entropy-coded on the device; the codestream
// headers, FrameHeader, DcGlobal, AcGlobal, TOC and ISOBMFF boxes are assembled
// on the host from the device histograms and section sizes (all O(1)/O(groups)/
// O(alphabet)); the body is gathered into one device buffer and copied out with
// a single bulk D2H. Returns false on a CUDA error, if a coded section exceeds
// its worst-case buffer, or if the frame has a single AC group.
bool encode_frame(const std::int32_t* q_device, std::size_t width, std::size_t height,
                  const bitstream::QuantParams& qp, std::vector<std::uint8_t>& out_file);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_FRAME_ENCODER_H_
