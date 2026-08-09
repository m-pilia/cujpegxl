// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_XYB_H_
#define CUJPEGXL_SRC_XYB_H_

#include <cstddef>
#include <cstdint>

namespace cujpegxl {

// Converts a device-resident NV12 image (BT.709 primaries, full range,
// 8-bit 4:2:0) to a device-resident planar XYB float image matching libjxl's
// opsin transform (raw ToXYB output, not ScaleXYB'd).
//
// All pointers are device addresses. `luma` is width x height bytes with the
// given row pitch; `chroma` is the interleaved Cb/Cr plane, (width/2) pairs per
// row over height/2 rows, with `chroma_pitch` bytes per row. `xyb` receives
// three tightly packed width*height float planes (X, then Y, then B),
// row-major.
//
// width and height must be even. Returns false on a CUDA error.
bool nv12_to_xyb(const std::uint8_t* luma, std::size_t luma_pitch, const std::uint8_t* chroma,
                 std::size_t chroma_pitch, std::size_t width, std::size_t height, float* xyb);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_XYB_H_
