// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_FRONTEND_TRANSFORM_H_
#define CUJPEGXL_SRC_FRONTEND_TRANSFORM_H_

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>

namespace cujpegxl {

// Runs the forward variable-size DCT of a planar XYB float image under a
// per-block transform decision, writing the unquantized coefficients as FP16 in
// the covered-block layout (src/vardct_layout): for each first-block of side N its
// N*N coefficients (forward_dctN transposed-raster layout) scatter across its
// covered 8x8 positions' 64-slot regions; covered blocks own no coefficients.
//
// `xyb` is three tightly packed width*height float planes (X, Y, B). `acs` is the
// per-8x8-block transform signal (width/8 * height/8, block raster; first-block
// holds its side, covered blocks ACS_COVERED). `coeffs` receives three
// channel-major FP16 planes of (width/8 * height/8) * 64 slots each. width and
// height are multiples of 8. Deterministic. Returns false on a CUDA error.
bool variable_forward_dct(const float* xyb, std::size_t width, std::size_t height,
                          const std::int8_t* acs, __half* coeffs);

// Driver: NV12 -> XYB -> bounded transform selection (select_transforms) ->
// variable forward DCT. `luma`/`chroma` are device NV12 planes (see nv12_to_xyb);
// `distance` drives the selection RD proxy. `coeffs` (three FP16 covered-block
// planes) and `acs` (width/8 * height/8 int8) receive the outputs. The XYB
// scratch is allocated internally. Gaborish-inverse pre-sharpening and the
// adaptive-quant field are applied elsewhere in the front end. Deterministic.
// Returns false on a CUDA error.
bool frontend_transform(const std::uint8_t* luma, std::size_t luma_pitch,
                        const std::uint8_t* chroma, std::size_t chroma_pitch, std::size_t width,
                        std::size_t height, float distance, __half* coeffs, std::int8_t* acs);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_FRONTEND_TRANSFORM_H_
