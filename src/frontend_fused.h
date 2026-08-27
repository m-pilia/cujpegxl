// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_FRONTEND_FUSED_H_
#define CUJPEGXL_SRC_FRONTEND_FUSED_H_

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>

namespace cujpegxl {

// Fused front end: collapses NV12 -> XYB -> adaptive quant -> gaborish ->
// transform selection -> forward DCT into one tile-resident kernel. Each CUDA
// block processes a 32x32 pixel region and writes only the quant field, transform
// decisions, and FP16 coefficients to global memory.
//
// `luma`/`chroma` are device NV12 planes (see nv12_to_xyb): luma is width*height
// bytes at `luma_pitch`; chroma is the interleaved Cb/Cr plane at `chroma_pitch`,
// which must be 32-byte aligned (texture requirement). `distance` sets the
// quantizer scales. `coeffs` receives three channel-major FP16 covered-block
// planes, `acs` receives the transform decisions, and `quant_field` receives the
// per-block quant integers. width and height must be multiples of 8. All pointers
// are device addresses. Deterministic. Returns false on a CUDA error.
bool encode_frontend(const std::uint8_t* luma, std::size_t luma_pitch, const std::uint8_t* chroma,
                     std::size_t chroma_pitch, std::size_t width, std::size_t height,
                     float distance, __half* coeffs, std::int8_t* acs,
                     std::int32_t* quant_field);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_FRONTEND_FUSED_H_
