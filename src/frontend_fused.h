// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_FRONTEND_FUSED_H_
#define CUJPEGXL_SRC_FRONTEND_FUSED_H_

#include <cstddef>
#include <cstdint>

namespace cujpegxl {

// Fused adaptive-quant front end: collapses NV12 -> XYB -> masking-based adaptive
// quant into one tile-resident kernel. Each CUDA block processes a tile of blocks,
// loads the tile's NV12 pixels (plus a one-pixel halo) once, computes XYB into
// shared memory, derives the per-block adaptive quant field (libjxl masking
// transcription; the pre-erosion/erosion spatial passes are approximated within
// the tile's shared-memory halo), and writes only the quant field to DRAM — the
// XYB intermediates never touch global memory. The transform-dependent forward
// DCT and quantization run separately (frontend_transform, quantize_residual).
//
// `luma`/`chroma` are device NV12 planes (see nv12_to_xyb): luma is width*height
// bytes at `luma_pitch`; chroma is the interleaved Cb/Cr plane at `chroma_pitch`,
// which must be 32-byte aligned (texture requirement). `distance` sets the
// quantizer scales. `quant_field` receives the per-block quant integers
// ((width/8)*(height/8), block raster order). width and height must be multiples
// of 8. All pointers are device addresses. Deterministic. Returns false on a CUDA
// error.
bool encode_frontend(const std::uint8_t* luma, std::size_t luma_pitch, const std::uint8_t* chroma,
                     std::size_t chroma_pitch, std::size_t width, std::size_t height,
                     float distance, std::int32_t* quant_field);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_FRONTEND_FUSED_H_
