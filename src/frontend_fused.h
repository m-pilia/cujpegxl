// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_FRONTEND_FUSED_H_
#define CUJPEGXL_SRC_FRONTEND_FUSED_H_

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "quant_calibration.h"

namespace cujpegxl {

// Linearly filtered texture over a device NV12 chroma plane (interleaved Cb/Cr,
// (width/2) pairs per row over height/2 rows at `chroma_pitch`, which must be
// 32-byte aligned), as the front end samples it. Destruction first waits for
// `stream`, where the front end reads the texture.
class ChromaTexture {
public:
    ChromaTexture(const std::uint8_t* chroma, std::size_t chroma_pitch, std::size_t width,
                  std::size_t height, cudaStream_t stream);
    ChromaTexture(const ChromaTexture&) = delete;
    ChromaTexture& operator=(const ChromaTexture&) = delete;
    ~ChromaTexture();

    bool valid() const { return valid_; }
    cudaTextureObject_t get() const { return texture_; }

private:
    cudaTextureObject_t texture_{};
    cudaStream_t stream_{nullptr};
    bool valid_{false};
};

// Bins of the adaptive quant field histogram encode_frontend builds.
inline constexpr std::uint32_t ADAPTIVE_FIELD_BINS = 4096;

// Fused front end: collapses NV12 -> XYB -> adaptive quant -> gaborish ->
// transform selection -> forward DCT into one tile-resident kernel. Each CUDA
// block processes a 32x32 pixel region and writes only the adaptive quant field,
// transform decisions, and FP16 coefficients to global memory. The quantizer
// calibration and the integer quant field follow from the adaptive field.
//
// `luma` is the device NV12 luma plane (see nv12_to_xyb), width*height bytes at
// `luma_pitch`; `chroma` samples the chroma plane of the same image. `distance`
// sets the quantizer scales. `coeffs` receives three channel-major FP16
// covered-block planes, `acs` receives the transform decisions,
// `adaptive_field` the per-block float quant field and `quant_field` its
// integers under `calibration`, whose global scale is libjxl's for the
// adaptive field's median and median absolute deviation, estimated from a
// histogram in `field_histogram` (ADAPTIVE_FIELD_BINS entries of scratch).
// width and height must be multiples of 8. All pointers are device addresses.
// Deterministic. Enqueues on `stream` without waiting; returns false on a
// launch error.
bool encode_frontend(const std::uint8_t* luma, std::size_t luma_pitch, const ChromaTexture& chroma,
                     std::size_t width, std::size_t height, float distance, __half* coeffs,
                     std::int8_t* acs, float* adaptive_field, std::uint32_t* field_histogram,
                     std::int32_t* quant_field, QuantCalibration* calibration, cudaStream_t stream);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_FRONTEND_FUSED_H_
