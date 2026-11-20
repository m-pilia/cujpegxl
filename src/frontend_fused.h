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

// Textures over a device NV12 image, as the front end samples it: the luma plane
// (width*height bytes at `luma_pitch`), point-sampled as the correctly rounded
// v / 255, and the chroma plane (interleaved Cb/Cr, (width+1)/2 pairs per row
// over (height+1)/2 rows at `chroma_pitch`), linearly filtered. Both pitches
// must be 32-byte aligned.
// Destruction first waits, blocking, for `stream`, where the front end reads the
// textures.
class Nv12Textures {
public:
    Nv12Textures(const std::uint8_t* luma, std::size_t luma_pitch, const std::uint8_t* chroma,
                 std::size_t chroma_pitch, std::size_t width, std::size_t height,
                 cudaStream_t stream);
    Nv12Textures(const Nv12Textures&) = delete;
    Nv12Textures& operator=(const Nv12Textures&) = delete;
    ~Nv12Textures();

    bool valid() const { return valid_; }
    cudaTextureObject_t luma() const { return luma_; }
    cudaTextureObject_t chroma() const { return chroma_; }

private:
    cudaTextureObject_t luma_{};
    cudaTextureObject_t chroma_{};
    cudaStream_t stream_{nullptr};
    cudaEvent_t released_{nullptr};
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
// `image` samples the device NV12 image (see nv12_to_xyb). `distance`
// sets the quantizer scales. `coeffs` receives three channel-major FP16
// covered-block planes, `acs` receives the transform decisions,
// `adaptive_field` the per-block float quant field and `quant_field` its
// integers under `calibration`, whose global scale is libjxl's for the
// adaptive field's median and median absolute deviation, estimated from a
// histogram in `field_histogram` (ADAPTIVE_FIELD_BINS entries of scratch).
// The outputs cover the frame padded to whole blocks (see padded_to_blocks).
// All pointers are device addresses.
// Deterministic. Enqueues on `stream` without waiting; returns false on a
// launch error.
bool encode_frontend(const Nv12Textures& image, std::size_t width, std::size_t height,
                     float distance, __half* coeffs, std::int8_t* acs, float* adaptive_field,
                     std::uint32_t* field_histogram, std::int32_t* quant_field,
                     QuantCalibration* calibration, cudaStream_t stream);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_FRONTEND_FUSED_H_
