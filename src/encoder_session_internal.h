// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_ENCODER_SESSION_INTERNAL_H_
#define CUJPEGXL_SRC_ENCODER_SESSION_INTERNAL_H_

#include "frame_encoder.h"

namespace cujpegxl {

bool encode_nv12_direct(
    const std::uint8_t* luma, std::size_t luma_pitch,
    const std::uint8_t* chroma, std::size_t chroma_pitch, std::size_t width,
    std::size_t height, std::int32_t device_ordinal, float distance,
    const bitstream::QuantParams& qp, std::vector<std::uint8_t>& out_file,
    std::vector<StageTiming>* stats);

bool encode_nv12_m3_direct(
    const std::uint8_t* luma, std::size_t luma_pitch,
    const std::uint8_t* chroma, std::size_t chroma_pitch, std::size_t width,
    std::size_t height, std::int32_t device_ordinal, float distance,
    const bitstream::QuantParams& qp, std::vector<std::uint8_t>& out_file,
    std::vector<StageTiming>* stats);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_ENCODER_SESSION_INTERNAL_H_
