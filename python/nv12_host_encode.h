// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_PYTHON_NV12_HOST_ENCODE_H_
#define CUJPEGXL_PYTHON_NV12_HOST_ENCODE_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/frame_encoder.h"

// Test-only convenience for the Python binding: production consumers feed
// device-resident NV12 through the C ABI, but the Python wrapper accepts a host
// NumPy array and uploads it here before driving the device encode.
namespace cujpegxl::pybind_support {

// Uploads a host NV12 image (`luma` is width*height bytes, `chroma` is the
// interleaved plane of width*(height/2) bytes) to `device_ordinal` and runs the
// full encode pipeline, returning the `.jxl` file bytes in `out`. When `stats`
// is non-null it receives the per-stage budget-model timings.
bool encode_nv12_host(const std::uint8_t* luma, const std::uint8_t* chroma, std::uint32_t width,
                      std::uint32_t height, float distance, std::int32_t device_ordinal,
                      std::vector<std::uint8_t>& out, std::vector<StageTiming>* stats);

}  // namespace cujpegxl::pybind_support

#endif  // CUJPEGXL_PYTHON_NV12_HOST_ENCODE_H_
