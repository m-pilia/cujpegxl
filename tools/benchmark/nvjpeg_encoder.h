// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_TOOLS_BENCHMARK_NVJPEG_ENCODER_H_
#define CUJPEGXL_TOOLS_BENCHMARK_NVJPEG_ENCODER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include <nvjpeg.h>

namespace cujpegxl::tools {

// Device copy of an NV12 frame in the planar Y/Cb/Cr layout nvjpegEncodeYUV
// takes. Errors throw std::runtime_error.
class DevicePlanarYuv {
public:
    DevicePlanarYuv(const std::uint8_t* nv12, std::uint32_t width, std::uint32_t height);
    ~DevicePlanarYuv();
    DevicePlanarYuv(const DevicePlanarYuv&) = delete;
    DevicePlanarYuv& operator=(const DevicePlanarYuv&) = delete;

    std::uint32_t width() const { return width_; }
    std::uint32_t height() const { return height_; }
    const nvjpegImage_t& image() const { return image_; }

private:
    std::uint32_t width_;
    std::uint32_t height_;
    nvjpegImage_t image_{};
};

struct JpegView {
    const std::uint8_t* data;
    std::size_t size;
};

// 4:2:0 nvJPEG encoder session at a fixed quality, reused across encodes. Errors
// throw std::runtime_error.
class NvjpegEncoder {
public:
    NvjpegEncoder(int quality, std::uint32_t width, std::uint32_t height);
    ~NvjpegEncoder();
    NvjpegEncoder(const NvjpegEncoder&) = delete;
    NvjpegEncoder& operator=(const NvjpegEncoder&) = delete;

    // Encodes synchronously; the returned bytes stay valid until the next call.
    JpegView encode(const DevicePlanarYuv& yuv);

private:
    void release();

    nvjpegHandle_t handle_{nullptr};
    nvjpegEncoderState_t state_{nullptr};
    nvjpegEncoderParams_t params_{nullptr};
    std::vector<std::uint8_t> buffer_{};
};

}  // namespace cujpegxl::tools

#endif  // CUJPEGXL_TOOLS_BENCHMARK_NVJPEG_ENCODER_H_
