// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// GPU validation of the encode path (encode_nv12): mixed {8,16,32} blocks
// + chroma-from-luma. Encodes a device NV12 image and decodes the result through
// libjxl's public decoder, and checks per-device determinism.

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include <jxl/decode.h>

#include <gtest/gtest.h>

#include "frame_encoder.h"

namespace cujpegxl {
namespace {

struct DeviceNv12 {
    std::uint8_t* luma{nullptr};
    std::uint8_t* chroma{nullptr};
    std::size_t luma_pitch{0};
    std::size_t chroma_pitch{0};
    ~DeviceNv12() {
        cudaFree(luma);
        cudaFree(chroma);
    }
};

// A photo-like NV12: smooth luma gradient with a textured band, so selection
// produces a genuine mix of block sizes and CfL has correlated chroma.
bool upload_nv12(std::size_t width, std::size_t height, DeviceNv12& out) {
    std::vector<std::uint8_t> luma(width * height, 0);
    std::vector<std::uint8_t> chroma(width * (height / 2), 0);
    for (std::size_t y{0}; y < height; ++y) {
        for (std::size_t x{0}; x < width; ++x) {
            const bool textured{x > width / 2};
            const int v{textured ? 96 + (((x + y) & 1) ? 48 : -48)
                                 : static_cast<int>(40 + 160 * x / width)};
            luma[y * width + x] = static_cast<std::uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
    for (std::size_t y{0}; y < height / 2; ++y) {
        for (std::size_t x{0}; x < width; x += 2) {
            chroma[y * width + x] = static_cast<std::uint8_t>(128 + 40 * y / (height / 2));  // Cb
            chroma[y * width + x + 1] = static_cast<std::uint8_t>(128 - 30 * x / width);     // Cr
        }
    }
    out.luma_pitch = width;
    out.chroma_pitch = width;
    if (cudaMalloc(&out.luma, luma.size()) != cudaSuccess ||
        cudaMalloc(&out.chroma, chroma.size()) != cudaSuccess) {
        return false;
    }
    return cudaMemcpy(out.luma, luma.data(), luma.size(), cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy(out.chroma, chroma.data(), chroma.size(), cudaMemcpyHostToDevice) ==
               cudaSuccess;
}

testing::AssertionResult decode_dims(const std::vector<std::uint8_t>& file, std::uint32_t& xs,
                                     std::uint32_t& ys) {
    JxlDecoder* dec{JxlDecoderCreate(nullptr)};
    if (JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) !=
        JXL_DEC_SUCCESS) {
        JxlDecoderDestroy(dec);
        return testing::AssertionFailure() << "SubscribeEvents failed";
    }
    JxlDecoderSetInput(dec, file.data(), file.size());
    JxlDecoderCloseInput(dec);
    const JxlPixelFormat format{3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    std::vector<std::uint8_t> pixels{};
    for (;;) {
        const JxlDecoderStatus status{JxlDecoderProcessInput(dec)};
        if (status == JXL_DEC_ERROR) {
            JxlDecoderDestroy(dec);
            return testing::AssertionFailure() << "decoder returned JXL_DEC_ERROR";
        }
        if (status == JXL_DEC_BASIC_INFO) {
            JxlBasicInfo info{};
            JxlDecoderGetBasicInfo(dec, &info);
            xs = info.xsize;
            ys = info.ysize;
        } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
            pixels.assign(static_cast<std::size_t>(xs) * ys * 3, 0);
            JxlDecoderSetImageOutBuffer(dec, &format, pixels.data(), pixels.size());
        } else if (status == JXL_DEC_FULL_IMAGE || status == JXL_DEC_SUCCESS) {
            break;
        } else {
            JxlDecoderDestroy(dec);
            return testing::AssertionFailure() << "unexpected decoder status " << status;
        }
    }
    JxlDecoderDestroy(dec);
    return testing::AssertionSuccess();
}

std::vector<std::uint8_t> encode(const DeviceNv12& nv12, std::size_t w, std::size_t h) {
    const bitstream::QuantParams qp{quant_params_for_distance(1.0f)};
    std::vector<std::uint8_t> file{};
    EXPECT_TRUE(encode_nv12(nv12.luma, nv12.luma_pitch, nv12.chroma, nv12.chroma_pitch, w, h, 0,
                               1.0f, qp, file, nullptr));
    return file;
}

class EncodeLadder : public testing::TestWithParam<std::pair<std::size_t, std::size_t>> {};

TEST_P(EncodeLadder, DecodesWithLibjxl) {
    const std::size_t w{GetParam().first};
    const std::size_t h{GetParam().second};
    DeviceNv12 nv12{};
    ASSERT_TRUE(upload_nv12(w, h, nv12));
    const std::vector<std::uint8_t> file{encode(nv12, w, h)};
    ASSERT_GT(file.size(), 0u);
    std::uint32_t xs{0};
    std::uint32_t ys{0};
    ASSERT_TRUE(decode_dims(file, xs, ys));
    EXPECT_EQ(xs, w);
    EXPECT_EQ(ys, h);
}

// The ladder rungs are integer 4K downscales with dimensions multiple of 8
// (540p is excluded upstream, 540 not being a multiple of 8), plus a small
// multi-group frame and a partial-edge frame (dimensions not multiples of 256).
INSTANTIATE_TEST_SUITE_P(Ladder, EncodeLadder,
                         testing::Values(std::make_pair(std::size_t{640}, std::size_t{384}),
                                         std::make_pair(std::size_t{1280}, std::size_t{720}),
                                         std::make_pair(std::size_t{1920}, std::size_t{1080})));

TEST(Encode, Deterministic) {
    const std::size_t w{1280};
    const std::size_t h{720};
    DeviceNv12 nv12{};
    ASSERT_TRUE(upload_nv12(w, h, nv12));
    const std::vector<std::uint8_t> a{encode(nv12, w, h)};
    const std::vector<std::uint8_t> b{encode(nv12, w, h)};
    EXPECT_EQ(a, b);
}

}  // namespace
}  // namespace cujpegxl
