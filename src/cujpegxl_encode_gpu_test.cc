// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// GPU validation of the C ABI encode path: uploads an NV12 image to the device,
// drives cujpegxl_encoder_submit/receive over the full pipeline, and decodes the
// resulting files through libjxl's public decoder to the expected dimensions.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include <jxl/decode.h>

#include <gtest/gtest.h>

#include "cujpegxl/cujpegxl.h"

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

bool upload_nv12(std::uint32_t width, std::uint32_t height, DeviceNv12& out) {
    std::vector<std::uint8_t> luma(static_cast<std::size_t>(width) * height, 0);
    std::vector<std::uint8_t> chroma(static_cast<std::size_t>(width) * (height / 2), 0);
    for (std::uint32_t y{0}; y < height; ++y) {
        for (std::uint32_t x{0}; x < width; ++x) {
            luma[static_cast<std::size_t>(y) * width + x] =
                static_cast<std::uint8_t>((x * 3 + y * 5) & 0xFF);
        }
    }
    for (std::uint32_t y{0}; y < height / 2; ++y) {
        for (std::uint32_t x{0}; x < width; ++x) {
            chroma[static_cast<std::size_t>(y) * width + x] =
                static_cast<std::uint8_t>((x * 7 + y * 11) & 0xFF);
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

testing::AssertionResult decode_dims(const std::uint8_t* data, std::size_t size, std::uint32_t& xs,
                                     std::uint32_t& ys) {
    JxlDecoder* dec{JxlDecoderCreate(nullptr)};
    if (JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) !=
        JXL_DEC_SUCCESS) {
        JxlDecoderDestroy(dec);
        return testing::AssertionFailure() << "SubscribeEvents failed";
    }
    JxlDecoderSetInput(dec, data, size);
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

TEST(CujpegxlEncodeGpu, EncodesDeviceNv12AndDecodes) {
    const cujpegxl_config config{512, 512, 1.0f, 0};

    DeviceNv12 nv12{};
    ASSERT_TRUE(upload_nv12(config.width, config.height, nv12));

    std::size_t bound{0};
    ASSERT_EQ(cujpegxl_max_output_size(&config, &bound), CUJPEGXL_OK);
    ASSERT_GT(bound, 0u);

    cujpegxl_encoder* encoder{nullptr};
    ASSERT_EQ(cujpegxl_encoder_create(&config, 1, &encoder), CUJPEGXL_OK);
    ASSERT_NE(encoder, nullptr);

    std::vector<std::uint8_t> buffer(bound, 0);
    cujpegxl_output output{buffer.data(), 0, buffer.size()};
    const cujpegxl_nv12_input input{reinterpret_cast<std::uintptr_t>(nv12.luma),
                                    reinterpret_cast<std::uintptr_t>(nv12.chroma), nv12.luma_pitch,
                                    nv12.chroma_pitch};

    ASSERT_EQ(cujpegxl_encoder_submit(encoder, &input, &output), CUJPEGXL_OK);
    cujpegxl_output* received{nullptr};
    ASSERT_EQ(cujpegxl_encoder_receive(encoder, &received), CUJPEGXL_OK);
    EXPECT_EQ(received, &output);
    EXPECT_GT(output.size, 0u);
    EXPECT_LE(output.size, output.capacity);

    std::uint32_t xs{0};
    std::uint32_t ys{0};
    ASSERT_TRUE(decode_dims(output.data, output.size, xs, ys));
    EXPECT_EQ(xs, config.width);
    EXPECT_EQ(ys, config.height);

    cujpegxl_encoder_destroy(encoder);
}

TEST(CujpegxlEncodeGpu, TooSmallBufferReports) {
    const cujpegxl_config config{512, 512, 1.0f, 0};

    DeviceNv12 nv12{};
    ASSERT_TRUE(upload_nv12(config.width, config.height, nv12));

    cujpegxl_encoder* encoder{nullptr};
    ASSERT_EQ(cujpegxl_encoder_create(&config, 1, &encoder), CUJPEGXL_OK);

    std::vector<std::uint8_t> buffer(16, 0);
    cujpegxl_output output{buffer.data(), 0, buffer.size()};
    const cujpegxl_nv12_input input{reinterpret_cast<std::uintptr_t>(nv12.luma),
                                    reinterpret_cast<std::uintptr_t>(nv12.chroma), nv12.luma_pitch,
                                    nv12.chroma_pitch};

    ASSERT_EQ(cujpegxl_encoder_submit(encoder, &input, &output), CUJPEGXL_OK);
    cujpegxl_output* received{nullptr};
    EXPECT_EQ(cujpegxl_encoder_receive(encoder, &received), CUJPEGXL_BUFFER_TOO_SMALL);
    EXPECT_EQ(received, &output);
    EXPECT_GT(output.size, buffer.size());

    cujpegxl_encoder_destroy(encoder);
}

TEST(CujpegxlEncodeGpu, RejectsUnsupportedResolution) {
    const cujpegxl_config config{100, 100, 1.0f, 0};
    cujpegxl_encoder* encoder{nullptr};
    EXPECT_EQ(cujpegxl_encoder_create(&config, 1, &encoder), CUJPEGXL_UNSUPPORTED_RESOLUTION);
    EXPECT_EQ(encoder, nullptr);
}

TEST(CujpegxlEncodeGpu, RejectsZeroPipelineDepth) {
    const cujpegxl_config config{512, 512, 1.0f, 0};
    cujpegxl_encoder* encoder{nullptr};
    EXPECT_EQ(cujpegxl_encoder_create(&config, 0, &encoder), CUJPEGXL_INVALID_ARGUMENT);
    EXPECT_EQ(encoder, nullptr);
}

TEST(CujpegxlEncodeGpu, RejectsDeviceChange) {
    const cujpegxl_config config{512, 512, 1.0f, 0};
    cujpegxl_encoder* encoder{nullptr};
    ASSERT_EQ(cujpegxl_encoder_create(&config, 1, &encoder), CUJPEGXL_OK);
    const cujpegxl_config other_device{512, 512, 1.0f, 1};
    EXPECT_EQ(cujpegxl_encoder_configure(encoder, &other_device), CUJPEGXL_INVALID_ARGUMENT);
    cujpegxl_encoder_destroy(encoder);
}

// Frames come back in submission order with their own outputs; the second
// frame, submitted after a configure, has the new distance.
TEST(CujpegxlEncodeGpu, PipelinesFramesInOrder) {
    const cujpegxl_config config{512, 512, 1.0f, 0};
    const cujpegxl_config coarse{512, 512, 4.0f, 0};

    DeviceNv12 nv12{};
    ASSERT_TRUE(upload_nv12(config.width, config.height, nv12));
    const cujpegxl_nv12_input input{reinterpret_cast<std::uintptr_t>(nv12.luma),
                                    reinterpret_cast<std::uintptr_t>(nv12.chroma), nv12.luma_pitch,
                                    nv12.chroma_pitch};
    std::size_t bound{0};
    ASSERT_EQ(cujpegxl_max_output_size(&config, &bound), CUJPEGXL_OK);
    std::vector<std::vector<std::uint8_t>> buffers(3, std::vector<std::uint8_t>(bound, 0));
    std::vector<cujpegxl_output> outputs{};
    for (std::vector<std::uint8_t>& b : buffers) {
        outputs.push_back(cujpegxl_output{b.data(), 0, b.size()});
    }

    cujpegxl_encoder* encoder{nullptr};
    ASSERT_EQ(cujpegxl_encoder_create(&config, 2, &encoder), CUJPEGXL_OK);
    cujpegxl_output* received{nullptr};
    EXPECT_EQ(cujpegxl_encoder_receive(encoder, &received), CUJPEGXL_PIPELINE_EMPTY);
    ASSERT_EQ(cujpegxl_encoder_submit(encoder, &input, &outputs[0]), CUJPEGXL_OK);
    ASSERT_EQ(cujpegxl_encoder_configure(encoder, &coarse), CUJPEGXL_OK);
    ASSERT_EQ(cujpegxl_encoder_submit(encoder, &input, &outputs[1]), CUJPEGXL_OK);
    EXPECT_EQ(cujpegxl_encoder_submit(encoder, &input, &outputs[2]), CUJPEGXL_PIPELINE_FULL);
    ASSERT_EQ(cujpegxl_encoder_receive(encoder, &received), CUJPEGXL_OK);
    EXPECT_EQ(received, &outputs[0]);
    ASSERT_EQ(cujpegxl_encoder_submit(encoder, &input, &outputs[2]), CUJPEGXL_OK);
    ASSERT_EQ(cujpegxl_encoder_receive(encoder, &received), CUJPEGXL_OK);
    EXPECT_EQ(received, &outputs[1]);
    ASSERT_EQ(cujpegxl_encoder_receive(encoder, &received), CUJPEGXL_OK);
    EXPECT_EQ(received, &outputs[2]);
    EXPECT_EQ(cujpegxl_encoder_receive(encoder, &received), CUJPEGXL_PIPELINE_EMPTY);
    cujpegxl_encoder_destroy(encoder);

    EXPECT_LT(outputs[1].size, outputs[0].size);
    EXPECT_EQ(outputs[2].size, outputs[1].size);
    EXPECT_TRUE(std::equal(buffers[1].begin(),
                           buffers[1].begin() + static_cast<std::ptrdiff_t>(outputs[1].size),
                           buffers[2].begin()));
    for (const cujpegxl_output& output : outputs) {
        std::uint32_t xs{0};
        std::uint32_t ys{0};
        ASSERT_TRUE(decode_dims(output.data, output.size, xs, ys));
        EXPECT_EQ(xs, config.width);
        EXPECT_EQ(ys, config.height);
    }
}

}  // namespace
}  // namespace cujpegxl
