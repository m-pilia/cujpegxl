// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// GPU validation of the C ABI encode path: uploads an NV12 image to the device,
// drives cujpegxl_encoder_encode over the full pipeline, and decodes the
// resulting file through libjxl's public decoder to the expected dimensions.

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
    ASSERT_EQ(cujpegxl_encoder_create(&config, &encoder), CUJPEGXL_OK);
    ASSERT_NE(encoder, nullptr);

    std::vector<std::uint8_t> buffer(bound, 0);
    cujpegxl_output output{buffer.data(), 0, buffer.size()};
    const cujpegxl_nv12_input input{reinterpret_cast<std::uintptr_t>(nv12.luma),
                                    reinterpret_cast<std::uintptr_t>(nv12.chroma), nv12.luma_pitch,
                                    nv12.chroma_pitch};

    ASSERT_EQ(cujpegxl_encoder_encode(encoder, &input, &output), CUJPEGXL_OK);
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
    ASSERT_EQ(cujpegxl_encoder_create(&config, &encoder), CUJPEGXL_OK);

    std::vector<std::uint8_t> buffer(16, 0);
    cujpegxl_output output{buffer.data(), 0, buffer.size()};
    const cujpegxl_nv12_input input{reinterpret_cast<std::uintptr_t>(nv12.luma),
                                    reinterpret_cast<std::uintptr_t>(nv12.chroma), nv12.luma_pitch,
                                    nv12.chroma_pitch};

    EXPECT_EQ(cujpegxl_encoder_encode(encoder, &input, &output), CUJPEGXL_BUFFER_TOO_SMALL);

    cujpegxl_encoder_destroy(encoder);
}

TEST(CujpegxlEncodeGpu, AsyncFutureSupportsWaitAndOutputRetry) {
    const cujpegxl_config frame_config{512, 512, 1.0f, 0};
    const cujpegxl_async_config config{frame_config, 2};

    DeviceNv12 nv12{};
    ASSERT_TRUE(upload_nv12(frame_config.width, frame_config.height, nv12));

    cujpegxl_encoder* encoder{nullptr};
    ASSERT_EQ(cujpegxl_async_encoder_create(&config, &encoder), CUJPEGXL_OK);
    const cujpegxl_nv12_input input{reinterpret_cast<std::uintptr_t>(nv12.luma),
                                    reinterpret_cast<std::uintptr_t>(nv12.chroma), nv12.luma_pitch,
                                    nv12.chroma_pitch};

    cujpegxl_future* future{nullptr};
    ASSERT_EQ(cujpegxl_encoder_submit(encoder, &input, 42, &future), CUJPEGXL_OK);
    ASSERT_NE(future, nullptr);
    ASSERT_EQ(cujpegxl_future_wait(future, 60'000'000'000), CUJPEGXL_OK);

    int ready{0};
    ASSERT_EQ(cujpegxl_future_ready(future, &ready), CUJPEGXL_OK);
    EXPECT_EQ(ready, 1);

    std::vector<std::uint8_t> buffer(16, 0);
    cujpegxl_output output{buffer.data(), 0, buffer.size()};
    EXPECT_EQ(cujpegxl_future_get(future, &output, nullptr), CUJPEGXL_BUFFER_TOO_SMALL);
    ASSERT_GT(output.size, buffer.size());
    buffer.resize(output.size);
    output.data = buffer.data();
    output.capacity = buffer.size();

    std::uint64_t sequence{0};
    ASSERT_EQ(cujpegxl_future_get(future, &output, &sequence), CUJPEGXL_OK);
    EXPECT_EQ(sequence, 42u);
    EXPECT_GT(output.size, 0u);

    cujpegxl_future_destroy(future);
    cujpegxl_encoder_destroy(encoder);
}

TEST(CujpegxlEncodeGpu, RejectsUnsupportedResolution) {
    const cujpegxl_config config{100, 100, 1.0f, 0};
    cujpegxl_encoder* encoder{nullptr};
    EXPECT_EQ(cujpegxl_encoder_create(&config, &encoder), CUJPEGXL_UNSUPPORTED_RESOLUTION);
    EXPECT_EQ(encoder, nullptr);
}

}  // namespace
}  // namespace cujpegxl
