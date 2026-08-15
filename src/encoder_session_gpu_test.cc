// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include "frame_encoder.h"

namespace cujpegxl {
namespace {

struct DeviceNv12 {
    std::uint8_t* luma{nullptr};
    std::uint8_t* chroma{nullptr};

    ~DeviceNv12() {
        cudaFree(luma);
        cudaFree(chroma);
    }
};

bool upload_frame(std::size_t width, std::size_t height, DeviceNv12& frame) {
    std::vector<std::uint8_t> luma(width * height, 96);
    std::vector<std::uint8_t> chroma(width * height / 2, 128);
    for (std::size_t y{0}; y < height; ++y) {
        for (std::size_t x{0}; x < width; ++x) {
            luma[y * width + x] = static_cast<std::uint8_t>(32 + (x * 127 / width + y) % 192);
        }
    }
    if (cudaMalloc(&frame.luma, luma.size()) != cudaSuccess ||
        cudaMalloc(&frame.chroma, chroma.size()) != cudaSuccess) {
        return false;
    }
    return cudaMemcpy(frame.luma, luma.data(), luma.size(), cudaMemcpyHostToDevice) ==
               cudaSuccess &&
           cudaMemcpy(frame.chroma, chroma.data(), chroma.size(), cudaMemcpyHostToDevice) ==
               cudaSuccess;
}

EncoderInput input_for(const DeviceNv12& frame, std::size_t width, std::size_t height,
                       std::uint64_t sequence) {
    return EncoderInput{.luma = frame.luma,
                        .luma_pitch = width,
                        .chroma = frame.chroma,
                        .chroma_pitch = width,
                        .width = width,
                        .height = height,
                        .distance = 1.0f,
                        .quant_params = quant_params_for_distance(1.0f),
                        .sequence = sequence};
}

TEST(EncoderSessionGpu, DepthOneMatchesSynchronousInterface) {
    constexpr std::size_t width{640};
    constexpr std::size_t height{384};
    DeviceNv12 frame{};
    ASSERT_TRUE(upload_frame(width, height, frame));

    std::vector<std::uint8_t> synchronous{};
    ASSERT_TRUE(encode_nv12(frame.luma, width, frame.chroma, width, width, height, 0, 1.0f,
                            quant_params_for_distance(1.0f), synchronous));

    std::unique_ptr<EncoderSession> session{
        EncoderSession::create(EncoderConfig{.device_ordinal = 0,
                                             .max_width = width,
                                             .max_height = height,
                                             .pipeline_depth = 1,
                                             .pipeline = EncoderPipeline::DCT8})};
    ASSERT_NE(session, nullptr);
    EncodedFrameFuture future{};
    EncoderInput input{input_for(frame, width, height, 17)};
    input.collect_stats = true;
    ASSERT_TRUE(session->try_encode(input, future));
    EXPECT_TRUE(future.valid());
    EncodedFrame output{};
    ASSERT_TRUE(future.get(output));
    EXPECT_EQ(output.sequence, 17u);
    EXPECT_EQ(output.bytes, synchronous);
    ASSERT_EQ(output.stats.size(), 3u);
    EXPECT_STREQ(output.stats[1].name, "entropy");
    EXPECT_EQ(output.stats[1].phases.size(), 17u);
    EXPECT_EQ(output.stats[1].metrics.size(), 11u);
    EXPECT_GT(output.stats[1].phases[1].gpu_us, 0.0);
    EXPECT_GT(output.stats[1].phases[2].gpu_us, 0.0);
    EXPECT_EQ(output.stats[1].phases[6].gpu_us, 0.0);
    EXPECT_EQ(output.stats[1].metrics[0].value, 1.0);
    EXPECT_EQ(output.stats[1].metrics[1].value, 2.0);
    const double retained_allocations{output.stats[1].metrics[5].value};
    EXPECT_FALSE(future.get(output));

    EncodedFrameFuture repeated_future{};
    input.sequence = 18;
    ASSERT_TRUE(session->try_encode(input, repeated_future));
    EncodedFrame repeated{};
    ASSERT_TRUE(repeated_future.get(repeated));
    EXPECT_EQ(repeated.bytes, synchronous);
    EXPECT_EQ(repeated.stats[1].metrics[5].value, retained_allocations);
}

TEST(EncoderSessionGpu, BoundedPipelineCompletesDeterministically) {
    constexpr std::size_t width{1920};
    constexpr std::size_t height{1080};
    constexpr std::size_t depth{3};
    DeviceNv12 frame{};
    ASSERT_TRUE(upload_frame(width, height, frame));
    std::unique_ptr<EncoderSession> session{
        EncoderSession::create(EncoderConfig{.device_ordinal = 0,
                                             .max_width = width,
                                             .max_height = height,
                                             .pipeline_depth = depth,
                                             .pipeline = EncoderPipeline::DCT8})};
    ASSERT_NE(session, nullptr);

    std::vector<EncodedFrameFuture> futures(depth);
    for (std::size_t i{0}; i < depth; ++i) {
        ASSERT_TRUE(session->try_encode(input_for(frame, width, height, i), futures[i]));
    }
    EncodedFrameFuture rejected{};
    EXPECT_FALSE(session->try_encode(input_for(frame, width, height, depth), rejected));

    std::vector<EncodedFrame> outputs(depth);
    for (std::size_t i{depth}; i > 0; --i) {
        ASSERT_TRUE(futures[i - 1].wait_for(std::chrono::seconds{10}));
        ASSERT_TRUE(futures[i - 1].get(outputs[i - 1]));
        EXPECT_EQ(outputs[i - 1].sequence, i - 1);
    }
    for (std::size_t i{1}; i < depth; ++i) {
        EXPECT_EQ(outputs[i].bytes, outputs[0].bytes);
    }
    session->flush();
}

TEST(EncoderSessionGpu, MixedPipelineSlotsOwnIndependentScratch) {
    constexpr std::size_t width{640};
    constexpr std::size_t height{384};
    constexpr std::size_t depth{2};
    DeviceNv12 frame{};
    ASSERT_TRUE(upload_frame(width, height, frame));
    std::unique_ptr<EncoderSession> session{
        EncoderSession::create(EncoderConfig{.device_ordinal = 0,
                                             .max_width = width,
                                             .max_height = height,
                                             .pipeline_depth = depth,
                                             .pipeline = EncoderPipeline::MIXED})};
    ASSERT_NE(session, nullptr);
    EncodedFrameFuture first{};
    EncodedFrameFuture second{};
    ASSERT_TRUE(session->try_encode(input_for(frame, width, height, 1), first));
    ASSERT_TRUE(session->try_encode(input_for(frame, width, height, 2), second));
    EncodedFrame first_output{};
    EncodedFrame second_output{};
    ASSERT_TRUE(first.get(first_output));
    ASSERT_TRUE(second.get(second_output));
    EXPECT_EQ(first_output.sequence, 1u);
    EXPECT_EQ(second_output.sequence, 2u);
    EXPECT_EQ(first_output.bytes, second_output.bytes);
}

}  // namespace
}  // namespace cujpegxl
