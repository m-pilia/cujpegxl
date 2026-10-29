// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// GPU validation of the encode path (FrameEncoder): mixed {8,16,32} blocks +
// chroma-from-luma. Encodes a device NV12 image and decodes the result through
// libjxl's public decoder, and checks per-device determinism across encodes,
// worker thread counts and pipeline depths.

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
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
// produces a genuine mix of block sizes and CfL has correlated chroma. A flat
// image instead quantizes every AC coefficient to zero.
bool upload_nv12(std::size_t width, std::size_t height, DeviceNv12& out, bool flat = false) {
    std::vector<std::uint8_t> luma(width * height, 0);
    std::vector<std::uint8_t> chroma(width * (height / 2), 0);
    for (std::size_t y{0}; y < height; ++y) {
        for (std::size_t x{0}; x < width; ++x) {
            const bool textured{x > width / 2};
            const int v{textured ? 96 + (((x + y) & 1) ? 48 : -48)
                                 : static_cast<int>(40 + 160 * x / width)};
            luma[y * width + x] =
                flat ? 100 : static_cast<std::uint8_t>(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
    for (std::size_t y{0}; y < height / 2; ++y) {
        for (std::size_t x{0}; x < width; x += 2) {
            chroma[y * width + x] =
                flat ? 128 : static_cast<std::uint8_t>(128 + 40 * y / (height / 2));  // Cb
            chroma[y * width + x + 1] =
                flat ? 128 : static_cast<std::uint8_t>(128 - 30 * x / width);  // Cr
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

bool submit(FrameEncoder& encoder, const DeviceNv12& nv12, std::size_t w, std::size_t h,
            float distance, std::vector<std::uint8_t>& file) {
    file.assign(max_encoded_size(w, h), 0);
    return encoder.submit(nv12.luma, nv12.luma_pitch, nv12.chroma, nv12.chroma_pitch, nullptr, w, h,
                          distance, file.data(), file.size());
}

std::vector<std::uint8_t> encode(FrameEncoder& encoder, const DeviceNv12& nv12, std::size_t w,
                                 std::size_t h, float distance = 1.0f) {
    std::vector<std::uint8_t> file{};
    if (!submit(encoder, nv12, w, h, distance, file)) {
        ADD_FAILURE() << "submit failed";
        return {};
    }
    const EncodeResult result{encoder.receive()};
    EXPECT_EQ(result.status, EncodeStatus::OK);
    file.resize(result.file_size);
    return file;
}

class EncodeLadder : public testing::TestWithParam<std::pair<std::size_t, std::size_t>> {};

TEST_P(EncodeLadder, DecodesWithLibjxl) {
    const std::size_t w{GetParam().first};
    const std::size_t h{GetParam().second};
    DeviceNv12 nv12{};
    ASSERT_TRUE(upload_nv12(w, h, nv12));
    FrameEncoder encoder{0, 4, 1};
    const std::vector<std::uint8_t> file{encode(encoder, nv12, w, h)};
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
    FrameEncoder encoder{0, 4, 1};
    const std::vector<std::uint8_t> a{encode(encoder, nv12, w, h)};
    const std::vector<std::uint8_t> b{encode(encoder, nv12, w, h)};
    EXPECT_EQ(a, b);
}

TEST(Encode, IndependentOfWorkerThreads) {
    const std::size_t w{1920};
    const std::size_t h{1080};
    DeviceNv12 nv12{};
    ASSERT_TRUE(upload_nv12(w, h, nv12));
    FrameEncoder single{0, 1, 1};
    const std::vector<std::uint8_t> expected{encode(single, nv12, w, h)};
    for (std::size_t threads : {2, 3, 8}) {
        FrameEncoder encoder{0, threads, 1};
        EXPECT_EQ(encode(encoder, nv12, w, h), expected) << threads << " threads";
    }
}

// All AC tokens are zero counts, one symbol per cluster: the AC groups take
// prefix codes, as libjxl codes such sets.
TEST(Encode, FlatImageDecodes) {
    const std::size_t w{640};
    const std::size_t h{384};
    DeviceNv12 nv12{};
    ASSERT_TRUE(upload_nv12(w, h, nv12, true));
    FrameEncoder encoder{0, 4, 1};
    const std::vector<std::uint8_t> file{encode(encoder, nv12, w, h)};
    std::uint32_t xs{0};
    std::uint32_t ys{0};
    ASSERT_TRUE(decode_dims(file, xs, ys));
    EXPECT_EQ(xs, w);
    EXPECT_EQ(ys, h);
}

struct PipelineFrame {
    std::size_t width;
    std::size_t height;
    float distance;
    bool flat;
};

// Frames of different sizes, contents and distances in flight together, so
// that consecutive frames reuse slots sized for other frames.
class Pipeline : public testing::Test {
protected:
    static constexpr PipelineFrame FRAMES[]{
        {1280, 720, 1.0f, false}, {640, 384, 4.0f, false}, {1920, 1080, 1.0f, false},
        {1280, 720, 2.5f, false}, {640, 384, 1.0f, true},  {1920, 1080, 4.0f, false},
    };
    static constexpr std::size_t NUM_FRAMES{std::size(FRAMES)};

    void SetUp() override {
        FrameEncoder serial{0, 4, 1};
        for (std::size_t i{0}; i < NUM_FRAMES; ++i) {
            const PipelineFrame& f{FRAMES[i]};
            images_[i] = std::make_unique<DeviceNv12>();
            ASSERT_TRUE(upload_nv12(f.width, f.height, *images_[i], f.flat));
            expected_[i] = encode(serial, *images_[i], f.width, f.height, f.distance);
            ASSERT_FALSE(expected_[i].empty());
        }
    }

    // Encodes the frames `rounds` times over, keeping `depth` in flight, and
    // checks each against its serial encode.
    void expect_serial_output(std::size_t threads, std::size_t depth, std::size_t rounds) {
        FrameEncoder encoder{0, threads, depth};
        std::vector<std::vector<std::uint8_t>> files(depth);
        std::size_t received{0};
        const auto receive_one = [&] {
            const std::vector<std::uint8_t>& file{files[received % depth]};
            const std::vector<std::uint8_t>& expected{expected_[received % NUM_FRAMES]};
            const EncodeResult result{encoder.receive()};
            ASSERT_EQ(result.status, EncodeStatus::OK);
            ASSERT_EQ(result.file_size, expected.size());
            EXPECT_TRUE(std::equal(expected.begin(), expected.end(), file.begin()))
                << "frame " << received << ", " << threads << " threads, depth " << depth;
            ++received;
        };
        for (std::size_t k{0}; k < rounds * NUM_FRAMES; ++k) {
            if (encoder.in_flight() == depth) {
                receive_one();
            }
            const PipelineFrame& f{FRAMES[k % NUM_FRAMES]};
            ASSERT_TRUE(submit(encoder, *images_[k % NUM_FRAMES], f.width, f.height, f.distance,
                               files[k % depth]));
        }
        while (encoder.in_flight() != 0) {
            receive_one();
        }
    }

    std::unique_ptr<DeviceNv12> images_[NUM_FRAMES];
    std::vector<std::uint8_t> expected_[NUM_FRAMES];
};

TEST_F(Pipeline, MatchesSerialEncodes) {
    for (std::size_t depth : {2, 3, 4}) {
        expect_serial_output(4, depth, 2);
    }
}

TEST_F(Pipeline, IndependentOfWorkerThreads) {
    for (std::size_t threads : {1, 3}) {
        expect_serial_output(threads, 3, 1);
    }
}

TEST_F(Pipeline, DestructionCompletesFramesInFlight) {
    std::vector<std::vector<std::uint8_t>> files(3);
    {
        FrameEncoder encoder{0, 4, 3};
        for (std::size_t i{0}; i < files.size(); ++i) {
            const PipelineFrame& f{FRAMES[i]};
            ASSERT_TRUE(submit(encoder, *images_[i], f.width, f.height, f.distance, files[i]));
        }
    }
    for (std::size_t i{0}; i < files.size(); ++i) {
        EXPECT_TRUE(std::equal(expected_[i].begin(), expected_[i].end(), files[i].begin()))
            << "frame " << i;
    }
}

}  // namespace
}  // namespace cujpegxl
