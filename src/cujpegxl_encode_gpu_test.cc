// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// GPU validation of the C ABI encode path: uploads an NV12 image to the device,
// drives cujpegxl_encoder_submit/receive over the full pipeline, and decodes the
// resulting files through libjxl's public decoder to the expected dimensions
// and, for smooth images, close to the source pixels.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <utility>
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

std::size_t chroma_row_bytes(std::uint32_t width) {
    return (static_cast<std::size_t>(width) + 1) / 2 * 2;
}

std::size_t chroma_rows(std::uint32_t height) {
    return (static_cast<std::size_t>(height) + 1) / 2;
}

// Tightly packed host NV12 planes.
struct HostNv12 {
    std::uint32_t width{0};
    std::uint32_t height{0};
    std::vector<std::uint8_t> luma;
    std::vector<std::uint8_t> chroma;

    HostNv12(std::uint32_t w, std::uint32_t h)
        : width{w},
          height{h},
          luma(static_cast<std::size_t>(w) * h, 0),
          chroma(chroma_row_bytes(w) * chroma_rows(h), 0) {}

    std::size_t y_index(std::size_t x, std::size_t y) const { return y * width + x; }
    std::size_t cb_index(std::size_t cx, std::size_t cy) const {
        return cy * chroma_row_bytes(width) + 2 * cx;
    }
};

HostNv12 sawtooth_nv12(std::uint32_t width, std::uint32_t height) {
    HostNv12 image{width, height};
    for (std::size_t y{0}; y < height; ++y) {
        for (std::size_t x{0}; x < width; ++x) {
            image.luma[image.y_index(x, y)] = static_cast<std::uint8_t>((x * 3 + y * 5) & 0xFF);
        }
    }
    for (std::size_t y{0}; y < chroma_rows(height); ++y) {
        for (std::size_t x{0}; x < chroma_row_bytes(width); ++x) {
            image.chroma[y * chroma_row_bytes(width) + x] =
                static_cast<std::uint8_t>((x * 7 + y * 11) & 0xFF);
        }
    }
    return image;
}

// Gradients over the whole frame, which an encode reproduces closely, so that
// a wrong pixel near the image edge stands out.
HostNv12 gradient_nv12(std::uint32_t width, std::uint32_t height) {
    HostNv12 image{width, height};
    for (std::size_t y{0}; y < height; ++y) {
        for (std::size_t x{0}; x < width; ++x) {
            image.luma[image.y_index(x, y)] =
                static_cast<std::uint8_t>(40 + 140 * x / width + 40 * y / height);
        }
    }
    for (std::size_t cy{0}; cy < chroma_rows(height); ++cy) {
        for (std::size_t cx{0}; cx < chroma_row_bytes(width) / 2; ++cx) {
            image.chroma[image.cb_index(cx, cy)] =
                static_cast<std::uint8_t>(104 + 48 * cx * 2 / width);
            image.chroma[image.cb_index(cx, cy) + 1] =
                static_cast<std::uint8_t>(148 - 40 * cy * 2 / height);
        }
    }
    return image;
}

// BT.709 full range, each chroma sample covering its 2x2 pixels.
std::vector<std::uint8_t> nv12_to_rgb(const HostNv12& image) {
    std::vector<std::uint8_t> rgb(image.luma.size() * 3, 0);
    const auto to_u8 = [](double v) {
        return static_cast<std::uint8_t>(std::lround(std::clamp(v, 0.0, 255.0)));
    };
    for (std::size_t y{0}; y < image.height; ++y) {
        for (std::size_t x{0}; x < image.width; ++x) {
            const std::size_t c{image.cb_index(x / 2, y / 2)};
            const double luma{static_cast<double>(image.luma[image.y_index(x, y)])};
            const double cb{image.chroma[c] - 128.0};
            const double cr{image.chroma[c + 1] - 128.0};
            std::uint8_t* const pixel{&rgb[(y * image.width + x) * 3]};
            pixel[0] = to_u8(luma + 1.5748 * cr);
            pixel[1] = to_u8(luma - 0.187324 * cb - 0.468124 * cr);
            pixel[2] = to_u8(luma + 1.8556 * cb);
        }
    }
    return rgb;
}

// `pitched` pads the rows past the width to cudaMallocPitch's aligned pitch;
// otherwise the rows are tight.
bool upload_nv12(const HostNv12& image, DeviceNv12& out, bool pitched = false) {
    const std::size_t luma_width{image.width};
    const std::size_t chroma_width{chroma_row_bytes(image.width)};
    const std::size_t rows{image.height};
    if (pitched) {
        if (cudaMallocPitch(&out.luma, &out.luma_pitch, luma_width, rows) != cudaSuccess ||
            cudaMallocPitch(&out.chroma, &out.chroma_pitch, chroma_width,
                            chroma_rows(image.height)) != cudaSuccess) {
            return false;
        }
    } else {
        out.luma_pitch = luma_width;
        out.chroma_pitch = chroma_width;
        if (cudaMalloc(&out.luma, image.luma.size()) != cudaSuccess ||
            cudaMalloc(&out.chroma, image.chroma.size()) != cudaSuccess) {
            return false;
        }
    }
    return cudaMemcpy2D(out.luma, out.luma_pitch, image.luma.data(), luma_width, luma_width, rows,
                        cudaMemcpyHostToDevice) == cudaSuccess &&
           cudaMemcpy2D(out.chroma, out.chroma_pitch, image.chroma.data(), chroma_width,
                        chroma_width, chroma_rows(image.height),
                        cudaMemcpyHostToDevice) == cudaSuccess;
}

void CUDART_CB delay_stream(void* /*user_data*/) {
    std::this_thread::sleep_for(std::chrono::milliseconds{200});
}

struct DecodedImage {
    std::uint32_t xs{0};
    std::uint32_t ys{0};
    std::vector<std::uint8_t> rgb;
};

testing::AssertionResult decode(const std::uint8_t* data, std::size_t size, DecodedImage& out) {
    JxlDecoder* dec{JxlDecoderCreate(nullptr)};
    if (JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) !=
        JXL_DEC_SUCCESS) {
        JxlDecoderDestroy(dec);
        return testing::AssertionFailure() << "SubscribeEvents failed";
    }
    JxlDecoderSetInput(dec, data, size);
    JxlDecoderCloseInput(dec);
    const JxlPixelFormat format{3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    for (;;) {
        const JxlDecoderStatus status{JxlDecoderProcessInput(dec)};
        if (status == JXL_DEC_ERROR) {
            JxlDecoderDestroy(dec);
            return testing::AssertionFailure() << "decoder returned JXL_DEC_ERROR";
        }
        if (status == JXL_DEC_BASIC_INFO) {
            JxlBasicInfo info{};
            JxlDecoderGetBasicInfo(dec, &info);
            out.xs = info.xsize;
            out.ys = info.ysize;
        } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
            out.rgb.assign(static_cast<std::size_t>(out.xs) * out.ys * 3, 0);
            JxlDecoderSetImageOutBuffer(dec, &format, out.rgb.data(), out.rgb.size());
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
    const cujpegxl_config config{512, 512, 1.0f};

    DeviceNv12 nv12{};
    ASSERT_TRUE(upload_nv12(sawtooth_nv12(config.width, config.height), nv12));

    std::size_t bound{0};
    ASSERT_EQ(cujpegxl_max_output_size(&config, &bound), CUJPEGXL_OK);
    ASSERT_GT(bound, 0u);

    cujpegxl_encoder* encoder{nullptr};
    ASSERT_EQ(cujpegxl_encoder_create(&config, 0, 1, &encoder), CUJPEGXL_OK);
    ASSERT_NE(encoder, nullptr);

    std::vector<std::uint8_t> buffer(bound, 0);
    cujpegxl_output output{buffer.data(), 0, buffer.size()};
    const cujpegxl_nv12_input input{nv12.luma, nv12.chroma, nv12.luma_pitch, nv12.chroma_pitch,
                                    nullptr};

    ASSERT_EQ(cujpegxl_encoder_submit(encoder, &input, &output), CUJPEGXL_OK);
    cujpegxl_output* received{nullptr};
    ASSERT_EQ(cujpegxl_encoder_receive(encoder, &received), CUJPEGXL_OK);
    EXPECT_EQ(received, &output);
    EXPECT_GT(output.size, 0u);
    EXPECT_LE(output.size, output.capacity);

    DecodedImage decoded{};
    ASSERT_TRUE(decode(output.data, output.size, decoded));
    EXPECT_EQ(decoded.xs, config.width);
    EXPECT_EQ(decoded.ys, config.height);

    cujpegxl_encoder_destroy(encoder);
}

TEST(CujpegxlEncodeGpu, TooSmallBufferReports) {
    const cujpegxl_config config{512, 512, 1.0f};

    DeviceNv12 nv12{};
    ASSERT_TRUE(upload_nv12(sawtooth_nv12(config.width, config.height), nv12));

    cujpegxl_encoder* encoder{nullptr};
    ASSERT_EQ(cujpegxl_encoder_create(&config, 0, 1, &encoder), CUJPEGXL_OK);

    std::vector<std::uint8_t> buffer(16, 0);
    cujpegxl_output output{buffer.data(), 0, buffer.size()};
    const cujpegxl_nv12_input input{nv12.luma, nv12.chroma, nv12.luma_pitch, nv12.chroma_pitch,
                                    nullptr};

    ASSERT_EQ(cujpegxl_encoder_submit(encoder, &input, &output), CUJPEGXL_OK);
    cujpegxl_output* received{nullptr};
    EXPECT_EQ(cujpegxl_encoder_receive(encoder, &received), CUJPEGXL_BUFFER_TOO_SMALL);
    EXPECT_EQ(received, &output);
    EXPECT_GT(output.size, buffer.size());

    cujpegxl_encoder_destroy(encoder);
}

// The planes are copied on a caller stream held back by a host delay, and the
// frame is submitted without synchronizing: an encoder that does not wait for
// that stream reads the planes before they are written.
TEST(CujpegxlEncodeGpu, WaitsForInputStream) {
    const cujpegxl_config config{512, 512, 1.0f};

    DeviceNv12 source{};
    ASSERT_TRUE(upload_nv12(sawtooth_nv12(config.width, config.height), source));
    const std::size_t luma_bytes{source.luma_pitch * config.height};
    const std::size_t chroma_bytes{source.chroma_pitch * (config.height / 2)};
    DeviceNv12 planes{};
    planes.luma_pitch = source.luma_pitch;
    planes.chroma_pitch = source.chroma_pitch;
    ASSERT_EQ(cudaMalloc(&planes.luma, luma_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&planes.chroma, chroma_bytes), cudaSuccess);
    ASSERT_EQ(cudaMemset(planes.luma, 0, luma_bytes), cudaSuccess);
    ASSERT_EQ(cudaMemset(planes.chroma, 0, chroma_bytes), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::size_t bound{0};
    ASSERT_EQ(cujpegxl_max_output_size(&config, &bound), CUJPEGXL_OK);
    std::vector<std::uint8_t> expected(bound, 0);
    std::vector<std::uint8_t> actual(bound, 0);
    cujpegxl_output expected_output{expected.data(), 0, expected.size()};
    cujpegxl_output actual_output{actual.data(), 0, actual.size()};

    cujpegxl_encoder* encoder{nullptr};
    ASSERT_EQ(cujpegxl_encoder_create(&config, 0, 1, &encoder), CUJPEGXL_OK);
    cujpegxl_output* received{nullptr};
    const cujpegxl_nv12_input synchronized{source.luma, source.chroma, source.luma_pitch,
                                           source.chroma_pitch, nullptr};
    ASSERT_EQ(cujpegxl_encoder_submit(encoder, &synchronized, &expected_output), CUJPEGXL_OK);
    ASSERT_EQ(cujpegxl_encoder_receive(encoder, &received), CUJPEGXL_OK);

    cudaStream_t stream{nullptr};
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaLaunchHostFunc(stream, delay_stream, nullptr), cudaSuccess);
    ASSERT_EQ(
        cudaMemcpyAsync(planes.luma, source.luma, luma_bytes, cudaMemcpyDeviceToDevice, stream),
        cudaSuccess);
    ASSERT_EQ(cudaMemcpyAsync(planes.chroma, source.chroma, chroma_bytes, cudaMemcpyDeviceToDevice,
                              stream),
              cudaSuccess);
    const cujpegxl_nv12_input unsynchronized{planes.luma, planes.chroma, planes.luma_pitch,
                                             planes.chroma_pitch, stream};
    ASSERT_EQ(cujpegxl_encoder_submit(encoder, &unsynchronized, &actual_output), CUJPEGXL_OK);
    ASSERT_EQ(cujpegxl_encoder_receive(encoder, &received), CUJPEGXL_OK);
    cujpegxl_encoder_destroy(encoder);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    ASSERT_EQ(actual_output.size, expected_output.size);
    EXPECT_TRUE(std::equal(expected.begin(),
                           expected.begin() + static_cast<std::ptrdiff_t>(expected_output.size),
                           actual.begin()));
}

struct EncodeCase {
    std::uint32_t width;
    std::uint32_t height;
    bool pitched;
};

class EncodesAnySize : public testing::TestWithParam<EncodeCase> {};

// The blocks past the image edge pad it with the edge pixels, so the decoded
// image stays close to the source up to its last row and column.
TEST_P(EncodesAnySize, DecodesCloseToSource) {
    // Coding error of the smooth source at distance 1, with margin.
    constexpr int MAX_ERROR{12};
    const EncodeCase& c{GetParam()};
    const cujpegxl_config config{c.width, c.height, 1.0f};
    const HostNv12 image{gradient_nv12(c.width, c.height)};
    DeviceNv12 nv12{};
    ASSERT_TRUE(upload_nv12(image, nv12, c.pitched));

    std::size_t bound{0};
    ASSERT_EQ(cujpegxl_max_output_size(&config, &bound), CUJPEGXL_OK);
    std::vector<std::uint8_t> buffer(bound, 0);
    cujpegxl_output output{buffer.data(), 0, buffer.size()};
    cujpegxl_encoder* encoder{nullptr};
    ASSERT_EQ(cujpegxl_encoder_create(&config, 0, 1, &encoder), CUJPEGXL_OK);
    const cujpegxl_nv12_input input{nv12.luma, nv12.chroma, nv12.luma_pitch, nv12.chroma_pitch,
                                    nullptr};
    ASSERT_EQ(cujpegxl_encoder_submit(encoder, &input, &output), CUJPEGXL_OK);
    cujpegxl_output* received{nullptr};
    ASSERT_EQ(cujpegxl_encoder_receive(encoder, &received), CUJPEGXL_OK);
    cujpegxl_encoder_destroy(encoder);

    DecodedImage decoded{};
    ASSERT_TRUE(decode(output.data, output.size, decoded));
    ASSERT_EQ(decoded.xs, config.width);
    ASSERT_EQ(decoded.ys, config.height);
    const std::vector<std::uint8_t> expected{nv12_to_rgb(image)};
    ASSERT_EQ(decoded.rgb.size(), expected.size());
    int max_error{0};
    std::size_t worst{0};
    for (std::size_t i{0}; i < expected.size(); ++i) {
        const int error{std::abs(int{decoded.rgb[i]} - int{expected[i]})};
        if (error > max_error) {
            max_error = error;
            worst = i;
        }
    }
    EXPECT_LE(max_error, MAX_ERROR)
        << "at x=" << worst / 3 % config.width << " y=" << worst / 3 / config.width;
}

// Sizes past multiples of 8 in either or both dimensions, down to single rows
// and columns, across an AC group (256px) and a DC group (2048px) edge, with
// tight odd chroma rows and padded ones.
INSTANTIATE_TEST_SUITE_P(Sizes, EncodesAnySize,
                         testing::Values(EncodeCase{264, 264, false}, EncodeCase{257, 1, false},
                                         EncodeCase{1, 257, false}, EncodeCase{257, 257, true},
                                         EncodeCase{643, 389, false}, EncodeCase{643, 389, true},
                                         EncodeCase{1366, 766, false}, EncodeCase{1923, 1081, true},
                                         EncodeCase{2049, 300, false}));

class RejectsResolution : public testing::TestWithParam<std::pair<std::uint32_t, std::uint32_t>> {};

// Frames of at most one AC group (256x256 pixels).
TEST_P(RejectsResolution, AtMostOneAcGroup) {
    const cujpegxl_config config{GetParam().first, GetParam().second, 1.0f};
    std::size_t bound{0};
    EXPECT_EQ(cujpegxl_max_output_size(&config, &bound), CUJPEGXL_UNSUPPORTED_RESOLUTION);
    cujpegxl_encoder* encoder{nullptr};
    EXPECT_EQ(cujpegxl_encoder_create(&config, 0, 1, &encoder), CUJPEGXL_UNSUPPORTED_RESOLUTION);
    EXPECT_EQ(encoder, nullptr);
}

INSTANTIATE_TEST_SUITE_P(SingleGroup, RejectsResolution,
                         testing::Values(std::make_pair(256u, 256u), std::make_pair(256u, 1u),
                                         std::make_pair(1u, 256u), std::make_pair(100u, 100u),
                                         std::make_pair(0u, 512u), std::make_pair(512u, 0u)));

// The chroma rows of an odd width end in a full U/V pair, past the width.
TEST(CujpegxlEncodeGpu, RejectsChromaPitchOfOddWidth) {
    const cujpegxl_config config{643, 389, 1.0f};

    DeviceNv12 nv12{};
    ASSERT_TRUE(upload_nv12(gradient_nv12(config.width, config.height), nv12));
    std::size_t bound{0};
    ASSERT_EQ(cujpegxl_max_output_size(&config, &bound), CUJPEGXL_OK);
    std::vector<std::uint8_t> buffer(bound, 0);
    cujpegxl_output output{buffer.data(), 0, buffer.size()};

    cujpegxl_encoder* encoder{nullptr};
    ASSERT_EQ(cujpegxl_encoder_create(&config, 0, 1, &encoder), CUJPEGXL_OK);
    const cujpegxl_nv12_input input{nv12.luma, nv12.chroma, nv12.luma_pitch, config.width, nullptr};
    EXPECT_EQ(cujpegxl_encoder_submit(encoder, &input, &output), CUJPEGXL_INVALID_ARGUMENT);
    cujpegxl_output* received{nullptr};
    EXPECT_EQ(cujpegxl_encoder_receive(encoder, &received), CUJPEGXL_PIPELINE_EMPTY);
    cujpegxl_encoder_destroy(encoder);
}

TEST(CujpegxlEncodeGpu, RejectsShortPitches) {
    const cujpegxl_config config{512, 512, 1.0f};

    DeviceNv12 nv12{};
    ASSERT_TRUE(upload_nv12(sawtooth_nv12(config.width, config.height), nv12));
    std::size_t bound{0};
    ASSERT_EQ(cujpegxl_max_output_size(&config, &bound), CUJPEGXL_OK);
    std::vector<std::uint8_t> buffer(bound, 0);
    cujpegxl_output output{buffer.data(), 0, buffer.size()};

    cujpegxl_encoder* encoder{nullptr};
    ASSERT_EQ(cujpegxl_encoder_create(&config, 0, 1, &encoder), CUJPEGXL_OK);
    const cujpegxl_nv12_input short_luma{nv12.luma, nv12.chroma, config.width - 1,
                                         nv12.chroma_pitch, nullptr};
    EXPECT_EQ(cujpegxl_encoder_submit(encoder, &short_luma, &output), CUJPEGXL_INVALID_ARGUMENT);
    const cujpegxl_nv12_input short_chroma{nv12.luma, nv12.chroma, nv12.luma_pitch,
                                           config.width - 1, nullptr};
    EXPECT_EQ(cujpegxl_encoder_submit(encoder, &short_chroma, &output), CUJPEGXL_INVALID_ARGUMENT);
    cujpegxl_output* received{nullptr};
    EXPECT_EQ(cujpegxl_encoder_receive(encoder, &received), CUJPEGXL_PIPELINE_EMPTY);
    cujpegxl_encoder_destroy(encoder);
}

TEST(CujpegxlEncodeGpu, RejectsZeroPipelineDepth) {
    const cujpegxl_config config{512, 512, 1.0f};
    cujpegxl_encoder* encoder{nullptr};
    EXPECT_EQ(cujpegxl_encoder_create(&config, 0, 0, &encoder), CUJPEGXL_INVALID_ARGUMENT);
    EXPECT_EQ(encoder, nullptr);
}

TEST(CujpegxlEncodeGpu, RejectsMissingDevice) {
    const cujpegxl_config config{512, 512, 1.0f};
    cujpegxl_encoder* encoder{nullptr};
    EXPECT_EQ(cujpegxl_encoder_create(&config, 9999, 1, &encoder), CUJPEGXL_NO_DEVICE);
    EXPECT_EQ(encoder, nullptr);
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// Frames come back in submission order with their own outputs; the second
// frame, submitted after a configure, has the new distance.
TEST(CujpegxlEncodeGpu, PipelinesFramesInOrder) {
    const cujpegxl_config config{512, 512, 1.0f};
    const cujpegxl_config coarse{512, 512, 4.0f};

    DeviceNv12 nv12{};
    ASSERT_TRUE(upload_nv12(sawtooth_nv12(config.width, config.height), nv12));
    const cujpegxl_nv12_input input{nv12.luma, nv12.chroma, nv12.luma_pitch, nv12.chroma_pitch,
                                    nullptr};
    std::size_t bound{0};
    ASSERT_EQ(cujpegxl_max_output_size(&config, &bound), CUJPEGXL_OK);
    std::vector<std::vector<std::uint8_t>> buffers(3, std::vector<std::uint8_t>(bound, 0));
    std::vector<cujpegxl_output> outputs{};
    for (std::vector<std::uint8_t>& b : buffers) {
        outputs.push_back(cujpegxl_output{b.data(), 0, b.size()});
    }

    cujpegxl_encoder* encoder{nullptr};
    ASSERT_EQ(cujpegxl_encoder_create(&config, 0, 2, &encoder), CUJPEGXL_OK);
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
        DecodedImage decoded{};
        ASSERT_TRUE(decode(output.data, output.size, decoded));
        EXPECT_EQ(decoded.xs, config.width);
        EXPECT_EQ(decoded.ys, config.height);
    }
}

}  // namespace
}  // namespace cujpegxl
