// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// End-to-end validation of the device frame encoder: for a set of quantized
// coefficients it must produce the exact ISOBMFF .jxl file the host oracle
// produces (write_container(write_vardct_codestream)), and that file must decode
// through libjxl's public decoder to the expected dimensions.

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include <jxl/decode.h>

#include <gtest/gtest.h>

#include "frame_encoder.h"
#include "src/bitstream/container.h"
#include "tools/bitstream/vardct_frame.h"

namespace cujpegxl {
namespace {

using bitstream::FrameCoefficients;
using bitstream::QuantParams;

std::vector<std::int32_t> flatten_coeffs(const FrameCoefficients& fc) {
    const std::size_t plane{fc.width * fc.height};
    std::vector<std::int32_t> q(3 * plane, 0);
    const std::size_t blocks{(fc.width / 8) * (fc.height / 8)};
    for (int c{0}; c < 3; ++c) {
        for (std::size_t b{0}; b < blocks; ++b) {
            q[static_cast<std::size_t>(c) * plane + b * 64] = fc.dc[c][b];
            for (std::size_t k{1}; k < 64; ++k) {
                q[static_cast<std::size_t>(c) * plane + b * 64 + k] = fc.ac[c][b * 64 + k];
            }
        }
    }
    return q;
}

FrameCoefficients make_frame(std::size_t w, std::size_t h) {
    FrameCoefficients fc{};
    fc.width = w;
    fc.height = h;
    fc.global_scale = 4096;
    fc.quant_dc = 32;
    fc.raw_quant_field = 32;
    const std::size_t bw{w / 8};
    const std::size_t bh{h / 8};
    for (int c{0}; c < 3; ++c) {
        fc.dc[c].assign(bw * bh, 0);
        fc.ac[c].assign(bw * bh * 64, 0);
    }
    for (std::size_t by{0}; by < bh; ++by) {
        for (std::size_t bx{0}; bx < bw; ++bx) {
            const std::size_t b{by * bw + bx};
            for (int c{0}; c < 3; ++c) {
                fc.dc[c][b] = static_cast<std::int32_t>((bx * 3 + by + c) % 517) - 258;
                std::int32_t* blk{&fc.ac[c][b * 64]};
                blk[1] = static_cast<std::int32_t>((bx + c) % 7) - 3;
                blk[8] = static_cast<std::int32_t>((by * 2 + c) % 11) - 5;
                blk[9] = static_cast<std::int32_t>((bx + by) % 5) - 2;
            }
        }
    }
    return fc;
}

bool encode_on_device(const FrameCoefficients& fc, std::vector<std::uint8_t>& out) {
    const std::vector<std::int32_t> q{flatten_coeffs(fc)};
    std::int32_t* d_q{nullptr};
    if (cudaMalloc(&d_q, q.size() * sizeof(std::int32_t)) != cudaSuccess) {
        return false;
    }
    bool ok{cudaMemcpy(d_q, q.data(), q.size() * sizeof(std::int32_t),
                       cudaMemcpyHostToDevice) == cudaSuccess};
    ok = ok && encode_frame(d_q, fc.width, fc.height,
                            QuantParams{fc.global_scale, fc.quant_dc, fc.raw_quant_field}, out);
    cudaFree(d_q);
    return ok;
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

void check(const FrameCoefficients& fc) {
    const std::vector<std::uint8_t> expected{
        bitstream::write_container(write_vardct_codestream(fc))};

    std::vector<std::uint8_t> got{};
    ASSERT_TRUE(encode_on_device(fc, got));

    ASSERT_EQ(got.size(), expected.size());
    for (std::size_t i{0}; i < expected.size(); ++i) {
        ASSERT_EQ(got[i], expected[i]) << "file byte " << i;
    }

    std::uint32_t xs{0};
    std::uint32_t ys{0};
    ASSERT_TRUE(decode_dims(got, xs, ys));
    EXPECT_EQ(xs, fc.width);
    EXPECT_EQ(ys, fc.height);
}

TEST(FrameEncoderGpu, SingleDcGroupMultiAcGroup) { check(make_frame(512, 512)); }

TEST(FrameEncoderGpu, PartialEdgeGroups) { check(make_frame(640, 384)); }

TEST(FrameEncoderGpu, MultiDcGroup) { check(make_frame(2560, 256)); }

TEST(FrameEncoderGpu, Deterministic) {
    const FrameCoefficients fc{make_frame(640, 384)};
    std::vector<std::uint8_t> a{};
    std::vector<std::uint8_t> b{};
    ASSERT_TRUE(encode_on_device(fc, a));
    ASSERT_TRUE(encode_on_device(fc, b));
    EXPECT_EQ(a, b);
}

}  // namespace
}  // namespace cujpegxl
