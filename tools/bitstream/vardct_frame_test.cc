// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// End-to-end validation of the VarDCT codestream writer: the emitted bytes are
// decoded with libjxl's public decoder (the same path djxl uses) and must
// produce an image of the expected dimensions without error. Also checks the
// DCT8 natural scan order against libjxl's own ComputeNaturalCoeffOrder.

#include <cstdint>
#include <vector>

#include <jxl/decode.h>

#include <gtest/gtest.h>

#include "lib/jxl/ac_strategy.h"
#include "lib/jxl/coeff_order.h"

#include "vardct_frame.h"

namespace cujpegxl::bitstream {
namespace {

struct Decoded {
    std::uint32_t xsize{0};
    std::uint32_t ysize{0};
    std::vector<std::uint8_t> pixels{};
};

testing::AssertionResult decode(const std::vector<std::uint8_t>& codestream, Decoded& out) {
    JxlDecoder* dec{JxlDecoderCreate(nullptr)};
    if (JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) !=
        JXL_DEC_SUCCESS) {
        JxlDecoderDestroy(dec);
        return testing::AssertionFailure() << "SubscribeEvents failed";
    }
    JxlDecoderSetInput(dec, codestream.data(), codestream.size());
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
            out.xsize = info.xsize;
            out.ysize = info.ysize;
        } else if (status == JXL_DEC_NEED_IMAGE_OUT_BUFFER) {
            out.pixels.assign(static_cast<std::size_t>(out.xsize) * out.ysize * 3, 0);
            if (JxlDecoderSetImageOutBuffer(dec, &format, out.pixels.data(),
                                            out.pixels.size()) != JXL_DEC_SUCCESS) {
                JxlDecoderDestroy(dec);
                return testing::AssertionFailure() << "SetImageOutBuffer failed";
            }
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

FrameCoefficients make_frame(std::size_t w, std::size_t h) {
    FrameCoefficients fc{};
    fc.width = w;
    fc.height = h;
    fc.global_scale = 4096;
    fc.quant_dc = 32;
    fc.raw_quant_field = 32;
    const std::size_t blocks{(w / 8) * (h / 8)};
    for (int c{0}; c < 3; ++c) {
        fc.dc[c].assign(blocks, 0);
        fc.ac[c].assign(blocks * 64, 0);
    }
    return fc;
}

TEST(VarDctFrame, NaturalOrderMatchesLibjxl) {
    jxl::AcStrategy acs{jxl::AcStrategy::FromRawStrategy(jxl::AcStrategyType::DCT)};
    jxl::coeff_order_t reference[64]{};
    acs.ComputeNaturalCoeffOrder(reference);
    const auto& order{dct8_natural_order()};
    for (std::size_t k{0}; k < 64; ++k) {
        EXPECT_EQ(order[k], reference[k]) << "scan position " << k;
    }
}

TEST(VarDctFrame, AllZeroFrameDecodes) {
    Decoded out{};
    ASSERT_TRUE(decode(write_vardct_codestream(make_frame(256, 256)), out));
    EXPECT_EQ(out.xsize, 256u);
    EXPECT_EQ(out.ysize, 256u);
}

TEST(VarDctFrame, NonZeroDcDecodes) {
    FrameCoefficients fc{make_frame(256, 256)};
    for (int c{0}; c < 3; ++c) {
        for (std::size_t i{0}; i < fc.dc[c].size(); ++i) {
            fc.dc[c][i] = static_cast<std::int32_t>((i % 5) - 2);
        }
    }
    Decoded out{};
    ASSERT_TRUE(decode(write_vardct_codestream(fc), out));
    EXPECT_EQ(out.xsize, 256u);
    EXPECT_EQ(out.ysize, 256u);
}

TEST(VarDctFrame, NonZeroAcDecodes) {
    FrameCoefficients fc{make_frame(256, 256)};
    const std::size_t bw{32};
    for (std::size_t by{0}; by < 32; ++by) {
        for (std::size_t bx{0}; bx < 32; ++bx) {
            for (int c{0}; c < 3; ++c) {
                std::int32_t* blk{&fc.ac[c][(by * bw + bx) * 64]};
                blk[1] = 3;    // a low-frequency AC coefficient
                blk[8] = -2;
                blk[20] = 1;
            }
        }
    }
    Decoded out{};
    ASSERT_TRUE(decode(write_vardct_codestream(fc), out));
    EXPECT_EQ(out.xsize, 256u);
    EXPECT_EQ(out.ysize, 256u);
}

TEST(VarDctFrame, SmallerDimensionsDecode) {
    Decoded out{};
    ASSERT_TRUE(decode(write_vardct_codestream(make_frame(64, 128)), out));
    EXPECT_EQ(out.xsize, 64u);
    EXPECT_EQ(out.ysize, 128u);
}

}  // namespace
}  // namespace cujpegxl::bitstream
