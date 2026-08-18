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

#include "src/vardct_layout.h"

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
            if (JxlDecoderSetImageOutBuffer(dec, &format, out.pixels.data(), out.pixels.size()) !=
                JXL_DEC_SUCCESS) {
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
                blk[1] = 3;  // a low-frequency AC coefficient
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

// Fills every AC group with a distinct low-frequency pattern so the multi-group
// token split (per-group AcGroup sections sharing one AcGlobal prefix code) is
// exercised, not just all-zero groups.
void fill_ac_pattern(FrameCoefficients& fc) {
    const std::size_t bw{fc.width / 8};
    const std::size_t bh{fc.height / 8};
    for (std::size_t by{0}; by < bh; ++by) {
        for (std::size_t bx{0}; bx < bw; ++bx) {
            const std::size_t block{by * bw + bx};
            for (int c{0}; c < 3; ++c) {
                std::int32_t* blk{&fc.ac[c][block * 64]};
                blk[1] = static_cast<std::int32_t>((bx + c) % 5) - 2;
                blk[8] = static_cast<std::int32_t>((by + c) % 3) - 1;
                blk[9] = static_cast<std::int32_t>((bx + by) % 7) - 3;
            }
            for (int c{0}; c < 3; ++c) {
                fc.dc[c][block] = static_cast<std::int32_t>((bx * 3 + by + c) % 9) - 4;
            }
        }
    }
}

TEST(VarDctFrame, MultiGroupAllZeroDecodes) {
    Decoded out{};
    ASSERT_TRUE(decode(write_vardct_codestream(make_frame(512, 512)), out));
    EXPECT_EQ(out.xsize, 512u);
    EXPECT_EQ(out.ysize, 512u);
}

TEST(VarDctFrame, MultiGroupNonZeroDecodes) {
    FrameCoefficients fc{make_frame(512, 512)};
    fill_ac_pattern(fc);
    Decoded out{};
    ASSERT_TRUE(decode(write_vardct_codestream(fc), out));
    EXPECT_EQ(out.xsize, 512u);
    EXPECT_EQ(out.ysize, 512u);
}

TEST(VarDctFrame, MultiGroupPartialEdgeGroupsDecode) {
    // 640x384 -> 80x48 blocks -> 3x2 AC groups, the right and bottom groups
    // being partial (80 % 32 = 16 wide, 48 % 32 = 16 tall).
    FrameCoefficients fc{make_frame(640, 384)};
    fill_ac_pattern(fc);
    Decoded out{};
    ASSERT_TRUE(decode(write_vardct_codestream(fc), out));
    EXPECT_EQ(out.xsize, 640u);
    EXPECT_EQ(out.ysize, 384u);
}

TEST(VarDctFrame, MultiGroupSingleRowOfGroupsDecodes) {
    FrameCoefficients fc{make_frame(1024, 256)};
    fill_ac_pattern(fc);
    Decoded out{};
    ASSERT_TRUE(decode(write_vardct_codestream(fc), out));
    EXPECT_EQ(out.xsize, 1024u);
    EXPECT_EQ(out.ysize, 256u);
}

// Marks the block at (bx, by) as a first-block of `side` and its interior as
// covered, in the fc.acs field (block raster, bw blocks per row).
void set_first_block(FrameCoefficients& fc, std::size_t bw, int side, std::size_t bx,
                     std::size_t by) {
    const std::size_t s{static_cast<std::size_t>(side) / 8};
    for (std::size_t dy{0}; dy < s; ++dy) {
        for (std::size_t dx{0}; dx < s; ++dx) {
            fc.acs[(by + dy) * bw + (bx + dx)] =
                (dy == 0 && dx == 0) ? static_cast<std::int8_t>(side) : ACS_COVERED;
        }
    }
}

// Sets one AC coefficient (raw scan index `raw`) of a first-block across all
// three channels, in the covered-block layout.
void set_ac(FrameCoefficients& fc, std::size_t bw, int side, std::size_t bx, std::size_t by,
            std::size_t raw, std::int32_t v) {
    const std::size_t slot{covered_plane_slot(side, bx, by, bw, raw)};
    for (int c{0}; c < 3; ++c) {
        fc.ac[c][slot] = v;
    }
}

// A frame mixing DCT32/DCT16/DCT8 with non-trivial CfL maps must decode with the
// unmodified libjxl decoder (the conformance bar for the ACS + CfL signaling).
FrameCoefficients make_mixed_frame(std::size_t w, std::size_t h) {
    FrameCoefficients fc{make_frame(w, h)};
    const std::size_t bw{w / 8};
    const std::size_t bh{h / 8};
    fc.acs.assign(bw * bh, 8);

    // One DCT32 and two DCT16 near the origin (well inside the first AC group),
    // rest DCT8. All footprints fit the image and their 256px AC group.
    set_first_block(fc, bw, 32, 0, 0);
    set_first_block(fc, bw, 16, 4, 0);
    set_first_block(fc, bw, 16, 0, 4);
    set_ac(fc, bw, 32, 0, 0, 16, 4);  // first AC coefficient past the 4x4 LLF
    set_ac(fc, bw, 32, 0, 0, 40, -2);
    set_ac(fc, bw, 16, 4, 0, 4, 3);  // first AC coefficient past the 2x2 LLF
    set_ac(fc, bw, 16, 0, 4, 5, -1);
    set_ac(fc, bw, 8, 10, 10, 1, 2);  // an ordinary DCT8 block

    const std::size_t cmw{(bw + 7) / 8};
    const std::size_t cmh{(bh + 7) / 8};
    fc.ytox_map.assign(cmw * cmh, 0);
    fc.ytob_map.assign(cmw * cmh, 0);
    fc.ytox_map[0] = -20;
    fc.ytob_map[0] = 15;
    if (cmw * cmh > 1) {
        fc.ytox_map[cmw * cmh - 1] = 7;
        fc.ytob_map[cmw * cmh - 1] = -9;
    }
    for (int c{0}; c < 3; ++c) {
        for (std::size_t i{0}; i < fc.dc[c].size(); ++i) {
            fc.dc[c][i] = static_cast<std::int32_t>((i % 7) - 3);
        }
    }
    return fc;
}

TEST(VarDctFrame, MixedBlocksAndCflSingleGroupDecodes) {
    Decoded out{};
    ASSERT_TRUE(decode(write_vardct_codestream(make_mixed_frame(256, 256)), out));
    EXPECT_EQ(out.xsize, 256u);
    EXPECT_EQ(out.ysize, 256u);
}

TEST(VarDctFrame, MixedBlocksAndCflMultiGroupDecodes) {
    Decoded out{};
    ASSERT_TRUE(decode(write_vardct_codestream(make_mixed_frame(512, 384)), out));
    EXPECT_EQ(out.xsize, 512u);
    EXPECT_EQ(out.ysize, 384u);
}

// The clustered-AC path must reconstruct byte-identical pixels to the
// single-histogram path: it changes only how the (unchanged) AC tokens are
// entropy-coded, so a correct context model + context map decodes to the same
// image through the unmodified libjxl decoder.
void expect_clustered_equivalent(const FrameCoefficients& fc) {
    Decoded single{};
    Decoded clustered{};
    ASSERT_TRUE(decode(write_vardct_codestream(fc, /*clustered_ac=*/false), single));
    ASSERT_TRUE(decode(write_vardct_codestream(fc, /*clustered_ac=*/true), clustered));
    ASSERT_EQ(single.xsize, clustered.xsize);
    ASSERT_EQ(single.ysize, clustered.ysize);
    ASSERT_EQ(single.pixels.size(), clustered.pixels.size());
    EXPECT_EQ(single.pixels, clustered.pixels);
}

TEST(VarDctFrameClustered, SingleGroupRichAcEquivalent) {
    FrameCoefficients fc{make_frame(256, 256)};
    fill_ac_pattern(fc);
    expect_clustered_equivalent(fc);
}

TEST(VarDctFrameClustered, MultiGroupRichAcEquivalent) {
    FrameCoefficients fc{make_frame(512, 384)};
    fill_ac_pattern(fc);
    expect_clustered_equivalent(fc);
}

TEST(VarDctFrameClustered, MixedBlocksEquivalent) {
    expect_clustered_equivalent(make_mixed_frame(512, 384));
}

TEST(VarDctFrameClustered, ReducesSizeOnRichAc) {
    FrameCoefficients fc{make_frame(512, 512)};
    fill_ac_pattern(fc);
    const std::size_t single{write_vardct_codestream(fc, /*clustered_ac=*/false).size()};
    const std::size_t clustered{write_vardct_codestream(fc, /*clustered_ac=*/true).size()};
    EXPECT_LT(clustered, single) << "single=" << single << " clustered=" << clustered;
}

}  // namespace
}  // namespace cujpegxl::bitstream
