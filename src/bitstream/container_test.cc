// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Validates the ISOBMFF container writer: the fixed framing bytes are checked
// directly, and a real codestream wrapped in the container is decoded with
// libjxl's public decoder (the path djxl uses) to confirm the boxes are
// well-formed and the codestream is recovered byte-for-byte.

#include <cstdint>
#include <vector>

#include <jxl/decode.h>

#include <gtest/gtest.h>

#include "container.h"

#include "tools/bitstream/vardct_frame.h"

namespace cujpegxl::bitstream {
namespace {

struct Decoded {
    std::uint32_t xsize{0};
    std::uint32_t ysize{0};
    std::vector<std::uint8_t> pixels{};
};

testing::AssertionResult decode(const std::vector<std::uint8_t>& file, Decoded& out) {
    JxlDecoder* dec{JxlDecoderCreate(nullptr)};
    if (JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) !=
        JXL_DEC_SUCCESS) {
        JxlDecoderDestroy(dec);
        return testing::AssertionFailure() << "SubscribeEvents failed";
    }
    JxlDecoderSetInput(dec, file.data(), file.size());
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

TEST(Container, HeaderMatchesJxlSignatureAndFtyp) {
    // The JXL signature box: length 12, type "JXL ", payload 0x0D0A870A.
    const std::array<std::uint8_t, 12> signature{0,   0,   0,    0x0c, 'J', 'X',
                                                 'L', ' ', 0x0d, 0x0a, 0x87, 0x0a};
    // The ftyp box: length 20, type "ftyp", major brand "jxl ", minor version 0,
    // compatible brand "jxl ".
    const std::array<std::uint8_t, 20> ftyp{0,   0,   0,   0x14, 'f', 't', 'y',
                                            'p', 'j', 'x', 'l', ' ', 0,   0,
                                            0,   0,   'j', 'x', 'l', ' '};
    for (std::size_t i{0}; i < signature.size(); ++i) {
        EXPECT_EQ(CONTAINER_HEADER[i], signature[i]) << "signature byte " << i;
    }
    for (std::size_t i{0}; i < ftyp.size(); ++i) {
        EXPECT_EQ(CONTAINER_HEADER[signature.size() + i], ftyp[i]) << "ftyp byte " << i;
    }
}

TEST(Container, SmallBoxFramingIsWellFormed) {
    const std::size_t codestream_size{1234};
    EXPECT_EQ(jxlc_box_header_size(codestream_size), 8u);
    EXPECT_EQ(container_size(codestream_size), 32u + 8u + codestream_size);

    const std::vector<std::uint8_t> framing{container_framing(codestream_size)};
    ASSERT_EQ(framing.size(), 40u);
    // jxlc box header immediately after the 32-byte prefix: 32-bit big-endian
    // size = content + 8, then the type "jxlc".
    const std::size_t box_size{codestream_size + 8};
    EXPECT_EQ(framing[32], static_cast<std::uint8_t>(box_size >> 24));
    EXPECT_EQ(framing[33], static_cast<std::uint8_t>(box_size >> 16));
    EXPECT_EQ(framing[34], static_cast<std::uint8_t>(box_size >> 8));
    EXPECT_EQ(framing[35], static_cast<std::uint8_t>(box_size));
    EXPECT_EQ(framing[36], 'j');
    EXPECT_EQ(framing[37], 'x');
    EXPECT_EQ(framing[38], 'l');
    EXPECT_EQ(framing[39], 'c');
}

TEST(Container, LargeBoxFramingUsesExtendedSize) {
    // At or above 2^32 - 8 bytes the 32-bit size field cannot hold the box size,
    // so the size field is 1 and an 8-byte extended size follows the type.
    const std::size_t codestream_size{0x100000000ull - 8};
    EXPECT_EQ(jxlc_box_header_size(codestream_size), 16u);

    const std::vector<std::uint8_t> framing{container_framing(codestream_size)};
    ASSERT_EQ(framing.size(), 48u);
    EXPECT_EQ(framing[32], 0u);
    EXPECT_EQ(framing[33], 0u);
    EXPECT_EQ(framing[34], 0u);
    EXPECT_EQ(framing[35], 1u);
    EXPECT_EQ(framing[36], 'j');
    EXPECT_EQ(framing[37], 'x');
    EXPECT_EQ(framing[38], 'l');
    EXPECT_EQ(framing[39], 'c');
    const std::uint64_t box_size{static_cast<std::uint64_t>(codestream_size) + 16};
    for (std::size_t i{0}; i < 8; ++i) {
        EXPECT_EQ(framing[40 + i], static_cast<std::uint8_t>(box_size >> (8 * (7 - i))))
            << "extended size byte " << i;
    }
}

TEST(Container, WrappedCodestreamDecodes) {
    const std::vector<std::uint8_t> codestream{write_vardct_codestream(make_frame(256, 256))};
    const std::vector<std::uint8_t> file{write_container(codestream)};

    ASSERT_EQ(file.size(), container_size(codestream.size()));
    for (std::size_t i{0}; i < codestream.size(); ++i) {
        ASSERT_EQ(file[40 + i], codestream[i]) << "codestream byte " << i;
    }

    Decoded out{};
    ASSERT_TRUE(decode(file, out));
    EXPECT_EQ(out.xsize, 256u);
    EXPECT_EQ(out.ysize, 256u);
}

TEST(Container, MultiGroupWrappedCodestreamDecodes) {
    const std::vector<std::uint8_t> codestream{write_vardct_codestream(make_frame(512, 512))};
    Decoded out{};
    ASSERT_TRUE(decode(write_container(codestream), out));
    EXPECT_EQ(out.xsize, 512u);
    EXPECT_EQ(out.ysize, 512u);
}

}  // namespace
}  // namespace cujpegxl::bitstream
