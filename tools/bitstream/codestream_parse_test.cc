// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Validates the host codestream header/TOC writer against libjxl's own parsers:
// the bytes we emit are read back with ReadSizeHeader/ReadImageMetadata/
// ReadFrameHeader and the decoded fields must match what we intended to write.
// This links the testonly libjxl build (never part of the runtime).

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "lib/jxl/base/span.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/headers.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/toc.h"

#include "codestream.h"

namespace cujpegxl::bitstream {
namespace {

TEST(CodestreamHeaders, RoundTripsThroughLibjxlParsers) {
    constexpr std::uint32_t kXsize{256};
    constexpr std::uint32_t kYsize{256};

    BitWriter w{};
    write_codestream_headers(w, kXsize, kYsize);
    write_frame_header(w);

    const std::vector<std::uint8_t>& bytes{w.bytes()};
    jxl::BitReader br{jxl::Bytes(bytes.data(), bytes.size())};

    ASSERT_EQ(br.ReadFixedBits<8>(), 0xFFu);
    ASSERT_EQ(br.ReadFixedBits<8>(), 0x0Au);

    jxl::SizeHeader size{};
    ASSERT_TRUE(jxl::ReadSizeHeader(&br, &size));
    EXPECT_EQ(size.xsize(), kXsize);
    EXPECT_EQ(size.ysize(), kYsize);

    jxl::CodecMetadata metadata{};
    ASSERT_TRUE(jxl::ReadImageMetadata(&br, &metadata.m));
    EXPECT_TRUE(metadata.m.xyb_encoded);
    EXPECT_EQ(metadata.m.bit_depth.bits_per_sample, 8u);
    EXPECT_TRUE(metadata.m.color_encoding.IsSRGB());
    metadata.size = size;

    // CustomTransformData follows ImageMetadata in the codestream.
    metadata.transform_data.nonserialized_xyb_encoded = metadata.m.xyb_encoded;
    ASSERT_TRUE(jxl::Bundle::Read(&br, &metadata.transform_data));

    jxl::FrameHeader fh{&metadata};
    ASSERT_TRUE(jxl::ReadFrameHeader(&br, &fh));
    EXPECT_EQ(fh.encoding, jxl::FrameEncoding::kVarDCT);
    EXPECT_EQ(fh.frame_type, jxl::FrameType::kRegularFrame);
    EXPECT_EQ(fh.color_transform, jxl::ColorTransform::kXYB);
    EXPECT_EQ(fh.passes.num_passes, 1u);
    EXPECT_EQ(fh.x_qm_scale, 2u);
    EXPECT_EQ(fh.b_qm_scale, 2u);
    EXPECT_TRUE(fh.is_last);
    ASSERT_TRUE(br.Close());
}

TEST(CodestreamHeaders, TocSingleSectionByteAligned) {
    BitWriter w{};
    write_toc_single_section(w, 12345);
    EXPECT_TRUE(w.byte_aligned());

    jxl::BitReader br{jxl::Bytes(w.bytes().data(), w.bytes().size())};
    EXPECT_EQ(br.ReadFixedBits<1>(), 0u);  // no permutation
    ASSERT_TRUE(br.JumpToByteBoundary());
    const std::uint32_t size{jxl::U32Coder::Read(jxl::kTocDist, &br)};
    EXPECT_EQ(size, 12345u);
    ASSERT_TRUE(br.Close());
}

}  // namespace
}  // namespace cujpegxl::bitstream
