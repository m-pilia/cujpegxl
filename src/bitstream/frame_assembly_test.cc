// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Pins the production host frame assembler to the byte-exact oracle: the DcGroup
// header blobs and prefix codes must match reference_dc_encode, and a full
// codestream reassembled from frame_assembly's globals/head plus the oracle's
// group bodies must equal write_vardct_codestream byte-for-byte.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "frame_assembly.h"
#include "tools/bitstream/vardct_frame.h"

namespace cujpegxl::bitstream {
namespace {

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
                blk[8] = static_cast<std::int32_t>((by + c) % 5) - 2;
                blk[9] = static_cast<std::int32_t>((bx + by) % 11) - 5;
            }
        }
    }
    return fc;
}

QuantParams params_of(const FrameCoefficients& fc) {
    return QuantParams{fc.global_scale, fc.quant_dc, fc.raw_quant_field};
}

// Pads a trimmed histogram to the device HISTOGRAM_STRIDE width.
std::vector<std::uint32_t> pad(const std::vector<std::uint32_t>& h) {
    std::vector<std::uint32_t> out(HISTOGRAM_STRIDE, 0);
    for (std::size_t i{0}; i < h.size(); ++i) {
        out[i] = h[i];
    }
    return out;
}

TEST(FrameAssembly, DcGroupBlobsMatchOracle) {
    const FrameCoefficients fc{make_frame(2560, 256)};  // 2 DC groups (one partial)
    const DcReference ref{reference_dc_encode(fc)};

    std::vector<std::uint32_t> dc_hists{};
    for (const DcGroupReference& r : ref.groups) {
        const std::vector<std::uint32_t> padded{pad(r.dc_histogram)};
        dc_hists.insert(dc_hists.end(), padded.begin(), padded.end());
    }

    const DcGroupBlobs blobs{
        build_dc_group_blobs(fc.width, fc.height, params_of(fc), dc_hists.data())};
    ASSERT_EQ(ref.groups.size(), dc_group_count(fc.width, fc.height));

    for (std::size_t g{0}; g < ref.groups.size(); ++g) {
        const DcGroupReference& r{ref.groups[g]};

        EXPECT_EQ(blobs.blob_pre_bits[g], r.blob_pre_bits) << "group " << g << " blob_pre bits";
        for (std::size_t i{0}; i < r.blob_pre.size(); ++i) {
            EXPECT_EQ(blobs.blob_pre[blobs.blob_pre_off[g] + i], r.blob_pre[i])
                << "group " << g << " blob_pre byte " << i;
        }
        EXPECT_EQ(blobs.blob_mid_bits[g], r.blob_mid_bits) << "group " << g << " blob_mid bits";
        for (std::size_t i{0}; i < r.blob_mid.size(); ++i) {
            EXPECT_EQ(blobs.blob_mid[blobs.blob_mid_off[g] + i], r.blob_mid[i])
                << "group " << g << " blob_mid byte " << i;
        }
        for (std::size_t i{0}; i < r.dc_depth.size(); ++i) {
            EXPECT_EQ(blobs.dc_depth[g * HISTOGRAM_STRIDE + i], r.dc_depth[i])
                << "group " << g << " dc_depth " << i;
            EXPECT_EQ(blobs.dc_bits[g * HISTOGRAM_STRIDE + i], r.dc_bits[i])
                << "group " << g << " dc_bits " << i;
        }
        for (std::size_t i{0}; i < r.acmeta_depth.size(); ++i) {
            EXPECT_EQ(blobs.acmeta_depth[g * HISTOGRAM_STRIDE + i], r.acmeta_depth[i])
                << "group " << g << " acmeta_depth " << i;
            EXPECT_EQ(blobs.acmeta_bits[g * HISTOGRAM_STRIDE + i], r.acmeta_bits[i])
                << "group " << g << " acmeta_bits " << i;
        }
    }
}

TEST(FrameAssembly, AcPrefixCodeMatchesOracle) {
    const FrameCoefficients fc{make_frame(512, 512)};
    const AcReference ref{reference_ac_encode(fc)};
    const AcGlobalResult ac{
        build_ac_global(pad(ref.histogram).data(), ac_group_count(fc.width, fc.height))};
    EXPECT_EQ(ac.depth, ref.depth);
    EXPECT_EQ(ac.bits, ref.bits);
}

// Reassembles a codestream from frame_assembly's globals and head plus the
// oracle's group bodies, and checks it equals the oracle's full codestream.
void check_reassembly(const FrameCoefficients& fc) {
    const std::vector<std::uint8_t> expected{write_vardct_codestream(fc)};

    const DcReference dc{reference_dc_encode(fc)};
    const AcReference ac{reference_ac_encode(fc)};

    const std::vector<std::uint8_t> dc_global{build_dc_global(params_of(fc))};
    const AcGlobalResult ac_global{
        build_ac_global(pad(ac.histogram).data(), ac_group_count(fc.width, fc.height))};

    std::vector<std::uint32_t> sizes{};
    sizes.push_back(static_cast<std::uint32_t>(dc_global.size()));
    for (const DcGroupReference& r : dc.groups) {
        sizes.push_back(static_cast<std::uint32_t>(r.section.size()));
    }
    sizes.push_back(static_cast<std::uint32_t>(ac_global.section.size()));
    for (const std::vector<std::uint8_t>& s : ac.group_streams) {
        sizes.push_back(static_cast<std::uint32_t>(s.size()));
    }

    std::vector<std::uint8_t> got{build_codestream_head(
        static_cast<std::uint32_t>(fc.width), static_cast<std::uint32_t>(fc.height), sizes)};
    got.insert(got.end(), dc_global.begin(), dc_global.end());
    for (const DcGroupReference& r : dc.groups) {
        got.insert(got.end(), r.section.begin(), r.section.end());
    }
    got.insert(got.end(), ac_global.section.begin(), ac_global.section.end());
    for (const std::vector<std::uint8_t>& s : ac.group_streams) {
        got.insert(got.end(), s.begin(), s.end());
    }

    ASSERT_EQ(got.size(), expected.size());
    for (std::size_t i{0}; i < expected.size(); ++i) {
        ASSERT_EQ(got[i], expected[i]) << "codestream byte " << i;
    }
}

TEST(FrameAssembly, ReassemblyMatchesOracleSingleDcGroup) {
    check_reassembly(make_frame(512, 512));
}

TEST(FrameAssembly, ReassemblyMatchesOracleMultiDcGroup) {
    check_reassembly(make_frame(2560, 256));
}

TEST(FrameAssembly, ReassemblyMatchesOraclePartialEdges) {
    check_reassembly(make_frame(640, 384));
}

}  // namespace
}  // namespace cujpegxl::bitstream
