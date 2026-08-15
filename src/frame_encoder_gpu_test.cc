// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// End-to-end validation of the device frame encoder: for a set of quantized
// coefficients it must produce the exact ISOBMFF .jxl file the host oracle
// produces, and that file must decode through libjxl's public decoder to the
// expected dimensions.

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include <jxl/decode.h>

#include <gtest/gtest.h>

#include "entropy.h"
#include "frame_encoder.h"
#include "src/bitstream/container.h"
#include "src/bitstream/frame_assembly.h"
#include "tools/bitstream/vardct_frame.h"

namespace cujpegxl {
namespace {

using bitstream::FrameCoefficients;
using bitstream::QuantParams;

std::vector<std::uint8_t> reference_file(const FrameCoefficients& fc) {
    const bitstream::DcReference dc{bitstream::reference_dc_encode(fc)};
    const bitstream::AcReference prefix{bitstream::reference_ac_encode(fc)};
    const bitstream::AcAnsReference ans{bitstream::reference_ac_ans_encode(fc)};
    const std::size_t num_groups{bitstream::ac_group_count(fc.width, fc.height)};
    const bitstream::AcGlobalResult prefix_global{
        bitstream::build_ac_global(prefix.histogram.data(), num_groups)};
    const bitstream::AcAnsGlobalResult ans_global{
        bitstream::build_ac_global_ans(ans.histogram.data(), num_groups)};
    std::size_t prefix_size{prefix_global.section.size()};
    std::size_t ans_size{ans_global.section.size()};
    for (std::size_t group{0}; group < num_groups; ++group) {
        prefix_size += prefix.group_streams[group].size();
        ans_size += ans.group_streams[group].size();
    }
    const bool use_ans{ans_size < prefix_size};
    const std::vector<std::uint8_t>& ac_global{
        use_ans ? ans_global.section : prefix_global.section};

    const std::vector<std::uint8_t> dc_global{
        bitstream::build_dc_global(QuantParams{fc.global_scale, fc.quant_dc})};
    std::vector<std::uint32_t> sizes{static_cast<std::uint32_t>(dc_global.size())};
    for (const bitstream::DcGroupReference& group : dc.groups) {
        sizes.push_back(static_cast<std::uint32_t>(group.section.size()));
    }
    sizes.push_back(static_cast<std::uint32_t>(ac_global.size()));
    for (std::size_t group{0}; group < num_groups; ++group) {
        sizes.push_back(static_cast<std::uint32_t>(
            use_ans ? ans.group_streams[group].size()
                    : prefix.group_streams[group].size()));
    }

    std::vector<std::uint8_t> codestream{bitstream::build_codestream_head(
        static_cast<std::uint32_t>(fc.width), static_cast<std::uint32_t>(fc.height), sizes)};
    codestream.insert(codestream.end(), dc_global.begin(), dc_global.end());
    for (const bitstream::DcGroupReference& group : dc.groups) {
        codestream.insert(codestream.end(), group.section.begin(), group.section.end());
    }
    codestream.insert(codestream.end(), ac_global.begin(), ac_global.end());
    for (std::size_t group{0}; group < num_groups; ++group) {
        const std::vector<std::uint8_t>& stream{
            use_ans ? ans.group_streams[group] : prefix.group_streams[group]};
        codestream.insert(codestream.end(), stream.begin(), stream.end());
    }
    return bitstream::write_container(codestream);
}

// Packs fc's AC (slots 1..63 of each block) into the device int16 AC layout
// (channel-major planes X, Y, B; AC_COEFFS_PER_BLOCK per block; slot k-1).
std::vector<std::int16_t> flatten_ac(const FrameCoefficients& fc) {
    const std::size_t blocks{(fc.width / 8) * (fc.height / 8)};
    std::vector<std::int16_t> ac(3 * blocks * AC_COEFFS_PER_BLOCK, 0);
    for (std::size_t c{0}; c < 3; ++c) {
        for (std::size_t b{0}; b < blocks; ++b) {
            for (std::size_t k{1}; k < 64; ++k) {
                ac[c * blocks * AC_COEFFS_PER_BLOCK + b * AC_COEFFS_PER_BLOCK + (k - 1)] =
                    static_cast<std::int16_t>(fc.ac[c][b * 64 + k]);
            }
        }
    }
    return ac;
}

// Packs fc's DC (slot 0 of each block) into the device int32 DC layout
// (channel-major planes X, Y, B; one DC per block).
std::vector<std::int32_t> flatten_dc(const FrameCoefficients& fc) {
    const std::size_t blocks{(fc.width / 8) * (fc.height / 8)};
    std::vector<std::int32_t> dc(3 * blocks, 0);
    for (std::size_t c{0}; c < 3; ++c) {
        for (std::size_t b{0}; b < blocks; ++b) {
            dc[c * blocks + b] = fc.dc[c][b];
        }
    }
    return dc;
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

bool encode_on_device(const FrameCoefficients& fc, std::vector<std::uint8_t>& out,
                      AcClusteringMode clustering =
                          AcClusteringMode::DATA_DRIVEN,
                      std::vector<StageTiming>* stats = nullptr) {
    const std::vector<std::int16_t> ac{flatten_ac(fc)};
    const std::vector<std::int32_t> dc{flatten_dc(fc)};
    const std::vector<std::int32_t> qf((fc.width / 8) * (fc.height / 8),
                                       static_cast<std::int32_t>(fc.raw_quant_field));
    std::int16_t* d_ac{nullptr};
    std::int32_t* d_dc{nullptr};
    std::int32_t* d_qf{nullptr};
    if (cudaMalloc(&d_ac, ac.size() * sizeof(std::int16_t)) != cudaSuccess ||
        cudaMalloc(&d_dc, dc.size() * sizeof(std::int32_t)) != cudaSuccess ||
        cudaMalloc(&d_qf, qf.size() * sizeof(std::int32_t)) != cudaSuccess) {
        return false;
    }
    bool ok{cudaMemcpy(d_ac, ac.data(), ac.size() * sizeof(std::int16_t), cudaMemcpyHostToDevice) ==
                cudaSuccess &&
            cudaMemcpy(d_dc, dc.data(), dc.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice) ==
                cudaSuccess &&
            cudaMemcpy(d_qf, qf.data(), qf.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice) ==
                cudaSuccess};
    ok = ok && encode_frame(d_ac, d_dc, fc.width, fc.height,
                            QuantParams{fc.global_scale, fc.quant_dc}, d_qf, out,
                            stats, clustering);
    cudaFree(d_ac);
    cudaFree(d_dc);
    cudaFree(d_qf);
    return ok;
}

TEST(FrameEncoderGpu, PrefixSelectedFrameEmitsPrefixOnly) {
    FrameCoefficients fc{make_frame(512, 256)};
    for (std::vector<std::int32_t>& channel : fc.ac) {
        std::fill(channel.begin(), channel.end(), 0);
    }
    std::vector<std::uint8_t> encoded{};
    std::vector<StageTiming> stats{};
    ASSERT_TRUE(encode_on_device(fc, encoded, AcClusteringMode::FIXED, &stats));
    EXPECT_EQ(encoded, reference_file(fc));
    ASSERT_EQ(stats.size(), 2u);
    ASSERT_GE(stats[0].phases.size(), 15u);
    ASSERT_GE(stats[0].metrics.size(), 9u);
    EXPECT_GT(stats[0].phases[6].gpu_us, 0.0);
    EXPECT_EQ(stats[0].metrics[8].value, 0.0);
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
    const std::vector<std::uint8_t> expected{reference_file(fc)};

    std::vector<std::uint8_t> got{};
    ASSERT_TRUE(encode_on_device(fc, got, AcClusteringMode::FIXED));

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

TEST(FrameEncoderGpu, SingleDcGroupMultiAcGroup) {
    check(make_frame(512, 512));
}

TEST(FrameEncoderGpu, PartialEdgeGroups) {
    check(make_frame(640, 384));
}

TEST(FrameEncoderGpu, MultiDcGroup) {
    check(make_frame(2560, 256));
}

TEST(FrameEncoderGpu, DataDrivenClusteringReducesSizeAndDecodes) {
    const FrameCoefficients fc{make_frame(640, 384)};
    std::vector<std::uint8_t> fixed{};
    std::vector<std::uint8_t> data_driven{};
    ASSERT_TRUE(encode_on_device(fc, fixed, AcClusteringMode::FIXED));
    ASSERT_TRUE(encode_on_device(fc, data_driven));
    EXPECT_LT(data_driven.size(), fixed.size());

    std::uint32_t xs{0};
    std::uint32_t ys{0};
    ASSERT_TRUE(decode_dims(data_driven, xs, ys));
    EXPECT_EQ(xs, fc.width);
    EXPECT_EQ(ys, fc.height);
}

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
