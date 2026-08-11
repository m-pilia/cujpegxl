// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Validates the device DcGroup entropy coder against the host bitstream
// reference: per DcGroup the device must reproduce the pooled DC histogram and
// the full byte-aligned section (extra_precision + VarDCTDC modular + AcMetadata
// count + AcMetadata modular) byte-for-byte, and repeated runs must be identical.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "dc_entropy_roundtrip.h"
#include "src/vardct_layout.h"
#include "tools/bitstream/vardct_frame.h"

namespace cujpegxl {
namespace {

using bitstream::DcReference;
using bitstream::FrameCoefficients;
using bitstream::reference_dc_encode;

// Flattens fc's DC (slot 0 of each block) into the device coefficient layout
// (plane-major over channels X, Y, B; blocks in raster order; 64 slots each,
// only slot 0 populated — the DC coder ignores the rest).
std::vector<std::int32_t> flatten_dc(const FrameCoefficients& fc) {
    const std::size_t plane{fc.width * fc.height};
    std::vector<std::int32_t> q(3 * plane, 0);
    const std::size_t blocks{(fc.width / 8) * (fc.height / 8)};
    for (int c{0}; c < 3; ++c) {
        for (std::size_t b{0}; b < blocks; ++b) {
            q[static_cast<std::size_t>(c) * plane + b * 64] = fc.dc[c][b];
        }
    }
    return q;
}

FrameCoefficients make_frame(std::size_t w, std::size_t h, std::uint32_t raw_quant_field) {
    FrameCoefficients fc{};
    fc.width = w;
    fc.height = h;
    fc.global_scale = 4096;
    fc.quant_dc = 32;
    fc.raw_quant_field = raw_quant_field;
    const std::size_t blocks{(w / 8) * (h / 8)};
    for (int c{0}; c < 3; ++c) {
        fc.dc[c].assign(blocks, 0);
        fc.ac[c].assign(blocks * 64, 0);
    }
    return fc;
}

// Deterministic DC pattern spanning small and large magnitudes so the hybrid
// split (raw trailing bits) and multi-bit prefix codes are exercised per group.
void fill_dc_pattern(FrameCoefficients& fc) {
    const std::size_t bw{fc.width / 8};
    const std::size_t bh{fc.height / 8};
    for (std::size_t by{0}; by < bh; ++by) {
        for (std::size_t bx{0}; bx < bw; ++bx) {
            const std::size_t block{by * bw + bx};
            fc.dc[0][block] = static_cast<std::int32_t>((bx * 3 + by) % 517) - 258;
            fc.dc[1][block] = static_cast<std::int32_t>((bx + by * 2) % 41) - 20;
            fc.dc[2][block] = static_cast<std::int32_t>((bx + by) % 7) - 3;
        }
    }
}

void check_matches_reference(const FrameCoefficients& fc) {
    const DcReference ref{reference_dc_encode(fc)};
    const std::vector<std::int32_t> q{flatten_dc(fc)};

    DcDeviceResult dev{};
    ASSERT_TRUE(dc_encode_device(q, fc.width, fc.height, fc.raw_quant_field, ref, dev));
    ASSERT_EQ(dev.group_sizes.size(), ref.groups.size());

    std::uint32_t offset{0};
    for (std::size_t g{0}; g < ref.groups.size(); ++g) {
        const bitstream::DcGroupReference& r{ref.groups[g]};

        for (std::size_t s{0}; s < r.dc_histogram.size(); ++s) {
            EXPECT_EQ(dev.histograms[g][s], r.dc_histogram[s])
                << "group " << g << " histogram symbol " << s;
        }
        for (std::size_t s{r.dc_histogram.size()}; s < dev.histograms[g].size(); ++s) {
            EXPECT_EQ(dev.histograms[g][s], 0u)
                << "group " << g << " unexpected histogram symbol " << s;
        }

        EXPECT_EQ(dev.group_sizes[g], r.section.size()) << "group " << g << " size";
        EXPECT_EQ(dev.group_offsets[g], offset) << "group " << g << " offset";
        offset += dev.group_sizes[g];
        ASSERT_LE(dev.group_offsets[g] + r.section.size(), dev.stream.size());
        for (std::size_t i{0}; i < r.section.size(); ++i) {
            EXPECT_EQ(dev.stream[dev.group_offsets[g] + i], r.section[i])
                << "group " << g << " byte " << i;
        }
    }
    EXPECT_EQ(dev.stream.size(), offset);
}

TEST(DcEntropyGpu, SingleGroupAllZero) {
    check_matches_reference(make_frame(256, 256, 32));
}

TEST(DcEntropyGpu, SingleGroupPattern) {
    FrameCoefficients fc{make_frame(256, 256, 32)};
    fill_dc_pattern(fc);
    check_matches_reference(fc);
}

TEST(DcEntropyGpu, QuantFieldOne) {
    FrameCoefficients fc{make_frame(256, 256, 1)};
    fill_dc_pattern(fc);
    check_matches_reference(fc);
}

TEST(DcEntropyGpu, TwoGroupsPartialEdge) {
    // 2560x256 -> 320x32 blocks -> 2 DC groups wide, the right one 64 blocks
    // wide (partial), both 32 blocks tall (partial vs the 256-block group).
    FrameCoefficients fc{make_frame(2560, 256, 32)};
    fill_dc_pattern(fc);
    check_matches_reference(fc);
}

TEST(DcEntropyGpu, GroupGridPartialEdges) {
    // 2560x2304 -> 320x288 blocks -> 2x2 DC groups, right column 64 wide and
    // bottom row 32 tall (both partial).
    FrameCoefficients fc{make_frame(2560, 2304, 48)};
    fill_dc_pattern(fc);
    check_matches_reference(fc);
}

std::vector<std::int32_t> compact_dc(const FrameCoefficients& fc) {
    const std::size_t blocks{(fc.width / 8) * (fc.height / 8)};
    std::vector<std::int32_t> dc(3 * blocks, 0);
    for (int c{0}; c < 3; ++c) {
        for (std::size_t b{0}; b < blocks; ++b) {
            dc[static_cast<std::size_t>(c) * blocks + b] = fc.dc[c][b];
        }
    }
    return dc;
}

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

FrameCoefficients make_mixed_dc_frame(std::size_t w, std::size_t h) {
    FrameCoefficients fc{make_frame(w, h, 32)};
    const std::size_t bw{w / 8};
    const std::size_t bh{h / 8};
    fc.acs.assign(bw * bh, 8);
    set_first_block(fc, bw, 32, 0, 0);
    set_first_block(fc, bw, 16, 4, 0);
    set_first_block(fc, bw, 16, 0, 4);
    fill_dc_pattern(fc);
    const std::size_t cmw{(bw + 7) / 8};
    const std::size_t cmh{(bh + 7) / 8};
    fc.ytox_map.assign(cmw * cmh, 0);
    fc.ytob_map.assign(cmw * cmh, 0);
    for (std::size_t i{0}; i < cmw * cmh; ++i) {
        fc.ytox_map[i] = static_cast<std::int8_t>((i % 13) - 6);
        fc.ytob_map[i] = static_cast<std::int8_t>((i % 9) - 4);
    }
    return fc;
}

void check_matches_reference_m3(const FrameCoefficients& fc) {
    const DcReference ref{reference_dc_encode(fc)};
    const std::size_t blocks{(fc.width / 8) * (fc.height / 8)};
    const std::vector<std::int32_t> quant_field(blocks,
                                                static_cast<std::int32_t>(fc.raw_quant_field));
    DcDeviceResult dev{};
    ASSERT_TRUE(dc_encode_device_m3(compact_dc(fc), fc.acs, fc.ytox_map, fc.ytob_map, quant_field,
                                    fc.width, fc.height, ref, dev));
    ASSERT_EQ(dev.group_sizes.size(), ref.groups.size());

    std::uint32_t offset{0};
    for (std::size_t g{0}; g < ref.groups.size(); ++g) {
        const bitstream::DcGroupReference& r{ref.groups[g]};
        for (std::size_t s{0}; s < r.dc_histogram.size(); ++s) {
            EXPECT_EQ(dev.histograms[g][s], r.dc_histogram[s]) << "group " << g << " dc symbol " << s;
        }
        for (std::size_t s{0}; s < r.acmeta_histogram.size(); ++s) {
            EXPECT_EQ(dev.acmeta_histograms[g][s], r.acmeta_histogram[s])
                << "group " << g << " acmeta symbol " << s;
        }
        EXPECT_EQ(dev.group_sizes[g], r.section.size()) << "group " << g << " size";
        EXPECT_EQ(dev.group_offsets[g], offset) << "group " << g << " offset";
        offset += dev.group_sizes[g];
        ASSERT_LE(dev.group_offsets[g] + r.section.size(), dev.stream.size());
        for (std::size_t i{0}; i < r.section.size(); ++i) {
            EXPECT_EQ(dev.stream[dev.group_offsets[g] + i], r.section[i])
                << "group " << g << " byte " << i;
        }
    }
    EXPECT_EQ(dev.stream.size(), offset);
}

TEST(DcEntropyGpuM3, Dct8OnlyMatchesReference) {
    FrameCoefficients fc{make_frame(256, 256, 32)};
    fill_dc_pattern(fc);
    fc.acs.assign((256 / 8) * (256 / 8), 8);
    check_matches_reference_m3(fc);
}

TEST(DcEntropyGpuM3, MixedSingleGroupMatchesReference) {
    check_matches_reference_m3(make_mixed_dc_frame(256, 256));
}

TEST(DcEntropyGpuM3, MixedMultiGroupMatchesReference) {
    check_matches_reference_m3(make_mixed_dc_frame(2560, 256));
}

TEST(DcEntropyGpu, Deterministic) {
    FrameCoefficients fc{make_frame(2560, 256, 32)};
    fill_dc_pattern(fc);
    const DcReference ref{reference_dc_encode(fc)};
    const std::vector<std::int32_t> q{flatten_dc(fc)};

    DcDeviceResult a{};
    DcDeviceResult b{};
    ASSERT_TRUE(dc_encode_device(q, fc.width, fc.height, fc.raw_quant_field, ref, a));
    ASSERT_TRUE(dc_encode_device(q, fc.width, fc.height, fc.raw_quant_field, ref, b));
    EXPECT_EQ(a.stream, b.stream);
    EXPECT_EQ(a.group_sizes, b.group_sizes);
    EXPECT_EQ(a.group_offsets, b.group_offsets);
}

}  // namespace
}  // namespace cujpegxl
