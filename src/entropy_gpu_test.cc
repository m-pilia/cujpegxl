// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Validates the device AC entropy encoder against the host bitstream reference:
// the pooled symbol histogram, per-group byte-aligned token streams and their
// scanned offsets must match byte-for-byte, and repeated runs must be identical.

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "entropy.h"
#include "entropy_roundtrip.h"
#include "src/vardct_layout.h"
#include "tools/bitstream/vardct_frame.h"

namespace cujpegxl {
namespace {

using bitstream::AcReference;
using bitstream::FrameCoefficients;
using bitstream::reference_ac_encode;

// Flattens fc.ac into the device coefficient layout (plane-major over channels
// X, Y, B; blocks in raster order; 64 coefficients each).
std::vector<std::int32_t> flatten_ac(const FrameCoefficients& fc) {
    std::vector<std::int32_t> q{};
    q.reserve(fc.ac[0].size() * 3);
    for (int c{0}; c < 3; ++c) {
        q.insert(q.end(), fc.ac[c].begin(), fc.ac[c].end());
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
    const std::size_t blocks{(w / 8) * (h / 8)};
    for (int c{0}; c < 3; ++c) {
        fc.dc[c].assign(blocks, 0);
        fc.ac[c].assign(blocks * 64, 0);
    }
    return fc;
}

// Deterministic per-coefficient pattern spanning small and large magnitudes so
// the hybrid-uint split (raw trailing bits) and multi-bit prefix codes are
// exercised across groups.
void fill_ac_pattern(FrameCoefficients& fc) {
    const std::size_t bw{fc.width / 8};
    const std::size_t bh{fc.height / 8};
    for (std::size_t by{0}; by < bh; ++by) {
        for (std::size_t bx{0}; bx < bw; ++bx) {
            const std::size_t block{by * bw + bx};
            for (int c{0}; c < 3; ++c) {
                std::int32_t* blk{&fc.ac[c][block * 64]};
                blk[1] = static_cast<std::int32_t>((bx + c) % 7) - 3;
                blk[8] = static_cast<std::int32_t>((by * 2 + c) % 11) - 5;
                blk[9] = static_cast<std::int32_t>((bx + by) % 5) - 2;
                blk[16] = static_cast<std::int32_t>((bx * 3 + by) % 260) - 130;
                blk[2] = static_cast<std::int32_t>((by + c) % 3) - 1;
            }
        }
    }
}

void check_matches_reference(const FrameCoefficients& fc) {
    const AcReference ref{reference_ac_encode(fc)};
    const std::vector<std::int32_t> q{flatten_ac(fc)};

    AcDeviceResult dev{};
    ASSERT_TRUE(ac_encode_device(q, fc.width, fc.height, ref.depth, ref.bits, dev));

    ASSERT_EQ(dev.group_sizes.size(), ref.group_streams.size());

    for (std::size_t s{0}; s < ref.histogram.size(); ++s) {
        EXPECT_EQ(dev.histogram[s], ref.histogram[s]) << "histogram symbol " << s;
    }
    for (std::size_t s{ref.histogram.size()}; s < dev.histogram.size(); ++s) {
        EXPECT_EQ(dev.histogram[s], 0u) << "unexpected histogram symbol " << s;
    }

    std::uint32_t offset{0};
    for (std::size_t g{0}; g < ref.group_streams.size(); ++g) {
        const std::vector<std::uint8_t>& expected{ref.group_streams[g]};
        EXPECT_EQ(dev.group_sizes[g], expected.size()) << "group " << g << " size";
        EXPECT_EQ(dev.group_offsets[g], offset) << "group " << g << " offset";
        offset += dev.group_sizes[g];
        ASSERT_LE(dev.group_offsets[g] + expected.size(), dev.stream.size());
        for (std::size_t i{0}; i < expected.size(); ++i) {
            EXPECT_EQ(dev.stream[dev.group_offsets[g] + i], expected[i])
                << "group " << g << " byte " << i;
        }
    }
    EXPECT_EQ(dev.stream.size(), offset);
}

TEST(EntropyGpu, SingleGroupAllZero) {
    check_matches_reference(make_frame(256, 256));
}

TEST(EntropyGpu, SingleGroupPattern) {
    FrameCoefficients fc{make_frame(256, 256)};
    fill_ac_pattern(fc);
    check_matches_reference(fc);
}

TEST(EntropyGpu, MultiGroupPattern) {
    FrameCoefficients fc{make_frame(512, 512)};
    fill_ac_pattern(fc);
    check_matches_reference(fc);
}

TEST(EntropyGpu, MultiGroupPartialEdges) {
    FrameCoefficients fc{make_frame(640, 384)};
    fill_ac_pattern(fc);
    check_matches_reference(fc);
}

// Marks a first-block of `side` and its covered interior in fc.acs.
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

// Fills a mixed DCT32/DCT16/DCT8 frame with a per-coefficient pattern in the
// covered-block layout, so the device mixed-block AC path is diffed against the
// host reference across strategies.
FrameCoefficients make_mixed_frame(std::size_t w, std::size_t h) {
    FrameCoefficients fc{make_frame(w, h)};
    const std::size_t bw{w / 8};
    const std::size_t bh{h / 8};
    fc.acs.assign(bw * bh, 8);
    set_first_block(fc, bw, 32, 0, 0);
    set_first_block(fc, bw, 16, 4, 0);
    set_first_block(fc, bw, 16, 0, 4);
    for (std::size_t by{0}; by < bh; ++by) {
        for (std::size_t bx{0}; bx < bw; ++bx) {
            const int side{fc.acs[by * bw + bx]};
            if (side == ACS_COVERED) {
                continue;
            }
            for (int raw{1}; raw < side * side; raw += 3) {
                const std::size_t slot{covered_plane_slot(side, bx, by, bw,
                                                          static_cast<std::size_t>(raw))};
                for (int c{0}; c < 3; ++c) {
                    fc.ac[c][slot] = static_cast<std::int32_t>((raw + bx + by + c) % 9) - 4;
                }
            }
        }
    }
    return fc;
}

void check_matches_reference_m3(const FrameCoefficients& fc) {
    const AcReference ref{reference_ac_encode(fc)};
    const std::vector<std::int32_t> q{flatten_ac(fc)};
    AcDeviceResult dev{};
    ASSERT_TRUE(ac_encode_device_m3(q, fc.acs, fc.width, fc.height, ref.depth, ref.bits, dev));

    for (std::size_t s{0}; s < ref.histogram.size(); ++s) {
        EXPECT_EQ(dev.histogram[s], ref.histogram[s]) << "histogram symbol " << s;
    }
    std::uint32_t offset{0};
    for (std::size_t g{0}; g < ref.group_streams.size(); ++g) {
        const std::vector<std::uint8_t>& expected{ref.group_streams[g]};
        EXPECT_EQ(dev.group_sizes[g], expected.size()) << "group " << g << " size";
        EXPECT_EQ(dev.group_offsets[g], offset) << "group " << g << " offset";
        offset += dev.group_sizes[g];
        for (std::size_t i{0}; i < expected.size(); ++i) {
            EXPECT_EQ(dev.stream[dev.group_offsets[g] + i], expected[i])
                << "group " << g << " byte " << i;
        }
    }
    EXPECT_EQ(dev.stream.size(), offset);
}

TEST(EntropyGpuM3, Dct8OnlyMatchesReference) {
    FrameCoefficients fc{make_frame(256, 256)};
    fill_ac_pattern(fc);
    fc.acs.assign((256 / 8) * (256 / 8), 8);  // explicit all-DCT8
    check_matches_reference_m3(fc);
}

TEST(EntropyGpuM3, MixedSingleGroupMatchesReference) {
    check_matches_reference_m3(make_mixed_frame(256, 256));
}

TEST(EntropyGpuM3, MixedMultiGroupMatchesReference) {
    check_matches_reference_m3(make_mixed_frame(512, 384));
}

TEST(EntropyGpu, Deterministic) {
    FrameCoefficients fc{make_frame(512, 384)};
    fill_ac_pattern(fc);
    const AcReference ref{reference_ac_encode(fc)};
    const std::vector<std::int32_t> q{flatten_ac(fc)};

    AcDeviceResult a{};
    AcDeviceResult b{};
    ASSERT_TRUE(ac_encode_device(q, fc.width, fc.height, ref.depth, ref.bits, a));
    ASSERT_TRUE(ac_encode_device(q, fc.width, fc.height, ref.depth, ref.bits, b));
    EXPECT_EQ(a.stream, b.stream);
    EXPECT_EQ(a.group_sizes, b.group_sizes);
    EXPECT_EQ(a.group_offsets, b.group_offsets);
    EXPECT_EQ(a.histogram, b.histogram);
}

}  // namespace
}  // namespace cujpegxl
