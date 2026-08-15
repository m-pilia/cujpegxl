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

using bitstream::AcAnsReference;
using bitstream::AcReference;
using bitstream::FrameCoefficients;
using bitstream::reference_ac_ans_encode;
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
                const std::size_t slot{
                    covered_plane_slot(side, bx, by, bw, static_cast<std::size_t>(raw))};
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

void check_context_histograms(const FrameCoefficients& fc, bool mixed) {
    const std::vector<std::uint32_t> expected{bitstream::reference_ac_context_histogram(fc)};
    const std::vector<std::int32_t> q{flatten_ac(fc)};
    std::vector<std::uint32_t> actual{};
    const bool ok{mixed ? ac_context_histogram_device_m3(q, fc.acs, fc.width, fc.height, actual)
                        : ac_context_histogram_device(q, fc.width, fc.height, actual)};
    ASSERT_TRUE(ok);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i{0}; i < expected.size(); ++i) {
        if (actual[i] != expected[i]) {
            ADD_FAILURE() << "context histogram bin " << i << ": device=" << actual[i]
                          << " host=" << expected[i];
            return;
        }
    }

    std::vector<std::uint32_t> repeated{};
    ASSERT_TRUE(mixed ? ac_context_histogram_device_m3(q, fc.acs, fc.width, fc.height, repeated)
                      : ac_context_histogram_device(q, fc.width, fc.height, repeated));
    EXPECT_EQ(repeated, actual);

    const AcReference clustered{reference_ac_encode(fc)};
    std::vector<std::uint32_t> collapsed(clustered.histogram.size(), 0);
    for (std::size_t context{0}; context < AC_NUM_CONTEXTS; ++context) {
        const std::size_t cluster{
            static_cast<std::size_t>(ac_cluster(static_cast<std::uint32_t>(context)))};
        for (std::size_t symbol{0}; symbol < AC_HISTOGRAM_SIZE; ++symbol) {
            collapsed[cluster * AC_HISTOGRAM_SIZE + symbol] +=
                actual[context * AC_HISTOGRAM_SIZE + symbol];
        }
    }
    EXPECT_EQ(collapsed, clustered.histogram);
}

TEST(EntropyGpu, PerContextHistogramMatchesHost) {
    FrameCoefficients fc{make_frame(256, 256)};
    fill_ac_pattern(fc);
    check_context_histograms(fc, false);
}

TEST(EntropyGpu, RuntimeContextMapMatchesHostBytes) {
    FrameCoefficients fc{make_frame(512, 384)};
    fill_ac_pattern(fc);
    constexpr std::size_t num_clusters{7};
    std::vector<std::uint8_t> context_map(AC_NUM_CONTEXTS);
    for (std::size_t context{0}; context < context_map.size(); ++context) {
        context_map[context] =
            static_cast<std::uint8_t>((context * 5 + context / 17) % num_clusters);
    }

    const AcReference ref{reference_ac_encode(fc, context_map, num_clusters)};
    const std::vector<std::int32_t> q{flatten_ac(fc)};
    AcDeviceResult dev{};
    ASSERT_TRUE(ac_encode_device_runtime_map(q, fc.width, fc.height, context_map, num_clusters,
                                             ref.depth, ref.bits, dev));
    EXPECT_EQ(dev.histogram, ref.histogram);
    ASSERT_EQ(dev.group_sizes.size(), ref.group_streams.size());
    for (std::size_t group{0}; group < ref.group_streams.size(); ++group) {
        EXPECT_EQ(dev.group_sizes[group], ref.group_streams[group].size());
        const std::size_t offset{dev.group_offsets[group]};
        ASSERT_LE(offset + ref.group_streams[group].size(), dev.stream.size());
        for (std::size_t byte{0}; byte < ref.group_streams[group].size(); ++byte) {
            EXPECT_EQ(dev.stream[offset + byte], ref.group_streams[group][byte])
                << "group " << group << " byte " << byte;
        }
    }
}

void check_ans_streams(const FrameCoefficients& fc, const std::vector<std::uint8_t>& context_map,
                       std::size_t num_clusters) {
    const AcAnsReference reference{reference_ac_ans_encode(fc, context_map, num_clusters)};
    const std::vector<std::int32_t> q{flatten_ac(fc)};
    AcDeviceResult device{};
    ASSERT_TRUE(ac_encode_device_ans_runtime_map(q, fc.width, fc.height, context_map,
                                                 reference.tables, device));
    AcDeviceResult repeated{};
    ASSERT_TRUE(ac_encode_device_ans_runtime_map(q, fc.width, fc.height, context_map,
                                                 reference.tables, repeated));
    EXPECT_EQ(repeated.group_sizes, device.group_sizes);
    EXPECT_EQ(repeated.group_offsets, device.group_offsets);
    EXPECT_EQ(repeated.stream, device.stream);
    ASSERT_EQ(device.group_sizes.size(), reference.group_streams.size());
    std::uint32_t expected_offset{0};
    for (std::size_t group{0}; group < reference.group_streams.size(); ++group) {
        EXPECT_EQ(device.group_offsets[group], expected_offset);
        EXPECT_EQ(device.group_sizes[group], reference.group_streams[group].size());
        expected_offset += device.group_sizes[group];
        ASSERT_LE(device.group_offsets[group] + reference.group_streams[group].size(),
                  device.stream.size());
        for (std::size_t byte{0}; byte < reference.group_streams[group].size(); ++byte) {
            EXPECT_EQ(device.stream[device.group_offsets[group] + byte],
                      reference.group_streams[group][byte])
                << "group " << group << " byte " << byte;
        }
    }
    EXPECT_EQ(device.stream.size(), expected_offset);
}

TEST(EntropyGpuAns, FixedContextMapMatchesHostBytes) {
    FrameCoefficients fc{make_frame(512, 384)};
    fill_ac_pattern(fc);
    const AcAnsReference reference{reference_ac_ans_encode(fc)};
    const std::vector<std::int32_t> q{flatten_ac(fc)};
    AcDeviceResult device{};
    ASSERT_TRUE(ac_encode_device_ans(q, fc.width, fc.height, reference.tables, device));
    ASSERT_EQ(device.group_sizes.size(), reference.group_streams.size());
    for (std::size_t group{0}; group < reference.group_streams.size(); ++group) {
        EXPECT_EQ(device.group_sizes[group], reference.group_streams[group].size());
        ASSERT_LE(device.group_offsets[group] + reference.group_streams[group].size(),
                  device.stream.size());
        for (std::size_t byte{0}; byte < reference.group_streams[group].size(); ++byte) {
            EXPECT_EQ(device.stream[device.group_offsets[group] + byte],
                      reference.group_streams[group][byte])
                << "group " << group << " byte " << byte;
        }
    }
}

TEST(EntropyGpuAns, RuntimeContextMapMatchesHostBytes) {
    FrameCoefficients fc{make_frame(512, 384)};
    fill_ac_pattern(fc);
    constexpr std::size_t NUM_CLUSTERS = 7;
    std::vector<std::uint8_t> context_map(AC_NUM_CONTEXTS);
    for (std::size_t context{0}; context < context_map.size(); ++context) {
        context_map[context] =
            static_cast<std::uint8_t>((context * 5 + context / 17) % NUM_CLUSTERS);
    }
    check_ans_streams(fc, context_map, NUM_CLUSTERS);
}

TEST(EntropyGpuAns, MixedBlocksMatchHostBytes) {
    const FrameCoefficients fc{make_mixed_frame(512, 384)};
    const AcAnsReference reference{reference_ac_ans_encode(fc)};
    const std::vector<std::int32_t> q{flatten_ac(fc)};
    AcDeviceResult device{};
    ASSERT_TRUE(ac_encode_device_m3_ans(q, fc.acs, fc.width, fc.height, reference.tables, device));
    AcDeviceResult repeated{};
    ASSERT_TRUE(
        ac_encode_device_m3_ans(q, fc.acs, fc.width, fc.height, reference.tables, repeated));
    EXPECT_EQ(repeated.group_sizes, device.group_sizes);
    EXPECT_EQ(repeated.group_offsets, device.group_offsets);
    EXPECT_EQ(repeated.stream, device.stream);
    ASSERT_EQ(device.group_sizes.size(), reference.group_streams.size());
    for (std::size_t group{0}; group < reference.group_streams.size(); ++group) {
        EXPECT_EQ(device.group_sizes[group], reference.group_streams[group].size());
        ASSERT_LE(device.group_offsets[group] + reference.group_streams[group].size(),
                  device.stream.size());
        for (std::size_t byte{0}; byte < reference.group_streams[group].size(); ++byte) {
            EXPECT_EQ(device.stream[device.group_offsets[group] + byte],
                      reference.group_streams[group][byte])
                << "group " << group << " byte " << byte;
        }
    }
}

TEST(EntropyGpuM3, PerContextHistogramMatchesHost) {
    check_context_histograms(make_mixed_frame(256, 256), true);
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
