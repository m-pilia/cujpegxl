// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Validates the Modular sub-stream encoder by decoding the emitted bytes with
// libjxl's ModularGenericDecompress and comparing the recovered channel samples.

#include <cstdint>
#include <cstdlib>
#include <functional>
#include <vector>

#include <jxl/memory_manager.h>

#include <gtest/gtest.h>

#include "lib/jxl/base/span.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/modular/encoding/encoding.h"
#include "lib/jxl/modular/modular_image.h"
#include "lib/jxl/modular/options.h"

#include "modular.h"

namespace cujpegxl::bitstream {
namespace {

void* test_alloc(void*, size_t size) { return std::malloc(size); }
void test_free(void*, void* p) { std::free(p); }

void round_trip(const std::vector<ModularChannel>& channels) {
    BitWriter bw{};
    write_modular_image(bw, channels);
    bw.zero_pad_to_byte();

    JxlMemoryManager mm{nullptr, &test_alloc, &test_free};
    auto image_or = jxl::Image::Create(&mm, 0, 0, 8, 0);
    ASSERT_TRUE(image_or.ok());
    jxl::Image image{std::move(image_or).value_()};
    for (const ModularChannel& ch : channels) {
        auto c_or = jxl::Channel::Create(&mm, ch.w, ch.h);
        ASSERT_TRUE(c_or.ok());
        image.channel.push_back(std::move(c_or).value_());
    }

    jxl::BitReader br{jxl::Bytes(bw.bytes().data(), bw.bytes().size())};
    jxl::ModularOptions options{};
    ASSERT_TRUE(jxl::ModularGenericDecompress(&br, image, /*header=*/nullptr,
                                              /*group_id=*/0, &options));
    EXPECT_TRUE(br.AllReadsWithinBounds());
    EXPECT_TRUE(br.Close());

    for (std::size_t c{0}; c < channels.size(); ++c) {
        const ModularChannel& ch{channels[c]};
        for (std::size_t y{0}; y < ch.h; ++y) {
            const std::int32_t* row{image.channel[c].Row(y)};
            for (std::size_t x{0}; x < ch.w; ++x) {
                EXPECT_EQ(row[x], ch.pixels[y * ch.w + x])
                    << "channel " << c << " at (" << x << "," << y << ")";
            }
        }
    }
}

ModularChannel make_channel(std::size_t w, std::size_t h,
                            const std::function<std::int32_t(std::size_t, std::size_t)>& f) {
    ModularChannel ch{w, h, std::vector<std::int32_t>(w * h)};
    for (std::size_t y{0}; y < h; ++y) {
        for (std::size_t x{0}; x < w; ++x) {
            ch.pixels[y * w + x] = f(x, y);
        }
    }
    return ch;
}

TEST(Modular, SingleChannelSmallValues) {
    round_trip({make_channel(8, 8, [](std::size_t x, std::size_t y) {
        return static_cast<std::int32_t>(x) - static_cast<std::int32_t>(y);
    })});
}

TEST(Modular, ThreeEqualChannels) {
    round_trip({
        make_channel(32, 32, [](std::size_t x, std::size_t y) { return (x * 7 + y * 3) % 41 - 20; }),
        make_channel(32, 32, [](std::size_t x, std::size_t y) { return (x + 2 * y) % 5; }),
        make_channel(32, 32, [](std::size_t, std::size_t) { return 0; }),
    });
}

TEST(Modular, MixedChannelSizes) {
    round_trip({
        make_channel(4, 4, [](std::size_t x, std::size_t y) { return x * y; }),
        make_channel(4, 4, [](std::size_t x, std::size_t y) { return -static_cast<std::int32_t>(x + y); }),
        make_channel(6, 2, [](std::size_t x, std::size_t) { return 1000 * (static_cast<std::int32_t>(x) - 3); }),
    });
}

TEST(Modular, AllZeroChannels) {
    round_trip({
        make_channel(32, 32, [](std::size_t, std::size_t) { return 0; }),
        make_channel(32, 32, [](std::size_t, std::size_t) { return 0; }),
    });
}

}  // namespace
}  // namespace cujpegxl::bitstream
