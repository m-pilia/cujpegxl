// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "transform_select.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "vardct_layout.h"

namespace cujpegxl {
namespace {

std::vector<std::int8_t> select_device(const std::vector<float>& y, std::size_t width,
                                       std::size_t height, float distance) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    float* d_y{nullptr};
    std::int8_t* d_acs{nullptr};
    cudaMalloc(&d_y, y.size() * sizeof(float));
    cudaMalloc(&d_acs, bw * bh);
    cudaMemset(d_acs, -1, bw * bh);
    cudaMemcpy(d_y, y.data(), y.size() * sizeof(float), cudaMemcpyHostToDevice);
    EXPECT_TRUE(select_transforms(d_y, width, height, distance, d_acs));
    std::vector<std::int8_t> acs(bw * bh);
    cudaMemcpy(acs.data(), d_acs, bw * bh, cudaMemcpyDeviceToHost);
    cudaFree(d_y);
    cudaFree(d_acs);
    return acs;
}

// Every in-image 8x8 block is covered by exactly one first-block whose square
// footprint stays inside the image; covered blocks carry ACS_COVERED.
void check_tiling(const std::vector<std::int8_t>& acs, std::size_t bw, std::size_t bh) {
    std::vector<int> owner(bw * bh, -1);
    for (std::size_t by{0}; by < bh; ++by) {
        for (std::size_t bx{0}; bx < bw; ++bx) {
            const std::int8_t s{acs[by * bw + bx]};
            if (s == ACS_COVERED) {
                continue;
            }
            ASSERT_TRUE(s == 8 || s == 16 || s == 32) << "bad strategy " << int{s};
            const std::size_t side{static_cast<std::size_t>(s) / 8};
            ASSERT_LE(bx + side, bw) << "first-block escapes width";
            ASSERT_LE(by + side, bh) << "first-block escapes height";
            for (std::size_t dy{0}; dy < side; ++dy) {
                for (std::size_t dx{0}; dx < side; ++dx) {
                    const std::size_t idx{(by + dy) * bw + (bx + dx)};
                    EXPECT_EQ(owner[idx], -1) << "block owned twice";
                    owner[idx] = static_cast<int>(by * bw + bx);
                    if (dy != 0 || dx != 0) {
                        EXPECT_EQ(acs[idx], ACS_COVERED) << "interior not marked covered";
                    }
                }
            }
        }
    }
    for (std::size_t i{0}; i < bw * bh; ++i) {
        EXPECT_NE(owner[i], -1) << "block " << i << " left unassigned";
    }
}

TEST(TransformSelect, DeviceMatchesHostAndTilesValidly) {
    const std::size_t width{96};
    const std::size_t height{64};
    std::vector<float> y(width * height);
    for (std::size_t i{0}; i < y.size(); ++i) {
        const std::size_t px{i % width};
        const std::size_t py{i / width};
        // Smooth gradient on the left half, high-frequency texture on the right.
        y[i] = px < width / 2 ? 0.2f + 0.3f * (static_cast<float>(px + py) / (width + height))
                              : 0.5f + 0.4f * (((px + py) & 1) ? 1.0f : -1.0f);
    }

    const std::vector<std::int8_t> dev{select_device(y, width, height, 1.0f)};
    std::vector<std::int8_t> host(width / 8 * (height / 8), -1);
    select_transforms_host(y.data(), width, height, 1.0f, host.data());

    EXPECT_EQ(dev, host);
    check_tiling(dev, width / 8, height / 8);
}

// A perfectly flat plane has no AC energy, so every region that fits coalesces
// into a single 32x32 (independent of the RD weight).
TEST(TransformSelect, FlatPlanePicksLargestBlocks) {
    const std::size_t width{64};
    const std::size_t height{64};
    const std::vector<float> y(width * height, 0.37f);
    const std::vector<std::int8_t> acs{select_device(y, width, height, 1.0f)};
    const std::size_t bw{width / 8};
    for (std::size_t by{0}; by < height / 8; ++by) {
        for (std::size_t bx{0}; bx < bw; ++bx) {
            const std::int8_t s{acs[by * bw + bx]};
            const bool first{bx % 4 == 0 && by % 4 == 0};
            EXPECT_EQ(s, first ? std::int8_t{32} : ACS_COVERED) << "bx=" << bx << " by=" << by;
        }
    }
}

// Non-multiple-of-32 dimensions: large candidates only where they fit, smaller
// blocks along the trailing edge, still a valid tiling matching the host.
TEST(TransformSelect, EdgeFallbackTilesValidly) {
    const std::size_t width{88};   // 11 blocks: not a multiple of 4
    const std::size_t height{72};  // 9 blocks
    std::vector<float> y(width * height);
    for (std::size_t i{0}; i < y.size(); ++i) {
        y[i] = 0.3f + 0.2f * static_cast<float>((i * 7) % 13) / 13.0f;
    }
    const std::vector<std::int8_t> dev{select_device(y, width, height, 1.0f)};
    std::vector<std::int8_t> host(width / 8 * (height / 8), -1);
    select_transforms_host(y.data(), width, height, 1.0f, host.data());
    EXPECT_EQ(dev, host);
    check_tiling(dev, width / 8, height / 8);
}

}  // namespace
}  // namespace cujpegxl
