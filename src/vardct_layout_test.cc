// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "vardct_layout.h"

#include <cstddef>
#include <set>
#include <vector>

#include <gtest/gtest.h>

namespace cujpegxl {
namespace {

// A first-block's coefficients scatter into, and gather back from, the covered
// blocks' storage without loss, for every square size.
TEST(CoveredBlockLayout, ScatterGatherRoundTrip) {
    const std::size_t bw{8};
    const std::size_t bh{8};
    std::vector<float> plane(bw * bh * COEFFS_PER_BLOCK, -1.0f);
    for (std::size_t block_dim : {8, 16, 32}) {
        const std::size_t count{block_dim * block_dim};
        std::vector<float> coeffs(count);
        for (std::size_t i{0}; i < count; ++i) {
            coeffs[i] = static_cast<float>(i) + 0.5f;
        }
        scatter_covered_block(block_dim, 0, 0, bw, coeffs.data(), plane.data());
        std::vector<float> back(count, 0.0f);
        gather_covered_block(block_dim, 0, 0, bw, plane.data(), back.data());
        EXPECT_EQ(coeffs, back) << "block_dim=" << block_dim;
    }
}

// The raw-index -> plane-slot map is a bijection onto exactly the covered
// blocks' slots (no collisions, no spill outside the covered region).
TEST(CoveredBlockLayout, SlotMapIsBijectionOverCoveredRegion) {
    const std::size_t bw{8};
    const std::size_t bx{2};
    const std::size_t by{1};
    for (std::size_t block_dim : {8, 16, 32}) {
        const std::size_t side{covered_blocks_side(block_dim)};
        const std::size_t count{block_dim * block_dim};
        std::set<std::size_t> slots{};
        std::set<std::size_t> expected_blocks{};
        for (std::size_t dy{0}; dy < side; ++dy) {
            for (std::size_t dx{0}; dx < side; ++dx) {
                expected_blocks.insert((by + dy) * bw + (bx + dx));
            }
        }
        for (std::size_t raw{0}; raw < count; ++raw) {
            const std::size_t slot{covered_plane_slot(block_dim, bx, by, bw, raw)};
            EXPECT_TRUE(slots.insert(slot).second) << "collision at raw=" << raw;
            EXPECT_TRUE(expected_blocks.count(slot / COEFFS_PER_BLOCK) == 1)
                << "raw=" << raw << " escaped covered region";
        }
        EXPECT_EQ(slots.size(), count);
    }
}

// DC-from-LLF of a constant-valued block reproduces that constant in every
// covered position (the DC of a flat region is its mean).
TEST(DcFromLlf, ConstantBlockYieldsConstantDc) {
    for (std::size_t block_dim : {8, 16, 32}) {
        const std::size_t side{covered_blocks_side(block_dim)};
        std::vector<float> coeffs(block_dim * block_dim, 0.0f);
        coeffs[0] = 3.25f;  // whole-block DC (mean); forward_dctN stores it at slot 0
        std::vector<float> dc(side * side, 0.0f);
        dc_from_llf(block_dim, coeffs.data(), dc.data());
        for (float v : dc) {
            EXPECT_NEAR(v, 3.25f, 1e-5f) << "block_dim=" << block_dim;
        }
    }
}

}  // namespace
}  // namespace cujpegxl
