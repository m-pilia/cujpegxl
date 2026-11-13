// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "dct_mma_check.h"
#include "device_support.h"

namespace cujpegxl {
namespace {

constexpr int TILE{32 * 32};

// coeff[fx * 32 + fy] = sum_{r,c} A[fy][r] A[fx][c] pixel[r][c] (dct_impl.cuh).
std::vector<double> reference_coefficients(const float* tile) {
    const double pi{std::acos(-1.0)};
    std::vector<double> a(TILE);
    for (int k{0}; k < 32; ++k) {
        for (int t{0}; t < 32; ++t) {
            a[k * 32 + t] = (k == 0 ? 1.0 : std::sqrt(2.0)) / 32.0 * std::cos(pi * (t + 0.5) * k / 32.0);
        }
    }
    std::vector<double> coeffs(TILE);
    for (int fx{0}; fx < 32; ++fx) {
        for (int fy{0}; fy < 32; ++fy) {
            double sum{0.0};
            for (int r{0}; r < 32; ++r) {
                for (int c{0}; c < 32; ++c) {
                    sum += a[fy * 32 + r] * a[fx * 32 + c] * tile[r * 32 + c];
                }
            }
            coeffs[fx * 32 + fy] = sum;
        }
    }
    return coeffs;
}

void expect_matches_definition(const std::vector<float>& tiles, double max_error) {
    const auto coeffs{dct32_mma_coefficients(tiles)};
    ASSERT_TRUE(coeffs.has_value());
    for (std::size_t tile{0}; tile < tiles.size() / TILE; ++tile) {
        const std::vector<double> want{reference_coefficients(tiles.data() + tile * TILE)};
        for (int k{0}; k < TILE; ++k) {
            ASSERT_NEAR((*coeffs)[tile * TILE + k], want[k], max_error)
                << "tile " << tile << ", fx " << k / 32 << ", fy " << k % 32;
        }
    }
}

bool tensor_cores_unsupported() {
    return query_device_support(0).arch != GpuArch::SM87;
}

TEST(DctMmaImplGpuTest, Dct32MatchesTheDefinition) {
    if (tensor_cores_unsupported()) {
        GTEST_SKIP() << "the tensor-core DCT runs on sm_87 only";
    }
    std::mt19937 rng{1234};
    std::uniform_real_distribution<float> uniform{0.0f, 1.0f};
    constexpr int RANDOM_TILES{32};
    std::vector<float> tiles((RANDOM_TILES + 3) * TILE);
    for (int i{0}; i < RANDOM_TILES * TILE; ++i) {
        tiles[i] = uniform(rng);
    }
    for (int i{0}; i < TILE; ++i) {
        const int r{i / 32};
        const int c{i % 32};
        tiles[RANDOM_TILES * TILE + i] = 0.5f;
        tiles[(RANDOM_TILES + 1) * TILE + i] = static_cast<float>(c) / 31.0f;
        tiles[(RANDOM_TILES + 2) * TILE + i] = (r + c) % 2 == 0 ? 1.0f : 0.0f;
    }
    // Pixels within 1 of the offset. Rounding them, both bases and the
    // vertical-pass result to FP16 (2^-12 relative each) bounds the error of a
    // coefficient by about 4 * 2^-12 = 9.8e-4, as no basis row's absolute sum
    // exceeds 1; an index or layout error gives errors of the order of the
    // coefficients.
    expect_matches_definition(tiles, 2.0e-3);
}

TEST(DctMmaImplGpuTest, Dct32ErrorScalesWithTheDeviationFromTheOffset) {
    if (tensor_cores_unsupported()) {
        GTEST_SKIP() << "the tensor-core DCT runs on sm_87 only";
    }
    std::mt19937 rng{1234};
    std::uniform_real_distribution<float> texture{-1.0e-3f, 1.0e-3f};
    constexpr int TILES{8};
    std::vector<float> tiles(TILES * TILE);
    for (float& pixel : tiles) {
        pixel = 0.8f + texture(rng);
    }
    // Pixels within 2e-3 of the offset: as above, about 4 * 2^-12 * 2e-3 =
    // 2e-6. Rounding the pixels themselves to FP16 would give errors of order
    // 2^-12 * 0.8 / 32 = 6e-6 per coefficient.
    expect_matches_definition(tiles, 4.0e-6);
}

}  // namespace
}  // namespace cujpegxl
