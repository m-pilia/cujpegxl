// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include <cmath>
#include <cstddef>
#include <optional>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "dct_mma_check.h"
#include "device_support.h"

namespace cujpegxl {
namespace {

// coeff[fx * side + fy] = sum_{r,c} A[fy % n][r] A[fx % n][c] pixel[r][c] over
// the pixels of the n x n block (fx / n, fy / n) of the side x side tile
// (dct_impl.cuh).
std::vector<double> reference_coefficients(const float* tile, int side, int n) {
    const double pi{std::acos(-1.0)};
    std::vector<double> a(static_cast<std::size_t>(n * n));
    for (int k{0}; k < n; ++k) {
        for (int t{0}; t < n; ++t) {
            a[k * n + t] = (k == 0 ? 1.0 : std::sqrt(2.0)) / n * std::cos(pi * (t + 0.5) * k / n);
        }
    }
    std::vector<double> coeffs(static_cast<std::size_t>(side * side));
    for (int fx{0}; fx < side; ++fx) {
        for (int fy{0}; fy < side; ++fy) {
            const int x0{fx / n * n};
            const int y0{fy / n * n};
            double sum{0.0};
            for (int r{0}; r < n; ++r) {
                for (int c{0}; c < n; ++c) {
                    sum +=
                        a[(fy % n) * n + r] * a[(fx % n) * n + c] * tile[(y0 + r) * side + x0 + c];
                }
            }
            coeffs[fx * side + fy] = sum;
        }
    }
    return coeffs;
}

void expect_matches_definition(const std::optional<std::vector<float>>& coeffs,
                               const std::vector<float>& tiles, int side, int n, double max_error) {
    ASSERT_TRUE(coeffs.has_value());
    const std::size_t samples{static_cast<std::size_t>(side * side)};
    for (std::size_t tile{0}; tile < tiles.size() / samples; ++tile) {
        const std::vector<double> want{
            reference_coefficients(tiles.data() + tile * samples, side, n)};
        for (std::size_t k{0}; k < samples; ++k) {
            ASSERT_NEAR((*coeffs)[tile * samples + k], want[k], max_error)
                << "tile " << tile << ", fx " << k / side << ", fy " << k % side;
        }
    }
}

// Random tiles in [0, 1), then a flat tile, a horizontal ramp and a
// checkerboard.
std::vector<float> test_tiles(int side) {
    const std::size_t samples{static_cast<std::size_t>(side * side)};
    std::mt19937 rng{1234};
    std::uniform_real_distribution<float> uniform{0.0f, 1.0f};
    constexpr std::size_t RANDOM_TILES{32};
    std::vector<float> tiles((RANDOM_TILES + 3) * samples);
    for (std::size_t i{0}; i < RANDOM_TILES * samples; ++i) {
        tiles[i] = uniform(rng);
    }
    for (std::size_t i{0}; i < samples; ++i) {
        const std::size_t r{i / side};
        const std::size_t c{i % side};
        tiles[RANDOM_TILES * samples + i] = 0.5f;
        tiles[(RANDOM_TILES + 1) * samples + i] = static_cast<float>(c) / (side - 1);
        tiles[(RANDOM_TILES + 2) * samples + i] = (r + c) % 2 == 0 ? 1.0f : 0.0f;
    }
    return tiles;
}

// Tiles of 0.8 plus a texture in [-1e-3, 1e-3).
std::vector<float> textured_tiles(int side) {
    std::mt19937 rng{1234};
    std::uniform_real_distribution<float> texture{-1.0e-3f, 1.0e-3f};
    std::vector<float> tiles(static_cast<std::size_t>(8 * side * side));
    for (float& pixel : tiles) {
        pixel = 0.8f + texture(rng);
    }
    return tiles;
}

// Pixels within 1 of the offset. Rounding them, both bases and the
// vertical-pass result to FP16 (2^-12 relative each) bounds the error of a
// coefficient by about 4 * 2^-12 = 9.8e-4, as no basis row's absolute sum
// exceeds 1; an index or layout error gives errors of the order of the
// coefficients.
constexpr double MAX_ERROR{2.0e-3};
// Pixels of textured_tiles are within 2e-3 of the offset: as above, about
// 4 * 2^-12 * 2e-3 = 2e-6. Rounding the pixels themselves to FP16 would give
// errors of order 2^-12 * 0.8 / n = 6e-6 (n = 32) to 2.4e-5 (n = 8) per
// coefficient.
constexpr double MAX_TEXTURED_ERROR{4.0e-6};

bool tensor_cores_unsupported() {
    return query_device_support(0).arch != GpuArch::SM87;
}

TEST(DctMmaImplGpuTest, Dct32MatchesTheDefinition) {
    if (tensor_cores_unsupported()) {
        GTEST_SKIP() << "the tensor-core DCT runs on sm_87 only";
    }
    const std::vector<float> tiles{test_tiles(32)};
    expect_matches_definition(dct32_mma_coefficients(tiles), tiles, 32, 32, MAX_ERROR);
}

TEST(DctMmaImplGpuTest, Dct32ErrorScalesWithTheDeviationFromTheOffset) {
    if (tensor_cores_unsupported()) {
        GTEST_SKIP() << "the tensor-core DCT runs on sm_87 only";
    }
    const std::vector<float> tiles{textured_tiles(32)};
    expect_matches_definition(dct32_mma_coefficients(tiles), tiles, 32, 32, MAX_TEXTURED_ERROR);
}

TEST(DctMmaImplGpuTest, RegionTransformsMatchTheDefinition) {
    if (tensor_cores_unsupported()) {
        GTEST_SKIP() << "the tensor-core DCT runs on sm_87 only";
    }
    const std::vector<float> regions{test_tiles(16)};
    for (const int n : {8, 16}) {
        SCOPED_TRACE(n);
        expect_matches_definition(region_mma_coefficients(regions, n), regions, 16, n, MAX_ERROR);
    }
}

TEST(DctMmaImplGpuTest, RegionErrorScalesWithTheDeviationFromTheOffset) {
    if (tensor_cores_unsupported()) {
        GTEST_SKIP() << "the tensor-core DCT runs on sm_87 only";
    }
    const std::vector<float> regions{textured_tiles(16)};
    for (const int n : {8, 16}) {
        SCOPED_TRACE(n);
        expect_matches_definition(region_mma_coefficients(regions, n), regions, 16, n,
                                  MAX_TEXTURED_ERROR);
    }
}

}  // namespace
}  // namespace cujpegxl
