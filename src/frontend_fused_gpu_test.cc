// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frontend_fused.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "quant_calibration.h"
#include "vardct_layout.h"

namespace cujpegxl {
namespace {

constexpr std::size_t WIDTH = 256;
constexpr std::size_t HEIGHT = 128;
constexpr std::size_t BLOCKS = (WIDTH / 8) * (HEIGHT / 8);

template <typename T>
T* device_alloc(std::size_t count) {
    T* device{nullptr};
    cudaMalloc(&device, count * sizeof(T));
    return device;
}

template <typename T>
std::vector<T> download(const T* device, std::size_t count) {
    std::vector<T> host(count);
    cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost);
    return host;
}

// NV12 noise whose amplitude grows across the columns, so that the adaptive
// field spreads over a wide range; flat gives a constant field.
std::vector<std::uint8_t> make_nv12(bool flat) {
    std::vector<std::uint8_t> nv12(WIDTH * HEIGHT * 3 / 2, 128);
    if (flat) {
        return nv12;
    }
    std::mt19937 rng{7};
    std::normal_distribution<float> noise{0.0f, 1.0f};
    for (std::size_t y{0}; y < HEIGHT; ++y) {
        for (std::size_t x{0}; x < WIDTH; ++x) {
            const float amplitude{60.0f * static_cast<float>(x * x) / (WIDTH * WIDTH)};
            const float v{100.0f + 0.3f * static_cast<float>(y) + amplitude * noise(rng)};
            nv12[y * WIDTH + x] = static_cast<std::uint8_t>(std::clamp(v, 0.0f, 255.0f));
        }
    }
    return nv12;
}

struct Frontend {
    std::vector<float> adaptive_field;
    std::vector<std::int32_t> quant_field;
    QuantCalibration calibration{};
};

[[nodiscard]] bool run_frontend(const std::vector<std::uint8_t>& nv12, float distance,
                                Frontend& out) {
    std::uint8_t* d_nv12{device_alloc<std::uint8_t>(nv12.size())};
    __half* d_coeffs{device_alloc<__half>(3 * BLOCKS * COEFFS_PER_BLOCK)};
    std::int8_t* d_acs{device_alloc<std::int8_t>(BLOCKS)};
    float* d_adaptive_field{device_alloc<float>(BLOCKS)};
    std::uint32_t* d_field_histogram{device_alloc<std::uint32_t>(ADAPTIVE_FIELD_BINS)};
    std::int32_t* d_quant_field{device_alloc<std::int32_t>(BLOCKS)};
    QuantCalibration* d_calibration{device_alloc<QuantCalibration>(1)};
    cudaMemcpy(d_nv12, nv12.data(), nv12.size(), cudaMemcpyHostToDevice);

    bool ok{false};
    {
        const ChromaTexture chroma{d_nv12 + WIDTH * HEIGHT, WIDTH, WIDTH, HEIGHT, nullptr};
        ok = encode_frontend(d_nv12, WIDTH, chroma, WIDTH, HEIGHT, distance, d_coeffs, d_acs,
                             d_adaptive_field, d_field_histogram, d_quant_field, d_calibration,
                             nullptr);
    }
    out.adaptive_field = download(d_adaptive_field, BLOCKS);
    out.quant_field = download(d_quant_field, BLOCKS);
    out.calibration = download(d_calibration, 1)[0];

    cudaFree(d_nv12);
    cudaFree(d_coeffs);
    cudaFree(d_acs);
    cudaFree(d_adaptive_field);
    cudaFree(d_field_histogram);
    cudaFree(d_quant_field);
    cudaFree(d_calibration);
    return ok && cudaDeviceSynchronize() == cudaSuccess;
}

// libjxl's order statistic of rank size / 2 (Quantizer::SetQuantField).
float order_statistic(std::vector<float> values) {
    const auto middle{values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2)};
    std::nth_element(values.begin(), middle, values.end());
    return *middle;
}

// The global scale is libjxl's for the exact median and median absolute
// deviation of the adaptive field, up to the histogram's bin width (2^-7 of
// the value) at the two order statistics, and the integer field follows it.
void expect_calibrated(bool flat, float distance) {
    Frontend f{};
    ASSERT_TRUE(run_frontend(make_nv12(flat), distance, f));

    const float median{order_statistic(f.adaptive_field)};
    std::vector<float> deviations(f.adaptive_field.size());
    std::transform(f.adaptive_field.begin(), f.adaptive_field.end(), deviations.begin(),
                   [median](float qf) { return std::fabs(qf - median); });
    const float deviation{order_statistic(deviations)};
    const QuantCalibration want{calibrate_quant(initial_quant_dc(distance), median, deviation)};

    constexpr float BIN_WIDTH = 1.0f / 128.0f;
    const float tolerance{static_cast<float>(K_GLOBAL_SCALE_DENOM) / K_QUANT_FIELD_TARGET *
                              BIN_WIDTH * (median + (median + deviation)) +
                          1.0f};
    EXPECT_NEAR(static_cast<float>(f.calibration.global_scale),
                static_cast<float>(want.global_scale), tolerance);
    EXPECT_FLOAT_EQ(f.calibration.global_scale_float,
                    static_cast<float>(f.calibration.global_scale) / K_GLOBAL_SCALE_DENOM);

    constexpr float ROUNDING_TOLERANCE = 1e-4f;
    for (std::size_t i{0}; i < BLOCKS; ++i) {
        const float steps{f.adaptive_field[i] * f.calibration.inv_global_scale + 0.5f};
        if (std::fabs(steps - std::round(steps)) < ROUNDING_TOLERANCE) {
            continue;
        }
        EXPECT_EQ(f.quant_field[i],
                  quant_field_value(f.adaptive_field[i], f.calibration.inv_global_scale))
            << "block " << i;
    }
}

TEST(FrontendFused, CalibratesTexturedFieldAtDistance1) {
    expect_calibrated(false, 1.0f);
}

TEST(FrontendFused, CalibratesTexturedFieldAtDistance4) {
    expect_calibrated(false, 4.0f);
}

TEST(FrontendFused, CalibratesFlatField) {
    expect_calibrated(true, 1.0f);
}

}  // namespace
}  // namespace cujpegxl
