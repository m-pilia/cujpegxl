// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frontend_transform.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "transform_select.h"
#include "vardct_layout.h"

namespace cujpegxl {
namespace {

// Reference forward DCT of one side*side block at (px0, py0) of a single float
// plane, matching forward_dctN / variable_forward_dct: coeff[fx*N+fy], the
// orthonormal 2D DCT divided by N.
float ref_coeff(const float* plane, std::size_t width, std::size_t px0, std::size_t py0, int n,
                int fx, int fy) {
    auto a = [n](int k, int t) {
        const double g = k == 0 ? 1.0 / n : 1.4142135623730951 / n;
        return g * std::cos(3.14159265358979323846 * (t + 0.5) * k / n);
    };
    double acc{0.0};
    for (int r{0}; r < n; ++r) {
        double rowsum{0.0};
        for (int col{0}; col < n; ++col) {
            rowsum += a(fx, col) * static_cast<double>(plane[(py0 + r) * width + (px0 + col)]);
        }
        acc += a(fy, r) * rowsum;
    }
    return static_cast<float>(acc);
}

std::vector<__half> run_dct(const std::vector<float>& xyb, std::size_t width, std::size_t height,
                            const std::vector<std::int8_t>& acs) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t coeff_count{3 * bw * bh * COEFFS_PER_BLOCK};
    float* d_xyb{nullptr};
    std::int8_t* d_acs{nullptr};
    __half* d_coeffs{nullptr};
    cudaMalloc(&d_xyb, xyb.size() * sizeof(float));
    cudaMalloc(&d_acs, acs.size());
    cudaMalloc(&d_coeffs, coeff_count * sizeof(__half));
    cudaMemset(d_coeffs, 0, coeff_count * sizeof(__half));
    cudaMemcpy(d_xyb, xyb.data(), xyb.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_acs, acs.data(), acs.size(), cudaMemcpyHostToDevice);
    EXPECT_TRUE(variable_forward_dct(d_xyb, width, height, d_acs, d_coeffs));
    std::vector<__half> coeffs(coeff_count);
    cudaMemcpy(coeffs.data(), d_coeffs, coeff_count * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaFree(d_xyb);
    cudaFree(d_acs);
    cudaFree(d_coeffs);
    return coeffs;
}

void place(std::vector<std::int8_t>& acs, std::size_t bw, int side, std::size_t bx,
           std::size_t by) {
    const std::size_t s{static_cast<std::size_t>(side) / 8};
    for (std::size_t dy{0}; dy < s; ++dy) {
        for (std::size_t dx{0}; dx < s; ++dx) {
            acs[(by + dy) * bw + (bx + dx)] =
                (dy == 0 && dx == 0) ? static_cast<std::int8_t>(side) : ACS_COVERED;
        }
    }
}

TEST(FrontendTransform, VariableDctMatchesReferencePerBlock) {
    const std::size_t width{96};
    const std::size_t height{64};
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t plane{width * height};
    std::vector<float> xyb(3 * plane);
    for (std::size_t i{0}; i < xyb.size(); ++i) {
        xyb[i] = 0.1f * std::sin(0.03f * static_cast<float>(i)) +
                 0.02f * static_cast<float>((i * 7) % 11);
    }
    std::vector<std::int8_t> acs(bw * bh, 8);
    place(acs, bw, 32, 0, 0);
    place(acs, bw, 16, 4, 0);

    const std::vector<__half> coeffs{run_dct(xyb, width, height, acs)};
    const std::size_t cplane{bw * bh * COEFFS_PER_BLOCK};
    for (std::size_t by{0}; by < bh; ++by) {
        for (std::size_t bx{0}; bx < bw; ++bx) {
            const int n{acs[by * bw + bx]};
            if (n == ACS_COVERED) {
                continue;
            }
            for (int c{0}; c < 3; ++c) {
                const float* p{xyb.data() + c * plane};
                for (int fx{0}; fx < n; ++fx) {
                    for (int fy{0}; fy < n; ++fy) {
                        const std::size_t raw{static_cast<std::size_t>(fx) * n + fy};
                        const float got{__half2float(
                            coeffs[c * cplane + covered_plane_slot(n, bx, by, bw, raw)])};
                        const float want{ref_coeff(p, width, bx * 8, by * 8, n, fx, fy)};
                        EXPECT_NEAR(got, want, 3e-3f + 2e-2f * std::fabs(want))
                            << "block (" << bx << "," << by << ") side " << n << " c" << c;
                    }
                }
            }
        }
    }
}

TEST(FrontendTransform, Deterministic) {
    const std::size_t width{64};
    const std::size_t height{64};
    const std::size_t bw{width / 8};
    std::vector<float> xyb(3 * width * height);
    for (std::size_t i{0}; i < xyb.size(); ++i) {
        xyb[i] = 0.05f * static_cast<float>((i * 13) % 17) - 0.2f;
    }
    const std::vector<std::int8_t> acs(bw * (height / 8), 8);
    const std::vector<__half> a{run_dct(xyb, width, height, acs)};
    const std::vector<__half> b{run_dct(xyb, width, height, acs)};
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i{0}; i < a.size(); ++i) {
        EXPECT_EQ(__half2float(a[i]), __half2float(b[i]));
    }
}

}  // namespace
}  // namespace cujpegxl
