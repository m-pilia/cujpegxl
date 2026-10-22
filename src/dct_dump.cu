// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Runs the front end's forward square DCT (dct_impl.cuh) on a planar XYB
// float32 file and writes the coefficients as float32: per plane, NxN blocks in
// raster order with N*N coefficients each in libjxl's transposed raster order.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "dct_impl.cuh"
#include "dump_io.h"

namespace {

// One thread per pixel row (horizontal pass), then per frequency fx (vertical
// pass).
template <int N>
__global__ void forward_dct_kernel(const float* __restrict__ xyb, std::size_t width,
                                   std::size_t height, float* __restrict__ coeffs) {
    __shared__ float rows[N * N];

    const std::size_t bx{blockIdx.x};
    const std::size_t by{blockIdx.y};
    const std::size_t plane{width * height};
    const float* src{xyb + blockIdx.z * plane};
    const int i{static_cast<int>(threadIdx.x)};

    float v[N];
#pragma unroll
    for (int t{0}; t < N; ++t) {
        v[t] = src[(by * N + i) * width + bx * N + t];
    }
    cujpegxl::dct1d(v);
#pragma unroll
    for (int fx{0}; fx < N; ++fx) {
        rows[fx * N + i] = v[fx];
    }
    __syncthreads();
#pragma unroll
    for (int t{0}; t < N; ++t) {
        v[t] = rows[i * N + t];
    }
    cujpegxl::dct1d(v);
    const std::size_t block_index{by * (width / N) + bx};
#pragma unroll
    for (int fy{0}; fy < N; ++fy) {
        coeffs[blockIdx.z * plane + block_index * (N * N) + i * N + fy] =
            v[fy] * cujpegxl::DCT2D_SCALE<N>;
    }
}

template <int N>
bool forward_dct(const float* xyb, std::size_t width, std::size_t height, float* coeffs) {
    const dim3 grid{static_cast<unsigned int>(width / N), static_cast<unsigned int>(height / N), 3};
    forward_dct_kernel<N><<<grid, N>>>(xyb, width, height, coeffs);
    const cudaError_t launch{cudaGetLastError()};
    const cudaError_t sync{cudaDeviceSynchronize()};
    return launch == cudaSuccess && sync == cudaSuccess;
}

}  // namespace

int run_dct_dump(int argc, char** argv) {
    if (argc != 7 || std::strcmp(argv[1], "dctn") != 0) {
        std::fprintf(stderr,
                     "usage: %s dctn <block_dim> <width> <height> <in_xyb_f32> <out_coeff_f32>\n",
                     argv[0]);
        return 2;
    }
    const std::size_t block_dim{std::strtoull(argv[2], nullptr, 10)};
    constexpr int arg{3};
    if (block_dim != 8 && block_dim != 16 && block_dim != 32) {
        std::fprintf(stderr, "dct_dump: block dim must be 8, 16, or 32\n");
        return 2;
    }
    const std::size_t width{std::strtoull(argv[arg], nullptr, 10)};
    const std::size_t height{std::strtoull(argv[arg + 1], nullptr, 10)};
    const std::size_t count{3 * width * height};

    std::vector<float> xyb{};
    if (!cujpegxl::read_raw(argv[arg + 2], count, &xyb)) {
        std::fprintf(stderr, "dct_dump: %s is not %zu floats\n", argv[arg + 2], count);
        return 1;
    }

    float* d_xyb{nullptr};
    float* d_coeffs{nullptr};
    if (cudaMalloc(&d_xyb, count * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_coeffs, count * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "dct_dump: cudaMalloc failed\n");
        return 1;
    }
    cudaMemcpy(d_xyb, xyb.data(), count * sizeof(float), cudaMemcpyHostToDevice);

    const bool ok{block_dim == 16   ? forward_dct<16>(d_xyb, width, height, d_coeffs)
                  : block_dim == 32 ? forward_dct<32>(d_xyb, width, height, d_coeffs)
                                    : forward_dct<8>(d_xyb, width, height, d_coeffs)};
    if (!ok) {
        std::fprintf(stderr, "dct_dump: forward DCT%zu failed\n", block_dim);
        return 1;
    }

    std::vector<float> coeffs(count);
    cudaMemcpy(coeffs.data(), d_coeffs, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_xyb);
    cudaFree(d_coeffs);

    if (!cujpegxl::write_raw(argv[arg + 3], coeffs.data(), coeffs.size())) {
        std::fprintf(stderr, "dct_dump: cannot write %s\n", argv[arg + 3]);
        return 1;
    }
    return 0;
}
