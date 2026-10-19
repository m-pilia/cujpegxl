// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Runs a forward square DCT on a planar XYB float32 file and writes the
// coefficients as float32.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "dct.h"
#include "dump_io.h"

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

    const bool ok{block_dim == 16   ? cujpegxl::forward_dct16(d_xyb, width, height, d_coeffs)
                  : block_dim == 32 ? cujpegxl::forward_dct32(d_xyb, width, height, d_coeffs)
                                    : cujpegxl::forward_dct8(d_xyb, width, height, d_coeffs)};
    if (!ok) {
        std::fprintf(stderr, "dct_dump: forward_dct%zu failed\n", block_dim);
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
