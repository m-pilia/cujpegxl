// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Testonly harness: runs the forward 8x8 DCT on a planar XYB float32 file and
// writes the resulting coefficients, so a Python test can diff them against the
// libjxl DCT oracle. Not part of the shipped runtime.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "dct.h"

namespace {

bool read_floats(const char* path, std::size_t count, std::vector<float>* out) {
    out->resize(count);
    std::FILE* file{std::fopen(path, "rb")};
    if (file == nullptr) {
        return false;
    }
    const std::size_t got{std::fread(out->data(), sizeof(float), count, file)};
    std::fclose(file);
    return got == count;
}

}  // namespace

int run_dct_dump(int argc, char** argv) {
    // `dct <w> <h> ...` runs DCT8; `dctn <block_dim> <w> <h> ...` selects the
    // square transform size, mirroring the libjxl oracle's subcommands.
    std::size_t block_dim{8};
    int arg{1};
    if (argc == 7 && std::strcmp(argv[1], "dctn") == 0) {
        block_dim = std::strtoull(argv[2], nullptr, 10);
        arg = 3;
    } else if (argc == 6 && std::strcmp(argv[1], "dct") == 0) {
        arg = 2;
    } else {
        std::fprintf(stderr,
                     "usage: %s dct <width> <height> <in_xyb_f32> <out_coeff_f32>\n"
                     "       %s dctn <block_dim> <width> <height> <in_xyb_f32> <out_coeff_f32>\n",
                     argv[0], argv[0]);
        return 2;
    }
    if (block_dim != 8 && block_dim != 16 && block_dim != 32) {
        std::fprintf(stderr, "dct_dump: block dim must be 8, 16, or 32\n");
        return 2;
    }
    const std::size_t width{std::strtoull(argv[arg], nullptr, 10)};
    const std::size_t height{std::strtoull(argv[arg + 1], nullptr, 10)};
    const std::size_t count{3 * width * height};

    std::vector<float> xyb{};
    if (!read_floats(argv[arg + 2], count, &xyb)) {
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

    const char* out_path{argv[arg + 3]};
    std::FILE* out{std::fopen(out_path, "wb")};
    if (out == nullptr) {
        std::fprintf(stderr, "dct_dump: cannot open %s for writing\n", out_path);
        return 1;
    }
    const std::size_t put{std::fwrite(coeffs.data(), sizeof(float), coeffs.size(), out)};
    std::fclose(out);
    return put == coeffs.size() ? 0 : 1;
}
