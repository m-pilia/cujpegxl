// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Testonly harness: runs the NV12->XYB kernel on a raw NV12 file and writes the
// resulting planar float32 XYB, so a Python test can diff it against the libjxl
// oracle. Not part of the shipped runtime.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "xyb.h"

namespace {

bool read_file(const char* path, std::vector<std::uint8_t>* out) {
    std::FILE* file{std::fopen(path, "rb")};
    if (file == nullptr) {
        return false;
    }
    std::fseek(file, 0, SEEK_END);
    const long size{std::ftell(file)};
    std::fseek(file, 0, SEEK_SET);
    out->resize(static_cast<std::size_t>(size));
    const std::size_t got{std::fread(out->data(), 1, out->size(), file)};
    std::fclose(file);
    return got == out->size();
}

}  // namespace

int run_nv12_to_xyb_dump(int argc, char** argv) {
    if (argc != 6 || std::strcmp(argv[1], "dump") != 0) {
        std::fprintf(stderr, "usage: %s dump <width> <height> <in_nv12> <out_xyb_f32>\n", argv[0]);
        return 2;
    }
    const std::size_t width{std::strtoull(argv[2], nullptr, 10)};
    const std::size_t height{std::strtoull(argv[3], nullptr, 10)};
    const std::size_t luma_size{width * height};
    const std::size_t chroma_size{width * (height / 2)};

    std::vector<std::uint8_t> nv12{};
    if (!read_file(argv[4], &nv12) || nv12.size() != luma_size + chroma_size) {
        std::fprintf(stderr, "dump: %s is not a %zux%zu NV12 buffer\n", argv[4], width, height);
        return 1;
    }

    std::uint8_t* d_luma{nullptr};
    std::uint8_t* d_chroma{nullptr};
    float* d_xyb{nullptr};
    if (cudaMalloc(&d_luma, luma_size) != cudaSuccess ||
        cudaMalloc(&d_chroma, chroma_size) != cudaSuccess ||
        cudaMalloc(&d_xyb, 3 * luma_size * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "dump: cudaMalloc failed\n");
        return 1;
    }
    cudaMemcpy(d_luma, nv12.data(), luma_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_chroma, nv12.data() + luma_size, chroma_size, cudaMemcpyHostToDevice);

    if (!cujpegxl::nv12_to_xyb(d_luma, width, d_chroma, width, width, height, d_xyb)) {
        std::fprintf(stderr, "dump: nv12_to_xyb failed\n");
        return 1;
    }

    std::vector<float> xyb(3 * luma_size);
    cudaMemcpy(xyb.data(), d_xyb, xyb.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_luma);
    cudaFree(d_chroma);
    cudaFree(d_xyb);

    std::FILE* out{std::fopen(argv[5], "wb")};
    if (out == nullptr) {
        std::fprintf(stderr, "dump: cannot open %s for writing\n", argv[5]);
        return 1;
    }
    const std::size_t put{std::fwrite(xyb.data(), sizeof(float), xyb.size(), out)};
    std::fclose(out);
    return put == xyb.size() ? 0 : 1;
}
