// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Runs the NV12->XYB kernel on a raw NV12 file and writes the planar float32
// XYB.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "dump_io.h"
#include "xyb.h"

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
    if (!cujpegxl::read_raw(argv[4], luma_size + chroma_size, &nv12)) {
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

    if (!cujpegxl::write_raw(argv[5], xyb.data(), xyb.size())) {
        std::fprintf(stderr, "dump: cannot write %s\n", argv[5]);
        return 1;
    }
    return 0;
}
