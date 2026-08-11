// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Testonly harness: runs forward DCT + uniform quantization on a planar XYB
// float32 file and writes the int32 quantized coefficients, for the
// self-consistency test. Not part of the shipped runtime.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "dct.h"
#include "quant.h"
#include "quant_calibration.h"

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

int run_quant_dump(int argc, char** argv) {
    if (argc != 7 || std::strcmp(argv[1], "quant") != 0) {
        std::fprintf(stderr,
                     "usage: %s quant <width> <height> <distance> <in_xyb_f32> <out_q_i32>\n",
                     argv[0]);
        return 2;
    }
    const std::size_t width{std::strtoull(argv[2], nullptr, 10)};
    const std::size_t height{std::strtoull(argv[3], nullptr, 10)};
    const float distance{std::strtof(argv[4], nullptr)};
    const std::size_t count{3 * width * height};

    std::vector<float> xyb{};
    if (!read_floats(argv[5], count, &xyb)) {
        std::fprintf(stderr, "quant_dump: %s is not %zu floats\n", argv[5], count);
        return 1;
    }

    // Uniform quant field (= calibrated raw_quant_field) so this harness exercises
    // the quantizer math independently of the adaptive-quant field.
    const std::size_t blocks{(width / 8) * (height / 8)};
    const std::vector<std::int32_t> quant_field(
        blocks, static_cast<std::int32_t>(cujpegxl::calibrate_quant(distance).raw_quant_field));

    float* d_xyb{nullptr};
    float* d_coeffs{nullptr};
    std::int32_t* d_qf{nullptr};
    std::int32_t* d_q{nullptr};
    if (cudaMalloc(&d_xyb, count * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_coeffs, count * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_qf, blocks * sizeof(std::int32_t)) != cudaSuccess ||
        cudaMalloc(&d_q, count * sizeof(std::int32_t)) != cudaSuccess) {
        std::fprintf(stderr, "quant_dump: cudaMalloc failed\n");
        return 1;
    }
    cudaMemcpy(d_xyb, xyb.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qf, quant_field.data(), blocks * sizeof(std::int32_t), cudaMemcpyHostToDevice);

    if (!cujpegxl::forward_dct8(d_xyb, width, height, d_coeffs) ||
        !cujpegxl::quantize_dct8(d_coeffs, width, height, distance, d_qf, d_q)) {
        std::fprintf(stderr, "quant_dump: kernel failed\n");
        return 1;
    }

    std::vector<std::int32_t> q(count);
    cudaMemcpy(q.data(), d_q, count * sizeof(std::int32_t), cudaMemcpyDeviceToHost);
    cudaFree(d_xyb);
    cudaFree(d_coeffs);
    cudaFree(d_qf);
    cudaFree(d_q);

    std::FILE* out{std::fopen(argv[6], "wb")};
    if (out == nullptr) {
        std::fprintf(stderr, "quant_dump: cannot open %s for writing\n", argv[6]);
        return 1;
    }
    const std::size_t put{std::fwrite(q.data(), sizeof(std::int32_t), q.size(), out)};
    std::fclose(out);
    return put == q.size() ? 0 : 1;
}
