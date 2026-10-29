// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Encodes one raw NV12 frame to a .jxl file via cujpegxl::FrameEncoder.
//
//   encode_to_file --dims=WxH --distance=F [--device=N] --output=OUT in.nv12

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "src/frame_encoder.h"
#include "src/worker_pool.h"

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "encode_to_file: %s\n", msg.c_str());
    std::exit(1);
}

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        die(std::string("cuda ") + what + " failed: " + cudaGetErrorString(err));
    }
}

std::string value_of(const std::string& arg) {
    return arg.substr(arg.find('=') + 1);
}

std::vector<std::uint8_t> read_file(const std::string& path, std::size_t expected) {
    std::vector<std::uint8_t> buf(expected);
    std::ifstream f{path, std::ios::binary};
    if (!f) {
        die("cannot open " + path);
    }
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(expected));
    if (static_cast<std::size_t>(f.gcount()) != expected) {
        die("short read on " + path);
    }
    return buf;
}

}  // namespace

int main(int argc, char** argv) {
    std::uint32_t width{0};
    std::uint32_t height{0};
    float distance{1.0f};
    int device{0};
    std::string output;
    std::string input;

    for (int i{1}; i < argc; ++i) {
        const std::string arg{argv[i]};
        if (arg.rfind("--dims=", 0) == 0) {
            const std::string v{value_of(arg)};
            if (std::sscanf(v.c_str(), "%ux%u", &width, &height) != 2) {
                die("expected --dims=WxH, got: " + v);
            }
        } else if (arg.rfind("--distance=", 0) == 0) {
            distance = std::strtof(value_of(arg).c_str(), nullptr);
        } else if (arg.rfind("--device=", 0) == 0) {
            device = std::atoi(value_of(arg).c_str());
        } else if (arg.rfind("--output=", 0) == 0) {
            output = value_of(arg);
        } else if (arg.rfind("--", 0) == 0) {
            die("unknown option: " + arg);
        } else {
            input = arg;
        }
    }

    if (width == 0 || height == 0) {
        die("--dims=WxH is required");
    }
    if (output.empty()) {
        die("--output=PATH is required");
    }
    if (input.empty()) {
        die("one NV12 input file is required");
    }
    if (!(distance > 0.0f)) {
        die("--distance must be positive");
    }

    const std::size_t luma_bytes{static_cast<std::size_t>(width) * height};
    const std::size_t chroma_bytes{static_cast<std::size_t>(width) * (height / 2)};
    const std::vector<std::uint8_t> host{read_file(input, luma_bytes + chroma_bytes)};

    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    std::uint8_t* d_luma{nullptr};
    std::uint8_t* d_chroma{nullptr};
    check_cuda(cudaMalloc(&d_luma, luma_bytes), "cudaMalloc luma");
    check_cuda(cudaMalloc(&d_chroma, chroma_bytes), "cudaMalloc chroma");
    check_cuda(cudaMemcpy(d_luma, host.data(), luma_bytes, cudaMemcpyHostToDevice), "H2D luma");
    check_cuda(cudaMemcpy(d_chroma, host.data() + luma_bytes, chroma_bytes, cudaMemcpyHostToDevice),
               "H2D chroma");

    std::vector<std::uint8_t> out_file(cujpegxl::max_encoded_size(width, height));
    cujpegxl::FrameEncoder encoder{device, cujpegxl::default_worker_threads(), 1};
    if (!encoder.submit(d_luma, width, d_chroma, width, width, height, distance, out_file.data(),
                        out_file.size())) {
        die("submit failed");
    }
    const cujpegxl::EncodeResult result{encoder.receive()};
    if (result.status != cujpegxl::EncodeStatus::OK) {
        die("encode failed");
    }

    std::ofstream out{output, std::ios::binary | std::ios::trunc};
    out.write(reinterpret_cast<const char*>(out_file.data()),
              static_cast<std::streamsize>(result.file_size));
    if (!out) {
        die("failed writing " + output);
    }

    cudaFree(d_luma);
    cudaFree(d_chroma);
    return 0;
}
