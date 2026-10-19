// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "nvjpeg_encoder.h"

#include <cstddef>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace cujpegxl::tools {
namespace {

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cuda ") + what +
                                 " failed: " + cudaGetErrorString(err));
    }
}

void check_nvjpeg(nvjpegStatus_t s, const char* what) {
    if (s != NVJPEG_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("nvjpeg ") + what + " failed (status " +
                                 std::to_string(static_cast<int>(s)) + ")");
    }
}

std::uint8_t* upload(const std::uint8_t* host, std::size_t bytes) {
    void* device{nullptr};
    check_cuda(cudaMalloc(&device, bytes), "cudaMalloc");
    if (cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(device);
        throw std::runtime_error("cuda cudaMemcpy failed");
    }
    return static_cast<std::uint8_t*>(device);
}

}  // namespace

DevicePlanarYuv::DevicePlanarYuv(const std::uint8_t* nv12, std::uint32_t width,
                                 std::uint32_t height)
    : width_{width}, height_{height} {
    const std::size_t luma_bytes{static_cast<std::size_t>(width) * height};
    const std::size_t chroma_w{width / 2};
    const std::size_t chroma_bytes{chroma_w * (height / 2)};
    std::vector<std::uint8_t> cb(chroma_bytes);
    std::vector<std::uint8_t> cr(chroma_bytes);
    const std::uint8_t* uv{nv12 + luma_bytes};
    for (std::size_t i{0}; i < chroma_bytes; ++i) {
        cb[i] = uv[2 * i];
        cr[i] = uv[2 * i + 1];
    }
    try {
        image_.channel[0] = upload(nv12, luma_bytes);
        image_.channel[1] = upload(cb.data(), chroma_bytes);
        image_.channel[2] = upload(cr.data(), chroma_bytes);
    } catch (...) {
        for (unsigned char* plane : image_.channel) {
            cudaFree(plane);
        }
        throw;
    }
    image_.pitch[0] = width;
    image_.pitch[1] = chroma_w;
    image_.pitch[2] = chroma_w;
}

DevicePlanarYuv::~DevicePlanarYuv() {
    for (unsigned char* plane : image_.channel) {
        cudaFree(plane);
    }
}

NvjpegEncoder::NvjpegEncoder(int quality, std::uint32_t width, std::uint32_t height) {
    if (quality < 1 || quality > 100) {
        throw std::invalid_argument("quality must be in [1, 100]");
    }
    try {
        check_nvjpeg(nvjpegCreateSimple(&handle_), "CreateSimple");
        check_nvjpeg(nvjpegEncoderStateCreate(handle_, &state_, nullptr), "EncoderStateCreate");
        check_nvjpeg(nvjpegEncoderParamsCreate(handle_, &params_, nullptr), "EncoderParamsCreate");
        check_nvjpeg(nvjpegEncoderParamsSetQuality(params_, quality, nullptr), "SetQuality");
        check_nvjpeg(nvjpegEncoderParamsSetSamplingFactors(params_, NVJPEG_CSS_420, nullptr),
                     "SetSamplingFactors");
        std::size_t max_size{0};
        check_nvjpeg(nvjpegEncodeGetBufferSize(handle_, params_, static_cast<int>(width),
                                               static_cast<int>(height), &max_size),
                     "EncodeGetBufferSize");
        buffer_.resize(max_size);
    } catch (...) {
        release();
        throw;
    }
}

NvjpegEncoder::~NvjpegEncoder() {
    release();
}

void NvjpegEncoder::release() {
    if (params_ != nullptr) {
        nvjpegEncoderParamsDestroy(params_);
        params_ = nullptr;
    }
    if (state_ != nullptr) {
        nvjpegEncoderStateDestroy(state_);
        state_ = nullptr;
    }
    if (handle_ != nullptr) {
        nvjpegDestroy(handle_);
        handle_ = nullptr;
    }
}

JpegView NvjpegEncoder::encode(const DevicePlanarYuv& yuv) {
    check_nvjpeg(
        nvjpegEncodeYUV(handle_, state_, params_, &yuv.image(), NVJPEG_CSS_420,
                        static_cast<int>(yuv.width()), static_cast<int>(yuv.height()), nullptr),
        "EncodeYUV");
    std::size_t length{buffer_.size()};
    check_nvjpeg(nvjpegEncodeRetrieveBitstream(handle_, state_, buffer_.data(), &length, nullptr),
                 "RetrieveBitstream");
    check_cuda(cudaStreamSynchronize(nullptr), "stream sync");
    return JpegView{buffer_.data(), length};
}

}  // namespace cujpegxl::tools
