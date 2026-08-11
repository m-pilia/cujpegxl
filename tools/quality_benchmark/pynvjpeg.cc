// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Testonly Python binding over the nvJPEG encoder for the quality benchmark.
// Mirrors the NV12 -> JPEG path proven in tools/benchmark/encode_bench.cc:
// deinterleaves the NV12 chroma plane into planar Cb/Cr, uploads to the device,
// and drives nvjpegEncodeYUV. Unlike the speed benchmark this runs a single
// encode per call (quality, not throughput, is what is measured downstream).

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cuda ") + what + " failed: " +
                                 cudaGetErrorString(err));
    }
}

void check_nvjpeg(nvjpegStatus_t s, const char* what) {
    if (s != NVJPEG_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("nvjpeg ") + what + " failed (status " +
                                 std::to_string(static_cast<int>(s)) + ")");
    }
}

py::bytes encode_jpeg(const py::array_t<std::uint8_t>& nv12, std::uint32_t width,
                      std::uint32_t height, int quality, int device) {
    const py::buffer_info info{nv12.request()};
    const std::size_t expected{static_cast<std::size_t>(width) * height * 3 / 2};
    if (static_cast<std::size_t>(info.size) != expected) {
        throw std::invalid_argument("nv12 array must hold width*height*3/2 bytes");
    }
    if ((nv12.flags() & py::array::c_style) == 0) {
        throw std::invalid_argument("nv12 array must be C-contiguous");
    }
    if (quality < 1 || quality > 100) {
        throw std::invalid_argument("quality must be in [1, 100]");
    }
    const auto* host{static_cast<const std::uint8_t*>(info.ptr)};

    check_cuda(cudaSetDevice(device), "cudaSetDevice");

    const std::size_t luma_bytes{static_cast<std::size_t>(width) * height};
    const std::size_t chroma_w{width / 2};
    const std::size_t chroma_h{height / 2};
    const std::size_t chroma_bytes{chroma_w * chroma_h};

    std::vector<std::uint8_t> cb(chroma_bytes);
    std::vector<std::uint8_t> cr(chroma_bytes);
    const std::uint8_t* uv{host + luma_bytes};
    for (std::size_t i{0}; i < chroma_bytes; ++i) {
        cb[i] = uv[2 * i];
        cr[i] = uv[2 * i + 1];
    }

    std::uint8_t* d_luma{nullptr};
    std::uint8_t* d_cb{nullptr};
    std::uint8_t* d_cr{nullptr};
    check_cuda(cudaMalloc(&d_luma, luma_bytes), "cudaMalloc luma");
    struct Guard {
        std::uint8_t* l;
        std::uint8_t* cb;
        std::uint8_t* cr;
        ~Guard() {
            if (l != nullptr) cudaFree(l);
            if (cb != nullptr) cudaFree(cb);
            if (cr != nullptr) cudaFree(cr);
        }
    } guard{d_luma, d_cb, d_cr};
    check_cuda(cudaMalloc(&d_cb, chroma_bytes), "cudaMalloc cb");
    guard.cb = d_cb;
    check_cuda(cudaMalloc(&d_cr, chroma_bytes), "cudaMalloc cr");
    guard.cr = d_cr;
    check_cuda(cudaMemcpy(d_luma, host, luma_bytes, cudaMemcpyHostToDevice), "memcpy luma");
    check_cuda(cudaMemcpy(d_cb, cb.data(), chroma_bytes, cudaMemcpyHostToDevice), "memcpy cb");
    check_cuda(cudaMemcpy(d_cr, cr.data(), chroma_bytes, cudaMemcpyHostToDevice), "memcpy cr");

    nvjpegHandle_t handle{nullptr};
    nvjpegEncoderState_t state{nullptr};
    nvjpegEncoderParams_t params{nullptr};
    check_nvjpeg(nvjpegCreateSimple(&handle), "CreateSimple");
    try {
        check_nvjpeg(nvjpegEncoderStateCreate(handle, &state, nullptr), "EncoderStateCreate");
        try {
            check_nvjpeg(nvjpegEncoderParamsCreate(handle, &params, nullptr), "EncoderParamsCreate");
            try {
                check_nvjpeg(nvjpegEncoderParamsSetQuality(params, quality, nullptr), "SetQuality");
                check_nvjpeg(nvjpegEncoderParamsSetSamplingFactors(params, NVJPEG_CSS_420, nullptr),
                             "SetSamplingFactors");

                nvjpegImage_t image{};
                image.channel[0] = d_luma;
                image.pitch[0] = width;
                image.channel[1] = d_cb;
                image.pitch[1] = chroma_w;
                image.channel[2] = d_cr;
                image.pitch[2] = chroma_w;

                check_nvjpeg(nvjpegEncodeYUV(handle, state, params, &image, NVJPEG_CSS_420,
                                             static_cast<int>(width), static_cast<int>(height),
                                             nullptr),
                             "EncodeYUV");
                std::size_t max_size{0};
                check_nvjpeg(nvjpegEncodeGetBufferSize(handle, params, static_cast<int>(width),
                                                       static_cast<int>(height), &max_size),
                             "EncodeGetBufferSize");
                std::vector<std::uint8_t> out(max_size);
                std::size_t length{out.size()};
                check_nvjpeg(nvjpegEncodeRetrieveBitstream(handle, state, out.data(), &length,
                                                           nullptr),
                             "RetrieveBitstream");
                check_cuda(cudaDeviceSynchronize(), "sync");
                out.resize(length);
                nvjpegEncoderParamsDestroy(params);
                params = nullptr;
                return py::bytes(reinterpret_cast<const char*>(out.data()), out.size());
            } catch (...) {
                if (params != nullptr) {
                    nvjpegEncoderParamsDestroy(params);
                }
                throw;
            }
        } catch (...) {
            if (state != nullptr) {
                nvjpegEncoderStateDestroy(state);
            }
            throw;
        }
    } catch (...) {
        nvjpegDestroy(handle);
        throw;
    }
}

}  // namespace

PYBIND11_MODULE(pynvjpeg, m) {
    m.doc() = "nvJPEG encoder binding for the quality benchmark";
    m.def("encode", &encode_jpeg, py::arg("nv12"), py::arg("width"), py::arg("height"),
          py::arg("quality"), py::arg("device") = 0);
}
