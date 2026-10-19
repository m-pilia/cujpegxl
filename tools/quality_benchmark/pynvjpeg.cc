// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <cuda_runtime.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "tools/benchmark/nvjpeg_encoder.h"

namespace py = pybind11;

namespace {

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
    if (cudaSetDevice(device) != cudaSuccess) {
        throw std::runtime_error("cudaSetDevice failed");
    }
    const cujpegxl::tools::DevicePlanarYuv yuv{static_cast<const std::uint8_t*>(info.ptr), width,
                                               height};
    cujpegxl::tools::NvjpegEncoder encoder{quality, width, height};
    const cujpegxl::tools::JpegView jpeg{encoder.encode(yuv)};
    return py::bytes(reinterpret_cast<const char*>(jpeg.data), jpeg.size);
}

}  // namespace

PYBIND11_MODULE(pynvjpeg, m) {
    m.doc() = "nvJPEG encoder binding for the quality benchmark";
    m.def("encode", &encode_jpeg, py::arg("nv12"), py::arg("width"), py::arg("height"),
          py::arg("quality"), py::arg("device") = 0);
}
