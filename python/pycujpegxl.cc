// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cujpegxl/cujpegxl.h"
#include "nv12_host_encode.h"

namespace py = pybind11;

namespace {

std::string query_backend(std::int32_t device_ordinal) {
    cujpegxl_backend backend{CUJPEGXL_BACKEND_UNKNOWN};
    const cujpegxl_status status{cujpegxl_query_backend(device_ordinal, &backend)};
    if (status != CUJPEGXL_OK) {
        throw std::runtime_error(std::string("query_backend failed: ") +
                                 cujpegxl_status_string(status));
    }
    return cujpegxl_backend_string(backend);
}

const std::uint8_t* nv12_planes(const py::array_t<std::uint8_t>& nv12, std::uint32_t width,
                                std::uint32_t height) {
    const py::buffer_info info{nv12.request()};
    const std::size_t expected{static_cast<std::size_t>(width) * height * 3 / 2};
    if (static_cast<std::size_t>(info.size) != expected) {
        throw std::invalid_argument("nv12 array must hold width*height*3/2 bytes");
    }
    if ((nv12.flags() & py::array::c_style) == 0) {
        throw std::invalid_argument("nv12 array must be C-contiguous");
    }
    return static_cast<const std::uint8_t*>(info.ptr);
}

py::bytes encode(const py::array_t<std::uint8_t>& nv12, std::uint32_t width, std::uint32_t height,
                 float distance, std::int32_t device_ordinal) {
    const std::uint8_t* luma{nv12_planes(nv12, width, height)};
    const std::uint8_t* chroma{luma + static_cast<std::size_t>(width) * height};
    std::vector<std::uint8_t> out{};
    if (!cujpegxl::pybind_support::encode_nv12_host(luma, chroma, width, height, distance,
                                                    device_ordinal, out, nullptr)) {
        throw std::runtime_error("encode failed");
    }
    return py::bytes(reinterpret_cast<const char*>(out.data()), out.size());
}

py::tuple encode_with_stats(const py::array_t<std::uint8_t>& nv12, std::uint32_t width,
                            std::uint32_t height, float distance, std::int32_t device_ordinal) {
    const std::uint8_t* luma{nv12_planes(nv12, width, height)};
    const std::uint8_t* chroma{luma + static_cast<std::size_t>(width) * height};
    std::vector<std::uint8_t> out{};
    std::vector<cujpegxl::StageTiming> stats{};
    if (!cujpegxl::pybind_support::encode_nv12_host(luma, chroma, width, height, distance,
                                                    device_ordinal, out, &stats)) {
        throw std::runtime_error("encode failed");
    }
    py::list records{};
    for (const cujpegxl::StageTiming& s : stats) {
        py::dict record{};
        record["name"] = std::string(s.name);
        record["bytes_moved"] = s.bytes_moved;
        record["gpu_us"] = s.gpu_us;
        record["cpu_us"] = s.cpu_us;
        records.append(record);
    }
    return py::make_tuple(py::bytes(reinterpret_cast<const char*>(out.data()), out.size()),
                          records);
}

}  // namespace

PYBIND11_MODULE(pycujpegxl, m) {
    m.doc() = "Thin Python bindings over the cujpegxl C ABI.";
    m.attr("API_VERSION") = py::int_(cujpegxl_api_version());
    m.def("api_version", &cujpegxl_api_version);
    m.def("query_backend", &query_backend, py::arg("device_ordinal") = 0);
    m.def("encode", &encode, py::arg("nv12"), py::arg("width"), py::arg("height"),
          py::arg("distance") = 1.0f, py::arg("device_ordinal") = 0);
    m.def("encode_with_stats", &encode_with_stats, py::arg("nv12"), py::arg("width"),
          py::arg("height"), py::arg("distance") = 1.0f, py::arg("device_ordinal") = 0);
}
