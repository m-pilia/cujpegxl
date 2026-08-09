// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdint>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>

#include "cujpegxl/cujpegxl.h"

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

}  // namespace

PYBIND11_MODULE(pycujpegxl, m) {
    m.doc() = "Thin Python bindings over the cujpegxl C ABI.";
    m.attr("API_VERSION") = py::int_(cujpegxl_api_version());
    m.def("api_version", &cujpegxl_api_version);
    m.def("query_backend", &query_backend, py::arg("device_ordinal") = 0);
}
