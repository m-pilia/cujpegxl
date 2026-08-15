// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cuda_runtime.h>

#include "cujpegxl/cujpegxl.h"
#include "nv12_host_encode.h"
#include "src/frame_encoder.h"

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

class PyFuture {
public:
    PyFuture(cujpegxl::EncodedFrameFuture future, std::uint8_t* luma, std::uint8_t* chroma,
             std::int32_t device)
        : future_{std::move(future)}, luma_{luma}, chroma_{chroma}, device_{device} {}

    ~PyFuture() {
        while (future_.valid() && !future_.ready()) {
            future_.wait_for(std::chrono::seconds{1});
        }
        release_input();
    }

    bool ready() const { return future_.ready(); }

    std::uint64_t sequence() const { return loaded_ ? output_.sequence : sequence_; }

    py::bytes result(double timeout_seconds) {
        if (!loaded_) {
            bool completed{false};
            {
                py::gil_scoped_release release{};
                if (timeout_seconds < 0.0) {
                    while (!(completed = future_.ready())) {
                        future_.wait_for(std::chrono::seconds{1});
                    }
                } else {
                    completed =
                        future_.wait_for(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::duration<double>{timeout_seconds}));
                }
                if (completed) {
                    loaded_ = future_.get(output_);
                }
            }
            if (!completed) {
                PyErr_SetString(PyExc_TimeoutError, "encode future timed out");
                throw py::error_already_set();
            }
            if (!loaded_) {
                throw std::runtime_error("asynchronous encode failed");
            }
            release_input();
        }
        return py::bytes(reinterpret_cast<const char*>(output_.bytes.data()), output_.bytes.size());
    }

    void set_sequence(std::uint64_t sequence) { sequence_ = sequence; }

private:
    void release_input() {
        if (luma_ == nullptr && chroma_ == nullptr) {
            return;
        }
        cudaSetDevice(device_);
        cudaFree(luma_);
        cudaFree(chroma_);
        luma_ = nullptr;
        chroma_ = nullptr;
    }

    cujpegxl::EncodedFrameFuture future_{};
    cujpegxl::EncodedFrame output_{};
    std::uint8_t* luma_{nullptr};
    std::uint8_t* chroma_{nullptr};
    std::int32_t device_{0};
    std::uint64_t sequence_{0};
    bool loaded_{false};
};

class PyEncoder {
public:
    PyEncoder(std::uint32_t width, std::uint32_t height, float distance, std::int32_t device,
              std::size_t pipeline_depth)
        : width_{width}, height_{height}, distance_{distance}, device_{device} {
        if (pipeline_depth == 0 || width == 0 || height == 0) {
            throw std::invalid_argument("invalid encoder configuration");
        }
        session_ = cujpegxl::EncoderSession::create(
            cujpegxl::EncoderConfig{.device_ordinal = device,
                                    .max_width = width,
                                    .max_height = height,
                                    .pipeline_depth = pipeline_depth,
                                    .pipeline = cujpegxl::EncoderPipeline::MIXED});
        if (session_ == nullptr) {
            throw std::runtime_error("failed to create encoder session");
        }
    }

    std::shared_ptr<PyFuture> submit(const py::array_t<std::uint8_t>& nv12, std::uint64_t sequence,
                                     bool block) {
        const std::uint8_t* host_luma{nv12_planes(nv12, width_, height_)};
        const std::size_t luma_bytes{static_cast<std::size_t>(width_) * height_};
        const std::size_t chroma_bytes{luma_bytes / 2};
        std::uint8_t* luma{nullptr};
        std::uint8_t* chroma{nullptr};
        cudaSetDevice(device_);
        if (cudaMalloc(&luma, luma_bytes) != cudaSuccess ||
            cudaMalloc(&chroma, chroma_bytes) != cudaSuccess ||
            cudaMemcpy(luma, host_luma, luma_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(chroma, host_luma + luma_bytes, chroma_bytes, cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            cudaFree(luma);
            cudaFree(chroma);
            throw std::runtime_error("failed to upload NV12 input");
        }
        return enqueue(luma, width_, chroma, width_, sequence, block, true);
    }

    std::shared_ptr<PyFuture> submit_device(std::uintptr_t luma, std::size_t luma_pitch,
                                            std::uintptr_t chroma, std::size_t chroma_pitch,
                                            std::uint64_t sequence, bool block) {
        if (luma == 0 || chroma == 0) {
            throw std::invalid_argument("device pointers must be nonzero");
        }
        return enqueue(reinterpret_cast<std::uint8_t*>(luma), luma_pitch,
                       reinterpret_cast<std::uint8_t*>(chroma), chroma_pitch, sequence, block,
                       false);
    }

    std::shared_ptr<PyFuture> try_submit(const py::array_t<std::uint8_t>& nv12,
                                         std::uint64_t sequence) {
        return submit(nv12, sequence, false);
    }

    std::shared_ptr<PyFuture> try_submit_device(std::uintptr_t luma, std::size_t luma_pitch,
                                                std::uintptr_t chroma, std::size_t chroma_pitch,
                                                std::uint64_t sequence) {
        return submit_device(luma, luma_pitch, chroma, chroma_pitch, sequence, false);
    }

    void flush() {
        py::gil_scoped_release release{};
        session_->flush();
    }

private:
    std::shared_ptr<PyFuture> enqueue(std::uint8_t* luma, std::size_t luma_pitch,
                                      std::uint8_t* chroma, std::size_t chroma_pitch,
                                      std::uint64_t sequence, bool block, bool owns_input) {
        const cujpegxl::EncoderInput input{
            .luma = luma,
            .luma_pitch = luma_pitch,
            .chroma = chroma,
            .chroma_pitch = chroma_pitch,
            .width = width_,
            .height = height_,
            .distance = distance_,
            .quant_params = cujpegxl::quant_params_for_distance(distance_),
            .sequence = sequence};
        cujpegxl::EncodedFrameFuture future{};
        bool accepted{false};
        {
            py::gil_scoped_release release{};
            accepted =
                block ? session_->encode(input, future) : session_->try_encode(input, future);
        }
        if (!accepted) {
            if (owns_input) {
                cudaFree(luma);
                cudaFree(chroma);
            }
            return nullptr;
        }
        std::shared_ptr<PyFuture> result{
            std::make_shared<PyFuture>(std::move(future), owns_input ? luma : nullptr,
                                       owns_input ? chroma : nullptr, device_)};
        result->set_sequence(sequence);
        return result;
    }
    std::uint32_t width_{0};
    std::uint32_t height_{0};
    float distance_{1.0f};
    std::int32_t device_{0};
    std::unique_ptr<cujpegxl::EncoderSession> session_{};
};

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
    py::class_<PyFuture, std::shared_ptr<PyFuture>>(m, "EncodeFuture")
        .def_property_readonly("ready", &PyFuture::ready)
        .def_property_readonly("sequence", &PyFuture::sequence)
        .def("result", &PyFuture::result, py::arg("timeout") = -1.0);
    py::class_<PyEncoder>(m, "Encoder")
        .def(py::init<std::uint32_t, std::uint32_t, float, std::int32_t, std::size_t>(),
             py::arg("width"), py::arg("height"), py::arg("distance") = 1.0f,
             py::arg("device_ordinal") = 0, py::arg("pipeline_depth") = 2)
        .def("submit", &PyEncoder::submit, py::arg("nv12"), py::arg("sequence") = 0,
             py::arg("block") = true)
        .def("try_submit", &PyEncoder::try_submit, py::arg("nv12"), py::arg("sequence") = 0)
        .def("submit_device", &PyEncoder::submit_device, py::arg("luma"), py::arg("luma_pitch"),
             py::arg("chroma"), py::arg("chroma_pitch"), py::arg("sequence") = 0,
             py::arg("block") = true,
             "Submit NV12 device pointers, which must remain valid until the "
             "future is ready.")
        .def("try_submit_device", &PyEncoder::try_submit_device, py::arg("luma"),
             py::arg("luma_pitch"), py::arg("chroma"), py::arg("chroma_pitch"),
             py::arg("sequence") = 0)
        .def("flush", &PyEncoder::flush);
}
