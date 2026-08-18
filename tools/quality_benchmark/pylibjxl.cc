// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Testonly Python bindings over libjxl for the quality benchmark: JXL
// encode/decode through the public C API, and the three image-quality metrics
// (butteraugli, ssimulacra2, PSNR) computed against the reference libjxl
// implementations so the numbers match cjxl's distance calibration.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <jxl/decode.h>
#include <jxl/encode.h>
#include <jxl/memory_manager.h>
#include <jxl/thread_parallel_runner.h>
#include <jxl/types.h>

#include "lib/jxl/butteraugli/butteraugli.h"
#include "lib/jxl/color_encoding_internal.h"
#include "lib/jxl/image.h"
#include "lib/jxl/image_bundle.h"
#include "lib/jxl/image_metadata.h"
#include "tools/ssimulacra2.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

// libjxl's SDR intensity target for 8-bit sRGB. Matches the value the encoder
// uses when computing butteraugli distance, so the scores here are on the same
// scale as `cjxl -d` and cujpegxl's `--distance`.
constexpr float kSdrIntensityTarget{255.0f};

void* mm_alloc(void*, size_t size) {
    return std::malloc(size);
}
void mm_free(void*, void* address) {
    std::free(address);
}

JxlMemoryManager make_memory_manager() {
    return JxlMemoryManager{nullptr, &mm_alloc, &mm_free};
}

struct Planes {
    std::uint32_t width{0};
    std::uint32_t height{0};
    const std::uint8_t* rgb{nullptr};
};

Planes take_rgb(const py::array_t<std::uint8_t>& image) {
    const py::buffer_info info{image.request()};
    if (info.ndim != 3 || info.shape[2] != 3) {
        throw std::invalid_argument("expected an HxWx3 uint8 image");
    }
    if ((image.flags() & py::array::c_style) == 0) {
        throw std::invalid_argument("image must be C-contiguous");
    }
    return Planes{static_cast<std::uint32_t>(info.shape[1]),
                  static_cast<std::uint32_t>(info.shape[0]),
                  static_cast<const std::uint8_t*>(info.ptr)};
}

void check_same_size(const Planes& a, const Planes& b) {
    if (a.width != b.width || a.height != b.height) {
        throw std::invalid_argument("images must have the same dimensions");
    }
}

jxl::Image3F image3f_from_rgb(JxlMemoryManager* mm, const Planes& p, bool linearize) {
    auto img_or = jxl::Image3F::Create(mm, p.width, p.height);
    if (!img_or.ok()) {
        throw std::runtime_error("Image3F::Create failed");
    }
    jxl::Image3F img = std::move(img_or).value_();
    const auto srgb_eotf = [](float v) {
        v /= 255.0f;
        return v <= 0.04045f ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
    };
    for (std::size_t c{0}; c < 3; ++c) {
        for (std::uint32_t y{0}; y < p.height; ++y) {
            float* row{img.PlaneRow(c, y)};
            const std::uint8_t* src{p.rgb + (static_cast<std::size_t>(y) * p.width) * 3};
            for (std::uint32_t x{0}; x < p.width; ++x) {
                row[x] = linearize ? srgb_eotf(src[x * 3 + c]) : src[x * 3 + c] / 255.0f;
            }
        }
    }
    return img;
}

// Builds an ImageBundle in sRGB [0,1] with SDR intensity target, the form both
// JxlButteraugliComparator and ComputeSSIMULACRA2 expect. `metadata` must
// outlive the returned bundle (both live in the caller's scope).
jxl::ImageBundle imagebundle_from_rgb(JxlMemoryManager* mm, jxl::ImageMetadata* metadata,
                                      const Planes& p) {
    metadata->SetUintSamples(8);
    metadata->color_encoding = jxl::ColorEncoding::SRGB(false);
    jxl::Image3F color{image3f_from_rgb(mm, p, /*linearize=*/false)};
    jxl::ImageBundle bundle(mm, metadata);
    if (!bundle.SetFromImage(std::move(color), jxl::ColorEncoding::SRGB(false))) {
        throw std::runtime_error("SetFromImage failed");
    }
    return bundle;
}

py::bytes encode_jxl(const py::array_t<std::uint8_t>& rgb, float distance) {
    const Planes p{take_rgb(rgb)};
    JxlEncoder* enc{JxlEncoderCreate(nullptr)};
    if (enc == nullptr) {
        throw std::runtime_error("JxlEncoderCreate failed");
    }
    struct EncoderGuard {
        JxlEncoder* e;
        void* runner;
        ~EncoderGuard() {
            JxlEncoderDestroy(e);
            if (runner != nullptr) {
                JxlThreadParallelRunnerDestroy(runner);
            }
        }
    } guard{enc, nullptr};

    guard.runner =
        JxlThreadParallelRunnerCreate(nullptr, JxlThreadParallelRunnerDefaultNumWorkerThreads());
    if (guard.runner == nullptr) {
        throw std::runtime_error("JxlThreadParallelRunnerCreate failed");
    }
    if (JxlEncoderSetParallelRunner(enc, JxlThreadParallelRunner, guard.runner) !=
        JXL_ENC_SUCCESS) {
        throw std::runtime_error("JxlEncoderSetParallelRunner failed");
    }

    JxlBasicInfo info;
    JxlEncoderInitBasicInfo(&info);
    info.xsize = p.width;
    info.ysize = p.height;
    info.bits_per_sample = 8;
    info.exponent_bits_per_sample = 0;
    info.num_color_channels = 3;
    info.uses_original_profile = JXL_FALSE;
    if (JxlEncoderSetBasicInfo(enc, &info) != JXL_ENC_SUCCESS) {
        throw std::runtime_error("JxlEncoderSetBasicInfo failed");
    }
    JxlColorEncoding color;
    JxlColorEncodingSetToSRGB(&color, JXL_FALSE);
    if (JxlEncoderSetColorEncoding(enc, &color) != JXL_ENC_SUCCESS) {
        throw std::runtime_error("JxlEncoderSetColorEncoding failed");
    }
    JxlEncoderFrameSettings* fs{JxlEncoderFrameSettingsCreate(enc, nullptr)};
    JxlEncoderFrameSettingsSetOption(fs, JXL_ENC_FRAME_SETTING_EFFORT, 7);
    if (JxlEncoderSetFrameDistance(fs, distance) != JXL_ENC_SUCCESS) {
        throw std::runtime_error("JxlEncoderSetFrameDistance failed");
    }
    JxlPixelFormat pixfmt{3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    if (JxlEncoderAddImageFrame(fs, &pixfmt, p.rgb,
                                static_cast<std::size_t>(p.width) * p.height * 3) !=
        JXL_ENC_SUCCESS) {
        throw std::runtime_error("JxlEncoderAddImageFrame failed");
    }
    JxlEncoderCloseInput(enc);

    std::vector<std::uint8_t> out{};
    std::array<std::uint8_t, 1 << 16> buf{};
    for (;;) {
        std::uint8_t* next{buf.data()};
        std::size_t avail{buf.size()};
        const JxlEncoderStatus status{JxlEncoderProcessOutput(enc, &next, &avail)};
        const std::size_t written{buf.size() - avail};
        out.insert(out.end(), buf.data(), buf.data() + written);
        if (status == JXL_ENC_SUCCESS) {
            break;
        }
        if (status != JXL_ENC_NEED_MORE_OUTPUT) {
            throw std::runtime_error("JxlEncoderProcessOutput failed");
        }
    }
    return py::bytes(reinterpret_cast<const char*>(out.data()), out.size());
}

py::object decode_jxl(const py::bytes& data) {
    std::string bytes{data};
    JxlDecoder* dec{JxlDecoderCreate(nullptr)};
    if (dec == nullptr) {
        throw std::runtime_error("JxlDecoderCreate failed");
    }
    struct DecoderGuard {
        JxlDecoder* d;
        ~DecoderGuard() { JxlDecoderDestroy(d); }
    } guard{dec};

    if (JxlDecoderSubscribeEvents(dec, JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) !=
        JXL_DEC_SUCCESS) {
        throw std::runtime_error("JxlDecoderSubscribeEvents failed");
    }
    JxlDecoderSetInput(dec, reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
    JxlDecoderCloseInput(dec);

    JxlPixelFormat pixfmt{3, JXL_TYPE_UINT8, JXL_NATIVE_ENDIAN, 0};
    std::uint32_t width{0};
    std::uint32_t height{0};
    std::vector<std::uint8_t> pixels{};
    for (;;) {
        const JxlDecoderStatus status{JxlDecoderProcessInput(dec)};
        if (status == JXL_DEC_SUCCESS) {
            break;
        }
        if (status == JXL_DEC_ERROR) {
            throw std::runtime_error("jxl decode error");
        }
        if (status == JXL_DEC_BASIC_INFO) {
            JxlBasicInfo info;
            JxlDecoderGetBasicInfo(dec, &info);
            width = info.xsize;
            height = info.ysize;
            std::size_t size{0};
            if (JxlDecoderImageOutBufferSize(dec, &pixfmt, &size) != JXL_DEC_SUCCESS) {
                throw std::runtime_error("JxlDecoderImageOutBufferSize failed");
            }
            pixels.resize(size);
            if (JxlDecoderSetImageOutBuffer(dec, &pixfmt, pixels.data(), size) != JXL_DEC_SUCCESS) {
                throw std::runtime_error("JxlDecoderSetImageOutBuffer failed");
            }
        } else if (status == JXL_DEC_FULL_IMAGE) {
            // pixels now populated; keep going until SUCCESS
        } else {
            throw std::runtime_error("unexpected jxl decoder status");
        }
    }
    if (width == 0 || height == 0) {
        throw std::runtime_error("decoded image has zero dimensions");
    }

    std::vector<py::ssize_t> shape{static_cast<py::ssize_t>(height),
                                   static_cast<py::ssize_t>(width), 3};
    py::array_t<std::uint8_t> out(shape);
    std::memcpy(out.mutable_data(), pixels.data(), pixels.size());
    return out;
}

double butteraugli_score(const py::array_t<std::uint8_t>& a, const py::array_t<std::uint8_t>& b) {
    const Planes pa{take_rgb(a)};
    const Planes pb{take_rgb(b)};
    check_same_size(pa, pb);
    JxlMemoryManager mm{make_memory_manager()};
    jxl::Image3F lin0{image3f_from_rgb(&mm, pa, /*linearize=*/true)};
    jxl::Image3F lin1{image3f_from_rgb(&mm, pb, /*linearize=*/true)};
    jxl::ButteraugliParams params;
    params.intensity_target = kSdrIntensityTarget;
    auto comp_or{jxl::ButteraugliComparator::Make(lin0, params)};
    if (!comp_or.ok()) {
        throw std::runtime_error("ButteraugliComparator::Make failed");
    }
    auto comp{std::move(comp_or).value_()};
    auto diffmap_or{jxl::ImageF::Create(&mm, pa.width, pa.height)};
    if (!diffmap_or.ok()) {
        throw std::runtime_error("ImageF::Create failed");
    }
    jxl::ImageF diffmap{std::move(diffmap_or).value_()};
    if (!comp->Diffmap(lin1, diffmap)) {
        throw std::runtime_error("butteraugli Diffmap failed");
    }
    return jxl::ButteraugliScoreFromDiffmap(diffmap, &params);
}

double ssimulacra2_score(const py::array_t<std::uint8_t>& a, const py::array_t<std::uint8_t>& b) {
    const Planes pa{take_rgb(a)};
    const Planes pb{take_rgb(b)};
    check_same_size(pa, pb);
    JxlMemoryManager mm{make_memory_manager()};
    jxl::ImageMetadata md0;
    jxl::ImageMetadata md1;
    jxl::ImageBundle b0{imagebundle_from_rgb(&mm, &md0, pa)};
    jxl::ImageBundle b1{imagebundle_from_rgb(&mm, &md1, pb)};
    auto result{ComputeSSIMULACRA2(b0, b1)};
    if (!result.ok()) {
        throw std::runtime_error("ComputeSSIMULACRA2 failed");
    }
    return std::move(result).value_().Score();
}

double psnr_score(const py::array_t<std::uint8_t>& a, const py::array_t<std::uint8_t>& b) {
    const Planes pa{take_rgb(a)};
    const Planes pb{take_rgb(b)};
    check_same_size(pa, pb);
    double sse{0.0};
    const std::size_t n{static_cast<std::size_t>(pa.width) * pa.height * 3};
    for (std::size_t i{0}; i < n; ++i) {
        const double d{static_cast<double>(pa.rgb[i]) - static_cast<double>(pb.rgb[i])};
        sse += d * d;
    }
    if (sse == 0.0) {
        return 99.0;
    }
    const double mse{sse / static_cast<double>(n)};
    return 10.0 * std::log10(255.0 * 255.0 / mse);
}

}  // namespace

PYBIND11_MODULE(pylibjxl, m) {
    m.doc() = "libjxl encode/decode + quality metrics for the quality benchmark";
    m.def("encode", &encode_jxl, py::arg("rgb"), py::arg("distance") = 1.0f);
    m.def("decode", &decode_jxl, py::arg("data"));
    m.def("butteraugli", &butteraugli_score, py::arg("reference"), py::arg("distorted"));
    m.def("ssimulacra2", &ssimulacra2_score, py::arg("reference"), py::arg("distorted"));
    m.def("psnr", &psnr_score, py::arg("reference"), py::arg("distorted"));
}
