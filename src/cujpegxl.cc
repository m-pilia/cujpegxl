// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "cujpegxl/cujpegxl.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <vector>

#include "cujpegxl/capability.h"
#include "frame_encoder.h"
#include "src/bitstream/frame_assembly.h"

struct cujpegxl_encoder {
    cujpegxl_config config;
    std::size_t pipeline_depth;
    std::unique_ptr<cujpegxl::EncoderSession> session;
};

struct cujpegxl_future {
    cujpegxl::EncodedFrameFuture future;
    cujpegxl::EncodedFrame output;
    bool loaded{false};
};

namespace {

cujpegxl_backend to_c_backend(cujpegxl::Backend backend) {
    switch (backend) {
        case cujpegxl::Backend::FP32_SIMT:
            return CUJPEGXL_BACKEND_FP32_SIMT;
        case cujpegxl::Backend::FP16_TENSOR:
            return CUJPEGXL_BACKEND_FP16_TENSOR;
        case cujpegxl::Backend::UNKNOWN:
            return CUJPEGXL_BACKEND_UNKNOWN;
    }
    return CUJPEGXL_BACKEND_UNKNOWN;
}

// Rejects configs the device pipeline cannot encode. Block-size (multiple of 8)
// and multi-AC-group constraints are resolution errors; a non-positive distance
// is an argument error.
cujpegxl_status validate_config(const cujpegxl_config& config) {
    if (!(config.distance > 0.0f) || !std::isfinite(config.distance)) {
        return CUJPEGXL_INVALID_ARGUMENT;
    }
    if (config.width == 0 || config.height == 0 || config.width % 8 != 0 ||
        config.height % 8 != 0) {
        return CUJPEGXL_UNSUPPORTED_RESOLUTION;
    }
    if (cujpegxl::bitstream::ac_group_count(config.width, config.height) <= 1) {
        return CUJPEGXL_UNSUPPORTED_RESOLUTION;
    }
    return CUJPEGXL_OK;
}

std::size_t worst_case_output(std::uint32_t width, std::uint32_t height) {
    const std::size_t bw{width / 8u};
    const std::size_t bh{height / 8u};
    const std::size_t num_ac{cujpegxl::bitstream::ac_group_count(width, height)};
    const std::size_t num_dc{cujpegxl::bitstream::dc_group_count(width, height)};
    // AC bodies (<=64 tokens * 6 bytes per block-channel, 3 channels) + DC/
    // AcMetadata bodies + per-group TOC entries and DcGroup header blobs +
    // container/headers/histogram-description slack.
    return 3 * bw * bh * 384 + 48 * bw * bh + (num_ac + num_dc) * 2048 + 16384;
}

std::unique_ptr<cujpegxl::EncoderSession> make_session(const cujpegxl_config& config,
                                                       std::size_t pipeline_depth) {
    return cujpegxl::EncoderSession::create(
        cujpegxl::EncoderConfig{.device_ordinal = config.device_ordinal,
                                .max_width = config.width,
                                .max_height = config.height,
                                .pipeline_depth = pipeline_depth,
                                .pipeline = cujpegxl::EncoderPipeline::MIXED});
}

cujpegxl::EncoderInput make_input(const cujpegxl_encoder& encoder, const cujpegxl_nv12_input& input,
                                  std::uint64_t sequence) {
    const cujpegxl_config& config{encoder.config};
    return cujpegxl::EncoderInput{
        .luma = reinterpret_cast<const std::uint8_t*>(input.luma),
        .luma_pitch = input.luma_pitch,
        .chroma = reinterpret_cast<const std::uint8_t*>(input.chroma),
        .chroma_pitch = input.chroma_pitch,
        .width = config.width,
        .height = config.height,
        .distance = config.distance,
        .quant_params = cujpegxl::quant_params_for_distance(config.distance),
        .sequence = sequence};
}

cujpegxl_status submit(cujpegxl_encoder* encoder, const cujpegxl_nv12_input* input,
                       std::uint64_t sequence, cujpegxl_future** out_future, bool blocking) {
    if (encoder == nullptr || input == nullptr || out_future == nullptr || input->luma == 0 ||
        input->chroma == 0) {
        return CUJPEGXL_INVALID_ARGUMENT;
    }
    *out_future = nullptr;
    auto* result{new (std::nothrow) cujpegxl_future{}};
    if (result == nullptr) {
        return CUJPEGXL_INTERNAL_ERROR;
    }
    const cujpegxl::EncoderInput encoder_input{make_input(*encoder, *input, sequence)};
    const bool accepted{blocking ? encoder->session->encode(encoder_input, result->future)
                                 : encoder->session->try_encode(encoder_input, result->future)};
    if (!accepted) {
        delete result;
        return blocking ? CUJPEGXL_INTERNAL_ERROR : CUJPEGXL_WOULD_BLOCK;
    }
    *out_future = result;
    return CUJPEGXL_OK;
}

}  // namespace

extern "C" {

uint32_t cujpegxl_api_version(void) {
    return CUJPEGXL_API_VERSION;
}

const char* cujpegxl_status_string(cujpegxl_status status) {
    switch (status) {
        case CUJPEGXL_OK:
            return "ok";
        case CUJPEGXL_NOT_IMPLEMENTED:
            return "not-implemented";
        case CUJPEGXL_INVALID_ARGUMENT:
            return "invalid-argument";
        case CUJPEGXL_UNSUPPORTED_RESOLUTION:
            return "unsupported-resolution";
        case CUJPEGXL_NO_DEVICE:
            return "no-device";
        case CUJPEGXL_INTERNAL_ERROR:
            return "internal-error";
        case CUJPEGXL_BUFFER_TOO_SMALL:
            return "buffer-too-small";
        case CUJPEGXL_WOULD_BLOCK:
            return "would-block";
        case CUJPEGXL_NOT_READY:
            return "not-ready";
        case CUJPEGXL_TIMEOUT:
            return "timeout";
    }
    return "internal-error";
}

const char* cujpegxl_backend_string(cujpegxl_backend backend) {
    switch (backend) {
        case CUJPEGXL_BACKEND_FP32_SIMT:
            return "fp32-simt";
        case CUJPEGXL_BACKEND_FP16_TENSOR:
            return "fp16-tensor";
        case CUJPEGXL_BACKEND_UNKNOWN:
            return "unknown";
    }
    return "unknown";
}

cujpegxl_status cujpegxl_query_backend(int32_t device_ordinal, cujpegxl_backend* out_backend) {
    if (out_backend == nullptr) {
        return CUJPEGXL_INVALID_ARGUMENT;
    }
    cujpegxl::DeviceCapability capability{};
    if (!cujpegxl::query_device_capability(device_ordinal, &capability)) {
        return CUJPEGXL_NO_DEVICE;
    }
    *out_backend = to_c_backend(cujpegxl::select_backend(capability));
    return CUJPEGXL_OK;
}

cujpegxl_status cujpegxl_max_output_size(const cujpegxl_config* config, size_t* out_bytes) {
    if (config == nullptr || out_bytes == nullptr) {
        return CUJPEGXL_INVALID_ARGUMENT;
    }
    const cujpegxl_status status{validate_config(*config)};
    if (status != CUJPEGXL_OK) {
        return status;
    }
    *out_bytes = worst_case_output(config->width, config->height);
    return CUJPEGXL_OK;
}

cujpegxl_status cujpegxl_encoder_create(const cujpegxl_config* config,
                                        cujpegxl_encoder** out_encoder) {
    if (config == nullptr || out_encoder == nullptr) {
        return CUJPEGXL_INVALID_ARGUMENT;
    }
    *out_encoder = nullptr;
    const cujpegxl_status status{validate_config(*config)};
    if (status != CUJPEGXL_OK) {
        return status;
    }
    cujpegxl::DeviceCapability capability{};
    if (!cujpegxl::query_device_capability(config->device_ordinal, &capability)) {
        return CUJPEGXL_NO_DEVICE;
    }
    std::unique_ptr<cujpegxl::EncoderSession> session{make_session(*config, 1)};
    if (session == nullptr) {
        return CUJPEGXL_INTERNAL_ERROR;
    }
    auto* encoder{new (std::nothrow) cujpegxl_encoder{*config, 1, std::move(session)}};
    if (encoder == nullptr) {
        return CUJPEGXL_INTERNAL_ERROR;
    }
    *out_encoder = encoder;
    return CUJPEGXL_OK;
}

cujpegxl_status cujpegxl_encoder_configure(cujpegxl_encoder* encoder,
                                           const cujpegxl_config* config) {
    if (encoder == nullptr || config == nullptr) {
        return CUJPEGXL_INVALID_ARGUMENT;
    }
    const cujpegxl_status status{validate_config(*config)};
    if (status != CUJPEGXL_OK) {
        return status;
    }
    std::unique_ptr<cujpegxl::EncoderSession> session{
        make_session(*config, encoder->pipeline_depth)};
    if (session == nullptr) {
        return CUJPEGXL_INTERNAL_ERROR;
    }
    encoder->session = std::move(session);
    encoder->config = *config;
    return CUJPEGXL_OK;
}

cujpegxl_status cujpegxl_encoder_encode(cujpegxl_encoder* encoder, const cujpegxl_nv12_input* input,
                                        cujpegxl_output* output) {
    if (encoder == nullptr || input == nullptr || output == nullptr || output->data == nullptr) {
        return CUJPEGXL_INVALID_ARGUMENT;
    }
    const cujpegxl_config& config{encoder->config};
    const cujpegxl_status status{validate_config(config)};
    if (status != CUJPEGXL_OK) {
        return status;
    }

    cujpegxl_future* future{nullptr};
    cujpegxl_status submit_status{submit(encoder, input, 0, &future, true)};
    if (submit_status != CUJPEGXL_OK) {
        return submit_status;
    }
    if (!future->future.get(future->output)) {
        cujpegxl_future_destroy(future);
        return CUJPEGXL_INTERNAL_ERROR;
    }
    future->loaded = true;
    cujpegxl_status get_status{cujpegxl_future_get(future, output, nullptr)};
    cujpegxl_future_destroy(future);
    return get_status;
}

void cujpegxl_encoder_destroy(cujpegxl_encoder* encoder) {
    delete encoder;
}

cujpegxl_status cujpegxl_async_encoder_create(const cujpegxl_async_config* config,
                                              cujpegxl_encoder** out_encoder) {
    if (config == nullptr || out_encoder == nullptr || config->pipeline_depth == 0) {
        return CUJPEGXL_INVALID_ARGUMENT;
    }
    *out_encoder = nullptr;
    const cujpegxl_status status{validate_config(config->frame)};
    if (status != CUJPEGXL_OK) {
        return status;
    }
    cujpegxl::DeviceCapability capability{};
    if (!cujpegxl::query_device_capability(config->frame.device_ordinal, &capability)) {
        return CUJPEGXL_NO_DEVICE;
    }
    std::unique_ptr<cujpegxl::EncoderSession> session{
        make_session(config->frame, config->pipeline_depth)};
    if (session == nullptr) {
        return CUJPEGXL_INTERNAL_ERROR;
    }
    auto* encoder{new (std::nothrow)
                      cujpegxl_encoder{config->frame, config->pipeline_depth, std::move(session)}};
    if (encoder == nullptr) {
        return CUJPEGXL_INTERNAL_ERROR;
    }
    *out_encoder = encoder;
    return CUJPEGXL_OK;
}

cujpegxl_status cujpegxl_encoder_try_submit(cujpegxl_encoder* encoder,
                                            const cujpegxl_nv12_input* input, uint64_t sequence,
                                            cujpegxl_future** out_future) {
    return submit(encoder, input, sequence, out_future, false);
}

cujpegxl_status cujpegxl_encoder_submit(cujpegxl_encoder* encoder, const cujpegxl_nv12_input* input,
                                        uint64_t sequence, cujpegxl_future** out_future) {
    return submit(encoder, input, sequence, out_future, true);
}

void cujpegxl_encoder_flush(cujpegxl_encoder* encoder) {
    if (encoder != nullptr) {
        encoder->session->flush();
    }
}

cujpegxl_status cujpegxl_future_ready(const cujpegxl_future* future, int* out_ready) {
    if (future == nullptr || out_ready == nullptr) {
        return CUJPEGXL_INVALID_ARGUMENT;
    }
    *out_ready = future->future.ready() ? 1 : 0;
    return CUJPEGXL_OK;
}

cujpegxl_status cujpegxl_future_wait(cujpegxl_future* future, uint64_t timeout_ns) {
    if (future == nullptr) {
        return CUJPEGXL_INVALID_ARGUMENT;
    }
    constexpr std::uint64_t max_timeout{
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())};
    const std::int64_t bounded_timeout{
        static_cast<std::int64_t>(timeout_ns > max_timeout ? max_timeout : timeout_ns)};
    return future->future.wait_for(std::chrono::nanoseconds{bounded_timeout}) ? CUJPEGXL_OK
                                                                              : CUJPEGXL_TIMEOUT;
}

cujpegxl_status cujpegxl_future_get(cujpegxl_future* future, cujpegxl_output* output,
                                    uint64_t* out_sequence) {
    if (future == nullptr || output == nullptr || output->data == nullptr) {
        return CUJPEGXL_INVALID_ARGUMENT;
    }
    if (!future->future.ready()) {
        return CUJPEGXL_NOT_READY;
    }
    if (!future->loaded) {
        if (!future->future.get(future->output)) {
            return CUJPEGXL_INTERNAL_ERROR;
        }
        future->loaded = true;
    }
    if (future->output.bytes.size() > output->capacity) {
        output->size = future->output.bytes.size();
        return CUJPEGXL_BUFFER_TOO_SMALL;
    }
    for (std::size_t i{0}; i < future->output.bytes.size(); ++i) {
        output->data[i] = future->output.bytes[i];
    }
    output->size = future->output.bytes.size();
    if (out_sequence != nullptr) {
        *out_sequence = future->output.sequence;
    }
    return CUJPEGXL_OK;
}

void cujpegxl_future_destroy(cujpegxl_future* future) {
    delete future;
}

}  // extern "C"
