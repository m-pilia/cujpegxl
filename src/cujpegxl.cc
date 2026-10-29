// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "cujpegxl/cujpegxl.h"

#include <cmath>
#include <cstddef>
#include <new>
#include <vector>

#include "device_support.h"
#include "frame_encoder.h"
#include "src/bitstream/frame_assembly.h"
#include "worker_pool.h"

struct cujpegxl_encoder {
    cujpegxl_config config;
    cujpegxl::FrameEncoder frame_encoder;
    // The outputs of the frames in flight, in a ring of pipeline_depth entries
    // from the oldest frame's at `oldest_output`.
    std::vector<cujpegxl_output*> outputs;
    std::size_t oldest_output;
};

namespace {

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

}  // namespace

extern "C" {

uint32_t cujpegxl_api_version(void) {
    return CUJPEGXL_API_VERSION;
}

const char* cujpegxl_status_string(cujpegxl_status status) {
    switch (status) {
        case CUJPEGXL_OK:
            return "ok";
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
        case CUJPEGXL_PIPELINE_FULL:
            return "pipeline-full";
        case CUJPEGXL_PIPELINE_EMPTY:
            return "pipeline-empty";
        case CUJPEGXL_UNSUPPORTED_DEVICE:
            return "unsupported-device";
    }
    return "internal-error";
}

cujpegxl_status cujpegxl_max_output_size(const cujpegxl_config* config, size_t* out_bytes) {
    if (config == nullptr || out_bytes == nullptr) {
        return CUJPEGXL_INVALID_ARGUMENT;
    }
    const cujpegxl_status status{validate_config(*config)};
    if (status != CUJPEGXL_OK) {
        return status;
    }
    *out_bytes = cujpegxl::max_encoded_size(config->width, config->height);
    return CUJPEGXL_OK;
}

cujpegxl_status cujpegxl_encoder_create(const cujpegxl_config* config, int32_t device_ordinal,
                                        uint32_t pipeline_depth, cujpegxl_encoder** out_encoder) {
    if (config == nullptr || out_encoder == nullptr || pipeline_depth == 0) {
        return CUJPEGXL_INVALID_ARGUMENT;
    }
    *out_encoder = nullptr;
    const cujpegxl_status status{validate_config(*config)};
    if (status != CUJPEGXL_OK) {
        return status;
    }
    switch (cujpegxl::query_device_support(device_ordinal)) {
        case cujpegxl::DeviceSupport::SUPPORTED:
            break;
        case cujpegxl::DeviceSupport::NO_DEVICE:
            return CUJPEGXL_NO_DEVICE;
        case cujpegxl::DeviceSupport::UNSUPPORTED:
            return CUJPEGXL_UNSUPPORTED_DEVICE;
    }
    auto* encoder{new (std::nothrow) cujpegxl_encoder{
        *config,
        cujpegxl::FrameEncoder{device_ordinal, cujpegxl::default_worker_threads(), pipeline_depth},
        std::vector<cujpegxl_output*>(pipeline_depth, nullptr), 0}};
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
    encoder->config = *config;
    return CUJPEGXL_OK;
}

cujpegxl_status cujpegxl_encoder_submit(cujpegxl_encoder* encoder, const cujpegxl_nv12_input* input,
                                        cujpegxl_output* output) {
    if (encoder == nullptr || input == nullptr || output == nullptr || output->data == nullptr ||
        input->luma == nullptr || input->chroma == nullptr) {
        return CUJPEGXL_INVALID_ARGUMENT;
    }
    const cujpegxl_config& config{encoder->config};
    const cujpegxl_status status{validate_config(config)};
    if (status != CUJPEGXL_OK) {
        return status;
    }
    if (input->luma_pitch < config.width || input->chroma_pitch < config.width) {
        return CUJPEGXL_INVALID_ARGUMENT;
    }
    cujpegxl::FrameEncoder& frame_encoder{encoder->frame_encoder};
    const std::size_t in_flight{frame_encoder.in_flight()};
    if (in_flight == frame_encoder.pipeline_depth()) {
        return CUJPEGXL_PIPELINE_FULL;
    }
    if (!frame_encoder.submit(static_cast<const std::uint8_t*>(input->luma), input->luma_pitch,
                              static_cast<const std::uint8_t*>(input->chroma), input->chroma_pitch,
                              static_cast<cudaStream_t>(input->stream), config.width, config.height,
                              config.distance, output->data, output->capacity)) {
        return CUJPEGXL_INTERNAL_ERROR;
    }
    encoder->outputs[(encoder->oldest_output + in_flight) % encoder->outputs.size()] = output;
    return CUJPEGXL_OK;
}

cujpegxl_status cujpegxl_encoder_receive(cujpegxl_encoder* encoder, cujpegxl_output** out_output) {
    if (encoder == nullptr || out_output == nullptr) {
        return CUJPEGXL_INVALID_ARGUMENT;
    }
    if (encoder->frame_encoder.in_flight() == 0) {
        return CUJPEGXL_PIPELINE_EMPTY;
    }
    const cujpegxl::EncodeResult result{encoder->frame_encoder.receive()};
    cujpegxl_output* const output{encoder->outputs[encoder->oldest_output]};
    encoder->oldest_output = (encoder->oldest_output + 1) % encoder->outputs.size();
    *out_output = output;
    switch (result.status) {
        case cujpegxl::EncodeStatus::OK:
            output->size = result.file_size;
            return CUJPEGXL_OK;
        case cujpegxl::EncodeStatus::BUFFER_TOO_SMALL:
            output->size = result.file_size;
            return CUJPEGXL_BUFFER_TOO_SMALL;
        case cujpegxl::EncodeStatus::FAILED:
            return CUJPEGXL_INTERNAL_ERROR;
    }
    return CUJPEGXL_INTERNAL_ERROR;
}

void cujpegxl_encoder_destroy(cujpegxl_encoder* encoder) {
    delete encoder;
}

}  // extern "C"
