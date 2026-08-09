// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "cujpegxl/cujpegxl.h"

#include "cujpegxl/capability.h"

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

cujpegxl_status cujpegxl_encoder_create(const cujpegxl_config* config,
                                        cujpegxl_encoder** out_encoder) {
    if (config == nullptr || out_encoder == nullptr) {
        return CUJPEGXL_INVALID_ARGUMENT;
    }
    *out_encoder = nullptr;
    return CUJPEGXL_NOT_IMPLEMENTED;
}

cujpegxl_status cujpegxl_encoder_configure(cujpegxl_encoder* encoder,
                                           const cujpegxl_config* config) {
    if (encoder == nullptr || config == nullptr) {
        return CUJPEGXL_INVALID_ARGUMENT;
    }
    return CUJPEGXL_NOT_IMPLEMENTED;
}

cujpegxl_status cujpegxl_encoder_encode(cujpegxl_encoder* encoder, const cujpegxl_nv12_input* input,
                                        cujpegxl_output* output) {
    if (encoder == nullptr || input == nullptr || output == nullptr) {
        return CUJPEGXL_INVALID_ARGUMENT;
    }
    return CUJPEGXL_NOT_IMPLEMENTED;
}

void cujpegxl_encoder_destroy(cujpegxl_encoder* encoder) {
    (void)encoder;
}

}  // extern "C"
