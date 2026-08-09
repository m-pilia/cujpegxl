/* Copyright (c) 2026 Martino Pilia */
/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef CUJPEGXL_CUJPEGXL_H_
#define CUJPEGXL_CUJPEGXL_H_

/* Public C ABI for the cujpegxl encoder.
 *
 * This header is deliberately free of any CUDA types or includes so that
 * non-CUDA consumers can compile and link against the shared library without a
 * CUDA toolkit. Device memory is passed as opaque integer addresses. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C ABI: snake_case tagged types and typedefs are required for C interop and
 * intentionally diverge from the C++ naming rules. */
/* NOLINTBEGIN(modernize-use-using,readability-identifier-naming) */

#define CUJPEGXL_API_VERSION 1u

typedef enum cujpegxl_status {
    CUJPEGXL_OK = 0,
    CUJPEGXL_NOT_IMPLEMENTED = 1,
    CUJPEGXL_INVALID_ARGUMENT = 2,
    CUJPEGXL_UNSUPPORTED_RESOLUTION = 3,
    CUJPEGXL_NO_DEVICE = 4,
    CUJPEGXL_INTERNAL_ERROR = 5
} cujpegxl_status;

typedef enum cujpegxl_backend {
    CUJPEGXL_BACKEND_UNKNOWN = 0,
    CUJPEGXL_BACKEND_FP32_SIMT = 1,
    CUJPEGXL_BACKEND_FP16_TENSOR = 2
} cujpegxl_backend;

typedef struct cujpegxl_encoder cujpegxl_encoder;

typedef struct cujpegxl_config {
    uint32_t width;
    uint32_t height;
    float distance;
    int32_t device_ordinal;
} cujpegxl_config;

/* NV12 (BT.709 full range) device input. `luma`/`chroma` are raw device
 * addresses held as integers to keep this header CUDA-free. */
typedef struct cujpegxl_nv12_input {
    uintptr_t luma;
    uintptr_t chroma;
    size_t luma_pitch;
    size_t chroma_pitch;
} cujpegxl_nv12_input;

typedef struct cujpegxl_output {
    uint8_t* data;
    size_t size;
    size_t capacity;
} cujpegxl_output;

uint32_t cujpegxl_api_version(void);
const char* cujpegxl_status_string(cujpegxl_status status);
const char* cujpegxl_backend_string(cujpegxl_backend backend);

cujpegxl_status cujpegxl_query_backend(int32_t device_ordinal, cujpegxl_backend* out_backend);

cujpegxl_status cujpegxl_encoder_create(const cujpegxl_config* config,
                                        cujpegxl_encoder** out_encoder);
cujpegxl_status cujpegxl_encoder_configure(cujpegxl_encoder* encoder,
                                           const cujpegxl_config* config);
cujpegxl_status cujpegxl_encoder_encode(cujpegxl_encoder* encoder, const cujpegxl_nv12_input* input,
                                        cujpegxl_output* output);
void cujpegxl_encoder_destroy(cujpegxl_encoder* encoder);

/* NOLINTEND(modernize-use-using,readability-identifier-naming) */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CUJPEGXL_CUJPEGXL_H_ */
