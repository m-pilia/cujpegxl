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

#define CUJPEGXL_API_VERSION 2u

typedef enum cujpegxl_status {
    CUJPEGXL_OK = 0,
    CUJPEGXL_NOT_IMPLEMENTED = 1,
    CUJPEGXL_INVALID_ARGUMENT = 2,
    CUJPEGXL_UNSUPPORTED_RESOLUTION = 3,
    CUJPEGXL_NO_DEVICE = 4,
    CUJPEGXL_INTERNAL_ERROR = 5,
    CUJPEGXL_BUFFER_TOO_SMALL = 6,
    CUJPEGXL_WOULD_BLOCK = 7,
    CUJPEGXL_NOT_READY = 8,
    CUJPEGXL_TIMEOUT = 9
} cujpegxl_status;

typedef enum cujpegxl_backend {
    CUJPEGXL_BACKEND_UNKNOWN = 0,
    CUJPEGXL_BACKEND_FP32_SIMT = 1,
    CUJPEGXL_BACKEND_FP16_TENSOR = 2
} cujpegxl_backend;

typedef struct cujpegxl_encoder cujpegxl_encoder;
typedef struct cujpegxl_future cujpegxl_future;

typedef struct cujpegxl_config {
    uint32_t width;
    uint32_t height;
    float distance;
    int32_t device_ordinal;
} cujpegxl_config;

typedef struct cujpegxl_async_config {
    cujpegxl_config frame;
    size_t pipeline_depth;
} cujpegxl_async_config;

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

/* Worst-case codestream size (in bytes) for `config`, so a caller can
 * preallocate the output buffer before `cujpegxl_encoder_encode`. */
cujpegxl_status cujpegxl_max_output_size(const cujpegxl_config* config, size_t* out_bytes);

cujpegxl_status cujpegxl_encoder_create(const cujpegxl_config* config,
                                        cujpegxl_encoder** out_encoder);
cujpegxl_status cujpegxl_encoder_configure(cujpegxl_encoder* encoder,
                                           const cujpegxl_config* config);
cujpegxl_status cujpegxl_encoder_encode(cujpegxl_encoder* encoder, const cujpegxl_nv12_input* input,
                                        cujpegxl_output* output);
void cujpegxl_encoder_destroy(cujpegxl_encoder* encoder);

/* Creates a bounded asynchronous encoder. Input device pointers must remain
 * valid until the corresponding future is ready. */
cujpegxl_status cujpegxl_async_encoder_create(const cujpegxl_async_config* config,
                                              cujpegxl_encoder** out_encoder);

/* Non-blocking submission returns CUJPEGXL_WOULD_BLOCK when the pipeline is
 * full. The blocking variant waits only for submission capacity. */
cujpegxl_status cujpegxl_encoder_try_submit(cujpegxl_encoder* encoder,
                                            const cujpegxl_nv12_input* input,
                                            uint64_t sequence,
                                            cujpegxl_future** out_future);
cujpegxl_status cujpegxl_encoder_submit(cujpegxl_encoder* encoder,
                                        const cujpegxl_nv12_input* input,
                                        uint64_t sequence,
                                        cujpegxl_future** out_future);
void cujpegxl_encoder_flush(cujpegxl_encoder* encoder);

cujpegxl_status cujpegxl_future_ready(const cujpegxl_future* future, int* out_ready);
cujpegxl_status cujpegxl_future_wait(cujpegxl_future* future, uint64_t timeout_ns);
cujpegxl_status cujpegxl_future_get(cujpegxl_future* future, cujpegxl_output* output,
                                    uint64_t* out_sequence);
void cujpegxl_future_destroy(cujpegxl_future* future);

/* NOLINTEND(modernize-use-using,readability-identifier-naming) */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CUJPEGXL_CUJPEGXL_H_ */
