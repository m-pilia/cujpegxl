/* Copyright (c) 2026 Martino Pilia */
/* SPDX-License-Identifier: BSD-3-Clause */

#ifndef CUJPEGXL_CUJPEGXL_H_
#define CUJPEGXL_CUJPEGXL_H_

/* Public C ABI for the cujpegxl encoder.
 *
 * This header is deliberately free of any CUDA types or includes so that
 * non-CUDA consumers can compile and link against the shared library without a
 * CUDA toolkit. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* C ABI: snake_case tagged types and typedefs are required for C interop and
 * intentionally diverge from the C++ naming rules. */
/* NOLINTBEGIN(modernize-use-using,readability-identifier-naming) */

#define CUJPEGXL_API_VERSION 5u

typedef enum cujpegxl_status {
    CUJPEGXL_OK = 0,
    CUJPEGXL_NOT_IMPLEMENTED = 1,
    CUJPEGXL_INVALID_ARGUMENT = 2,
    CUJPEGXL_UNSUPPORTED_RESOLUTION = 3,
    CUJPEGXL_NO_DEVICE = 4,
    CUJPEGXL_INTERNAL_ERROR = 5,
    CUJPEGXL_BUFFER_TOO_SMALL = 6,
    CUJPEGXL_PIPELINE_FULL = 7,
    CUJPEGXL_PIPELINE_EMPTY = 8
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

/* NV12 (BT.709 full range) input. `luma`/`chroma` are device pointers. */
typedef struct cujpegxl_nv12_input {
    const void* luma;
    const void* chroma;
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
 * preallocate the output buffers of `cujpegxl_encoder_submit`. */
cujpegxl_status cujpegxl_max_output_size(const cujpegxl_config* config, size_t* out_bytes);

/* An encoder is a pipeline of up to `pipeline_depth` (at least 1) frames in
 * flight, submitted and not yet received: the device work of a frame runs while
 * the host entropy-codes the frames submitted before it. Each frame in flight
 * holds its own device and pinned host memory. The output does not depend on
 * the depth. An encoder is not thread-safe. */
cujpegxl_status cujpegxl_encoder_create(const cujpegxl_config* config, uint32_t pipeline_depth,
                                        cujpegxl_encoder** out_encoder);
/* Applies to the frames submitted afterwards. The device is fixed at creation:
 * a different `device_ordinal` is an invalid argument. */
cujpegxl_status cujpegxl_encoder_configure(cujpegxl_encoder* encoder,
                                           const cujpegxl_config* config);
/* Starts encoding `input` into `output->data` (`output->capacity` bytes) and
 * returns without waiting for it; CUJPEGXL_PIPELINE_FULL if `pipeline_depth`
 * frames are in flight. The input planes must stay unchanged, and `output` and
 * its buffer valid, until the frame is received. */
cujpegxl_status cujpegxl_encoder_submit(cujpegxl_encoder* encoder, const cujpegxl_nv12_input* input,
                                        cujpegxl_output* output);
/* Waits for the oldest frame in flight and returns its status;
 * CUJPEGXL_PIPELINE_EMPTY if no frame is in flight. `*out_output` receives the
 * frame's `output` as passed to `cujpegxl_encoder_submit`, with `size` set to
 * the file size on CUJPEGXL_OK and to the capacity the file needs on
 * CUJPEGXL_BUFFER_TOO_SMALL. */
cujpegxl_status cujpegxl_encoder_receive(cujpegxl_encoder* encoder, cujpegxl_output** out_output);

/* Waits for the frames in flight to complete. */
void cujpegxl_encoder_destroy(cujpegxl_encoder* encoder);

/* NOLINTEND(modernize-use-using,readability-identifier-naming) */

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CUJPEGXL_CUJPEGXL_H_ */
