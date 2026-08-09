// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "cujpegxl/capability.h"

namespace cujpegxl {

const char* backend_name(Backend backend) {
    switch (backend) {
        case Backend::FP32_SIMT:
            return "fp32-simt";
        case Backend::FP16_TENSOR:
            return "fp16-tensor";
        case Backend::UNKNOWN:
            return "unknown";
    }
    return "unknown";
}

}  // namespace cujpegxl
