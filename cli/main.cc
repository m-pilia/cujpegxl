// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdio>

#include "cujpegxl/cujpegxl.h"

// Internal development CLI: prints the ABI version and the backend the encoder
// would select on the requested device. Not part of the shipped deliverable.
int main() {
    std::printf("cujpegxl api_version=%u\n", cujpegxl_api_version());

    cujpegxl_backend backend{CUJPEGXL_BACKEND_UNKNOWN};
    const cujpegxl_status status{cujpegxl_query_backend(0, &backend)};
    std::printf("backend_query_status=%s backend=%s\n", cujpegxl_status_string(status),
                cujpegxl_backend_string(backend));
    return 0;
}
