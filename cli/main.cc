// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdio>

#include "cujpegxl/cujpegxl.h"

// Internal development CLI: prints the ABI version. Not part of the shipped
// deliverable.
int main() {
    std::printf("cujpegxl api_version=%u\n", cujpegxl_api_version());
    return 0;
}
