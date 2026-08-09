/* Copyright (c) 2026 Martino Pilia */
/* SPDX-License-Identifier: BSD-3-Clause */

#include <stdio.h>

#include "cujpegxl/cujpegxl.h"

/* Compiled as C to prove the public header needs no C++ and no CUDA. */
int main(void) {
    printf("cujpegxl api_version=%u\n", cujpegxl_api_version());
    return 0;
}
