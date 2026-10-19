// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Writes the baked dequant weight matrix for one square DCT size
// (3 * block_dim^2 float32, channel-major X, Y, B).

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "dump_io.h"
#include "quant_weights_dct16.h"
#include "quant_weights_dct32.h"
#include "quant_weights_dct8.h"

int main(int argc, char** argv) {
    if (argc != 4 || std::strcmp(argv[1], "quantmatrix") != 0) {
        std::fprintf(stderr, "usage: %s quantmatrix <block_dim> <out_weights_f32>\n", argv[0]);
        return 2;
    }
    const std::size_t block_dim{std::strtoull(argv[2], nullptr, 10)};
    const float* weights{nullptr};
    if (block_dim == 8) {
        weights = &cujpegxl::DCT8_DEQUANT_WEIGHTS[0][0];
    } else if (block_dim == 16) {
        weights = &cujpegxl::DCT16_DEQUANT_WEIGHTS[0][0];
    } else if (block_dim == 32) {
        weights = &cujpegxl::DCT32_DEQUANT_WEIGHTS[0][0];
    } else {
        std::fprintf(stderr, "quant_weights_dump: block dim must be 8, 16, or 32\n");
        return 2;
    }
    if (!cujpegxl::write_raw(argv[3], weights, 3 * block_dim * block_dim)) {
        std::fprintf(stderr, "quant_weights_dump: cannot write %s\n", argv[3]);
        return 1;
    }
    return 0;
}
