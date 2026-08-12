// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Testonly harness: writes the baked DCT8/16/32 dequant weight matrix for one
// block size (3 * block_dim^2 float32, channel-major X, Y, B) so a Python test
// can diff it against the libjxl quantmatrix oracle. Not shipped.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "quant_weights_dct16.h"
#include "quant_weights_dct32.h"
#include "quant_weights_dct8.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <block_dim> <out_weights_f32>\n", argv[0]);
        return 2;
    }
    const std::size_t block_dim{std::strtoull(argv[1], nullptr, 10)};
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

    const std::size_t count{3 * block_dim * block_dim};
    std::FILE* out{std::fopen(argv[2], "wb")};
    if (out == nullptr) {
        std::fprintf(stderr, "quant_weights_dump: cannot open %s for writing\n", argv[2]);
        return 1;
    }
    const std::size_t put{std::fwrite(weights, sizeof(float), count, out)};
    std::fclose(out);
    return put == count ? 0 : 1;
}
