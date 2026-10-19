// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Runs dc_from_llf on one block's coefficients and writes the resulting DC
// values as float32.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "dump_io.h"
#include "vardct_layout.h"

int main(int argc, char** argv) {
    if (argc != 5 || std::strcmp(argv[1], "dcfromllf") != 0) {
        std::fprintf(stderr, "usage: %s dcfromllf <block_dim> <in_coeff_f32> <out_dc_f32>\n",
                     argv[0]);
        return 2;
    }
    const std::size_t block_dim{std::strtoull(argv[2], nullptr, 10)};
    if (block_dim != 8 && block_dim != 16 && block_dim != 32) {
        std::fprintf(stderr, "vardct_layout_dump: block dim must be 8, 16, or 32\n");
        return 2;
    }

    std::vector<float> coeffs{};
    if (!cujpegxl::read_raw(argv[3], block_dim * block_dim, &coeffs)) {
        std::fprintf(stderr, "vardct_layout_dump: %s is not %zu floats\n", argv[3],
                     block_dim * block_dim);
        return 1;
    }

    const std::size_t side{cujpegxl::covered_blocks_side(block_dim)};
    std::vector<float> dc(side * side);
    cujpegxl::dc_from_llf(block_dim, coeffs.data(), dc.data());

    if (!cujpegxl::write_raw(argv[4], dc.data(), dc.size())) {
        std::fprintf(stderr, "vardct_layout_dump: cannot write %s\n", argv[4]);
        return 1;
    }
    return 0;
}
