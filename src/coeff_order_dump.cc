// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Writes natural_coeff_order(block_dim) as raw uint32.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "coeff_order.h"
#include "dump_io.h"

int main(int argc, char** argv) {
    if (argc != 4 || std::strcmp(argv[1], "coefforder") != 0) {
        std::fprintf(stderr, "usage: %s coefforder <block_dim> <out_order_u32>\n", argv[0]);
        return 2;
    }
    const std::size_t block_dim{std::strtoull(argv[2], nullptr, 10)};
    if (block_dim != 8 && block_dim != 16 && block_dim != 32) {
        std::fprintf(stderr, "coeff_order_dump: block dim must be 8, 16, or 32\n");
        return 2;
    }
    const std::vector<std::uint32_t> order{cujpegxl::natural_coeff_order(block_dim)};
    if (!cujpegxl::write_raw(argv[3], order.data(), order.size())) {
        std::fprintf(stderr, "coeff_order_dump: cannot write %s\n", argv[3]);
        return 1;
    }
    return 0;
}
