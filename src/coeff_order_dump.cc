// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Testonly harness: writes natural_coeff_order(block_dim) as raw uint32 so a
// Python test can diff it against the libjxl coefforder oracle. Not shipped.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "coeff_order.h"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <block_dim> <out_order_u32>\n", argv[0]);
        return 2;
    }
    const std::size_t block_dim{std::strtoull(argv[1], nullptr, 10)};
    if (block_dim != 8 && block_dim != 16 && block_dim != 32) {
        std::fprintf(stderr, "coeff_order_dump: block dim must be 8, 16, or 32\n");
        return 2;
    }

    const std::vector<std::uint32_t> order{cujpegxl::natural_coeff_order(block_dim)};
    std::FILE* out{std::fopen(argv[2], "wb")};
    if (out == nullptr) {
        std::fprintf(stderr, "coeff_order_dump: cannot open %s for writing\n", argv[2]);
        return 1;
    }
    const std::size_t put{std::fwrite(order.data(), sizeof(std::uint32_t), order.size(), out)};
    std::fclose(out);
    return put == order.size() ? 0 : 1;
}
