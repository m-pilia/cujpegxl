// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Testonly harness: runs dc_from_llf on one block's coefficients and writes the
// resulting DC values, so a Python test can diff them against the libjxl
// DCFromLowestFrequencies oracle. Not shipped.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vardct_layout.h"

namespace {

bool read_floats(const char* path, std::size_t count, std::vector<float>* out) {
    out->resize(count);
    std::FILE* file{std::fopen(path, "rb")};
    if (file == nullptr) {
        return false;
    }
    const std::size_t got{std::fread(out->data(), sizeof(float), count, file)};
    std::fclose(file);
    return got == count;
}

}  // namespace

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
    if (!read_floats(argv[3], block_dim * block_dim, &coeffs)) {
        std::fprintf(stderr, "vardct_layout_dump: %s is not %zu floats\n", argv[3],
                     block_dim * block_dim);
        return 1;
    }

    const std::size_t side{cujpegxl::covered_blocks_side(block_dim)};
    std::vector<float> dc(side * side);
    cujpegxl::dc_from_llf(block_dim, coeffs.data(), dc.data());

    std::FILE* out{std::fopen(argv[4], "wb")};
    if (out == nullptr) {
        std::fprintf(stderr, "vardct_layout_dump: cannot open %s for writing\n", argv[4]);
        return 1;
    }
    const std::size_t put{std::fwrite(dc.data(), sizeof(float), dc.size(), out)};
    std::fclose(out);
    return put == dc.size() ? 0 : 1;
}
