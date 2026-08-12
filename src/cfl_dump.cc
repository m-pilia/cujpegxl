// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Testonly harness: writes cfl_ytox_ratio / cfl_ytob_ratio for every signed-int8
// map value so a Python test can diff them against the libjxl cfl oracle. The
// layout matches the oracle: 256 YtoX ratios (map -128..127) then 256 YtoB. Not
// shipped.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cfl.h"

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <out_ratios_f32>\n", argv[0]);
        return 2;
    }
    std::vector<float> ratios{};
    ratios.reserve(512);
    for (int m{-128}; m <= 127; ++m) {
        ratios.push_back(cujpegxl::cfl_ytox_ratio(m));
    }
    for (int m{-128}; m <= 127; ++m) {
        ratios.push_back(cujpegxl::cfl_ytob_ratio(m));
    }
    std::FILE* out{std::fopen(argv[1], "wb")};
    if (out == nullptr) {
        std::fprintf(stderr, "cfl_dump: cannot open %s for writing\n", argv[1]);
        return 1;
    }
    const std::size_t put{std::fwrite(ratios.data(), sizeof(float), ratios.size(), out)};
    std::fclose(out);
    return put == ratios.size() ? 0 : 1;
}
