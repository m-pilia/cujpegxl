// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Writes cfl_ytox_ratio / cfl_ytob_ratio for every signed-int8 map value as
// float32: 256 YtoX ratios (map -128..127) followed by 256 YtoB.

#include <cstdio>
#include <cstring>
#include <vector>

#include "cfl.h"
#include "dump_io.h"

int main(int argc, char** argv) {
    if (argc != 3 || std::strcmp(argv[1], "cfl") != 0) {
        std::fprintf(stderr, "usage: %s cfl <out_ratios_f32>\n", argv[0]);
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
    if (!cujpegxl::write_raw(argv[2], ratios.data(), ratios.size())) {
        std::fprintf(stderr, "cfl_dump: cannot write %s\n", argv[2]);
        return 1;
    }
    return 0;
}
