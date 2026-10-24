// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

struct RgbBenchArgs {
    std::uint32_t width;
    std::uint32_t height;
    std::int32_t iterations;
    std::int32_t warmup;
    float distance;
    std::int32_t effort;
    std::int32_t device;
    std::vector<std::string> files;
};

RgbBenchArgs parse_rgb_bench_args(int argc, char** argv, bool has_device);

int run_rgb_bench(const RgbBenchArgs& args, std::string_view codec, bool uses_device,
                  const std::function<std::size_t(const std::vector<std::uint8_t>&)>& encode);
