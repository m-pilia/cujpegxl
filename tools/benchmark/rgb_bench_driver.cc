// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "tools/benchmark/rgb_bench_driver.h"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <string>

#include "tools/benchmark/benchmark_stats.h"

namespace {

[[noreturn]] void die(const std::string& message) {
    std::fprintf(stderr, "rgb_bench: %s\n", message.c_str());
    std::exit(1);
}

long parse_integer(const char* value, const char* name) {
    errno = 0;
    char* end{nullptr};
    const long result{std::strtol(value, &end, 10)};
    if (errno != 0 || end == value || *end != '\0') {
        die(std::string("invalid ") + name + ": " + value);
    }
    return result;
}

float parse_distance(const char* value) {
    errno = 0;
    char* end{nullptr};
    const float distance{std::strtof(value, &end)};
    if (errno != 0 || end == value || *end != '\0' || !(distance > 0.0f) ||
        !std::isfinite(distance)) {
        die(std::string("invalid distance: ") + value);
    }
    return distance;
}

std::vector<std::uint8_t> load_rgb(const std::string& path, std::size_t expected_size) {
    std::ifstream stream{path, std::ios::binary};
    if (!stream) {
        die("cannot open " + path);
    }
    std::vector<std::uint8_t> rgb(expected_size);
    stream.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    if (static_cast<std::size_t>(stream.gcount()) != rgb.size()) {
        die("short read on " + path);
    }
    return rgb;
}

}  // namespace

RgbBenchArgs parse_rgb_bench_args(int argc, char** argv, bool has_device) {
    const int first_file{has_device ? 8 : 7};
    if (argc <= first_file) {
        die("usage: WIDTH HEIGHT ITERATIONS WARMUP DISTANCE EFFORT [DEVICE] FILE.rgb [...]");
    }
    const long width{parse_integer(argv[1], "width")};
    const long height{parse_integer(argv[2], "height")};
    const long iterations{parse_integer(argv[3], "iterations")};
    const long warmup{parse_integer(argv[4], "warmup")};
    const float distance{parse_distance(argv[5])};
    const long effort{parse_integer(argv[6], "effort")};
    const long device{has_device ? parse_integer(argv[7], "device") : 0};
    if (width < 1 || width > std::numeric_limits<std::uint32_t>::max() || height < 1 ||
        height > std::numeric_limits<std::uint32_t>::max() || iterations < 1 ||
        iterations > std::numeric_limits<std::int32_t>::max() || warmup < 0 ||
        warmup > std::numeric_limits<std::int32_t>::max() || effort < 1 || effort > 10 ||
        device < 0 || device > std::numeric_limits<std::int32_t>::max() ||
        static_cast<std::size_t>(width) >
            std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(height) / 3) {
        die("invalid dimensions, iterations, warmup, effort, or device");
    }
    return RgbBenchArgs{
        .width = static_cast<std::uint32_t>(width),
        .height = static_cast<std::uint32_t>(height),
        .iterations = static_cast<std::int32_t>(iterations),
        .warmup = static_cast<std::int32_t>(warmup),
        .distance = distance,
        .effort = static_cast<std::int32_t>(effort),
        .device = static_cast<std::int32_t>(device),
        .files = {argv + first_file, argv + argc},
    };
}

int run_rgb_bench(const RgbBenchArgs& args, std::string_view codec, bool uses_device,
                  const std::function<std::size_t(const std::vector<std::uint8_t>&)>& encode) {
    const std::size_t expected_size{static_cast<std::size_t>(args.width) * args.height * 3};
    std::vector<std::vector<std::uint8_t>> images;
    images.reserve(args.files.size());
    for (const std::string& file : args.files) {
        images.push_back(load_rgb(file, expected_size));
    }

    std::printf("rgb_bench\n");
    std::printf("  codec:      %.*s\n", static_cast<int>(codec.size()), codec.data());
    std::printf("  dims:       %ux%u\n", args.width, args.height);
    std::printf("  images:     %zu\n", images.size());
    std::printf("  iterations: %d\n", args.iterations);
    std::printf("  warmup:     %d\n", args.warmup);
    std::printf("  distance:   %.2f\n", args.distance);
    std::printf("  effort:     %d\n", args.effort);
    if (uses_device) {
        std::printf("  device:     %d\n", args.device);
    }
    std::printf("  profile:    false\n");

    for (std::int32_t i{0}; i < args.warmup; ++i) {
        encode(images[static_cast<std::size_t>(i) % images.size()]);
    }
    std::vector<double> times_us;
    times_us.reserve(static_cast<std::size_t>(args.iterations));
    for (std::int32_t i{0}; i < args.iterations; ++i) {
        const auto start{std::chrono::steady_clock::now()};
        const std::size_t encoded_size{encode(images[static_cast<std::size_t>(i) % images.size()])};
        const auto end{std::chrono::steady_clock::now()};
        const double elapsed_us{std::chrono::duration<double, std::micro>(end - start).count()};
        times_us.push_back(elapsed_us);
        std::fprintf(stderr, "[%.*s] iter %d/%d: %.0f us (%zu bytes)\n",
                     static_cast<int>(codec.size()), codec.data(), i + 1, args.iterations,
                     elapsed_us, encoded_size);
    }
    benchmark_stats::print_stats(times_us, codec);
    return 0;
}
