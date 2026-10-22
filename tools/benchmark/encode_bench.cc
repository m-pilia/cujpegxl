// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Benchmark the GPU encode time of NV12 images against cujpegxl or nvJPEG.
//
// NV12 input bytes are loaded from disk into host memory and uploaded to the
// device once, then each timed iteration drives only the on-device encode path
// (cujpegxl: XYB -> DCT -> quantize -> entropy -> ISOBMFF assembly; nvJPEG:
// nvjpegEncodeYUV + bitstream retrieve). All NV12 files passed to a single
// invocation must share the same --dims. Iterations alternate across the input
// files so that any per-image caching effects average out across the run.
//
// cujpegxl encodes the iterations as one stream of frames with up to
// --pipeline-depth frames in flight, warmup frames first. The reported time per
// frame is the interval between consecutive completions, the steady-state
// throughput; the latency of each frame, from submission to completion, is
// reported separately. At depth 1 the two coincide up to the loop overhead.
//
// For cujpegxl the tool additionally supports a --profile mode intended for
// capture under NVIDIA Nsight Systems (nsys) or Nsight Compute (ncu): it
// defaults iterations to 1, emits an NVTX range around every encode (also
// emitted in the default mode, since NVTX is zero-cost when no profiler is
// attached), and prints the per-stage StageTiming breakdown for the final
// iteration. Profiling support for nvJPEG is intentionally omitted: nvJPEG is a
// fixed vendor baseline and not an optimization target for this project.

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include "src/bitstream/frame_assembly.h"
#include "src/frame_encoder.h"
#include "src/worker_pool.h"
#include "tools/benchmark/nvjpeg_encoder.h"

namespace {

constexpr const char* USAGE =
    "encode_bench: measure GPU encode time of NV12 images with cujpegxl or nvJPEG.\n"
    "\n"
    "Usage:\n"
    "  encode_bench --codec=<cujpegxl|nvjpeg> --dims=WxH [options] <file1.nv12> [...]\n"
    "\n"
    "Input files are raw NV12 (BT.709 full range, 4:2:0) byte streams of size\n"
    "W*H*3/2. The encoder reads them as device-resident inputs: H2D upload is\n"
    "performed once at startup and excluded from the timed window.\n"
    "\n"
    "Options:\n"
    "  --codec=<cujpegxl|nvjpeg>   Encoder to benchmark (required).\n"
    "  --dims=WxH                  Frame dimensions in pixels (required).\n"
    "  --iterations=N              Timed encode iterations (default 5).\n"
    "  --warmup=N                  Untimed warmup encode iterations (default 1).\n"
    "  --distance=F                cujpegxl Butteraugli distance (default 1.0).\n"
    "  --pipeline-depth=N          cujpegxl frames in flight (default 4).\n"
    "  --quality=N                 nvJPEG quality 1..100 (default 90).\n"
    "  --device=N                  CUDA device ordinal (default 0).\n"
    "  --profile                   cujpegxl only: defaults iterations to 1 (unless\n"
    "                              --iterations is given explicitly), wraps each\n"
    "                              encode in an NVTX range for nsys/ncu capture, and\n"
    "                              prints the per-stage StageTiming breakdown.\n"
    "  -h, --help                  Show this help.\n";

struct Args {
    std::string codec;
    std::uint32_t width{0};
    std::uint32_t height{0};
    int iterations{5};
    int warmup{1};
    float distance{1.0f};
    std::size_t pipeline_depth{4};
    int quality{90};
    int device{0};
    bool profile{false};
    bool iterations_explicit{false};
    std::vector<std::string> files;
};

[[noreturn]] void die(const std::string& msg) {
    std::fprintf(stderr, "encode_bench: %s\n", msg.c_str());
    std::exit(1);
}

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        die(std::string("cuda ") + what + " failed: " + cudaGetErrorString(err));
    }
}

bool starts_with(std::string_view s, std::string_view prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

bool parse_long(const char* s, long* out) {
    errno = 0;
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (errno != 0 || end == s || *end != '\0') {
        return false;
    }
    *out = v;
    return true;
}

bool parse_float(const char* s, float* out) {
    errno = 0;
    char* end = nullptr;
    float v = std::strtof(s, &end);
    if (errno != 0 || end == s || *end != '\0') {
        return false;
    }
    *out = v;
    return true;
}

std::string take_value(int argc, char** argv, int& i, const std::string& arg) {
    const auto eq = arg.find('=');
    if (eq != std::string::npos) {
        return arg.substr(eq + 1);
    }
    if (i + 1 < argc) {
        return argv[++i];
    }
    die("missing value for " + arg);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i{1}; i < argc; ++i) {
        const std::string arg{argv[i]};
        if (arg == "-h" || arg == "--help") {
            std::printf("%s\n", USAGE);
            std::exit(0);
        } else if (starts_with(arg, "--codec")) {
            a.codec = take_value(argc, argv, i, arg);
        } else if (starts_with(arg, "--dims")) {
            const std::string v{take_value(argc, argv, i, arg)};
            const auto x = v.find('x');
            if (x == std::string::npos) {
                die("expected --dims=WxH, got: " + v);
            }
            long w{0};
            long h{0};
            const std::string w_part{v.substr(0, x)};
            const std::string h_part{v.substr(x + 1)};
            if (!parse_long(w_part.c_str(), &w) || !parse_long(h_part.c_str(), &h) || w <= 0 ||
                h <= 0) {
                die("invalid dims: " + v);
            }
            a.width = static_cast<std::uint32_t>(w);
            a.height = static_cast<std::uint32_t>(h);
        } else if (starts_with(arg, "--iterations")) {
            long v{0};
            if (!parse_long(take_value(argc, argv, i, arg).c_str(), &v)) {
                die("--iterations requires an integer");
            }
            a.iterations = static_cast<int>(v);
            a.iterations_explicit = true;
        } else if (starts_with(arg, "--warmup")) {
            long v{0};
            if (!parse_long(take_value(argc, argv, i, arg).c_str(), &v)) {
                die("--warmup requires an integer");
            }
            a.warmup = static_cast<int>(v);
        } else if (starts_with(arg, "--distance")) {
            float v{0.0f};
            if (!parse_float(take_value(argc, argv, i, arg).c_str(), &v)) {
                die("--distance requires a number");
            }
            a.distance = v;
        } else if (starts_with(arg, "--pipeline-depth")) {
            long v{0};
            if (!parse_long(take_value(argc, argv, i, arg).c_str(), &v) || v < 1) {
                die("--pipeline-depth requires a positive integer");
            }
            a.pipeline_depth = static_cast<std::size_t>(v);
        } else if (starts_with(arg, "--quality")) {
            long v{0};
            if (!parse_long(take_value(argc, argv, i, arg).c_str(), &v)) {
                die("--quality requires an integer");
            }
            a.quality = static_cast<int>(v);
        } else if (starts_with(arg, "--device")) {
            long v{0};
            if (!parse_long(take_value(argc, argv, i, arg).c_str(), &v)) {
                die("--device requires an integer");
            }
            a.device = static_cast<int>(v);
        } else if (arg == "--profile") {
            a.profile = true;
        } else if (starts_with(arg, "--")) {
            die("unknown option: " + arg);
        } else {
            a.files.push_back(arg);
        }
    }
    return a;
}

struct BenchImage {
    std::vector<std::uint8_t> host;
    std::uint8_t* d_luma{nullptr};
    std::uint8_t* d_chroma{nullptr};
    std::unique_ptr<cujpegxl::tools::DevicePlanarYuv> planar;
};

BenchImage load_nv12(const std::string& path, std::uint32_t width, std::uint32_t height) {
    const std::size_t expected{static_cast<std::size_t>(width) * height * 3 / 2};
    BenchImage img{};
    img.host.resize(expected);
    std::ifstream f{path, std::ios::binary};
    if (!f) {
        die("cannot open: " + path);
    }
    f.read(reinterpret_cast<char*>(img.host.data()), static_cast<std::streamsize>(expected));
    if (static_cast<std::size_t>(f.gcount()) != expected) {
        die("short read on " + path + ": expected " + std::to_string(expected) + " bytes");
    }
    return img;
}

void upload_nv12(BenchImage& img, std::uint32_t width, std::uint32_t height) {
    const std::size_t luma_bytes{static_cast<std::size_t>(width) * height};
    const std::size_t chroma_bytes{static_cast<std::size_t>(width) * (height / 2)};
    check_cuda(cudaMalloc(&img.d_luma, luma_bytes), "cudaMalloc luma");
    check_cuda(cudaMalloc(&img.d_chroma, chroma_bytes), "cudaMalloc chroma");
    check_cuda(cudaMemcpy(img.d_luma, img.host.data(), luma_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy luma");
    check_cuda(cudaMemcpy(img.d_chroma, img.host.data() + luma_bytes, chroma_bytes,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy chroma");
}

void free_image(BenchImage& img) {
    if (img.d_luma != nullptr) {
        cudaFree(img.d_luma);
        img.d_luma = nullptr;
    }
    if (img.d_chroma != nullptr) {
        cudaFree(img.d_chroma);
        img.d_chroma = nullptr;
    }
    img.planar.reset();
}

double mean_of(const std::vector<double>& xs) {
    assert(!xs.empty());
    return std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());
}

double stddev_of(const std::vector<double>& xs) {
    if (xs.size() < 2) {
        return 0.0;
    }
    const double m{mean_of(xs)};
    double s{0.0};
    for (const double x : xs) {
        const double d{x - m};
        s += d * d;
    }
    return std::sqrt(s / static_cast<double>(xs.size() - 1));
}

double percentile_of(std::vector<double> xs, double p) {
    assert(!xs.empty());
    assert(p >= 0.0 && p <= 1.0);
    std::sort(xs.begin(), xs.end());
    const auto idx{static_cast<std::size_t>(std::llround(p * (xs.size() - 1)))};
    return xs[idx];
}

void print_header(const Args& a, std::size_t image_count) {
    std::printf("encode_bench\n");
    std::printf("  codec:      %s\n", a.codec.c_str());
    std::printf("  dims:       %ux%u\n", a.width, a.height);
    std::printf("  images:     %zu\n", image_count);
    std::printf("  iterations: %d\n", a.iterations);
    std::printf("  warmup:     %d\n", a.warmup);
    if (a.codec == "cujpegxl") {
        std::printf("  distance:   %.2f\n", a.distance);
        std::printf("  depth:      %zu\n", a.pipeline_depth);
    } else {
        std::printf("  quality:    %d\n", a.quality);
    }
    std::printf("  device:     %d\n", a.device);
    std::printf("  profile:    %s\n", a.profile ? "true" : "false");
}

void print_stats(const std::vector<double>& times_us, std::string_view codec) {
    const double total_us{std::accumulate(times_us.begin(), times_us.end(), 0.0)};
    const double mean_us{mean_of(times_us)};
    const double fps{mean_us > 0.0 ? 1.0e6 / mean_us : 0.0};
    std::printf("\nresults (%.*s, microseconds):\n", static_cast<int>(codec.size()), codec.data());
    std::printf("  n:          %zu\n", times_us.size());
    std::printf("  total:      %.0f\n", total_us);
    std::printf("  mean:       %.0f   (%.2f fps)\n", mean_us, fps);
    std::printf("  stddev:     %.0f\n", stddev_of(times_us));
    std::printf("  min:        %.0f\n", *std::min_element(times_us.begin(), times_us.end()));
    std::printf("  p50:        %.0f\n", percentile_of(times_us, 0.50));
    std::printf("  p99:        %.0f\n", percentile_of(times_us, 0.99));
    std::printf("  max:        %.0f\n", *std::max_element(times_us.begin(), times_us.end()));
}

int run(int argc, char** argv) {
    Args args{parse_args(argc, argv)};

    if (args.codec.empty()) {
        die("--codec=<cujpegxl|nvjpeg> is required");
    }
    if (args.codec != "cujpegxl" && args.codec != "nvjpeg") {
        die("--codec must be 'cujpegxl' or 'nvjpeg'");
    }
    if (args.width == 0 || args.height == 0) {
        die("--dims=WxH is required");
    }
    if (args.files.empty()) {
        die("at least one NV12 input file is required");
    }
    if (args.width % 2 != 0 || args.height % 2 != 0) {
        die("dims must be even (NV12 4:2:0 constraint)");
    }
    if (args.width % 8 != 0 || args.height % 8 != 0) {
        die("dims must be multiples of 8 (cujpegxl block size constraint)");
    }
    if (args.iterations < 1) {
        die("--iterations must be >= 1");
    }
    if (args.warmup < 0) {
        die("--warmup must be >= 0");
    }
    if (args.profile && args.codec != "cujpegxl") {
        die("--profile is only supported with --codec=cujpegxl");
    }
    if (args.profile && !args.iterations_explicit) {
        args.iterations = 1;
    }
    if (args.codec == "cujpegxl" && (!(args.distance > 0.0f) || !std::isfinite(args.distance))) {
        die("--distance must be a positive finite number");
    }
    if (args.codec == "nvjpeg" && (args.quality < 1 || args.quality > 100)) {
        die("--quality must be in [1, 100]");
    }

    check_cuda(cudaSetDevice(args.device), "cudaSetDevice");
    check_cuda(cudaDeviceSynchronize(), "warmup sync");

    std::vector<BenchImage> images{};
    images.reserve(args.files.size());
    for (const std::string& path : args.files) {
        BenchImage img{load_nv12(path, args.width, args.height)};
        if (args.codec == "cujpegxl") {
            upload_nv12(img, args.width, args.height);
        } else {
            img.planar = std::make_unique<cujpegxl::tools::DevicePlanarYuv>(
                img.host.data(), args.width, args.height);
        }
        images.push_back(std::move(img));
    }

    print_header(args, images.size());

    std::vector<double> times_us{};
    times_us.reserve(static_cast<std::size_t>(args.iterations));
    std::vector<double> latencies_us{};
    std::vector<cujpegxl::StageTiming> cujpegxl_stages{};
    if (args.codec == "cujpegxl") {
        const cujpegxl::bitstream::QuantParams qp{
            cujpegxl::quant_params_for_distance(args.distance)};
        const std::size_t depth{args.pipeline_depth};
        cujpegxl::FrameEncoder encoder{args.device, cujpegxl::default_worker_threads(), depth};
        // Per slot of frames in flight: output buffer, stage timings, NVTX range.
        std::vector<std::vector<std::uint8_t>> outputs(
            depth, std::vector<std::uint8_t>(cujpegxl::max_encoded_size(args.width, args.height)));
        std::vector<std::vector<cujpegxl::StageTiming>> stages(depth);
        std::vector<nvtxRangeId_t> ranges(depth);
        const std::size_t num_frames{static_cast<std::size_t>(args.warmup + args.iterations)};
        std::vector<std::chrono::steady_clock::time_point> submitted(num_frames);
        std::chrono::steady_clock::time_point last_completion{};
        std::size_t next_received{0};

        // The first encode after handle creation pays one-time CUDA context
        // init and kernel JIT; the warmup frames run before the timed ones in
        // the same stream, so that the timed frames see a full pipeline.
        const auto receive_one = [&] {
            const cujpegxl::EncodeResult result{encoder.receive()};
            const auto now{std::chrono::steady_clock::now()};
            const std::size_t f{next_received};
            ++next_received;
            nvtxRangeEnd(ranges[f % depth]);
            if (result.status != cujpegxl::EncodeStatus::OK) {
                die("cujpegxl encode failed");
            }
            if (f >= static_cast<std::size_t>(args.warmup)) {
                const double latency{
                    std::chrono::duration<double, std::micro>(now - submitted[f]).count()};
                times_us.push_back(
                    std::chrono::duration<double, std::micro>(now - last_completion).count());
                latencies_us.push_back(latency);
            }
            if (f + 1 == num_frames) {
                cujpegxl_stages = stages[f % depth];
            }
            last_completion = now;
        };
        check_cuda(cudaDeviceSynchronize(), "pre-stream sync");
        last_completion = std::chrono::steady_clock::now();
        for (std::size_t f{0}; f < num_frames; ++f) {
            if (encoder.in_flight() == depth) {
                receive_one();
            }
            const BenchImage& img{images[f % images.size()]};
            stages[f % depth].clear();
            submitted[f] = std::chrono::steady_clock::now();
            ranges[f % depth] = nvtxRangeStartA("cujpegxl_encode");
            if (!encoder.submit(img.d_luma, args.width, img.d_chroma, args.width, args.width,
                                args.height, args.distance, qp, outputs[f % depth].data(),
                                outputs[f % depth].size(),
                                args.profile ? &stages[f % depth] : nullptr)) {
                die("cujpegxl submit failed");
            }
        }
        while (encoder.in_flight() != 0) {
            receive_one();
        }
        for (std::size_t i{0}; i < times_us.size(); ++i) {
            std::fprintf(stderr, "[cujpegxl] iter %zu/%d: %.0f us (latency %.0f us)\n", i + 1,
                         args.iterations, times_us[i], latencies_us[i]);
        }
    } else {
        cujpegxl::tools::NvjpegEncoder nvjpeg{args.quality, args.width, args.height};
        auto encode_one = [&](const BenchImage& img) -> double {
            nvtxRangePushA("nvjpeg_encode");
            const auto t0{std::chrono::steady_clock::now()};
            nvjpeg.encode(*img.planar);
            const auto t1{std::chrono::steady_clock::now()};
            nvtxRangePop();
            return std::chrono::duration<double, std::micro>(t1 - t0).count();
        };

        // Warmup: the first encode after handle creation pays one-time CUDA
        // context init and nvjpeg state setup.
        for (int w{0}; w < args.warmup; ++w) {
            encode_one(images[static_cast<std::size_t>(w) % images.size()]);
        }
        check_cuda(cudaDeviceSynchronize(), "post-warmup sync");

        // Timed loop: alternate across images so cache effects are not
        // correlated with a single input.
        for (int i{0}; i < args.iterations; ++i) {
            const BenchImage& img{images[static_cast<std::size_t>(i) % images.size()]};
            check_cuda(cudaDeviceSynchronize(), "pre-iter sync");
            const double us{encode_one(img)};
            check_cuda(cudaDeviceSynchronize(), "post-iter sync");
            times_us.push_back(us);
            std::fprintf(stderr, "[nvjpeg] iter %d/%d: %.0f us\n", i + 1, args.iterations, us);
        }
    }

    print_stats(times_us, args.codec);
    if (!latencies_us.empty()) {
        std::printf("  latency:    mean %.0f   p50 %.0f   max %.0f\n", mean_of(latencies_us),
                    percentile_of(latencies_us, 0.50),
                    *std::max_element(latencies_us.begin(), latencies_us.end()));
    }

    if (args.profile && args.codec == "cujpegxl" && !cujpegxl_stages.empty()) {
        std::printf("\ncujpegxl per-stage timings (last iteration):\n");
        for (const cujpegxl::StageTiming& s : cujpegxl_stages) {
            std::printf("  %-10s gpu=%9.0f us  cpu=%7.0f us  bytes=%zu\n", s.name, s.gpu_us,
                        s.cpu_us, s.bytes_moved);
        }
        std::printf("\nCapture a detailed profile with:\n");
        std::printf(
            "  nsys profile --trace=cuda,nvtx --stats=true ./encode_bench --codec=cujpegxl "
            "--dims=%ux%u --profile %s\n",
            args.width, args.height, args.files.front().c_str());
        std::printf(
            "  ncu --set full --kernel-name regex:.* ./encode_bench --codec=cujpegxl "
            "--dims=%ux%u --profile %s\n",
            args.width, args.height, args.files.front().c_str());
    }

    for (BenchImage& img : images) {
        free_image(img);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        die(e.what());
    }
}
