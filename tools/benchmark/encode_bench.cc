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
#include <deque>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <nvtx3/nvToolsExt.h>

#include "src/bitstream/frame_assembly.h"
#include "src/frame_encoder.h"

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
    "  --quality=N                 nvJPEG quality 1..100 (default 90).\n"
    "  --device=N                  CUDA device ordinal (default 0).\n"
    "  --pipeline-depth=N          cujpegxl in-flight frame limit (default 1).\n"
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
    int quality{90};
    int device{0};
    int pipeline_depth{1};
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

void check_nvjpeg(nvjpegStatus_t s, const char* what) {
    if (s != NVJPEG_STATUS_SUCCESS) {
        die(std::string("nvjpeg ") + what + " failed (status " +
            std::to_string(static_cast<int>(s)) + ")");
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
        } else if (starts_with(arg, "--pipeline-depth")) {
            long v{0};
            if (!parse_long(take_value(argc, argv, i, arg).c_str(), &v)) {
                die("--pipeline-depth requires an integer");
            }
            a.pipeline_depth = static_cast<int>(v);
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
    std::uint8_t* d_cb{nullptr};
    std::uint8_t* d_cr{nullptr};
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

// Deinterleaves the NV12 chroma plane (Cb0 Cr0 Cb1 Cr1 ...) into separate
// planar Cb and Cr buffers matching nvjpegEncodeYUV's expected image layout.
void deinterleave_nv12_chroma(const BenchImage& img, std::uint32_t width, std::uint32_t height,
                              std::vector<std::uint8_t>& cb, std::vector<std::uint8_t>& cr) {
    const std::size_t chroma_w{width / 2};
    const std::size_t chroma_h{height / 2};
    cb.resize(chroma_w * chroma_h);
    cr.resize(chroma_w * chroma_h);
    const std::uint8_t* uv{img.host.data() + static_cast<std::size_t>(width) * height};
    for (std::size_t i{0}; i < chroma_w * chroma_h; ++i) {
        cb[i] = uv[2 * i];
        cr[i] = uv[2 * i + 1];
    }
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

void upload_planar_yuv(BenchImage& img, std::uint32_t width, std::uint32_t height) {
    const std::size_t luma_bytes{static_cast<std::size_t>(width) * height};
    const std::size_t chroma_bytes{(static_cast<std::size_t>(width) / 2) * (height / 2)};
    std::vector<std::uint8_t> cb{};
    std::vector<std::uint8_t> cr{};
    deinterleave_nv12_chroma(img, width, height, cb, cr);
    check_cuda(cudaMalloc(&img.d_luma, luma_bytes), "cudaMalloc luma");
    check_cuda(cudaMalloc(&img.d_cb, chroma_bytes), "cudaMalloc cb");
    check_cuda(cudaMalloc(&img.d_cr, chroma_bytes), "cudaMalloc cr");
    check_cuda(cudaMemcpy(img.d_luma, img.host.data(), luma_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy luma");
    check_cuda(cudaMemcpy(img.d_cb, cb.data(), chroma_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy cb");
    check_cuda(cudaMemcpy(img.d_cr, cr.data(), chroma_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy cr");
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
    if (img.d_cb != nullptr) {
        cudaFree(img.d_cb);
        img.d_cb = nullptr;
    }
    if (img.d_cr != nullptr) {
        cudaFree(img.d_cr);
        img.d_cr = nullptr;
    }
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

// RAII handle bundle for nvJPEG. The encoder state and params are reused across
// iterations, matching the steady-state usage pattern for nvJPEG benchmarks.
struct NvjpegSession {
    nvjpegHandle_t handle{nullptr};
    nvjpegEncoderState_t state{nullptr};
    nvjpegEncoderParams_t params{nullptr};
    nvjpegImage_t image{};
    std::vector<std::uint8_t> output;

    explicit NvjpegSession(int quality, std::uint32_t width, std::uint32_t height) {
        check_nvjpeg(nvjpegCreateSimple(&handle), "CreateSimple");
        check_nvjpeg(nvjpegEncoderStateCreate(handle, &state, nullptr), "EncoderStateCreate");
        check_nvjpeg(nvjpegEncoderParamsCreate(handle, &params, nullptr), "EncoderParamsCreate");
        check_nvjpeg(nvjpegEncoderParamsSetQuality(params, quality, nullptr), "SetQuality");
        check_nvjpeg(nvjpegEncoderParamsSetSamplingFactors(params, NVJPEG_CSS_420, nullptr),
                     "SetSamplingFactors");
        std::size_t max_size{0};
        check_nvjpeg(nvjpegEncodeGetBufferSize(handle, params, static_cast<int>(width),
                                               static_cast<int>(height), &max_size),
                     "EncodeGetBufferSize");
        output.resize(max_size);
        for (std::size_t i{0}; i < NVJPEG_MAX_COMPONENT; ++i) {
            image.channel[i] = nullptr;
            image.pitch[i] = 0;
        }
    }

    ~NvjpegSession() {
        if (params != nullptr) {
            nvjpegEncoderParamsDestroy(params);
        }
        if (state != nullptr) {
            nvjpegEncoderStateDestroy(state);
        }
        if (handle != nullptr) {
            nvjpegDestroy(handle);
        }
    }

    NvjpegSession(const NvjpegSession&) = delete;
    NvjpegSession& operator=(const NvjpegSession&) = delete;
};

void print_header(const Args& a, std::size_t image_count) {
    std::printf("encode_bench\n");
    std::printf("  codec:      %s\n", a.codec.c_str());
    std::printf("  dims:       %ux%u\n", a.width, a.height);
    std::printf("  images:     %zu\n", image_count);
    std::printf("  iterations: %d\n", a.iterations);
    std::printf("  warmup:     %d\n", a.warmup);
    if (a.codec == "cujpegxl") {
        std::printf("  distance:   %.2f\n", a.distance);
    } else {
        std::printf("  quality:    %d\n", a.quality);
    }
    std::printf("  device:     %d\n", a.device);
    if (a.codec == "cujpegxl") {
        std::printf("  pipeline:   %d\n", a.pipeline_depth);
    }
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

}  // namespace

int main(int argc, char** argv) {
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
    if (args.pipeline_depth < 1) {
        die("--pipeline-depth must be >= 1");
    }
    if (args.codec != "cujpegxl" && args.pipeline_depth != 1) {
        die("--pipeline-depth is only supported with --codec=cujpegxl");
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
            upload_planar_yuv(img, args.width, args.height);
        }
        images.push_back(std::move(img));
    }

    print_header(args, images.size());

    const cujpegxl::bitstream::QuantParams qp{
        args.codec == "cujpegxl" ? cujpegxl::quant_params_for_distance(args.distance)
                                 : cujpegxl::bitstream::QuantParams{}};
    std::vector<std::uint8_t> cujpegxl_out{};
    std::vector<cujpegxl::StageTiming> cujpegxl_stages{};
    std::unique_ptr<cujpegxl::EncoderSession> cujpegxl_session{};
    if (args.codec == "cujpegxl") {
        cujpegxl_session = cujpegxl::EncoderSession::create(
            cujpegxl::EncoderConfig{.device_ordinal = args.device,
                                    .max_width = args.width,
                                    .max_height = args.height,
                                    .pipeline_depth = static_cast<std::size_t>(args.pipeline_depth),
                                    .pipeline = cujpegxl::EncoderPipeline::DCT8});
        if (cujpegxl_session == nullptr) {
            die("failed to create cujpegxl encoder session");
        }
    }

    std::unique_ptr<NvjpegSession> nvjpeg{};
    if (args.codec == "nvjpeg") {
        nvjpeg.reset(new NvjpegSession(args.quality, args.width, args.height));
    }

    auto encode_cujpegxl = [&](const BenchImage& img) -> double {
        const auto t0{std::chrono::steady_clock::now()};
        cujpegxl::EncodedFrameFuture future{};
        const cujpegxl::EncoderInput input{.luma = img.d_luma,
                                           .luma_pitch = args.width,
                                           .chroma = img.d_chroma,
                                           .chroma_pitch = args.width,
                                           .width = args.width,
                                           .height = args.height,
                                           .distance = args.distance,
                                           .quant_params = qp,
                                           .sequence = 0,
                                           .collect_stats = args.profile};
        cujpegxl::EncodedFrame output{};
        const bool ok{cujpegxl_session->encode(input, future) && future.get(output)};
        const auto t1{std::chrono::steady_clock::now()};
        if (!ok) {
            die("cujpegxl encode failed");
        }
        cujpegxl_out = std::move(output.bytes);
        cujpegxl_stages.insert(cujpegxl_stages.end(), output.stats.begin(), output.stats.end());
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    };

    auto encode_nvjpeg = [&](const BenchImage& img) -> double {
        const std::uint32_t chroma_w{args.width / 2};
        nvjpeg->image.channel[0] = img.d_luma;
        nvjpeg->image.pitch[0] = args.width;
        nvjpeg->image.channel[1] = img.d_cb;
        nvjpeg->image.pitch[1] = chroma_w;
        nvjpeg->image.channel[2] = img.d_cr;
        nvjpeg->image.pitch[2] = chroma_w;
        const auto t0{std::chrono::steady_clock::now()};
        check_nvjpeg(nvjpegEncodeYUV(nvjpeg->handle, nvjpeg->state, nvjpeg->params, &nvjpeg->image,
                                     NVJPEG_CSS_420, static_cast<int>(args.width),
                                     static_cast<int>(args.height), nullptr),
                     "EncodeYUV");
        std::size_t length{nvjpeg->output.size()};
        check_nvjpeg(nvjpegEncodeRetrieveBitstream(nvjpeg->handle, nvjpeg->state,
                                                   nvjpeg->output.data(), &length, nullptr),
                     "RetrieveBitstream");
        check_cuda(cudaStreamSynchronize(nullptr), "stream sync");
        const auto t1{std::chrono::steady_clock::now()};
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    };

    auto encode_one = [&](const BenchImage& img) -> double {
        const char* range_name{args.codec == "cujpegxl" ? "cujpegxl_encode" : "nvjpeg_encode"};
        nvtxRangePushA(range_name);
        const double us{args.codec == "cujpegxl" ? encode_cujpegxl(img) : encode_nvjpeg(img)};
        nvtxRangePop();
        return us;
    };

    // Warmup: run `warmup` encode iterations (alternating across images). The
    // first encode after handle/session creation pays one-time CUDA context
    // init, kernel JIT and nvjpeg state setup; one warmup encode is enough to
    // reach steady state regardless of the image count.
    for (int w{0}; w < args.warmup; ++w) {
        encode_one(images[static_cast<std::size_t>(w) % images.size()]);
    }
    check_cuda(cudaDeviceSynchronize(), "post-warmup sync");

    // Timed loop: alternate across images so cache effects are not correlated
    // with a single input.
    std::vector<double> times_us{};
    times_us.reserve(static_cast<std::size_t>(args.iterations));
    if (args.codec != "cujpegxl" || args.pipeline_depth == 1) {
        for (int i{0}; i < args.iterations; ++i) {
            const BenchImage& img{images[static_cast<std::size_t>(i) % images.size()]};
            check_cuda(cudaDeviceSynchronize(), "pre-iter sync");
            const double us{encode_one(img)};
            check_cuda(cudaDeviceSynchronize(), "post-iter sync");
            times_us.push_back(us);
            std::fprintf(stderr, "[%s] iter %d/%d: %.0f us\n", args.codec.c_str(), i + 1,
                         args.iterations, us);
        }
    } else {
        using TimePoint = std::chrono::steady_clock::time_point;
        struct Pending {
            cujpegxl::EncodedFrameFuture future{};
            TimePoint submitted{};
            std::uint64_t sequence{0};
        };
        std::deque<Pending> pending{};
        std::vector<TimePoint> completions{};
        completions.reserve(static_cast<std::size_t>(args.iterations));
        int submitted{0};
        int completed{0};
        const TimePoint batch_start{std::chrono::steady_clock::now()};
        while (completed < args.iterations) {
            while (submitted < args.iterations &&
                   pending.size() < static_cast<std::size_t>(args.pipeline_depth)) {
                const BenchImage& img{images[static_cast<std::size_t>(submitted) % images.size()]};
                const cujpegxl::EncoderInput input{
                    .luma = img.d_luma,
                    .luma_pitch = args.width,
                    .chroma = img.d_chroma,
                    .chroma_pitch = args.width,
                    .width = args.width,
                    .height = args.height,
                    .distance = args.distance,
                    .quant_params = qp,
                    .sequence = static_cast<std::uint64_t>(submitted),
                    .collect_stats = args.profile};
                cujpegxl::EncodedFrameFuture future{};
                if (!cujpegxl_session->try_encode(input, future)) {
                    die("pipeline rejected a submission below its configured depth");
                }
                pending.push_back({std::move(future), std::chrono::steady_clock::now(),
                                   static_cast<std::uint64_t>(submitted)});
                ++submitted;
            }

            Pending current{std::move(pending.front())};
            pending.pop_front();
            cujpegxl::EncodedFrame output{};
            if (!current.future.get(output) || output.sequence != current.sequence) {
                die("pipelined cujpegxl encode failed or completed out of order");
            }
            const TimePoint completed_at{std::chrono::steady_clock::now()};
            const double latency_us{
                std::chrono::duration<double, std::micro>(completed_at - current.submitted)
                    .count()};
            times_us.push_back(latency_us);
            completions.push_back(completed_at);
            cujpegxl_out = std::move(output.bytes);
            cujpegxl_stages.insert(cujpegxl_stages.end(), output.stats.begin(), output.stats.end());
            ++completed;
        }
        const TimePoint batch_end{std::chrono::steady_clock::now()};
        const double batch_us{
            std::chrono::duration<double, std::micro>(batch_end - batch_start).count()};
        std::printf("\npipeline throughput:\n");
        std::printf("  batch:       %.0f us\n", batch_us);
        std::printf("  throughput:  %.2f fps\n", 1.0e6 * args.iterations / batch_us);
        const std::size_t depth{static_cast<std::size_t>(args.pipeline_depth)};
        if (completions.size() > 2 * depth) {
            const std::size_t first{depth - 1};
            const std::size_t last{completions.size() - depth};
            const double steady_us{
                std::chrono::duration<double, std::micro>(completions[last] - completions[first])
                    .count()};
            const std::size_t steady_frames{last - first};
            std::printf("  steady:      %.2f fps (%zu frames)\n", 1.0e6 * steady_frames / steady_us,
                        steady_frames);
        }
    }

    print_stats(times_us, args.pipeline_depth == 1 ? args.codec : "cujpegxl latency");

    if (args.profile && args.codec == "cujpegxl" && !cujpegxl_stages.empty()) {
        const std::size_t stages_per_call{3};
        const std::size_t offset{cujpegxl_stages.size() >= stages_per_call
                                     ? cujpegxl_stages.size() - stages_per_call
                                     : 0};
        std::printf("\ncujpegxl per-stage timings (last iteration):\n");
        for (std::size_t i{offset}; i < cujpegxl_stages.size(); ++i) {
            const cujpegxl::StageTiming& s{cujpegxl_stages[i]};
            std::printf("  %-10s gpu=%9.0f us  cpu=%7.0f us  bytes=%zu\n", s.name, s.gpu_us,
                        s.cpu_us, s.bytes_moved);
            for (const cujpegxl::PhaseTiming& phase : s.phases) {
                std::printf("    %-24s gpu=%9.0f us  cpu=%7.0f us\n", phase.name, phase.gpu_us,
                            phase.cpu_us);
            }
            for (const cujpegxl::ProfileMetric& metric : s.metrics) {
                std::printf("    %-24s %.0f\n", metric.name, metric.value);
            }
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
