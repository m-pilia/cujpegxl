// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Sub-stage timing harness. The production encode path emits only three lumped
// StageTiming records (frontend / entropy / assembly), which is too coarse to
// attribute a per-workstream runtime change. This harness calls the *public*
// sub-stage functions directly and wall-clock times each one; the production
// functions are sync-free, so `timed` issues its own cudaDeviceSynchronize after
// each sub-call, making the blocking wall time that sub-stage's GPU time. It
// deliberately mirrors the encode_nv12 + encode_frame orchestration so
// the internal sub-stages of encode_frame can be timed without instrumenting
// production files.
//
// Numbers are for *relative* attribution: scratch is re-allocated each iteration
// (no stream-ordered mempool), so absolute per-resolution sums run a little above
// the mempool pipeline totals reported by //tools/benchmark:benchmark --profile,
// which remains the canonical top-line metric.
//
// Usage:
//   bazelisk run //tools/stage_profiling:stage_profiler -- \
//       --dims=3840x2160 --iterations=20 --warmup=5 --distance=1.0 \
//       frame0.nv12 [frame1.nv12 ...]
//
// Generate representative NV12 inputs from the checked-in 4K corpus with
// //tools/corpus:corpus_prep.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "src/ac_context.h"
#include "src/bitstream/frame_assembly.h"
#include "src/entropy.h"
#include "src/frontend_fused.h"
#include "src/frontend_quantize.h"
#include "src/quant_calibration.h"
#include "src/vardct_layout.h"

namespace {

using Clock = std::chrono::steady_clock;

double us_since(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

struct Stage {
    const char* name;
    double us{0.0};
};

template <typename Fn>
void timed(Stage& stage, bool record, Fn&& fn) {
    const Clock::time_point start{Clock::now()};
    fn();
    cudaDeviceSynchronize();
    if (record) {
        stage.us += us_since(start);
    }
}

template <typename T>
T* device_alloc(std::size_t count) {
    void* p{nullptr};
    cudaMalloc(&p, count * sizeof(T));
    return static_cast<T*>(p);
}

template <typename T>
T* upload(const std::vector<T>& host) {
    if (host.empty()) {
        return nullptr;
    }
    T* device{device_alloc<T>(host.size())};
    cudaMemcpy(device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
    return device;
}

std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream in{path, std::ios::binary};
    return std::vector<std::uint8_t>{std::istreambuf_iterator<char>(in),
                                     std::istreambuf_iterator<char>()};
}

struct Args {
    std::size_t width{1920};
    std::size_t height{1080};
    int iterations{20};
    int warmup{5};
    float distance{1.0f};
    std::vector<std::string> inputs;
};

Args parse_args(int argc, char** argv) {
    Args args{};
    for (int i{1}; i < argc; ++i) {
        const std::string a{argv[i]};
        if (a.rfind("--dims=", 0) == 0) {
            std::sscanf(a.c_str() + 7, "%zux%zu", &args.width, &args.height);
        } else if (a.rfind("--iterations=", 0) == 0) {
            args.iterations = std::atoi(a.c_str() + 13);
        } else if (a.rfind("--warmup=", 0) == 0) {
            args.warmup = std::atoi(a.c_str() + 9);
        } else if (a.rfind("--distance=", 0) == 0) {
            args.distance = static_cast<float>(std::atof(a.c_str() + 11));
        } else {
            args.inputs.push_back(a);
        }
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    using namespace cujpegxl;
    const Args args{parse_args(argc, argv)};
    if (args.inputs.empty()) {
        std::fprintf(stderr,
                     "usage: stage_profiler --dims=WxH [--iterations=N] [--warmup=N] "
                     "[--distance=F] <file.nv12> [...]\n");
        return 2;
    }

    const std::size_t width{args.width};
    const std::size_t height{args.height};
    const std::size_t ysize{width * height};

    std::vector<std::vector<std::uint8_t>> frames;
    for (const std::string& path : args.inputs) {
        std::vector<std::uint8_t> bytes{read_file(path)};
        if (bytes.size() != ysize * 3 / 2) {
            std::fprintf(stderr, "%s: expected %zu NV12 bytes, got %zu\n", path.c_str(),
                         ysize * 3 / 2, bytes.size());
            return 1;
        }
        frames.push_back(std::move(bytes));
    }
    std::printf("frames=%zu dims=%zux%zu distance=%.2f iters=%d\n", frames.size(), width, height,
                args.distance, args.iterations);

    cudaSetDevice(0);
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t blocks{bw * bh};
    const std::size_t cmw{(bw + 7) / 8};
    const std::size_t cmh{(bh + 7) / 8};
    const std::size_t num_ac{bitstream::ac_group_count(width, height)};
    const std::size_t num_dc{bitstream::dc_group_count(width, height)};

    std::uint8_t* d_luma{device_alloc<std::uint8_t>(ysize)};
    std::uint8_t* d_chroma{device_alloc<std::uint8_t>(ysize / 2)};
    __half* d_coeffs{device_alloc<__half>(3 * blocks * COEFFS_PER_BLOCK)};
    std::int8_t* d_acs{device_alloc<std::int8_t>(blocks)};
    std::int8_t* d_mx{device_alloc<std::int8_t>(cmw * cmh)};
    std::int8_t* d_mb{device_alloc<std::int8_t>(cmw * cmh)};
    std::int32_t* d_qf{device_alloc<std::int32_t>(blocks)};
    std::int16_t* d_ac16{device_alloc<std::int16_t>(3 * blocks * COEFFS_PER_BLOCK)};
    std::int32_t* d_dc32{device_alloc<std::int32_t>(3 * blocks)};
    std::uint32_t* d_ac_hist{device_alloc<std::uint32_t>(AC_NUM_CLUSTERS * AC_HISTOGRAM_SIZE)};
    std::uint32_t* d_dc_hist{device_alloc<std::uint32_t>(num_dc * AC_HISTOGRAM_SIZE)};
    std::uint32_t* d_am_hist{device_alloc<std::uint32_t>(num_dc * AC_HISTOGRAM_SIZE)};

    const QuantCalibration cal{calibrate_quant(args.distance)};
    const bitstream::QuantParams qp{cal.global_scale, cal.quant_dc};

    Stage s_fe{"frontend_fused"};
    Stage s_cfl{"estimate_cfl_covered"};
    Stage s_qz{"quantize_residual"};
    Stage s_ach{"ac_build_histogram"};
    Stage s_dch{"dc_build_histograms"};
    Stage s_amh{"acmeta_build_histograms"};
    Stage s_d2h{"histogram D2H + acs D2H"};
    Stage s_host{"host prefix build (CPU)"};
    Stage s_up{"upload depth/bits"};
    Stage s_ace{"ac_encode_groups"};
    Stage s_dce{"dc_encode_groups"};
    Stage s_asm{"assembly gather + D2H"};

    const std::size_t xdg{(bw + 255) / 256};

    for (int it{0}; it < args.warmup + args.iterations; ++it) {
        const bool record{it >= args.warmup};
        const std::vector<std::uint8_t>& frame{
            frames[static_cast<std::size_t>(it) % frames.size()]};
        cudaMemcpy(d_luma, frame.data(), ysize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_chroma, frame.data() + ysize, ysize / 2, cudaMemcpyHostToDevice);

        // Front-end sub-stages, in encode_nv12 order.
        timed(s_fe, record, [&] {
            encode_frontend(d_luma, width, d_chroma, width, width, height, args.distance, d_coeffs,
                            d_acs, d_qf);
        });
        timed(s_cfl, record,
              [&] { estimate_cfl_covered(d_coeffs, d_acs, width, height, d_mx, d_mb); });
        timed(s_qz, record, [&] {
            quantize_residual(d_coeffs, d_acs, d_mx, d_mb, d_qf, width, height, args.distance,
                              d_ac16, d_dc32);
        });

        // Entropy histograms.
        timed(s_ach, record, [&] { ac_build_histogram(d_ac16, d_acs, width, height, d_ac_hist); });
        timed(s_dch, record, [&] { dc_build_histograms(d_dc32, width, height, d_dc_hist); });
        timed(s_amh, record,
              [&] { acmeta_build_histograms(d_qf, d_acs, d_mx, d_mb, width, height, d_am_hist); });

        std::vector<std::uint32_t> ac_hist(AC_NUM_CLUSTERS * AC_HISTOGRAM_SIZE);
        std::vector<std::uint32_t> dc_hist(num_dc * AC_HISTOGRAM_SIZE);
        std::vector<std::uint32_t> am_hist(num_dc * AC_HISTOGRAM_SIZE);
        std::vector<std::int8_t> acs_host(blocks, 8);
        timed(s_d2h, record, [&] {
            cudaMemcpy(ac_hist.data(), d_ac_hist, ac_hist.size() * sizeof(std::uint32_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(dc_hist.data(), d_dc_hist, dc_hist.size() * sizeof(std::uint32_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(am_hist.data(), d_am_hist, am_hist.size() * sizeof(std::uint32_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(acs_host.data(), d_acs, blocks, cudaMemcpyDeviceToHost);
        });

        std::vector<std::size_t> fbc(num_dc, 0);
        bitstream::AcGlobalResult global{};
        std::vector<std::uint8_t> dc_global;
        bitstream::DcGroupBlobs blobs{};
        timed(s_host, record, [&] {
            for (std::size_t g{0}; g < num_dc; ++g) {
                const std::size_t bx0{(g % xdg) * 256};
                const std::size_t by0{(g / xdg) * 256};
                const std::size_t dgw{std::min<std::size_t>(256, bw - bx0)};
                const std::size_t dgh{std::min<std::size_t>(256, bh - by0)};
                for (std::size_t by{by0}; by < by0 + dgh; ++by) {
                    for (std::size_t bx{bx0}; bx < bx0 + dgw; ++bx) {
                        if (acs_host[by * bw + bx] != ACS_COVERED) {
                            ++fbc[g];
                        }
                    }
                }
            }
            global = bitstream::build_ac_global(ac_hist.data(), num_ac);
            dc_global = bitstream::build_dc_global(qp);
            blobs = bitstream::build_dc_group_blobs(width, height, dc_hist.data(), am_hist.data(),
                                                    fbc.data());
        });

        std::uint8_t* d_ac_depth{nullptr};
        std::uint16_t* d_ac_bits{nullptr};
        std::uint8_t* d_dc_depth{nullptr};
        std::uint16_t* d_dc_bits{nullptr};
        std::uint8_t* d_am_depth{nullptr};
        std::uint16_t* d_am_bits{nullptr};
        std::uint8_t* d_pre{nullptr};
        std::uint32_t* d_pre_off{nullptr};
        std::uint32_t* d_pre_bits{nullptr};
        std::uint8_t* d_mid{nullptr};
        std::uint32_t* d_mid_off{nullptr};
        std::uint32_t* d_mid_bits{nullptr};
        timed(s_up, record, [&] {
            d_ac_depth = upload(global.depth);
            d_ac_bits = upload(global.bits);
            d_dc_depth = upload(blobs.dc_depth);
            d_dc_bits = upload(blobs.dc_bits);
            d_am_depth = upload(blobs.acmeta_depth);
            d_am_bits = upload(blobs.acmeta_bits);
            d_pre = upload(blobs.blob_pre);
            d_pre_off = upload(blobs.blob_pre_off);
            d_pre_bits = upload(blobs.blob_pre_bits);
            d_mid = upload(blobs.blob_mid);
            d_mid_off = upload(blobs.blob_mid_off);
            d_mid_bits = upload(blobs.blob_mid_bits);
        });

        const std::size_t ac_capacity{3 * bw * bh * 384 + num_ac * 64 + 4096};
        const std::size_t dc_capacity{48 * bw * bh + blobs.blob_pre.size() + blobs.blob_mid.size() +
                                      num_dc * 64 + 4096};
        std::uint8_t* d_ac_body{device_alloc<std::uint8_t>(ac_capacity)};
        std::uint8_t* d_dc_body{device_alloc<std::uint8_t>(dc_capacity)};
        std::uint32_t* d_ac_sizes{device_alloc<std::uint32_t>(num_ac)};
        std::uint32_t* d_ac_offsets{device_alloc<std::uint32_t>(num_ac)};
        std::uint32_t* d_dc_sizes{device_alloc<std::uint32_t>(num_dc)};
        std::uint32_t* d_dc_offsets{device_alloc<std::uint32_t>(num_dc)};

        std::size_t ac_total{0};
        std::size_t dc_total{0};
        timed(s_ace, record, [&] {
            ac_encode_groups(d_ac16, d_acs, width, height, d_ac_depth, d_ac_bits, d_ac_body,
                             ac_capacity, d_ac_sizes, d_ac_offsets, &ac_total);
        });
        timed(s_dce, record, [&] {
            dc_encode_groups(d_dc32, width, height, d_qf, d_acs, d_mx, d_mb, d_dc_depth, d_dc_bits,
                             d_am_depth, d_am_bits, d_pre, d_pre_off, d_pre_bits, d_mid, d_mid_off,
                             d_mid_bits, d_dc_body, dc_capacity, d_dc_sizes, d_dc_offsets,
                             &dc_total);
        });

        const std::size_t body_size{dc_global.size() + dc_total + global.section.size() + ac_total};
        std::uint8_t* d_body{device_alloc<std::uint8_t>(body_size)};
        std::vector<std::uint32_t> ac_sizes(num_ac);
        std::vector<std::uint32_t> dc_sizes(num_dc);
        std::vector<std::uint8_t> out_body(body_size);
        timed(s_asm, record, [&] {
            cudaMemcpy(ac_sizes.data(), d_ac_sizes, num_ac * sizeof(std::uint32_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(dc_sizes.data(), d_dc_sizes, num_dc * sizeof(std::uint32_t),
                       cudaMemcpyDeviceToHost);
            std::size_t at{0};
            cudaMemcpy(d_body + at, dc_global.data(), dc_global.size(), cudaMemcpyHostToDevice);
            at += dc_global.size();
            cudaMemcpy(d_body + at, d_dc_body, dc_total, cudaMemcpyDeviceToDevice);
            at += dc_total;
            cudaMemcpy(d_body + at, global.section.data(), global.section.size(),
                       cudaMemcpyHostToDevice);
            at += global.section.size();
            cudaMemcpy(d_body + at, d_ac_body, ac_total, cudaMemcpyDeviceToDevice);
            cudaMemcpy(out_body.data(), d_body, body_size, cudaMemcpyDeviceToHost);
        });

        for (void* p : {static_cast<void*>(d_ac_body), static_cast<void*>(d_dc_body),
                        static_cast<void*>(d_ac_sizes), static_cast<void*>(d_ac_offsets),
                        static_cast<void*>(d_dc_sizes), static_cast<void*>(d_dc_offsets),
                        static_cast<void*>(d_body), static_cast<void*>(d_ac_depth),
                        static_cast<void*>(d_ac_bits), static_cast<void*>(d_dc_depth),
                        static_cast<void*>(d_dc_bits), static_cast<void*>(d_am_depth),
                        static_cast<void*>(d_am_bits), static_cast<void*>(d_pre),
                        static_cast<void*>(d_pre_off), static_cast<void*>(d_pre_bits),
                        static_cast<void*>(d_mid), static_cast<void*>(d_mid_off),
                        static_cast<void*>(d_mid_bits)}) {
            cudaFree(p);
        }
    }

    const Stage* stages[]{&s_fe,  &s_cfl,  &s_qz, &s_ach, &s_dch, &s_amh,
                          &s_d2h, &s_host, &s_up, &s_ace, &s_dce, &s_asm};
    std::printf("\nper-substage mean over %d iters (us):\n", args.iterations);
    double sum{0.0};
    for (const Stage* stage : stages) {
        const double mean{stage->us / args.iterations};
        std::printf("  %-30s %9.1f\n", stage->name, mean);
        sum += mean;
    }
    std::printf("  %-30s %9.1f\n", "[sum]", sum);
    return 0;
}
