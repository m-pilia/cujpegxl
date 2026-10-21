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
// Numbers are for *relative* attribution: scratch is allocated once up front and
// every sub-stage is followed by a device sync, so the sum differs from the
// pipelined totals reported by //tools/benchmark:benchmark --profile, which
// remains the canonical top-line metric.
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
#include "src/bitstream/container.h"
#include "src/bitstream/frame_assembly.h"
#include "src/entropy.h"
#include "src/frame_encoder.h"
#include "src/frontend_fused.h"
#include "src/frontend_quantize.h"
#include "src/prefix_build.h"
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
    // Mirrors frame_encoder.cu: without it every sync returns the entropy wrappers'
    // cudaMallocAsync scratch to the driver and the next frame re-allocates it.
    cudaMemPool_t pool{};
    cudaDeviceGetDefaultMemPool(&pool, 0);
    std::uint64_t threshold{UINT64_MAX};
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
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
    std::int32_t* d_nz_grid{device_alloc<std::int32_t>(3 * blocks)};
    std::int32_t* d_nz_exact{device_alloc<std::int32_t>(3 * blocks)};
    std::uint32_t* d_ac_hist{device_alloc<std::uint32_t>(AC_NUM_CLUSTERS * AC_HISTOGRAM_SIZE)};
    std::uint32_t* d_dc_hist{device_alloc<std::uint32_t>(num_dc * AC_HISTOGRAM_SIZE)};
    std::uint32_t* d_am_hist{device_alloc<std::uint32_t>(num_dc * AC_HISTOGRAM_SIZE)};

    const QuantCalibration cal{calibrate_quant(args.distance)};
    const bitstream::QuantParams qp{cal.global_scale, cal.quant_dc};

    std::size_t* d_fb_pos{device_alloc<std::size_t>(num_dc * DC_FIRST_BLOCK_STRIDE)};
    std::size_t* d_fbc{device_alloc<std::size_t>(num_dc)};

    const SectionCapacities caps{section_capacities(width, height)};
    const bitstream::AcContextMapBlob ctxmap{bitstream::build_ac_context_map_blob()};
    std::uint8_t* d_ctxmap{upload(ctxmap.bytes)};
    std::uint8_t* d_ac_depth{
        device_alloc<std::uint8_t>(AC_NUM_CLUSTERS * bitstream::HISTOGRAM_STRIDE)};
    std::uint16_t* d_ac_bits{
        device_alloc<std::uint16_t>(AC_NUM_CLUSTERS * bitstream::HISTOGRAM_STRIDE)};
    std::uint8_t* d_ac_section{device_alloc<std::uint8_t>(caps.ac_global)};
    std::uint32_t* d_section_sizes{device_alloc<std::uint32_t>(num_dc + 1 + num_ac)};
    std::uint32_t* d_dc_sizes{d_section_sizes};
    std::uint32_t* d_ac_global_size{d_section_sizes + num_dc};
    std::uint32_t* d_ac_sizes{d_section_sizes + num_dc + 1};
    std::uint8_t* d_dc_depth{device_alloc<std::uint8_t>(num_dc * bitstream::HISTOGRAM_STRIDE)};
    std::uint16_t* d_dc_bits{device_alloc<std::uint16_t>(num_dc * bitstream::HISTOGRAM_STRIDE)};
    std::uint8_t* d_am_depth{device_alloc<std::uint8_t>(num_dc * bitstream::HISTOGRAM_STRIDE)};
    std::uint16_t* d_am_bits{device_alloc<std::uint16_t>(num_dc * bitstream::HISTOGRAM_STRIDE)};
    std::uint8_t* d_pre{device_alloc<std::uint8_t>(num_dc * DC_BLOB_STRIDE)};
    std::uint32_t* d_pre_off{device_alloc<std::uint32_t>(num_dc)};
    std::uint32_t* d_pre_bits{device_alloc<std::uint32_t>(num_dc)};
    std::uint8_t* d_mid{device_alloc<std::uint8_t>(num_dc * DC_BLOB_STRIDE)};
    std::uint32_t* d_mid_off{device_alloc<std::uint32_t>(num_dc)};
    std::uint32_t* d_mid_bits{device_alloc<std::uint32_t>(num_dc)};

    std::uint8_t* d_ac_body{device_alloc<std::uint8_t>(caps.ac_groups)};
    std::uint8_t* d_dc_body{device_alloc<std::uint8_t>(caps.dc_groups)};
    std::uint32_t* d_ac_offsets{device_alloc<std::uint32_t>(num_ac)};
    std::uint32_t* d_dc_offsets{device_alloc<std::uint32_t>(num_dc)};
    std::uint8_t* d_dc_scratch{device_alloc<std::uint8_t>(dc_encode_scratch_bytes(width, height))};
    std::uint8_t* d_body{
        device_alloc<std::uint8_t>(caps.ac_groups + caps.dc_groups + caps.ac_global + 4096)};
    std::vector<std::uint8_t> file(max_encoded_size(width, height));

    Stage s_fe{"frontend_fused"};
    Stage s_cfl{"estimate_cfl_covered"};
    Stage s_qz{"quantize_residual"};
    Stage s_nz{"ac_nonzero_grids"};
    Stage s_ach{"ac_build_histogram"};
    Stage s_dch{"dc_build_histograms"};
    Stage s_amh{"acmeta_build_histograms"};
    Stage s_fbc{"dc_first_blocks"};
    Stage s_acg{"build_ac_global_device"};
    Stage s_dcb{"build_dc_group_blobs_device"};
    Stage s_dcg{"build_dc_global (CPU)"};
    Stage s_ace{"ac_encode_groups"};
    Stage s_dce{"dc_encode_groups"};
    Stage s_sz{"section sizes D2H"};
    Stage s_asm{"assembly head + gather + D2H"};

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

        // Entropy sub-stages, in encode_frame order.
        timed(s_nz, record,
              [&] { ac_nonzero_grids(d_ac16, d_acs, width, height, d_nz_grid, d_nz_exact); });
        timed(s_ach, record, [&] {
            ac_build_histogram(d_ac16, d_acs, d_nz_grid, d_nz_exact, width, height, d_ac_hist);
        });
        timed(s_dch, record, [&] { dc_build_histograms(d_dc32, width, height, d_dc_hist); });
        timed(s_amh, record,
              [&] { acmeta_build_histograms(d_qf, d_acs, d_mx, d_mb, width, height, d_am_hist); });
        timed(s_fbc, record, [&] { dc_first_blocks(d_acs, width, height, d_fb_pos, d_fbc); });
        timed(s_acg, record, [&] {
            build_ac_global_device(d_ac_hist, num_ac, d_ctxmap, ctxmap.nbits, d_ac_depth, d_ac_bits,
                                   d_ac_section, caps.ac_global, d_ac_global_size);
        });
        timed(s_dcb, record, [&] {
            build_dc_group_blobs_device(width, height, d_dc_hist, d_am_hist, d_fbc, d_dc_depth,
                                        d_dc_bits, d_am_depth, d_am_bits, d_pre, d_pre_off,
                                        d_pre_bits, d_mid, d_mid_off, d_mid_bits);
        });
        std::vector<std::uint8_t> dc_global;
        timed(s_dcg, record, [&] { dc_global = bitstream::build_dc_global(qp); });

        std::size_t ac_total{0};
        std::size_t dc_total{0};
        timed(s_ace, record, [&] {
            ac_encode_groups(d_ac16, d_acs, d_nz_grid, d_nz_exact, width, height, d_ac_depth,
                             d_ac_bits, d_ac_body, caps.ac_groups, d_ac_sizes, d_ac_offsets,
                             &ac_total);
        });
        timed(s_dce, record, [&] {
            dc_encode_groups(d_dc32, width, height, d_qf, d_acs, d_mx, d_mb, d_fb_pos, d_fbc,
                             d_dc_depth, d_dc_bits, d_am_depth, d_am_bits, d_pre, d_pre_off,
                             d_pre_bits, d_mid, d_mid_off, d_mid_bits, d_dc_scratch, d_dc_body,
                             caps.dc_groups, d_dc_sizes, d_dc_offsets, &dc_total);
        });

        std::vector<std::uint32_t> section_sizes(1 + num_dc + 1 + num_ac);
        timed(s_sz, record, [&] {
            section_sizes[0] = static_cast<std::uint32_t>(dc_global.size());
            cudaMemcpy(section_sizes.data() + 1, d_section_sizes,
                       (num_dc + 1 + num_ac) * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
        });

        timed(s_asm, record, [&] {
            const std::uint32_t ac_global_size{section_sizes[1 + num_dc]};
            const std::vector<std::uint8_t> head{bitstream::build_codestream_head(
                static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height),
                section_sizes)};
            const std::size_t body_size{dc_global.size() + dc_total + ac_global_size + ac_total};
            std::size_t at{0};
            cudaMemcpyAsync(d_body + at, dc_global.data(), dc_global.size(), cudaMemcpyHostToDevice,
                            nullptr);
            at += dc_global.size();
            cudaMemcpyAsync(d_body + at, d_dc_body, dc_total, cudaMemcpyDeviceToDevice, nullptr);
            at += dc_total;
            cudaMemcpyAsync(d_body + at, d_ac_section, ac_global_size, cudaMemcpyDeviceToDevice,
                            nullptr);
            at += ac_global_size;
            cudaMemcpyAsync(d_body + at, d_ac_body, ac_total, cudaMemcpyDeviceToDevice, nullptr);
            const std::size_t codestream_size{head.size() + body_size};
            const std::vector<std::uint8_t> framing{bitstream::container_framing(codestream_size)};
            std::uint8_t* const head_dst{std::copy(framing.begin(), framing.end(), file.data())};
            std::uint8_t* const body_dst{std::copy(head.begin(), head.end(), head_dst)};
            cudaMemcpy(body_dst, d_body, body_size, cudaMemcpyDeviceToHost);
        });
    }

    const Stage* stages[]{&s_fe,  &s_cfl, &s_qz,  &s_nz,  &s_ach, &s_dch, &s_amh, &s_fbc,
                          &s_acg, &s_dcb, &s_dcg, &s_ace, &s_dce, &s_sz,  &s_asm};
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
