// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Sub-stage timing harness. The production encode path emits only three lumped
// StageTiming records (frontend / entropy / assembly), which is too coarse to
// attribute a per-workstream runtime change. This harness calls the *public*
// sub-stage functions directly and wall-clock times each one; the production
// functions are sync-free, so `timed` issues its own cudaDeviceSynchronize after
// each sub-call, making the blocking wall time that sub-stage's GPU time (or,
// for the host sub-stages, their CPU time). It deliberately mirrors the
// FrameEncoder orchestration, serialized, so its internal sub-stages can be
// timed without instrumenting production files.
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
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "src/ac_context.h"
#include "src/bitstream/container.h"
#include "src/bitstream/frame_assembly.h"
#include "src/bitstream/token_stream.h"
#include "src/entropy.h"
#include "src/frame_encoder.h"
#include "src/frontend_fused.h"
#include "src/frontend_quantize.h"
#include "src/prefix_build.h"
#include "src/quant_calibration.h"
#include "src/vardct_layout.h"
#include "src/worker_pool.h"

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
T* pinned_alloc(std::size_t count) {
    void* p{nullptr};
    cudaHostAlloc(&p, count * sizeof(T), cudaHostAllocDefault);
    return static_cast<T*>(p);
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
    std::size_t threads{cujpegxl::default_worker_threads()};
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
        } else if (a.rfind("--threads=", 0) == 0) {
            args.threads = static_cast<std::size_t>(std::atoi(a.c_str() + 10));
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
                     "[--distance=F] [--threads=N] <file.nv12> [...]\n");
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

    std::uint8_t* d_fb_scratch{
        device_alloc<std::uint8_t>(dc_first_blocks_scratch_bytes(width, height))};
    std::size_t* d_fb_pos{device_alloc<std::size_t>(num_dc * DC_FIRST_BLOCK_STRIDE)};
    std::size_t* d_fbc{device_alloc<std::size_t>(num_dc)};

    const SectionCapacities caps{section_capacities(width, height)};
    constexpr std::size_t HIST_WORDS{AC_NUM_CLUSTERS * AC_HISTOGRAM_SIZE};
    const std::size_t layout_words{3 * num_ac + 2};
    std::uint64_t* d_item_sizes{device_alloc<std::uint64_t>(3 * blocks)};
    std::uint8_t* d_layout_scratch{
        device_alloc<std::uint8_t>(ac_token_layout_scratch_bytes(width, height))};
    std::uint64_t* d_chunk_base{device_alloc<std::uint64_t>(ac_token_chunks(width, height))};
    std::uint32_t* d_layout{device_alloc<std::uint32_t>(layout_words)};
    std::uint16_t* d_entries{device_alloc<std::uint16_t>(ac_max_tokens(width, height))};
    std::uint64_t* d_raw{device_alloc<std::uint64_t>(ac_max_raw_words(width, height))};
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
    std::uint8_t* d_dc_body{device_alloc<std::uint8_t>(caps.dc_groups)};
    std::uint32_t* d_dc_sizes{device_alloc<std::uint32_t>(num_dc)};
    std::uint32_t* d_dc_offsets{device_alloc<std::uint32_t>(num_dc)};
    std::uint8_t* d_dc_scratch{device_alloc<std::uint8_t>(dc_encode_scratch_bytes(width, height))};

    std::uint32_t* h_ac_meta{pinned_alloc<std::uint32_t>(HIST_WORDS + layout_words)};
    std::uint16_t* h_entries{pinned_alloc<std::uint16_t>(ac_max_tokens(width, height))};
    std::uint64_t* h_raw{pinned_alloc<std::uint64_t>(ac_max_raw_words(width, height) + 1)};
    std::uint32_t* h_dc_sizes{pinned_alloc<std::uint32_t>(num_dc)};
    std::uint8_t* h_dc_body{pinned_alloc<std::uint8_t>(caps.dc_groups)};
    const std::uint32_t* const token_offset{h_ac_meta + HIST_WORDS};
    const std::uint32_t* const raw_offset{token_offset + num_ac + 1};
    const std::uint32_t* const raw_bits{raw_offset + num_ac + 1};

    WorkerPool workers{args.threads};
    std::vector<std::uint8_t> file(max_encoded_size(width, height));

    Stage s_fe{"frontend_fused"};
    Stage s_qz{"quantize_residual"};
    Stage s_ach{"ac_build_histogram"};
    Stage s_acl{"ac_token_layout"};
    Stage s_acp{"ac_pack_tokens"};
    Stage s_act{"AC tokens D2H"};
    Stage s_dch{"dc_build_histograms"};
    Stage s_amh{"acmeta_build_histograms"};
    Stage s_fbc{"dc_first_blocks"};
    Stage s_dcb{"build_dc_group_blobs_device"};
    Stage s_dcg{"build_dc_global (CPU)"};
    Stage s_dce{"dc_encode_groups"};
    Stage s_dct{"DcGroups D2H"};
    Stage s_acg{"AcGlobal + ANS tables (CPU)"};
    Stage s_acc{"code_token_streams (CPU)"};
    Stage s_asm{"assembly (CPU)"};

    bitstream::TokenCode code{};
    std::vector<bitstream::TokenStream> streams(num_ac);
    std::vector<bitstream::CodedStream> coded(num_ac);
    std::vector<std::size_t> slot_end(num_ac);
    std::vector<std::size_t> order(num_ac);
    std::vector<std::uint64_t> slot_words{};

    for (int it{0}; it < args.warmup + args.iterations; ++it) {
        const bool record{it >= args.warmup};
        const std::vector<std::uint8_t>& frame{
            frames[static_cast<std::size_t>(it) % frames.size()]};
        cudaMemcpy(d_luma, frame.data(), ysize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_chroma, frame.data() + ysize, ysize / 2, cudaMemcpyHostToDevice);

        // Front-end sub-stages, in FrameEncoder order.
        timed(s_fe, record, [&] {
            const ChromaTexture chroma{d_chroma, width, width, height, nullptr};
            encode_frontend(d_luma, width, chroma, width, height, args.distance, d_coeffs, d_acs,
                            d_qf, nullptr);
        });
        timed(s_qz, record, [&] {
            quantize_residual(d_coeffs, d_acs, d_qf, width, height, args.distance, d_mx, d_mb,
                              d_ac16, d_dc32, d_nz_exact, d_nz_grid, nullptr);
        });

        // Entropy sub-stages, serialized.
        timed(s_ach, record, [&] {
            ac_build_histogram(d_ac16, d_acs, d_nz_grid, d_nz_exact, width, height, d_ac_hist,
                               d_item_sizes, nullptr);
        });
        timed(s_acl, record, [&] {
            ac_token_layout(d_item_sizes, width, height, d_layout_scratch, d_chunk_base, d_layout,
                            nullptr);
        });
        timed(s_acp, record, [&] {
            ac_pack_tokens(d_ac16, d_acs, d_nz_grid, d_nz_exact, width, height, d_item_sizes,
                           d_chunk_base, d_layout, 0, num_ac, d_entries, d_raw, nullptr);
        });
        timed(s_act, record, [&] {
            cudaMemcpy(h_ac_meta, d_ac_hist, HIST_WORDS * sizeof(std::uint32_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_ac_meta + HIST_WORDS, d_layout, layout_words * sizeof(std::uint32_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_entries, d_entries, token_offset[num_ac] * sizeof(std::uint16_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(h_raw, d_raw, raw_offset[num_ac] * sizeof(std::uint64_t),
                       cudaMemcpyDeviceToHost);
        });
        timed(s_dch, record,
              [&] { dc_build_histograms(d_dc32, width, height, d_dc_hist, nullptr); });
        timed(s_amh, record, [&] {
            acmeta_build_histograms(d_qf, d_acs, d_mx, d_mb, width, height, d_am_hist, nullptr);
        });
        timed(s_fbc, record, [&] {
            dc_first_blocks(d_acs, width, height, d_fb_scratch, d_fb_pos, d_fbc, nullptr);
        });
        timed(s_dcb, record, [&] {
            build_dc_group_blobs_device(width, height, d_dc_hist, d_am_hist, d_fbc, d_dc_depth,
                                        d_dc_bits, d_am_depth, d_am_bits, d_pre, d_pre_off,
                                        d_pre_bits, d_mid, d_mid_off, d_mid_bits, nullptr);
        });
        std::vector<std::uint8_t> dc_global;
        timed(s_dcg, record, [&] { dc_global = bitstream::build_dc_global(qp); });
        timed(s_dce, record, [&] {
            dc_encode_groups(d_dc32, width, height, d_qf, d_acs, d_mx, d_mb, d_fb_pos, d_fbc,
                             d_dc_depth, d_dc_bits, d_am_depth, d_am_bits, d_pre, d_pre_off,
                             d_pre_bits, d_mid, d_mid_off, d_mid_bits, d_dc_scratch, d_dc_body,
                             caps.dc_groups, d_dc_sizes, d_dc_offsets, nullptr);
        });
        std::size_t dc_total{0};
        timed(s_dct, record, [&] {
            cudaMemcpy(h_dc_sizes, d_dc_sizes, num_dc * sizeof(std::uint32_t),
                       cudaMemcpyDeviceToHost);
            dc_total = std::accumulate(h_dc_sizes, h_dc_sizes + num_dc, std::size_t{0});
            cudaMemcpy(h_dc_body, d_dc_body, dc_total, cudaMemcpyDeviceToHost);
        });

        std::vector<std::uint8_t> ac_global;
        timed(s_acg, record, [&] {
            bitstream::AcGlobalBuilder tables{h_ac_meta, num_ac, code};
            std::atomic<std::size_t> next{0};
            workers.run([&](std::size_t) {
                for (std::size_t c{next.fetch_add(1)}; c < AC_NUM_CLUSTERS; c = next.fetch_add(1)) {
                    tables.build_cluster(c);
                }
            });
            ac_global = tables.finish();
        });

        timed(s_acc, record, [&] {
            std::size_t slot_total{0};
            for (std::size_t g{0}; g < num_ac; ++g) {
                streams[g] = bitstream::TokenStream{h_entries + token_offset[g],
                                                    token_offset[g + 1] - token_offset[g],
                                                    h_raw + raw_offset[g], raw_bits[g]};
                slot_total += bitstream::coded_stream_capacity_words(streams[g]);
                slot_end[g] = slot_total;
            }
            if (slot_words.size() < slot_total) {
                slot_words.resize(slot_total);
            }
            std::iota(order.begin(), order.end(), std::size_t{0});
            std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                return streams[a].num_tokens > streams[b].num_tokens;
            });
            bitstream::StreamQueue queue{};
            queue.streams = streams.data();
            queue.num_streams = num_ac;
            queue.order = order.data();
            queue.slot_words = slot_words.data();
            queue.slot_end = slot_end.data();
            queue.coded = coded.data();
            queue.ready.store(num_ac);
            workers.run([&](std::size_t) { bitstream::code_token_streams(code, queue); });
        });

        timed(s_asm, record, [&] {
            std::vector<std::uint32_t> section_sizes{};
            section_sizes.push_back(static_cast<std::uint32_t>(dc_global.size()));
            section_sizes.insert(section_sizes.end(), h_dc_sizes, h_dc_sizes + num_dc);
            section_sizes.push_back(static_cast<std::uint32_t>(ac_global.size()));
            std::size_t ac_total{0};
            for (const bitstream::CodedStream& c : coded) {
                section_sizes.push_back(static_cast<std::uint32_t>(c.size_bytes()));
                ac_total += c.size_bytes();
            }
            const std::vector<std::uint8_t> head{bitstream::build_codestream_head(
                static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height),
                section_sizes)};
            const std::size_t codestream_size{head.size() + dc_global.size() + dc_total +
                                              ac_global.size() + ac_total};
            const std::vector<std::uint8_t> framing{bitstream::container_framing(codestream_size)};
            std::uint8_t* at{std::copy(framing.begin(), framing.end(), file.data())};
            at = std::copy(head.begin(), head.end(), at);
            at = std::copy(dc_global.begin(), dc_global.end(), at);
            std::memcpy(at, h_dc_body, dc_total);
            at = std::copy(ac_global.begin(), ac_global.end(), at + dc_total);
            std::vector<std::size_t> offsets(num_ac);
            for (std::size_t g{0}; g < num_ac; ++g) {
                offsets[g] = static_cast<std::size_t>(at - file.data());
                at += coded[g].size_bytes();
            }
            std::atomic<std::size_t> next{0};
            workers.run([&](std::size_t) {
                for (std::size_t g{next.fetch_add(1)}; g < num_ac; g = next.fetch_add(1)) {
                    bitstream::place_coded_stream(coded[g], file.data() + offsets[g]);
                }
            });
        });
    }

    const Stage* stages[]{&s_fe,  &s_qz,  &s_ach, &s_acl, &s_acp, &s_act, &s_dch, &s_amh,
                          &s_fbc, &s_dcb, &s_dcg, &s_dce, &s_dct, &s_acg, &s_acc, &s_asm};
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
