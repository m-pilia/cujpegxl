// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frame_encoder.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <thread>

#include <cuda_runtime.h>

#include <cuda_fp16.h>

#include "ac_context.h"
#include "entropy.h"
#include "frontend_fused.h"
#include "frontend_quantize.h"
#include "prefix_build.h"
#include "quant_calibration.h"
#include "src/bitstream/container.h"
#include "src/bitstream/frame_assembly.h"
#include "src/bitstream/token_stream.h"
#include "vardct_layout.h"
#include "worker_pool.h"

namespace cujpegxl {
namespace {

using Clock = std::chrono::steady_clock;

constexpr EncodeResult ENCODE_FAILED{EncodeStatus::FAILED, 0};

// The AC groups are packed and copied to the host in this many batches, so that
// the host codes a batch while the device packs and copies the next ones.
constexpr std::size_t AC_BATCHES{4};

double us_since(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

// Retain freed blocks in the device's default stream-ordered memory pool instead
// of returning them to the driver, so the per-frame scratch (the ~300 MB
// frontend planes and the entropy/assembly bodies) is reused across encodes
// rather than re-allocated every frame.
void retain_default_mempool(int device) {
    cudaMemPool_t pool{};
    if (cudaDeviceGetDefaultMemPool(&pool, device) != cudaSuccess) {
        return;
    }
    std::uint64_t threshold{UINT64_MAX};
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
}

// Forks `side` off the legacy default stream: work enqueued on `side` starts
// after the default stream's work enqueued so far, and join() makes the
// default stream wait for `side`'s work. Destruction joins as well, so on an
// early return the DeviceScope frees (ordered on the default stream) cannot
// overtake work still pending on `side`.
class StreamFork {
public:
    explicit StreamFork(cudaStream_t side) : side_{side} {
        forked_ = cudaEventCreateWithFlags(&event_, cudaEventDisableTiming) == cudaSuccess &&
                  cudaEventRecord(event_, 0) == cudaSuccess &&
                  cudaStreamWaitEvent(side_, event_, 0) == cudaSuccess;
    }
    StreamFork(const StreamFork&) = delete;
    StreamFork& operator=(const StreamFork&) = delete;
    ~StreamFork() {
        if (forked_ && !joined_ && !join()) {
            cudaStreamSynchronize(side_);
        }
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }

    bool forked() const { return forked_; }

    bool join() {
        joined_ = cudaEventRecord(event_, side_) == cudaSuccess &&
                  cudaStreamWaitEvent(0, event_, 0) == cudaSuccess;
        return joined_;
    }

private:
    cudaStream_t side_{nullptr};
    cudaEvent_t event_{nullptr};
    bool forked_{false};
    bool joined_{false};
};

// A device allocation freed when the encode returns, keeping the orchestrator's
// many scratch buffers out of the happy-path control flow. Backed by the
// stream-ordered pool allocator so the freed blocks are recycled across frames.
struct DeviceScope {
    std::vector<void*> ptrs{};
    template <typename T>
    T* alloc(std::size_t count) {
        void* p{nullptr};
        if (cudaMallocAsync(&p, count * sizeof(T), 0) != cudaSuccess) {
            return nullptr;
        }
        ptrs.push_back(p);
        return static_cast<T*>(p);
    }
    void track(void* p) { ptrs.push_back(p); }
    ~DeviceScope() {
        for (void* p : ptrs) {
            cudaFreeAsync(p, 0);
        }
    }
};

// Grow-only page-locked host buffer of `T`, kept across frames.
template <typename T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    ~PinnedBuffer() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
        }
    }

    // Makes room for `count` elements; the contents are not kept.
    bool reserve(std::size_t count) {
        if (count <= capacity_) {
            return true;
        }
        if (data_ != nullptr) {
            cudaFreeHost(data_);
            data_ = nullptr;
            capacity_ = 0;
        }
        // Headroom, so that frames of slowly growing size do not reallocate
        // every time.
        const std::size_t grown{count + count / 4};
        void* p{nullptr};
        if (cudaHostAlloc(&p, grown * sizeof(T), cudaHostAllocPortable) != cudaSuccess) {
            return false;
        }
        data_ = static_cast<T*>(p);
        capacity_ = grown;
        return true;
    }

    T* data() const { return data_; }

private:
    T* data_{nullptr};
    std::size_t capacity_{0};
};

enum class Placement : std::uint8_t { PENDING, PLACE, SKIP };

}  // namespace

SectionCapacities section_capacities(std::size_t width, std::size_t height) {
    const std::size_t blocks{(width / 8) * (height / 8)};
    const std::size_t num_ac{bitstream::ac_group_count(width, height)};
    const std::size_t num_dc{bitstream::dc_group_count(width, height)};
    return SectionCapacities{3 * blocks * 384 + num_ac * 64 + 4096,
                             48 * blocks + 2 * num_dc * DC_BLOB_STRIDE + num_dc * 64 + 4096,
                             std::size_t{1} << 16};
}

std::size_t max_encoded_size(std::size_t width, std::size_t height) {
    // The container framing, codestream headers, FrameHeader and DcGlobal take
    // well under FIXED_HEADERS_BOUND; a TOC entry takes at most 4 bytes.
    constexpr std::size_t FIXED_HEADERS_BOUND{1024};
    constexpr std::size_t TOC_ENTRY_BOUND{4};
    const std::size_t num_sections{bitstream::dc_group_count(width, height) +
                                   bitstream::ac_group_count(width, height) + 2};
    const SectionCapacities caps{section_capacities(width, height)};
    return caps.ac_groups + caps.dc_groups + caps.ac_global + FIXED_HEADERS_BOUND +
           TOC_ENTRY_BOUND * num_sections;
}

bitstream::QuantParams quant_params_for_distance(float distance) {
    const QuantCalibration cal{calibrate_quant(distance)};
    return bitstream::QuantParams{cal.global_scale, cal.quant_dc};
}

struct FrameEncoder::State {
    explicit State(std::size_t worker_threads) : pool{worker_threads} {}
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    ~State() { release_device_objects(); }

    bool bind_device(int device_ordinal);
    void release_device_objects();

    EncodeResult encode_frame(const std::int16_t* ac_device, const std::int32_t* dc_device,
                              const std::int32_t* nz_grid, const std::int32_t* nz_exact,
                              const std::int8_t* acs, const std::int8_t* ytox_map,
                              const std::int8_t* ytob_map, std::size_t width, std::size_t height,
                              const bitstream::QuantParams& qp, const std::int32_t* quant_field,
                              std::uint8_t* out_file, std::size_t out_capacity,
                              std::vector<StageTiming>* stats);

    // Releases the AC groups to `queue` batch by batch as their copies land,
    // each time all the batches landed so far, longest groups first. False on a
    // CUDA error.
    template <typename BatchBegin>
    bool release_batches(bitstream::StreamQueue& queue, BatchBegin batch_begin);

    WorkerPool pool;
    int device{-1};
    cudaStream_t dc_stream{nullptr};
    cudaStream_t copy_stream{nullptr};
    cudaEvent_t layout_copied{nullptr};
    cudaEvent_t dc_sizes_copied{nullptr};
    cudaEvent_t dc_body_copied{nullptr};
    std::array<cudaEvent_t, AC_BATCHES> batch_packed{};
    std::array<cudaEvent_t, AC_BATCHES> batch_copied{};

    // The AC cluster histograms followed by the AC token layout.
    PinnedBuffer<std::uint32_t> ac_meta;
    PinnedBuffer<std::uint32_t> dc_sizes;
    PinnedBuffer<std::uint8_t> dc_body;
    PinnedBuffer<std::uint16_t> entries;
    PinnedBuffer<std::uint64_t> raw;

    bitstream::TokenCode code;
    std::vector<bitstream::TokenStream> streams;
    std::vector<std::size_t> order;
    std::vector<bitstream::CodedStream> coded;
    std::vector<std::size_t> slot_end;
    std::vector<std::uint64_t> slot_words;
    std::vector<std::size_t> ac_offsets;
    std::vector<std::uint32_t> section_sizes;
};

bool FrameEncoder::State::bind_device(int device_ordinal) {
    if (device == device_ordinal) {
        return true;
    }
    release_device_objects();
    const auto create_event = [](cudaEvent_t& e) {
        return cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
    };
    bool ok{cudaStreamCreateWithFlags(&dc_stream, cudaStreamNonBlocking) == cudaSuccess &&
            cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking) == cudaSuccess &&
            create_event(layout_copied) && create_event(dc_sizes_copied) &&
            create_event(dc_body_copied)};
    for (std::size_t b{0}; b < AC_BATCHES; ++b) {
        ok = ok && create_event(batch_packed[b]) && create_event(batch_copied[b]);
    }
    if (!ok) {
        release_device_objects();
        return false;
    }
    device = device_ordinal;
    return true;
}

void FrameEncoder::State::release_device_objects() {
    const auto destroy_event = [](cudaEvent_t& e) {
        if (e != nullptr) {
            cudaEventDestroy(e);
            e = nullptr;
        }
    };
    destroy_event(layout_copied);
    destroy_event(dc_sizes_copied);
    destroy_event(dc_body_copied);
    for (std::size_t b{0}; b < AC_BATCHES; ++b) {
        destroy_event(batch_packed[b]);
        destroy_event(batch_copied[b]);
    }
    for (cudaStream_t* s : {&dc_stream, &copy_stream}) {
        if (*s != nullptr) {
            cudaStreamDestroy(*s);
            *s = nullptr;
        }
    }
    device = -1;
}

template <typename BatchBegin>
bool FrameEncoder::State::release_batches(bitstream::StreamQueue& queue, BatchBegin batch_begin) {
    std::size_t landed{0};
    while (landed < AC_BATCHES) {
        if (cudaEventSynchronize(batch_copied[landed]) != cudaSuccess) {
            return false;
        }
        ++landed;
        for (cudaError_t status{}; landed < AC_BATCHES; ++landed) {
            status = cudaEventQuery(batch_copied[landed]);
            if (status == cudaErrorNotReady) {
                break;
            }
            if (status != cudaSuccess) {
                return false;
            }
        }
        const std::size_t begin{queue.ready.load(std::memory_order_relaxed)};
        const std::size_t end{batch_begin(landed)};
        std::iota(order.begin() + static_cast<std::ptrdiff_t>(begin),
                  order.begin() + static_cast<std::ptrdiff_t>(end), begin);
        std::stable_sort(order.begin() + static_cast<std::ptrdiff_t>(begin),
                         order.begin() + static_cast<std::ptrdiff_t>(end),
                         [&](std::size_t a, std::size_t b) {
                             return streams[a].num_tokens > streams[b].num_tokens;
                         });
        queue.ready.store(end, std::memory_order_release);
    }
    return true;
}

EncodeResult FrameEncoder::State::encode_frame(
    const std::int16_t* ac_device, const std::int32_t* dc_device, const std::int32_t* nz_grid,
    const std::int32_t* nz_exact, const std::int8_t* acs, const std::int8_t* ytox_map,
    const std::int8_t* ytob_map, std::size_t width, std::size_t height,
    const bitstream::QuantParams& qp, const std::int32_t* quant_field, std::uint8_t* out_file,
    std::size_t out_capacity, std::vector<StageTiming>* stats) {
    const std::size_t num_ac{bitstream::ac_group_count(width, height)};
    const std::size_t num_dc{bitstream::dc_group_count(width, height)};
    if (num_ac <= 1) {
        return ENCODE_FAILED;
    }
    const SectionCapacities caps{section_capacities(width, height)};
    const std::size_t blocks{(width / 8) * (height / 8)};
    constexpr std::size_t HIST_WORDS{AC_NUM_CLUSTERS * AC_HISTOGRAM_SIZE};
    const std::size_t layout_words{3 * num_ac + 2};

    DeviceScope scope{};
    StageTiming entropy{"entropy", 0, 0.0, 0.0};
    StageTiming assembly{"assembly", 0, 0.0, 0.0};
    const Clock::time_point entropy_gpu_start{Clock::now()};

    // Everything the DC path and the copies touch is allocated before the forks:
    // an allocation ordered on the default stream after a fork would not be
    // ordered before the side stream's use.
    std::uint32_t* d_ac_hist{scope.alloc<std::uint32_t>(HIST_WORDS)};
    std::uint64_t* d_item_sizes{scope.alloc<std::uint64_t>(3 * blocks)};
    std::uint64_t* d_chunk_base{scope.alloc<std::uint64_t>(ac_token_chunks(width, height))};
    std::uint32_t* d_layout{scope.alloc<std::uint32_t>(layout_words)};
    std::uint16_t* d_entries{scope.alloc<std::uint16_t>(ac_max_tokens(width, height))};
    std::uint64_t* d_raw{scope.alloc<std::uint64_t>(ac_max_raw_words(width, height))};
    std::uint32_t* d_dc_hist{scope.alloc<std::uint32_t>(num_dc * AC_HISTOGRAM_SIZE)};
    std::uint32_t* d_am_hist{scope.alloc<std::uint32_t>(num_dc * AC_HISTOGRAM_SIZE)};
    std::size_t* d_fb_pos{scope.alloc<std::size_t>(num_dc * DC_FIRST_BLOCK_STRIDE)};
    std::size_t* d_fbc{scope.alloc<std::size_t>(num_dc)};
    std::uint8_t* d_dc_depth{scope.alloc<std::uint8_t>(num_dc * bitstream::HISTOGRAM_STRIDE)};
    std::uint16_t* d_dc_bits{scope.alloc<std::uint16_t>(num_dc * bitstream::HISTOGRAM_STRIDE)};
    std::uint8_t* d_am_depth{scope.alloc<std::uint8_t>(num_dc * bitstream::HISTOGRAM_STRIDE)};
    std::uint16_t* d_am_bits{scope.alloc<std::uint16_t>(num_dc * bitstream::HISTOGRAM_STRIDE)};
    std::uint8_t* d_pre{scope.alloc<std::uint8_t>(num_dc * DC_BLOB_STRIDE)};
    std::uint32_t* d_pre_off{scope.alloc<std::uint32_t>(num_dc)};
    std::uint32_t* d_pre_bits{scope.alloc<std::uint32_t>(num_dc)};
    std::uint8_t* d_mid{scope.alloc<std::uint8_t>(num_dc * DC_BLOB_STRIDE)};
    std::uint32_t* d_mid_off{scope.alloc<std::uint32_t>(num_dc)};
    std::uint32_t* d_mid_bits{scope.alloc<std::uint32_t>(num_dc)};
    std::uint8_t* d_dc_body{scope.alloc<std::uint8_t>(caps.dc_groups)};
    std::uint32_t* d_dc_sizes{scope.alloc<std::uint32_t>(num_dc)};
    std::uint32_t* d_dc_offsets{scope.alloc<std::uint32_t>(num_dc)};
    std::uint8_t* d_dc_scratch{scope.alloc<std::uint8_t>(dc_encode_scratch_bytes(width, height))};
    if (!d_ac_hist || !d_item_sizes || !d_chunk_base || !d_layout || !d_entries || !d_raw ||
        !d_dc_hist || !d_am_hist || !d_fb_pos || !d_fbc || !d_dc_depth || !d_dc_bits ||
        !d_am_depth || !d_am_bits || !d_pre || !d_pre_off || !d_pre_bits || !d_mid || !d_mid_off ||
        !d_mid_bits || !d_dc_body || !d_dc_sizes || !d_dc_offsets || !d_dc_scratch ||
        !ac_meta.reserve(HIST_WORDS + layout_words) || !dc_sizes.reserve(num_dc)) {
        return ENCODE_FAILED;
    }

    // The DC path (DcGroup histograms, prefix codes and sections) and the AC
    // token copies run on their own streams alongside the AC path.
    StreamFork dc_fork{dc_stream};
    StreamFork copy_fork{copy_stream};
    if (!dc_fork.forked() || !copy_fork.forked()) {
        return ENCODE_FAILED;
    }
    if (!dc_build_histograms(dc_device, width, height, d_dc_hist, dc_stream) ||
        !acmeta_build_histograms(quant_field, acs, ytox_map, ytob_map, width, height, d_am_hist,
                                 dc_stream) ||
        !dc_first_blocks(acs, width, height, d_fb_pos, d_fbc, dc_stream) ||
        !build_dc_group_blobs_device(width, height, d_dc_hist, d_am_hist, d_fbc, d_dc_depth,
                                     d_dc_bits, d_am_depth, d_am_bits, d_pre, d_pre_off, d_pre_bits,
                                     d_mid, d_mid_off, d_mid_bits, dc_stream) ||
        !dc_encode_groups(dc_device, width, height, quant_field, acs, ytox_map, ytob_map, d_fb_pos,
                          d_fbc, d_dc_depth, d_dc_bits, d_am_depth, d_am_bits, d_pre, d_pre_off,
                          d_pre_bits, d_mid, d_mid_off, d_mid_bits, d_dc_scratch, d_dc_body,
                          caps.dc_groups, d_dc_sizes, d_dc_offsets, dc_stream) ||
        cudaMemcpyAsync(dc_sizes.data(), d_dc_sizes, num_dc * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, dc_stream) != cudaSuccess ||
        cudaEventRecord(dc_sizes_copied, dc_stream) != cudaSuccess) {
        return ENCODE_FAILED;
    }

    if (!ac_build_histogram(ac_device, acs, nz_grid, nz_exact, width, height, d_ac_hist,
                            d_item_sizes) ||
        !ac_token_layout(d_item_sizes, width, height, d_chunk_base, d_layout) ||
        cudaMemcpyAsync(ac_meta.data(), d_ac_hist, HIST_WORDS * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, 0) != cudaSuccess ||
        cudaMemcpyAsync(ac_meta.data() + HIST_WORDS, d_layout, layout_words * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, 0) != cudaSuccess ||
        cudaEventRecord(layout_copied, 0) != cudaSuccess) {
        return ENCODE_FAILED;
    }
    const auto batch_begin = [num_ac](std::size_t b) { return b * num_ac / AC_BATCHES; };
    for (std::size_t b{0}; b < AC_BATCHES; ++b) {
        if ((batch_begin(b) != batch_begin(b + 1) &&
             !ac_pack_tokens(ac_device, acs, nz_grid, nz_exact, width, height, d_item_sizes,
                             d_chunk_base, d_layout, batch_begin(b), batch_begin(b + 1), d_entries,
                             d_raw)) ||
            cudaEventRecord(batch_packed[b], 0) != cudaSuccess) {
            return ENCODE_FAILED;
        }
    }

    const std::vector<std::uint8_t> dc_global{bitstream::build_dc_global(qp)};
    if (cudaEventSynchronize(layout_copied) != cudaSuccess) {
        return ENCODE_FAILED;
    }
    entropy.gpu_us = us_since(entropy_gpu_start);
    const Clock::time_point entropy_cpu_start{Clock::now()};

    const std::uint32_t* const histograms{ac_meta.data()};
    const std::uint32_t* const token_offset{histograms + HIST_WORDS};
    const std::uint32_t* const raw_offset{token_offset + num_ac + 1};
    const std::uint32_t* const raw_bits{raw_offset + num_ac + 1};
    // One raw word past the last group's: the coder may read it.
    if (!entries.reserve(token_offset[num_ac]) || !raw.reserve(raw_offset[num_ac] + 1)) {
        return ENCODE_FAILED;
    }
    for (std::size_t b{0}; b < AC_BATCHES; ++b) {
        const std::size_t g0{batch_begin(b)};
        const std::size_t g1{batch_begin(b + 1)};
        const std::size_t num_tokens{token_offset[g1] - token_offset[g0]};
        const std::size_t num_raw{raw_offset[g1] - raw_offset[g0]};
        if (cudaStreamWaitEvent(copy_stream, batch_packed[b], 0) != cudaSuccess ||
            (num_tokens != 0 &&
             cudaMemcpyAsync(entries.data() + token_offset[g0], d_entries + token_offset[g0],
                             num_tokens * sizeof(std::uint16_t), cudaMemcpyDeviceToHost,
                             copy_stream) != cudaSuccess) ||
            (num_raw != 0 && cudaMemcpyAsync(raw.data() + raw_offset[g0], d_raw + raw_offset[g0],
                                             num_raw * sizeof(std::uint64_t),
                                             cudaMemcpyDeviceToHost, copy_stream) != cudaSuccess) ||
            cudaEventRecord(batch_copied[b], copy_stream) != cudaSuccess) {
            return ENCODE_FAILED;
        }
    }

    bitstream::AcGlobalBuilder tables{histograms, num_ac, code};
    streams.resize(num_ac);
    order.resize(num_ac);
    coded.assign(num_ac, bitstream::CodedStream{});
    slot_end.resize(num_ac);
    ac_offsets.resize(num_ac);
    std::size_t slot_total{0};
    for (std::size_t g{0}; g < num_ac; ++g) {
        streams[g] = bitstream::TokenStream{entries.data() + token_offset[g],
                                            token_offset[g + 1] - token_offset[g],
                                            raw.data() + raw_offset[g], raw_bits[g]};
        slot_total += bitstream::coded_stream_capacity_words(streams[g]);
        slot_end[g] = slot_total;
    }
    if (slot_words.size() < slot_total) {
        slot_words.resize(slot_total);
    }
    bitstream::StreamQueue queue{};
    queue.streams = streams.data();
    queue.num_streams = num_ac;
    queue.order = order.data();
    queue.slot_words = slot_words.data();
    queue.slot_end = slot_end.data();
    queue.coded = coded.data();

    std::atomic<std::size_t> next_cluster{0};
    std::atomic<std::size_t> tables_done{0};
    std::atomic<bool> code_ready{false};
    std::vector<std::uint8_t> ac_global{};
    std::atomic<std::size_t> coders_done{0};
    std::atomic<Placement> placement{Placement::PENDING};
    std::atomic<std::size_t> next_placed{0};
    EncodeResult result{ENCODE_FAILED};
    const std::size_t num_threads{pool.size()};

    // All threads build the clusters' ANS tables, then thread 0 completes the
    // AcGlobal and releases the batches as their copies land, longest groups
    // first, and fetches the DcGroups while the others code; once all are done
    // it lays out the file, and every thread places a share of the AC groups.
    pool.run([&](std::size_t t) {
        for (std::size_t c{next_cluster.fetch_add(1, std::memory_order_relaxed)};
             c < AC_NUM_CLUSTERS; c = next_cluster.fetch_add(1, std::memory_order_relaxed)) {
            tables.build_cluster(c);
        }
        tables_done.fetch_add(1, std::memory_order_acq_rel);
        if (t == 0) {
            while (tables_done.load(std::memory_order_acquire) != num_threads) {
                std::this_thread::yield();
            }
            ac_global = tables.finish();
            code_ready.store(true, std::memory_order_release);
            if (!release_batches(queue, batch_begin)) {
                queue.cancelled.store(true, std::memory_order_release);
            }
        } else {
            while (!code_ready.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
        }
        bitstream::code_token_streams(code, queue);
        coders_done.fetch_add(1, std::memory_order_acq_rel);
        if (t != 0) {
            while (placement.load(std::memory_order_acquire) == Placement::PENDING) {
                std::this_thread::yield();
            }
        } else {
            bool dc_ok{!queue.cancelled.load(std::memory_order_acquire) &&
                       cudaEventSynchronize(dc_sizes_copied) == cudaSuccess};
            std::size_t dc_total{0};
            if (dc_ok) {
                // The DC encoder skips sections over its capacity without failing.
                dc_total =
                    std::accumulate(dc_sizes.data(), dc_sizes.data() + num_dc, std::size_t{0});
                dc_ok = dc_total <= caps.dc_groups && dc_body.reserve(dc_total) &&
                        cudaMemcpyAsync(dc_body.data(), d_dc_body, dc_total, cudaMemcpyDeviceToHost,
                                        dc_stream) == cudaSuccess &&
                        cudaEventRecord(dc_body_copied, dc_stream) == cudaSuccess;
            }
            while (coders_done.load(std::memory_order_acquire) != num_threads) {
                std::this_thread::yield();
            }
            entropy.cpu_us = us_since(entropy_cpu_start);
            const Clock::time_point assembly_start{Clock::now()};
            if (!dc_ok || cudaEventSynchronize(dc_body_copied) != cudaSuccess) {
                placement.store(Placement::SKIP, std::memory_order_release);
                return;
            }
            section_sizes.clear();
            section_sizes.push_back(static_cast<std::uint32_t>(dc_global.size()));
            section_sizes.insert(section_sizes.end(), dc_sizes.data(), dc_sizes.data() + num_dc);
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
            const std::size_t file_size{framing.size() + codestream_size};
            if (file_size > out_capacity) {
                result = EncodeResult{EncodeStatus::BUFFER_TOO_SMALL, file_size};
                placement.store(Placement::SKIP, std::memory_order_release);
                return;
            }
            std::uint8_t* at{std::copy(framing.begin(), framing.end(), out_file)};
            at = std::copy(head.begin(), head.end(), at);
            at = std::copy(dc_global.begin(), dc_global.end(), at);
            std::memcpy(at, dc_body.data(), dc_total);
            at = std::copy(ac_global.begin(), ac_global.end(), at + dc_total);
            std::size_t offset{static_cast<std::size_t>(at - out_file)};
            for (std::size_t g{0}; g < num_ac; ++g) {
                ac_offsets[g] = offset;
                offset += coded[g].size_bytes();
            }
            result = EncodeResult{EncodeStatus::OK, file_size};
            assembly.cpu_us = us_since(assembly_start);
            placement.store(Placement::PLACE, std::memory_order_release);
        }
        if (placement.load(std::memory_order_acquire) == Placement::PLACE) {
            for (std::size_t g{next_placed.fetch_add(1, std::memory_order_relaxed)}; g < num_ac;
                 g = next_placed.fetch_add(1, std::memory_order_relaxed)) {
                bitstream::place_coded_stream(coded[g], out_file + ac_offsets[g]);
            }
        }
    });

    if (stats != nullptr) {
        stats->push_back(entropy);
        stats->push_back(assembly);
    }
    return result;
}

FrameEncoder::FrameEncoder(std::size_t worker_threads)
    : state_{std::make_unique<State>(worker_threads)} {}

FrameEncoder::~FrameEncoder() = default;

EncodeResult FrameEncoder::encode_nv12(const std::uint8_t* luma, std::size_t luma_pitch,
                                       const std::uint8_t* chroma, std::size_t chroma_pitch,
                                       std::size_t width, std::size_t height,
                                       std::int32_t device_ordinal, float distance,
                                       const bitstream::QuantParams& qp, std::uint8_t* out_file,
                                       std::size_t out_capacity, std::vector<StageTiming>* stats) {
    if (cudaSetDevice(device_ordinal) != cudaSuccess || !state_->bind_device(device_ordinal)) {
        return ENCODE_FAILED;
    }
    retain_default_mempool(device_ordinal);

    DeviceScope scope{};
    const std::size_t blocks{(width / 8) * (height / 8)};
    const std::size_t cmw{((width / 8) + 7) / 8};
    const std::size_t cmh{((height / 8) + 7) / 8};
    __half* d_coeffs{scope.alloc<__half>(3 * blocks * COEFFS_PER_BLOCK)};
    std::int8_t* d_acs{scope.alloc<std::int8_t>(blocks)};
    std::int8_t* d_mx{scope.alloc<std::int8_t>(cmw * cmh)};
    std::int8_t* d_mb{scope.alloc<std::int8_t>(cmw * cmh)};
    std::int32_t* d_qf{scope.alloc<std::int32_t>(blocks)};
    std::int16_t* d_ac16{scope.alloc<std::int16_t>(3 * blocks * COEFFS_PER_BLOCK)};
    std::int32_t* d_dc32{scope.alloc<std::int32_t>(3 * blocks)};
    std::int32_t* d_nz_exact{scope.alloc<std::int32_t>(3 * blocks)};
    std::int32_t* d_nz_grid{scope.alloc<std::int32_t>(3 * blocks)};
    if (!d_coeffs || !d_acs || !d_mx || !d_mb || !d_qf || !d_ac16 || !d_dc32 || !d_nz_exact ||
        !d_nz_grid) {
        return ENCODE_FAILED;
    }

    constexpr std::size_t CHROMA_PITCH_ALIGNMENT{32};
    const std::uint8_t* chroma_src{chroma};
    std::size_t chroma_src_pitch{chroma_pitch};
    if (chroma_pitch % CHROMA_PITCH_ALIGNMENT != 0) {
        const std::size_t aligned_pitch{(width + 511) & ~std::size_t{511}};
        std::uint8_t* aligned{scope.alloc<std::uint8_t>(aligned_pitch * (height / 2))};
        if (!aligned || cudaMemcpy2DAsync(aligned, aligned_pitch, chroma, chroma_pitch, width,
                                          height / 2, cudaMemcpyDeviceToDevice, 0) != cudaSuccess) {
            return ENCODE_FAILED;
        }
        chroma_src = aligned;
        chroma_src_pitch = aligned_pitch;
    }

    const ChromaTexture chroma_texture{chroma_src, chroma_src_pitch, width, height};
    const Clock::time_point frontend_start{Clock::now()};
    if (!encode_frontend(luma, luma_pitch, chroma_texture, width, height, distance, d_coeffs, d_acs,
                         d_qf) ||
        !quantize_residual(d_coeffs, d_acs, d_qf, width, height, distance, d_mx, d_mb, d_ac16,
                           d_dc32, d_nz_exact, d_nz_grid)) {
        return ENCODE_FAILED;
    }
    if (stats != nullptr) {
        // The frontend runs sync-free (its kernels only enqueue), so bracket the
        // stage with an instrumentation-only sync to keep the per-stage wall-clock
        // attribution valid. Production (stats == nullptr) stays fully pipelined.
        cudaDeviceSynchronize();
        stats->push_back(StageTiming{"frontend", 0, us_since(frontend_start), 0.0});
    }
    return state_->encode_frame(d_ac16, d_dc32, d_nz_grid, d_nz_exact, d_acs, d_mx, d_mb, width,
                                height, qp, d_qf, out_file, out_capacity, stats);
}

}  // namespace cujpegxl
