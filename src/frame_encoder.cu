// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frame_encoder.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <numeric>
#include <optional>
#include <thread>

#include <cuda_runtime.h>

#include <cuda_fp16.h>

#include "ac_clustering.h"
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

constexpr std::size_t CHROMA_PITCH_ALIGNMENT{32};

double us_since(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

std::size_t batch_begin(std::size_t num_ac, std::size_t b) {
    return b * num_ac / AC_BATCHES;
}

std::size_t aligned_chroma_pitch(std::size_t width) {
    return (width + 511) & ~std::size_t{511};
}

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

// Grow-only device allocation, kept across frames. Persistent scratch keeps the
// frames in flight independent: per-frame stream-ordered allocations would let
// the pool hand a block freed by one frame's stream to another's, ordering the
// two.
class DeviceArena {
public:
    DeviceArena() = default;
    DeviceArena(const DeviceArena&) = delete;
    DeviceArena& operator=(const DeviceArena&) = delete;
    ~DeviceArena() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    // Makes room for `bytes`; the contents are not kept.
    bool reserve(std::size_t bytes) {
        if (bytes <= capacity_) {
            return true;
        }
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
            capacity_ = 0;
        }
        void* p{nullptr};
        if (cudaMalloc(&p, bytes) != cudaSuccess) {
            return false;
        }
        data_ = static_cast<std::uint8_t*>(p);
        capacity_ = bytes;
        return true;
    }

    std::uint8_t* data() const { return data_; }

private:
    std::uint8_t* data_{nullptr};
    std::size_t capacity_{0};
};

// Hands out consecutive aligned ranges from address `base`; with base 0 it only
// measures their total size.
class ArenaCarver {
public:
    explicit ArenaCarver(std::uintptr_t base) : base_{base} {}

    template <typename T>
    T* take(std::size_t count) {
        constexpr std::size_t ALIGNMENT{256};
        offset_ = (offset_ + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
        T* const p{reinterpret_cast<T*>(base_ + offset_)};
        offset_ += count * sizeof(T);
        return p;
    }

    std::size_t size() const { return offset_; }

private:
    std::uintptr_t base_;
    std::size_t offset_{0};
};

struct FrameBuffers {
    std::uint8_t* aligned_chroma{nullptr};
    __half* coeffs{nullptr};
    std::int8_t* acs{nullptr};
    std::int8_t* ytox_map{nullptr};
    std::int8_t* ytob_map{nullptr};
    std::int32_t* quant_field{nullptr};
    std::int16_t* ac{nullptr};
    std::int32_t* dc{nullptr};
    std::int32_t* nz_exact{nullptr};
    std::int32_t* nz_grid{nullptr};
    std::uint32_t* context_hist{nullptr};
    std::uint8_t* cluster_scratch{nullptr};
    AcClustering* clustering{nullptr};
    std::uint64_t* item_sizes{nullptr};
    std::uint8_t* layout_scratch{nullptr};
    std::uint64_t* chunk_base{nullptr};
    std::uint32_t* layout{nullptr};
    std::uint16_t* entries{nullptr};
    std::uint64_t* raw{nullptr};
    std::uint32_t* dc_hist{nullptr};
    std::uint32_t* am_hist{nullptr};
    std::uint8_t* fb_scratch{nullptr};
    std::size_t* fb_pos{nullptr};
    std::size_t* fbc{nullptr};
    std::uint8_t* dc_depth{nullptr};
    std::uint16_t* dc_bits{nullptr};
    std::uint8_t* am_depth{nullptr};
    std::uint16_t* am_bits{nullptr};
    std::uint8_t* pre{nullptr};
    std::uint32_t* pre_off{nullptr};
    std::uint32_t* pre_bits{nullptr};
    std::uint8_t* mid{nullptr};
    std::uint32_t* mid_off{nullptr};
    std::uint32_t* mid_bits{nullptr};
    std::uint8_t* dc_body{nullptr};
    std::uint32_t* dc_sizes{nullptr};
    std::uint32_t* dc_offsets{nullptr};
    std::uint8_t* dc_scratch{nullptr};
};

FrameBuffers carve_frame_buffers(ArenaCarver& carver, std::size_t width, std::size_t height) {
    const std::size_t blocks{(width / 8) * (height / 8)};
    const std::size_t color_tiles{(((width / 8) + 7) / 8) * (((height / 8) + 7) / 8)};
    const std::size_t num_ac{bitstream::ac_group_count(width, height)};
    const std::size_t num_dc{bitstream::dc_group_count(width, height)};
    const std::size_t codes{num_dc * bitstream::HISTOGRAM_STRIDE};
    return FrameBuffers{
        .aligned_chroma = carver.take<std::uint8_t>(aligned_chroma_pitch(width) * (height / 2)),
        .coeffs = carver.take<__half>(3 * blocks * COEFFS_PER_BLOCK),
        .acs = carver.take<std::int8_t>(blocks),
        .ytox_map = carver.take<std::int8_t>(color_tiles),
        .ytob_map = carver.take<std::int8_t>(color_tiles),
        .quant_field = carver.take<std::int32_t>(blocks),
        .ac = carver.take<std::int16_t>(3 * blocks * COEFFS_PER_BLOCK),
        .dc = carver.take<std::int32_t>(3 * blocks),
        .nz_exact = carver.take<std::int32_t>(3 * blocks),
        .nz_grid = carver.take<std::int32_t>(3 * blocks),
        .context_hist = carver.take<std::uint32_t>(AC_NUM_CONTEXTS * AC_CONTEXT_ALPHABET),
        .cluster_scratch = carver.take<std::uint8_t>(ac_cluster_scratch_bytes()),
        .clustering = carver.take<AcClustering>(1),
        .item_sizes = carver.take<std::uint64_t>(3 * blocks),
        .layout_scratch = carver.take<std::uint8_t>(ac_token_layout_scratch_bytes(width, height)),
        .chunk_base = carver.take<std::uint64_t>(ac_token_chunks(width, height)),
        .layout = carver.take<std::uint32_t>(3 * num_ac + 2),
        .entries = carver.take<std::uint16_t>(ac_max_tokens(width, height)),
        .raw = carver.take<std::uint64_t>(ac_max_raw_words(width, height)),
        .dc_hist = carver.take<std::uint32_t>(num_dc * AC_HISTOGRAM_SIZE),
        .am_hist = carver.take<std::uint32_t>(num_dc * AC_HISTOGRAM_SIZE),
        .fb_scratch = carver.take<std::uint8_t>(dc_first_blocks_scratch_bytes(width, height)),
        .fb_pos = carver.take<std::size_t>(num_dc * DC_FIRST_BLOCK_STRIDE),
        .fbc = carver.take<std::size_t>(num_dc),
        .dc_depth = carver.take<std::uint8_t>(codes),
        .dc_bits = carver.take<std::uint16_t>(codes),
        .am_depth = carver.take<std::uint8_t>(codes),
        .am_bits = carver.take<std::uint16_t>(codes),
        .pre = carver.take<std::uint8_t>(num_dc * DC_BLOB_STRIDE),
        .pre_off = carver.take<std::uint32_t>(num_dc),
        .pre_bits = carver.take<std::uint32_t>(num_dc),
        .mid = carver.take<std::uint8_t>(num_dc * DC_BLOB_STRIDE),
        .mid_off = carver.take<std::uint32_t>(num_dc),
        .mid_bits = carver.take<std::uint32_t>(num_dc),
        .dc_body = carver.take<std::uint8_t>(section_capacities(width, height).dc_groups),
        .dc_sizes = carver.take<std::uint32_t>(num_dc),
        .dc_offsets = carver.take<std::uint32_t>(num_dc),
        .dc_scratch = carver.take<std::uint8_t>(dc_encode_scratch_bytes(width, height)),
    };
}

// A frame in flight: its parameters and result, and the resources it uses. The
// device work runs on `stream`, with the DC path on `dc_stream` and the AC
// token copies on `copy_stream`.
struct Slot {
    Slot() = default;
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
    ~Slot() {
        chroma.reset();
        release();
    }

    // Creates the streams and events on the current device, once.
    bool init();
    void release();

    bool ready{false};
    cudaStream_t stream{nullptr};
    cudaStream_t dc_stream{nullptr};
    cudaStream_t copy_stream{nullptr};
    cudaEvent_t forked{nullptr};
    cudaEvent_t layout_copied{nullptr};
    cudaEvent_t dc_sizes_copied{nullptr};
    cudaEvent_t dc_body_copied{nullptr};
    std::array<cudaEvent_t, AC_BATCHES> batch_packed{};
    std::array<cudaEvent_t, AC_BATCHES> batch_copied{};

    DeviceArena arena;
    FrameBuffers device{};
    PinnedBuffer<AcClustering> clustering;
    PinnedBuffer<std::uint32_t> layout;
    PinnedBuffer<std::uint32_t> dc_sizes;
    PinnedBuffer<std::uint8_t> dc_body;
    PinnedBuffer<std::uint16_t> entries;
    PinnedBuffer<std::uint64_t> raw;
    std::optional<ChromaTexture> chroma;

    std::size_t width{0};
    std::size_t height{0};
    bitstream::QuantParams qp{};
    std::uint8_t* out_file{nullptr};
    std::size_t out_capacity{0};
    std::vector<StageTiming>* stats{nullptr};
    Clock::time_point entropy_start{};
    Clock::time_point layout_landed{};
    bool copies_started{false};
    EncodeResult result{};
};

bool Slot::init() {
    if (ready) {
        return true;
    }
    release();
    const auto create_event = [](cudaEvent_t& e) {
        return cudaEventCreateWithFlags(&e, cudaEventDisableTiming) == cudaSuccess;
    };
    // Non-blocking, so that the slots' streams do not synchronize with each other
    // through the legacy default stream.
    ready = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess &&
            cudaStreamCreateWithFlags(&dc_stream, cudaStreamNonBlocking) == cudaSuccess &&
            cudaStreamCreateWithFlags(&copy_stream, cudaStreamNonBlocking) == cudaSuccess &&
            create_event(forked) && create_event(layout_copied) && create_event(dc_sizes_copied) &&
            create_event(dc_body_copied);
    for (std::size_t b{0}; b < AC_BATCHES; ++b) {
        ready = ready && create_event(batch_packed[b]) && create_event(batch_copied[b]);
    }
    return ready;
}

void Slot::release() {
    const auto destroy_event = [](cudaEvent_t& e) {
        if (e != nullptr) {
            cudaEventDestroy(e);
            e = nullptr;
        }
    };
    for (cudaEvent_t* e : {&forked, &layout_copied, &dc_sizes_copied, &dc_body_copied}) {
        destroy_event(*e);
    }
    for (std::size_t b{0}; b < AC_BATCHES; ++b) {
        destroy_event(batch_packed[b]);
        destroy_event(batch_copied[b]);
    }
    for (cudaStream_t* s : {&stream, &dc_stream, &copy_stream}) {
        if (*s != nullptr) {
            cudaStreamDestroy(*s);
            *s = nullptr;
        }
    }
    ready = false;
}

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

// Frame k uses slot k % depth. The caller thread enqueues a frame's device work
// in submit(); the driver thread then runs each frame's host stage in order, on
// the worker pool as its thread 0.
struct FrameEncoder::State {
    State(int device_ordinal, std::size_t worker_threads, std::size_t pipeline_depth)
        : device{device_ordinal},
          depth{pipeline_depth},
          slots{std::make_unique<Slot[]>(pipeline_depth)},
          pool{worker_threads},
          driver{[this] { drive(); }} {}
    State(const State&) = delete;
    State& operator=(const State&) = delete;
    ~State() {
        {
            const std::lock_guard<std::mutex> lock{mutex};
            stop = true;
        }
        submitted_cv.notify_one();
        driver.join();
    }

    bool enqueue_frame(Slot& slot, const std::uint8_t* luma, std::size_t luma_pitch,
                       const std::uint8_t* chroma, std::size_t chroma_pitch, float distance);
    void drive();
    EncodeResult code_frame(Slot& slot, std::uint64_t frame);
    bool start_copies(Slot& slot, bool wait);
    void start_next_copies(std::uint64_t frame);
    bool release_batches(Slot& slot, bitstream::StreamQueue& queue);

    const int device;
    const std::size_t depth;
    const std::unique_ptr<Slot[]> slots;
    WorkerPool pool;

    bitstream::TokenCode code;
    std::vector<bitstream::TokenStream> streams;
    std::vector<std::size_t> order;
    std::vector<bitstream::CodedStream> coded;
    std::vector<std::size_t> slot_end;
    std::vector<std::uint64_t> slot_words;
    std::vector<std::size_t> ac_offsets;
    std::vector<std::uint32_t> section_sizes;

    // Frame counts since construction; submitted and received change only on
    // the caller thread, completed only on the driver thread.
    std::mutex mutex;
    std::condition_variable submitted_cv;
    std::condition_variable completed_cv;
    std::uint64_t submitted{0};
    std::uint64_t completed{0};
    std::uint64_t received{0};
    bool stop{false};
    std::thread driver;
};

bool FrameEncoder::State::enqueue_frame(Slot& slot, const std::uint8_t* luma,
                                        std::size_t luma_pitch, const std::uint8_t* chroma,
                                        std::size_t chroma_pitch, float distance) {
    const std::size_t width{slot.width};
    const std::size_t height{slot.height};
    const std::size_t num_ac{bitstream::ac_group_count(width, height)};
    const std::size_t num_dc{bitstream::dc_group_count(width, height)};
    const std::size_t layout_words{3 * num_ac + 2};

    ArenaCarver measure{0};
    carve_frame_buffers(measure, width, height);
    if (!slot.arena.reserve(measure.size()) || !slot.clustering.reserve(1) ||
        !slot.layout.reserve(layout_words) || !slot.dc_sizes.reserve(num_dc)) {
        return false;
    }
    ArenaCarver carver{reinterpret_cast<std::uintptr_t>(slot.arena.data())};
    slot.device = carve_frame_buffers(carver, width, height);
    const FrameBuffers& d{slot.device};

    const std::uint8_t* chroma_src{chroma};
    std::size_t chroma_src_pitch{chroma_pitch};
    if (chroma_pitch % CHROMA_PITCH_ALIGNMENT != 0) {
        chroma_src_pitch = aligned_chroma_pitch(width);
        if (cudaMemcpy2DAsync(d.aligned_chroma, chroma_src_pitch, chroma, chroma_pitch, width,
                              height / 2, cudaMemcpyDeviceToDevice, slot.stream) != cudaSuccess) {
            return false;
        }
        chroma_src = d.aligned_chroma;
    }
    slot.chroma.emplace(chroma_src, chroma_src_pitch, width, height, slot.stream);

    const Clock::time_point frontend_start{Clock::now()};
    if (!encode_frontend(luma, luma_pitch, *slot.chroma, width, height, distance, d.coeffs, d.acs,
                         d.quant_field, slot.stream) ||
        !quantize_residual(d.coeffs, d.acs, d.quant_field, width, height, distance, d.ytox_map,
                           d.ytob_map, d.ac, d.dc, d.nz_exact, d.nz_grid, slot.stream)) {
        return false;
    }
    if (slot.stats != nullptr) {
        // Instrumentation-only wait, so that the front end's wall time is its own.
        if (cudaStreamSynchronize(slot.stream) != cudaSuccess) {
            return false;
        }
        slot.stats->push_back(StageTiming{"frontend", 0, us_since(frontend_start), 0.0});
    }
    slot.entropy_start = Clock::now();

    // The DC path (DcGroup histograms, prefix codes and sections) runs on its own
    // stream alongside the AC path.
    if (cudaEventRecord(slot.forked, slot.stream) != cudaSuccess ||
        cudaStreamWaitEvent(slot.dc_stream, slot.forked, 0) != cudaSuccess ||
        !dc_build_histograms(d.dc, width, height, d.dc_hist, slot.dc_stream) ||
        !acmeta_build_histograms(d.quant_field, d.acs, d.ytox_map, d.ytob_map, width, height,
                                 d.am_hist, slot.dc_stream) ||
        !dc_first_blocks(d.acs, width, height, d.fb_scratch, d.fb_pos, d.fbc, slot.dc_stream) ||
        !build_dc_group_blobs_device(width, height, d.dc_hist, d.am_hist, d.fbc, d.dc_depth,
                                     d.dc_bits, d.am_depth, d.am_bits, d.pre, d.pre_off, d.pre_bits,
                                     d.mid, d.mid_off, d.mid_bits, slot.dc_stream) ||
        !dc_encode_groups(d.dc, width, height, d.quant_field, d.acs, d.ytox_map, d.ytob_map,
                          d.fb_pos, d.fbc, d.dc_depth, d.dc_bits, d.am_depth, d.am_bits, d.pre,
                          d.pre_off, d.pre_bits, d.mid, d.mid_off, d.mid_bits, d.dc_scratch,
                          d.dc_body, section_capacities(width, height).dc_groups, d.dc_sizes,
                          d.dc_offsets, slot.dc_stream) ||
        cudaMemcpyAsync(slot.dc_sizes.data(), d.dc_sizes, num_dc * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, slot.dc_stream) != cudaSuccess ||
        cudaEventRecord(slot.dc_sizes_copied, slot.dc_stream) != cudaSuccess) {
        return false;
    }

    if (!ac_build_histogram(d.ac, d.acs, d.nz_grid, d.nz_exact, width, height, d.context_hist,
                            d.item_sizes, slot.stream) ||
        !ac_token_layout(d.item_sizes, width, height, d.layout_scratch, d.chunk_base, d.layout,
                         slot.stream) ||
        !ac_cluster_contexts(d.context_hist, d.cluster_scratch, d.clustering, slot.stream) ||
        cudaMemcpyAsync(slot.clustering.data(), d.clustering, sizeof(AcClustering),
                        cudaMemcpyDeviceToHost, slot.stream) != cudaSuccess ||
        cudaMemcpyAsync(slot.layout.data(), d.layout, layout_words * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, slot.stream) != cudaSuccess ||
        cudaEventRecord(slot.layout_copied, slot.stream) != cudaSuccess) {
        return false;
    }
    for (std::size_t b{0}; b < AC_BATCHES; ++b) {
        const std::size_t g0{batch_begin(num_ac, b)};
        const std::size_t g1{batch_begin(num_ac, b + 1)};
        if ((g0 != g1 && !ac_pack_tokens(d.ac, d.acs, d.nz_grid, d.nz_exact, width, height,
                                         d.clustering->context_map, d.item_sizes, d.chunk_base,
                                         d.layout, g0, g1, d.entries, d.raw, slot.stream)) ||
            cudaEventRecord(slot.batch_packed[b], slot.stream) != cudaSuccess) {
            return false;
        }
    }
    return true;
}

void FrameEncoder::State::drive() {
    // A failure shows in the frames' CUDA calls.
    cudaSetDevice(device);
    std::unique_lock<std::mutex> lock{mutex};
    for (;;) {
        submitted_cv.wait(lock, [this] { return stop || completed != submitted; });
        if (completed == submitted) {
            return;
        }
        const std::uint64_t frame{completed};
        lock.unlock();
        Slot& slot{slots[frame % depth]};
        const EncodeResult result{code_frame(slot, frame)};
        lock.lock();
        slot.result = result;
        ++completed;
        completed_cv.notify_one();
    }
}

// Enqueues the copies of the slot's AC token batches once its token layout has
// landed; with `wait` false, only if it already has. False on a CUDA error.
bool FrameEncoder::State::start_copies(Slot& slot, bool wait) {
    if (slot.copies_started) {
        return true;
    }
    if (wait) {
        if (cudaEventSynchronize(slot.layout_copied) != cudaSuccess) {
            return false;
        }
    } else {
        const cudaError_t status{cudaEventQuery(slot.layout_copied)};
        if (status == cudaErrorNotReady) {
            return true;
        }
        if (status != cudaSuccess) {
            return false;
        }
    }
    slot.layout_landed = Clock::now();
    const std::size_t num_ac{bitstream::ac_group_count(slot.width, slot.height)};
    const std::uint32_t* const token_offset{slot.layout.data()};
    const std::uint32_t* const raw_offset{token_offset + num_ac + 1};
    // One raw word past the last group's: the coder may read it.
    if (!slot.entries.reserve(token_offset[num_ac]) || !slot.raw.reserve(raw_offset[num_ac] + 1)) {
        return false;
    }
    for (std::size_t b{0}; b < AC_BATCHES; ++b) {
        const std::size_t g0{batch_begin(num_ac, b)};
        const std::size_t g1{batch_begin(num_ac, b + 1)};
        const std::size_t num_tokens{token_offset[g1] - token_offset[g0]};
        const std::size_t num_raw{raw_offset[g1] - raw_offset[g0]};
        if (cudaStreamWaitEvent(slot.copy_stream, slot.batch_packed[b], 0) != cudaSuccess ||
            (num_tokens != 0 &&
             cudaMemcpyAsync(slot.entries.data() + token_offset[g0],
                             slot.device.entries + token_offset[g0],
                             num_tokens * sizeof(std::uint16_t), cudaMemcpyDeviceToHost,
                             slot.copy_stream) != cudaSuccess) ||
            (num_raw != 0 &&
             cudaMemcpyAsync(slot.raw.data() + raw_offset[g0], slot.device.raw + raw_offset[g0],
                             num_raw * sizeof(std::uint64_t), cudaMemcpyDeviceToHost,
                             slot.copy_stream) != cudaSuccess) ||
            cudaEventRecord(slot.batch_copied[b], slot.copy_stream) != cudaSuccess) {
            return false;
        }
    }
    slot.copies_started = true;
    return true;
}

// Starts the next frame's copies if its token layout has already landed, so that
// they overlap the end of this frame's host stage instead of delaying the next
// one's. A failure shows again when the next frame's host stage starts them.
void FrameEncoder::State::start_next_copies(std::uint64_t frame) {
    {
        const std::lock_guard<std::mutex> lock{mutex};
        if (frame + 1 == submitted) {
            return;
        }
    }
    start_copies(slots[(frame + 1) % depth], false);
}

// Releases the AC groups to `queue` batch by batch as their copies land, each
// time all the batches landed so far, longest groups first. False on a CUDA
// error.
bool FrameEncoder::State::release_batches(Slot& slot, bitstream::StreamQueue& queue) {
    const std::size_t num_ac{queue.num_streams};
    std::size_t landed{0};
    while (landed < AC_BATCHES) {
        if (cudaEventSynchronize(slot.batch_copied[landed]) != cudaSuccess) {
            return false;
        }
        ++landed;
        for (cudaError_t status{}; landed < AC_BATCHES; ++landed) {
            status = cudaEventQuery(slot.batch_copied[landed]);
            if (status == cudaErrorNotReady) {
                break;
            }
            if (status != cudaSuccess) {
                return false;
            }
        }
        const std::size_t begin{queue.ready.load(std::memory_order_relaxed)};
        const std::size_t end{batch_begin(num_ac, landed)};
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

EncodeResult FrameEncoder::State::code_frame(Slot& slot, std::uint64_t frame) {
    const std::size_t width{slot.width};
    const std::size_t height{slot.height};
    const std::size_t num_ac{bitstream::ac_group_count(width, height)};
    const std::size_t num_dc{bitstream::dc_group_count(width, height)};
    const SectionCapacities caps{section_capacities(width, height)};
    StageTiming entropy{"entropy", 0, 0.0, 0.0};
    StageTiming assembly{"assembly", 0, 0.0, 0.0};

    if (!start_copies(slot, true)) {
        return ENCODE_FAILED;
    }
    entropy.gpu_us =
        std::chrono::duration<double, std::micro>(slot.layout_landed - slot.entropy_start).count();
    const Clock::time_point entropy_cpu_start{Clock::now()};
    const std::vector<std::uint8_t> dc_global{bitstream::build_dc_global(slot.qp)};

    const std::uint32_t* const token_offset{slot.layout.data()};
    const std::uint32_t* const raw_offset{token_offset + num_ac + 1};
    const std::uint32_t* const raw_bits{raw_offset + num_ac + 1};
    bitstream::AcGlobalBuilder tables{*slot.clustering.data(), num_ac, code};
    streams.resize(num_ac);
    order.resize(num_ac);
    coded.assign(num_ac, bitstream::CodedStream{});
    slot_end.resize(num_ac);
    ac_offsets.resize(num_ac);
    std::size_t slot_total{0};
    for (std::size_t g{0}; g < num_ac; ++g) {
        streams[g] = bitstream::TokenStream{slot.entries.data() + token_offset[g],
                                            token_offset[g + 1] - token_offset[g],
                                            slot.raw.data() + raw_offset[g], raw_bits[g]};
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

    std::atomic<std::size_t> next_table{0};
    std::atomic<std::size_t> tables_done{0};
    std::atomic<bool> code_ready{false};
    std::vector<std::uint8_t> ac_global{};
    std::atomic<std::size_t> coders_done{0};
    std::atomic<Placement> placement{Placement::PENDING};
    std::atomic<std::size_t> next_placed{0};
    EncodeResult result{ENCODE_FAILED};
    const std::size_t num_threads{pool.size()};

    // All threads build the context map, the longest job, first and the
    // clusters' ANS tables, then thread 0 completes the AcGlobal and releases the batches as their
    // copies land, longest groups first, and fetches the DcGroups while the others code; once all
    // are done it lays out the file, and every thread places a share of the AC groups.
    pool.run([&](std::size_t t) {
        for (std::size_t c{next_table.fetch_add(1, std::memory_order_relaxed)};
             c <= tables.num_clusters(); c = next_table.fetch_add(1, std::memory_order_relaxed)) {
            if (c == 0) {
                tables.build_context_map();
            } else {
                tables.build_cluster(c - 1);
            }
        }
        tables_done.fetch_add(1, std::memory_order_acq_rel);
        if (t == 0) {
            while (tables_done.load(std::memory_order_acquire) != num_threads) {
                std::this_thread::yield();
            }
            ac_global = tables.finish();
            code_ready.store(true, std::memory_order_release);
            if (!release_batches(slot, queue)) {
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
            start_next_copies(frame);
            bool dc_ok{!queue.cancelled.load(std::memory_order_acquire) &&
                       cudaEventSynchronize(slot.dc_sizes_copied) == cudaSuccess};
            std::size_t dc_total{0};
            if (dc_ok) {
                // The DC encoder skips sections over its capacity without failing.
                dc_total = std::accumulate(slot.dc_sizes.data(), slot.dc_sizes.data() + num_dc,
                                           std::size_t{0});
                dc_ok = dc_total <= caps.dc_groups && slot.dc_body.reserve(dc_total) &&
                        cudaMemcpyAsync(slot.dc_body.data(), slot.device.dc_body, dc_total,
                                        cudaMemcpyDeviceToHost, slot.dc_stream) == cudaSuccess &&
                        cudaEventRecord(slot.dc_body_copied, slot.dc_stream) == cudaSuccess;
            }
            while (coders_done.load(std::memory_order_acquire) != num_threads) {
                std::this_thread::yield();
            }
            entropy.cpu_us = us_since(entropy_cpu_start);
            const Clock::time_point assembly_start{Clock::now()};
            if (!dc_ok || cudaEventSynchronize(slot.dc_body_copied) != cudaSuccess) {
                placement.store(Placement::SKIP, std::memory_order_release);
                return;
            }
            section_sizes.clear();
            section_sizes.push_back(static_cast<std::uint32_t>(dc_global.size()));
            section_sizes.insert(section_sizes.end(), slot.dc_sizes.data(),
                                 slot.dc_sizes.data() + num_dc);
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
            if (file_size > slot.out_capacity) {
                result = EncodeResult{EncodeStatus::BUFFER_TOO_SMALL, file_size};
                placement.store(Placement::SKIP, std::memory_order_release);
                return;
            }
            std::uint8_t* at{std::copy(framing.begin(), framing.end(), slot.out_file)};
            at = std::copy(head.begin(), head.end(), at);
            at = std::copy(dc_global.begin(), dc_global.end(), at);
            std::memcpy(at, slot.dc_body.data(), dc_total);
            at = std::copy(ac_global.begin(), ac_global.end(), at + dc_total);
            std::size_t offset{static_cast<std::size_t>(at - slot.out_file)};
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
                bitstream::place_coded_stream(coded[g], slot.out_file + ac_offsets[g]);
            }
        }
    });

    if (slot.stats != nullptr) {
        slot.stats->push_back(entropy);
        slot.stats->push_back(assembly);
    }
    return result;
}

FrameEncoder::FrameEncoder(std::int32_t device_ordinal, std::size_t worker_threads,
                           std::size_t pipeline_depth)
    : state_{std::make_unique<State>(device_ordinal, worker_threads, pipeline_depth)} {
    assert(pipeline_depth >= 1);
}

FrameEncoder::~FrameEncoder() = default;

std::size_t FrameEncoder::pipeline_depth() const {
    return state_->depth;
}

std::size_t FrameEncoder::in_flight() const {
    const std::lock_guard<std::mutex> lock{state_->mutex};
    return static_cast<std::size_t>(state_->submitted - state_->received);
}

bool FrameEncoder::submit(const std::uint8_t* luma, std::size_t luma_pitch,
                          const std::uint8_t* chroma, std::size_t chroma_pitch, std::size_t width,
                          std::size_t height, float distance, const bitstream::QuantParams& qp,
                          std::uint8_t* out_file, std::size_t out_capacity,
                          std::vector<StageTiming>* stats) {
    assert(in_flight() < pipeline_depth());
    State& s{*state_};
    if (bitstream::ac_group_count(width, height) <= 1) {
        return false;
    }
    Slot& slot{s.slots[s.submitted % s.depth]};
    // The slot's previous frame has completed, but after a failure its device
    // work may still be running.
    if (cudaSetDevice(s.device) != cudaSuccess || !slot.init() ||
        cudaStreamSynchronize(slot.stream) != cudaSuccess ||
        cudaStreamSynchronize(slot.dc_stream) != cudaSuccess ||
        cudaStreamSynchronize(slot.copy_stream) != cudaSuccess) {
        return false;
    }
    slot.width = width;
    slot.height = height;
    slot.qp = qp;
    slot.out_file = out_file;
    slot.out_capacity = out_capacity;
    slot.stats = stats;
    slot.copies_started = false;
    if (!s.enqueue_frame(slot, luma, luma_pitch, chroma, chroma_pitch, distance)) {
        return false;
    }
    {
        const std::lock_guard<std::mutex> lock{s.mutex};
        ++s.submitted;
    }
    s.submitted_cv.notify_one();
    return true;
}

EncodeResult FrameEncoder::receive() {
    assert(in_flight() > 0);
    State& s{*state_};
    std::unique_lock<std::mutex> lock{s.mutex};
    s.completed_cv.wait(lock, [&s] { return s.completed != s.received; });
    const EncodeResult result{s.slots[s.received % s.depth].result};
    ++s.received;
    return result;
}

}  // namespace cujpegxl
