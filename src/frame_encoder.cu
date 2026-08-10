// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frame_encoder.h"

#include <chrono>

#include <cuda_runtime.h>

#include "dct.h"
#include "entropy.h"
#include "quant.h"
#include "src/bitstream/container.h"
#include "xyb.h"

namespace cujpegxl {
namespace {

using Clock = std::chrono::steady_clock;

double us_since(Clock::time_point start) {
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
}

template <typename T>
bool upload(const std::vector<T>& host, T** device) {
    if (host.empty()) {
        *device = nullptr;
        return true;
    }
    if (cudaMalloc(device, host.size() * sizeof(T)) != cudaSuccess) {
        return false;
    }
    return cudaMemcpy(*device, host.data(), host.size() * sizeof(T),
                      cudaMemcpyHostToDevice) == cudaSuccess;
}

// A device allocation freed when the encode returns, keeping the orchestrator's
// many scratch buffers out of the happy-path control flow.
struct DeviceScope {
    std::vector<void*> ptrs{};
    template <typename T>
    T* alloc(std::size_t count) {
        void* p{nullptr};
        if (cudaMalloc(&p, count * sizeof(T)) != cudaSuccess) {
            return nullptr;
        }
        ptrs.push_back(p);
        return static_cast<T*>(p);
    }
    void track(void* p) { ptrs.push_back(p); }
    ~DeviceScope() {
        for (void* p : ptrs) {
            cudaFree(p);
        }
    }
};

}  // namespace

bool encode_frame(const std::int32_t* q_device, std::size_t width, std::size_t height,
                  const bitstream::QuantParams& qp, std::vector<std::uint8_t>& out_file,
                  std::vector<StageTiming>* stats) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t num_ac{bitstream::ac_group_count(width, height)};
    const std::size_t num_dc{bitstream::dc_group_count(width, height)};
    if (num_ac <= 1) {
        return false;  // combined-section (single-group) layout not handled here
    }

    DeviceScope scope{};
    StageTiming entropy{"entropy", 0, 0.0, 0.0};
    StageTiming assembly{"assembly", 0, 0.0, 0.0};
    const Clock::time_point entropy_gpu_start{Clock::now()};

    // Device histograms -> host.
    std::uint32_t* d_ac_hist{scope.alloc<std::uint32_t>(AC_HISTOGRAM_SIZE)};
    std::uint32_t* d_dc_hist{scope.alloc<std::uint32_t>(num_dc * AC_HISTOGRAM_SIZE)};
    if (!d_ac_hist || !d_dc_hist) {
        return false;
    }
    if (!ac_build_histogram(q_device, width, height, d_ac_hist) ||
        !dc_build_histograms(q_device, width, height, d_dc_hist)) {
        return false;
    }
    std::vector<std::uint32_t> ac_hist(AC_HISTOGRAM_SIZE, 0);
    std::vector<std::uint32_t> dc_hist(num_dc * AC_HISTOGRAM_SIZE, 0);
    if (cudaMemcpy(ac_hist.data(), d_ac_hist, ac_hist.size() * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(dc_hist.data(), d_dc_hist, dc_hist.size() * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    entropy.gpu_us += us_since(entropy_gpu_start);

    // Host globals + per-group prefix codes and header blobs.
    const Clock::time_point entropy_cpu_start{Clock::now()};
    const bitstream::AcGlobalResult ac_global{bitstream::build_ac_global(ac_hist.data(), num_ac)};
    const std::vector<std::uint8_t> dc_global{bitstream::build_dc_global(qp)};
    const bitstream::DcGroupBlobs blobs{
        bitstream::build_dc_group_blobs(width, height, qp, dc_hist.data())};
    entropy.cpu_us += us_since(entropy_cpu_start);

    // Upload the entropy-coder inputs.
    const Clock::time_point entropy_encode_start{Clock::now()};
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
    if (!upload(ac_global.depth, &d_ac_depth) || !upload(ac_global.bits, &d_ac_bits) ||
        !upload(blobs.dc_depth, &d_dc_depth) || !upload(blobs.dc_bits, &d_dc_bits) ||
        !upload(blobs.acmeta_depth, &d_am_depth) || !upload(blobs.acmeta_bits, &d_am_bits) ||
        !upload(blobs.blob_pre, &d_pre) || !upload(blobs.blob_pre_off, &d_pre_off) ||
        !upload(blobs.blob_pre_bits, &d_pre_bits) || !upload(blobs.blob_mid, &d_mid) ||
        !upload(blobs.blob_mid_off, &d_mid_off) || !upload(blobs.blob_mid_bits, &d_mid_bits)) {
        return false;
    }
    for (void* p : {static_cast<void*>(d_ac_depth), static_cast<void*>(d_ac_bits),
                    static_cast<void*>(d_dc_depth), static_cast<void*>(d_dc_bits),
                    static_cast<void*>(d_am_depth), static_cast<void*>(d_am_bits),
                    static_cast<void*>(d_pre), static_cast<void*>(d_pre_off),
                    static_cast<void*>(d_pre_bits), static_cast<void*>(d_mid),
                    static_cast<void*>(d_mid_off), static_cast<void*>(d_mid_bits)}) {
        if (p) {
            scope.track(p);
        }
    }

    // Entropy-code the AC groups and DcGroups into their own device bodies.
    // Worst-case bounds: an AC block-channel emits <= 64 tokens of <= 6 bytes; a
    // DcGroup block contributes a handful of DC + AcMetadata tokens.
    const std::size_t ac_capacity{3 * bw * bh * 384 + num_ac * 64 + 4096};
    const std::size_t dc_capacity{48 * bw * bh + blobs.blob_pre.size() + blobs.blob_mid.size() +
                                  num_dc * 64 + 4096};
    std::uint8_t* d_ac_body{scope.alloc<std::uint8_t>(ac_capacity)};
    std::uint8_t* d_dc_body{scope.alloc<std::uint8_t>(dc_capacity)};
    std::uint32_t* d_ac_sizes{scope.alloc<std::uint32_t>(num_ac)};
    std::uint32_t* d_ac_offsets{scope.alloc<std::uint32_t>(num_ac)};
    std::uint32_t* d_dc_sizes{scope.alloc<std::uint32_t>(num_dc)};
    std::uint32_t* d_dc_offsets{scope.alloc<std::uint32_t>(num_dc)};
    if (!d_ac_body || !d_dc_body || !d_ac_sizes || !d_ac_offsets || !d_dc_sizes ||
        !d_dc_offsets) {
        return false;
    }

    std::size_t ac_total{0};
    std::size_t dc_total{0};
    if (!ac_encode_groups(q_device, width, height, d_ac_depth, d_ac_bits, ac_global.depth.size(),
                          d_ac_body, ac_capacity, d_ac_sizes, d_ac_offsets, &ac_total)) {
        return false;
    }
    if (!dc_encode_groups(q_device, width, height, qp.raw_quant_field, d_dc_depth, d_dc_bits,
                          d_am_depth, d_am_bits, d_pre, d_pre_off, d_pre_bits, d_mid, d_mid_off,
                          d_mid_bits, d_dc_body, dc_capacity, d_dc_sizes, d_dc_offsets,
                          &dc_total)) {
        return false;
    }

    std::vector<std::uint32_t> ac_sizes(num_ac, 0);
    std::vector<std::uint32_t> dc_sizes(num_dc, 0);
    if (cudaMemcpy(ac_sizes.data(), d_ac_sizes, num_ac * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(dc_sizes.data(), d_dc_sizes, num_dc * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    entropy.gpu_us += us_since(entropy_encode_start);
    entropy.bytes_moved = 2 * 3 * bw * bh * 64 * sizeof(std::int32_t) + ac_total + dc_total;

    // Section sizes in codestream order: DcGlobal, DcGroups, AcGlobal, AcGroups.
    const Clock::time_point assembly_head_start{Clock::now()};
    std::vector<std::uint32_t> section_sizes{};
    section_sizes.push_back(static_cast<std::uint32_t>(dc_global.size()));
    for (std::uint32_t s : dc_sizes) {
        section_sizes.push_back(s);
    }
    section_sizes.push_back(static_cast<std::uint32_t>(ac_global.section.size()));
    for (std::uint32_t s : ac_sizes) {
        section_sizes.push_back(s);
    }

    const std::vector<std::uint8_t> head{bitstream::build_codestream_head(
        static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), section_sizes)};
    assembly.cpu_us += us_since(assembly_head_start);

    // Gather the body [DcGlobal | DcGroups | AcGlobal | AcGroups] into one device
    // buffer (byte-aligned sections -> byte concatenation), then a single D2H.
    const std::size_t body_size{dc_global.size() + dc_total + ac_global.section.size() + ac_total};
    const Clock::time_point assembly_gather_start{Clock::now()};
    std::uint8_t* d_body{scope.alloc<std::uint8_t>(body_size)};
    if (!d_body) {
        return false;
    }
    std::size_t at{0};
    const bool gathered{
        cudaMemcpy(d_body + at, dc_global.data(), dc_global.size(), cudaMemcpyHostToDevice) ==
            cudaSuccess &&
        (at += dc_global.size(),
         cudaMemcpy(d_body + at, d_dc_body, dc_total, cudaMemcpyDeviceToDevice) == cudaSuccess) &&
        (at += dc_total,
         cudaMemcpy(d_body + at, ac_global.section.data(), ac_global.section.size(),
                    cudaMemcpyHostToDevice) == cudaSuccess) &&
        (at += ac_global.section.size(),
         cudaMemcpy(d_body + at, d_ac_body, ac_total, cudaMemcpyDeviceToDevice) == cudaSuccess)};
    if (!gathered) {
        return false;
    }
    assembly.gpu_us += us_since(assembly_gather_start);

    const Clock::time_point assembly_finish_start{Clock::now()};
    const std::size_t codestream_size{head.size() + body_size};
    std::vector<std::uint8_t> file{bitstream::container_framing(codestream_size)};
    const std::size_t framing{file.size()};
    file.resize(framing + codestream_size);
    std::copy(head.begin(), head.end(), file.begin() + framing);
    assembly.cpu_us += us_since(assembly_finish_start);

    const Clock::time_point assembly_d2h_start{Clock::now()};
    if (cudaMemcpy(file.data() + framing + head.size(), d_body, body_size,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    assembly.gpu_us += us_since(assembly_d2h_start);
    assembly.bytes_moved = 2 * body_size;

    if (stats != nullptr) {
        stats->push_back(entropy);
        stats->push_back(assembly);
    }

    out_file = std::move(file);
    return true;
}

bitstream::QuantParams quant_params_for_distance(float distance) {
    (void)distance;
    return bitstream::QuantParams{4096, 32, 32};
}

bool encode_nv12(const std::uint8_t* luma, std::size_t luma_pitch, const std::uint8_t* chroma,
                 std::size_t chroma_pitch, std::size_t width, std::size_t height,
                 std::int32_t device_ordinal, float distance, const bitstream::QuantParams& qp,
                 std::vector<std::uint8_t>& out_file, std::vector<StageTiming>* stats) {
    if (cudaSetDevice(device_ordinal) != cudaSuccess) {
        return false;
    }

    DeviceScope scope{};
    const std::size_t plane{width * height};
    float* d_xyb{scope.alloc<float>(3 * plane)};
    float* d_coeffs{scope.alloc<float>(3 * plane)};
    std::int32_t* d_q{scope.alloc<std::int32_t>(3 * plane)};
    if (!d_xyb || !d_coeffs || !d_q) {
        return false;
    }

    // The chroma pitch2D texture requires a 32-byte-aligned row pitch; re-pitch
    // the plane into an aligned device buffer when the caller's is not.
    constexpr std::size_t CHROMA_PITCH_ALIGNMENT{32};
    const std::uint8_t* chroma_src{chroma};
    std::size_t chroma_src_pitch{chroma_pitch};
    if (chroma_pitch % CHROMA_PITCH_ALIGNMENT != 0) {
        void* aligned{nullptr};
        std::size_t aligned_pitch{0};
        if (cudaMallocPitch(&aligned, &aligned_pitch, width, height / 2) != cudaSuccess) {
            return false;
        }
        scope.track(aligned);
        if (cudaMemcpy2D(aligned, aligned_pitch, chroma, chroma_pitch, width, height / 2,
                         cudaMemcpyDeviceToDevice) != cudaSuccess) {
            return false;
        }
        chroma_src = static_cast<const std::uint8_t*>(aligned);
        chroma_src_pitch = aligned_pitch;
    }

    const Clock::time_point frontend_start{Clock::now()};
    if (!nv12_to_xyb(luma, luma_pitch, chroma_src, chroma_src_pitch, width, height, d_xyb) ||
        !forward_dct8(d_xyb, width, height, d_coeffs) ||
        !quantize_dct8(d_coeffs, width, height, distance, d_q)) {
        return false;
    }
    if (stats != nullptr) {
        StageTiming frontend{"frontend", 0, us_since(frontend_start), 0.0};
        frontend.bytes_moved = width * height + width * height / 2 +
                               5 * 3 * plane * sizeof(float);
        stats->push_back(frontend);
    }
    return encode_frame(d_q, width, height, qp, out_file, stats);
}

}  // namespace cujpegxl
