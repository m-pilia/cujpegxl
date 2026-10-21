// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frame_encoder.h"

#include <algorithm>
#include <chrono>
#include <cstdint>

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
#include "vardct_layout.h"

namespace cujpegxl {
namespace {

using Clock = std::chrono::steady_clock;

constexpr EncodeResult ENCODE_FAILED{EncodeStatus::FAILED, 0};

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

// The data-independent AC context-map bit-blob, built on the host once and kept
// resident on the device so every frame's AcGlobal kernel blits it instead of
// re-serializing the 7425-entry map. Lazily initialized on the device active at
// first encode (the single target device).
struct DeviceCtxMapBlob {
    std::uint8_t* bytes{nullptr};
    std::uint32_t nbits{0};
};

const DeviceCtxMapBlob& ac_ctxmap_blob_device() {
    static const DeviceCtxMapBlob blob{[] {
        const bitstream::AcContextMapBlob host{bitstream::build_ac_context_map_blob()};
        DeviceCtxMapBlob b{};
        b.nbits = static_cast<std::uint32_t>(host.nbits);
        if (cudaMalloc(&b.bytes, host.bytes.size()) != cudaSuccess ||
            cudaMemcpy(b.bytes, host.bytes.data(), host.bytes.size(), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            b.bytes = nullptr;
        }
        return b;
    }()};
    return blob;
}

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

EncodeResult encode_frame(const std::int16_t* ac_device, const std::int32_t* dc_device,
                          const std::int8_t* acs, const std::int8_t* ytox_map,
                          const std::int8_t* ytob_map, std::size_t width, std::size_t height,
                          const bitstream::QuantParams& qp, const std::int32_t* quant_field,
                          std::uint8_t* out_file, std::size_t out_capacity,
                          std::vector<StageTiming>* stats) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t num_ac{bitstream::ac_group_count(width, height)};
    const std::size_t num_dc{bitstream::dc_group_count(width, height)};
    if (num_ac <= 1) {
        return ENCODE_FAILED;
    }
    const SectionCapacities caps{section_capacities(width, height)};
    int device{0};
    cudaGetDevice(&device);
    retain_default_mempool(device);

    DeviceScope scope{};
    StageTiming entropy{"entropy", 0, 0.0, 0.0};
    StageTiming assembly{"assembly", 0, 0.0, 0.0};
    const Clock::time_point entropy_gpu_start{Clock::now()};

    std::uint32_t* d_ac_hist{scope.alloc<std::uint32_t>(AC_NUM_CLUSTERS * AC_HISTOGRAM_SIZE)};
    std::uint32_t* d_dc_hist{scope.alloc<std::uint32_t>(num_dc * AC_HISTOGRAM_SIZE)};
    std::uint32_t* d_am_hist{scope.alloc<std::uint32_t>(num_dc * AC_HISTOGRAM_SIZE)};
    std::size_t* d_fb_pos{scope.alloc<std::size_t>(num_dc * DC_FIRST_BLOCK_STRIDE)};
    std::size_t* d_fbc{scope.alloc<std::size_t>(num_dc)};
    std::int32_t* d_nz_grid{scope.alloc<std::int32_t>(3 * bw * bh)};
    std::int32_t* d_nz_exact{scope.alloc<std::int32_t>(3 * bw * bh)};
    if (!d_ac_hist || !d_dc_hist || !d_am_hist || !d_fb_pos || !d_fbc || !d_nz_grid ||
        !d_nz_exact) {
        return ENCODE_FAILED;
    }
    if (!ac_nonzero_grids(ac_device, acs, width, height, d_nz_grid, d_nz_exact) ||
        !ac_build_histogram(ac_device, acs, d_nz_grid, d_nz_exact, width, height, d_ac_hist) ||
        !dc_build_histograms(dc_device, width, height, d_dc_hist) ||
        !acmeta_build_histograms(quant_field, acs, ytox_map, ytob_map, width, height, d_am_hist) ||
        !dc_first_blocks(acs, width, height, d_fb_pos, d_fbc)) {
        return ENCODE_FAILED;
    }

    // Prefix-code build and header serialization run on-device, so the AC/DC/
    // AcMetadata histograms stay resident: no histogram device-to-host copy, host
    // Huffman construction, or depth/bits/blob upload between the histogram and
    // encode kernels. DcGlobal depends only on the quantizer, so it stays on host.
    const DeviceCtxMapBlob& ctxmap{ac_ctxmap_blob_device()};
    std::uint8_t* d_ac_depth{
        scope.alloc<std::uint8_t>(AC_NUM_CLUSTERS * bitstream::HISTOGRAM_STRIDE)};
    std::uint16_t* d_ac_bits{
        scope.alloc<std::uint16_t>(AC_NUM_CLUSTERS * bitstream::HISTOGRAM_STRIDE)};
    std::uint8_t* d_ac_section{scope.alloc<std::uint8_t>(caps.ac_global)};
    // Section sizes after DcGlobal in codestream order (DcGroups, AcGlobal,
    // AcGroups), so one device-to-host copy fetches them all.
    std::uint32_t* d_section_sizes{scope.alloc<std::uint32_t>(num_dc + 1 + num_ac)};
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
    if (!ctxmap.bytes || !d_ac_depth || !d_ac_bits || !d_ac_section || !d_section_sizes ||
        !d_dc_depth || !d_dc_bits || !d_am_depth || !d_am_bits || !d_pre || !d_pre_off ||
        !d_pre_bits || !d_mid || !d_mid_off || !d_mid_bits) {
        return ENCODE_FAILED;
    }
    std::uint32_t* d_dc_sizes{d_section_sizes};
    std::uint32_t* d_ac_global_size{d_section_sizes + num_dc};
    std::uint32_t* d_ac_sizes{d_section_sizes + num_dc + 1};
    if (!build_ac_global_device(d_ac_hist, num_ac, ctxmap.bytes, ctxmap.nbits, d_ac_depth,
                                d_ac_bits, d_ac_section, caps.ac_global, d_ac_global_size) ||
        !build_dc_group_blobs_device(width, height, d_dc_hist, d_am_hist, d_fbc, d_dc_depth,
                                     d_dc_bits, d_am_depth, d_am_bits, d_pre, d_pre_off, d_pre_bits,
                                     d_mid, d_mid_off, d_mid_bits)) {
        return ENCODE_FAILED;
    }
    entropy.gpu_us += us_since(entropy_gpu_start);

    const Clock::time_point entropy_cpu_start{Clock::now()};
    const std::vector<std::uint8_t> dc_global{bitstream::build_dc_global(qp)};
    entropy.cpu_us += us_since(entropy_cpu_start);

    const Clock::time_point entropy_encode_start{Clock::now()};
    std::uint8_t* d_ac_body{scope.alloc<std::uint8_t>(caps.ac_groups)};
    std::uint8_t* d_dc_body{scope.alloc<std::uint8_t>(caps.dc_groups)};
    std::uint32_t* d_ac_offsets{scope.alloc<std::uint32_t>(num_ac)};
    std::uint32_t* d_dc_offsets{scope.alloc<std::uint32_t>(num_dc)};
    std::uint8_t* d_dc_scratch{scope.alloc<std::uint8_t>(dc_encode_scratch_bytes(width, height))};
    if (!d_ac_body || !d_dc_body || !d_ac_offsets || !d_dc_offsets || !d_dc_scratch) {
        return ENCODE_FAILED;
    }

    std::size_t ac_total{0};
    std::size_t dc_total{0};
    const bool ac_encoded{ac_encode_groups(ac_device, acs, d_nz_grid, d_nz_exact, width, height,
                                           d_ac_depth, d_ac_bits, d_ac_body, caps.ac_groups,
                                           d_ac_sizes, d_ac_offsets, &ac_total)};
    if (!ac_encoded) {
        return ENCODE_FAILED;
    }
    if (!dc_encode_groups(dc_device, width, height, quant_field, acs, ytox_map, ytob_map, d_fb_pos,
                          d_fbc, d_dc_depth, d_dc_bits, d_am_depth, d_am_bits, d_pre, d_pre_off,
                          d_pre_bits, d_mid, d_mid_off, d_mid_bits, d_dc_scratch, d_dc_body,
                          caps.dc_groups, d_dc_sizes, d_dc_offsets, &dc_total)) {
        return ENCODE_FAILED;
    }

    std::vector<std::uint32_t> section_sizes(1 + num_dc + 1 + num_ac, 0);
    section_sizes[0] = static_cast<std::uint32_t>(dc_global.size());
    if (cudaMemcpy(section_sizes.data() + 1, d_section_sizes,
                   (num_dc + 1 + num_ac) * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return ENCODE_FAILED;
    }
    const std::uint32_t ac_global_size{section_sizes[1 + num_dc]};
    entropy.gpu_us += us_since(entropy_encode_start);

    const Clock::time_point assembly_head_start{Clock::now()};
    const std::vector<std::uint8_t> head{bitstream::build_codestream_head(
        static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), section_sizes)};
    assembly.cpu_us += us_since(assembly_head_start);

    const std::size_t body_size{dc_global.size() + dc_total + ac_global_size + ac_total};
    const std::size_t codestream_size{head.size() + body_size};
    const std::vector<std::uint8_t> framing{bitstream::container_framing(codestream_size)};
    const std::size_t file_size{framing.size() + codestream_size};
    if (file_size > out_capacity) {
        return EncodeResult{EncodeStatus::BUFFER_TOO_SMALL, file_size};
    }

    const Clock::time_point assembly_gather_start{Clock::now()};
    std::uint8_t* d_body{scope.alloc<std::uint8_t>(body_size)};
    if (!d_body) {
        return ENCODE_FAILED;
    }
    std::size_t at{0};
    const bool gathered{
        cudaMemcpyAsync(d_body + at, dc_global.data(), dc_global.size(), cudaMemcpyHostToDevice,
                        0) == cudaSuccess &&
        (at += dc_global.size(), cudaMemcpyAsync(d_body + at, d_dc_body, dc_total,
                                                 cudaMemcpyDeviceToDevice, 0) == cudaSuccess) &&
        (at += dc_total, cudaMemcpyAsync(d_body + at, d_ac_section, ac_global_size,
                                         cudaMemcpyDeviceToDevice, 0) == cudaSuccess) &&
        (at += ac_global_size, cudaMemcpyAsync(d_body + at, d_ac_body, ac_total,
                                               cudaMemcpyDeviceToDevice, 0) == cudaSuccess)};
    if (!gathered) {
        return ENCODE_FAILED;
    }
    if (stats != nullptr) {
        cudaDeviceSynchronize();
    }
    assembly.gpu_us += us_since(assembly_gather_start);

    std::uint8_t* const head_dst{std::copy(framing.begin(), framing.end(), out_file)};
    std::uint8_t* const body_dst{std::copy(head.begin(), head.end(), head_dst)};
    if (cudaMemcpy(body_dst, d_body, body_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return ENCODE_FAILED;
    }
    if (stats != nullptr) {
        stats->push_back(entropy);
        stats->push_back(assembly);
    }
    return EncodeResult{EncodeStatus::OK, file_size};
}

bitstream::QuantParams quant_params_for_distance(float distance) {
    const QuantCalibration cal{calibrate_quant(distance)};
    return bitstream::QuantParams{cal.global_scale, cal.quant_dc};
}

EncodeResult encode_nv12(const std::uint8_t* luma, std::size_t luma_pitch,
                         const std::uint8_t* chroma, std::size_t chroma_pitch, std::size_t width,
                         std::size_t height, std::int32_t device_ordinal, float distance,
                         const bitstream::QuantParams& qp, std::uint8_t* out_file,
                         std::size_t out_capacity, std::vector<StageTiming>* stats) {
    if (cudaSetDevice(device_ordinal) != cudaSuccess) {
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
    if (!d_coeffs || !d_acs || !d_mx || !d_mb || !d_qf || !d_ac16 || !d_dc32) {
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

    const Clock::time_point frontend_start{Clock::now()};
    if (!encode_frontend(luma, luma_pitch, chroma_src, chroma_src_pitch, width, height, distance,
                         d_coeffs, d_acs, d_qf) ||
        !estimate_cfl_covered(d_coeffs, d_acs, width, height, d_mx, d_mb) ||
        !quantize_residual(d_coeffs, d_acs, d_mx, d_mb, d_qf, width, height, distance, d_ac16,
                           d_dc32)) {
        return ENCODE_FAILED;
    }
    if (stats != nullptr) {
        // The frontend now runs sync-free past the fused kernel (cfl/quantize only
        // enqueue), so bracket the stage with an instrumentation-only sync to keep
        // the per-stage wall-clock attribution valid. Production (stats == nullptr)
        // stays fully pipelined.
        cudaDeviceSynchronize();
        stats->push_back(StageTiming{"frontend", 0, us_since(frontend_start), 0.0});
    }
    return encode_frame(d_ac16, d_dc32, d_acs, d_mx, d_mb, width, height, qp, d_qf, out_file,
                        out_capacity, stats);
}

}  // namespace cujpegxl
