// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "frame_encoder.h"

#include <algorithm>
#include <chrono>
#include <cstdint>

#include <cuda_runtime.h>

#include <cuda_fp16.h>

#include "ac_context.h"
#include "ac_histogram_cluster.h"
#include "ac_histogram_exchange.h"
#include "ac_precluster.h"
#include "entropy.h"
#include "frontend_fused.h"
#include "frontend_quantize.h"
#include "frontend_transform.h"
#include "quant_calibration.h"
#include "src/bitstream/container.h"
#include "vardct_layout.h"

namespace cujpegxl {
namespace {

using Clock = std::chrono::steady_clock;

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

template <typename T>
bool upload(const std::vector<T>& host, T** device) {
    if (host.empty()) {
        *device = nullptr;
        return true;
    }
    if (cudaMallocAsync(device, host.size() * sizeof(T), 0) != cudaSuccess) {
        return false;
    }
    return cudaMemcpy(*device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) ==
           cudaSuccess;
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

struct AcEntropyPlan {
    bool ready{false};
    bool data_driven{false};
    std::vector<std::uint8_t> context_map{};
    std::vector<std::uint32_t> histograms{};
    bitstream::AcGlobalResult global{};
};

std::uint64_t ac_prefix_cost(const AcEntropyPlan& plan) {
    std::uint64_t bits{plan.global.section.size() * 8};
    for (std::size_t i{0}; i < plan.histograms.size(); ++i) {
        bits += static_cast<std::uint64_t>(plan.histograms[i]) *
                plan.global.depth[i];
    }
    return bits;
}

struct CachedHistogramExchange {
    AcHistogramExchange exchange{};
    std::int32_t device{-1};

    bool ensure(std::int32_t requested_device) {
        if (exchange.ownership() != AcHistogramOwnership::UNINITIALIZED &&
            device == requested_device) {
            return true;
        }
        exchange.shutdown();
        device = -1;
        if (!exchange.initialize(AcHistogramExchangeMode::AUTO, requested_device)) {
            return false;
        }
        device = requested_device;
        return true;
    }
};

bool build_data_driven_ac_plan(const std::int16_t* ac_device,
                               const std::int8_t* acs, std::size_t width,
                               std::size_t height, std::size_t num_ac_groups,
                               std::int32_t device, AcEntropyPlan& plan,
                               double& gpu_us, double& cpu_us) {
    static thread_local CachedHistogramExchange cached{};
    if (!cached.ensure(device)) {
        return false;
    }

    AcHistogramExchange& exchange{cached.exchange};
    const Clock::time_point gpu_start{Clock::now()};
    const bool histogram_ok{
        acs == nullptr
            ? ac_build_context_histograms(ac_device, width, height,
                                          exchange.gpu_data())
            : ac_build_context_histograms_m3(ac_device, acs, width, height,
                                             exchange.gpu_data())};
    if (!histogram_ok || !exchange.release_to_cpu(nullptr) ||
        !exchange.acquire_for_cpu()) {
        exchange.shutdown();
        cached.device = -1;
        return false;
    }
    gpu_us += us_since(gpu_start);

    const Clock::time_point cpu_start{Clock::now()};
    AcPreclusterResult preclusters{};
    AcClusterResult clusters{};
    ac_precluster(exchange.cpu_data(), preclusters);
    const bool clustered{ac_cluster_histograms(preclusters, AcClusterConfig{},
                                                clusters)};
    if (!clustered) {
        exchange.release_to_gpu();
        return false;
    }

    AcEntropyPlan dynamic{};
    dynamic.ready = true;
    dynamic.data_driven = true;
    dynamic.context_map.assign(clusters.context_map.begin(),
                               clusters.context_map.end());
    dynamic.histograms.assign(
        clusters.histograms.begin(),
        clusters.histograms.begin() +
            clusters.num_clusters * AC_HISTOGRAM_SIZE);
    dynamic.global = bitstream::build_ac_global(
        dynamic.histograms.data(), dynamic.context_map.data(),
        clusters.num_clusters, num_ac_groups);

    AcEntropyPlan fixed{};
    fixed.ready = true;
    fixed.histograms.assign(AC_NUM_CLUSTERS * AC_HISTOGRAM_SIZE, 0);
    const std::uint32_t* context_histograms{exchange.cpu_data()};
    for (std::size_t context{0}; context < AC_NUM_CONTEXTS; ++context) {
        const std::size_t cluster{static_cast<std::size_t>(
            ac_cluster(static_cast<std::uint32_t>(context)))};
        for (std::size_t symbol{0}; symbol < AC_HISTOGRAM_SIZE; ++symbol) {
            fixed.histograms[cluster * AC_HISTOGRAM_SIZE + symbol] +=
                context_histograms[context * AC_HISTOGRAM_SIZE + symbol];
        }
    }
    fixed.global = bitstream::build_ac_global(fixed.histograms.data(),
                                              num_ac_groups);
    plan = ac_prefix_cost(dynamic) < ac_prefix_cost(fixed) ? std::move(dynamic)
                                                           : std::move(fixed);
    exchange.release_to_gpu();
    cpu_us += us_since(cpu_start);
    return true;
}

}  // namespace

bool encode_frame(const std::int16_t* ac_device, const std::int32_t* dc_device, std::size_t width,
                  std::size_t height, const bitstream::QuantParams& qp,
                  const std::int32_t* quant_field, std::vector<std::uint8_t>& out_file,
                  std::vector<StageTiming>* stats, AcClusteringMode clustering) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t num_ac{bitstream::ac_group_count(width, height)};
    const std::size_t num_dc{bitstream::dc_group_count(width, height)};
    if (num_ac <= 1) {
        return false;  // combined-section (single-group) layout not handled here
    }

    int device{0};
    cudaGetDevice(&device);
    retain_default_mempool(device);

    DeviceScope scope{};
    StageTiming entropy{"entropy", 0, 0.0, 0.0};
    StageTiming assembly{"assembly", 0, 0.0, 0.0};

    AcEntropyPlan ac_plan{};
    if (clustering == AcClusteringMode::DATA_DRIVEN) {
        build_data_driven_ac_plan(ac_device, nullptr, width, height, num_ac,
                                  device, ac_plan, entropy.gpu_us,
                                  entropy.cpu_us);
    }
    const Clock::time_point entropy_gpu_start{Clock::now()};

    // Device histograms -> host.
    std::uint32_t* d_ac_hist{
        ac_plan.ready
            ? nullptr
            : scope.alloc<std::uint32_t>(AC_NUM_CLUSTERS * AC_HISTOGRAM_SIZE)};
    std::uint32_t* d_dc_hist{scope.alloc<std::uint32_t>(num_dc * AC_HISTOGRAM_SIZE)};
    std::uint32_t* d_am_hist{scope.alloc<std::uint32_t>(num_dc * AC_HISTOGRAM_SIZE)};
    if ((!ac_plan.ready && !d_ac_hist) || !d_dc_hist || !d_am_hist) {
        return false;
    }
    if ((!ac_plan.ready &&
         !ac_build_histogram(ac_device, width, height, d_ac_hist)) ||
        !dc_build_histograms(dc_device, width, height, d_dc_hist) ||
        !acmeta_build_histograms(quant_field, nullptr, nullptr, nullptr, width, height, d_am_hist)) {
        return false;
    }
    std::vector<std::uint32_t> ac_hist{};
    if (!ac_plan.ready) {
        ac_hist.assign(AC_NUM_CLUSTERS * AC_HISTOGRAM_SIZE, 0);
    }
    std::vector<std::uint32_t> dc_hist(num_dc * AC_HISTOGRAM_SIZE, 0);
    std::vector<std::uint32_t> am_hist(num_dc * AC_HISTOGRAM_SIZE, 0);
    if ((!ac_plan.ready &&
         cudaMemcpy(ac_hist.data(), d_ac_hist,
                    ac_hist.size() * sizeof(std::uint32_t),
                    cudaMemcpyDeviceToHost) != cudaSuccess) ||
        cudaMemcpy(dc_hist.data(), d_dc_hist, dc_hist.size() * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(am_hist.data(), d_am_hist, am_hist.size() * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    entropy.gpu_us += us_since(entropy_gpu_start);

    // Host globals + per-group prefix codes and header blobs.
    const Clock::time_point entropy_cpu_start{Clock::now()};
    if (!ac_plan.ready) {
        ac_plan.histograms = std::move(ac_hist);
        ac_plan.global = bitstream::build_ac_global(ac_plan.histograms.data(),
                                                    num_ac);
        ac_plan.ready = true;
    }
    const std::vector<std::uint8_t> dc_global{bitstream::build_dc_global(qp)};
    const bitstream::DcGroupBlobs blobs{
        bitstream::build_dc_group_blobs(width, height, dc_hist.data(), am_hist.data())};
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
    std::uint8_t* d_ac_context_map{nullptr};
    if (!upload(ac_plan.global.depth, &d_ac_depth) ||
        !upload(ac_plan.global.bits, &d_ac_bits) ||
        !upload(ac_plan.context_map, &d_ac_context_map) ||
        !upload(blobs.dc_depth, &d_dc_depth) || !upload(blobs.dc_bits, &d_dc_bits) ||
        !upload(blobs.acmeta_depth, &d_am_depth) || !upload(blobs.acmeta_bits, &d_am_bits) ||
        !upload(blobs.blob_pre, &d_pre) || !upload(blobs.blob_pre_off, &d_pre_off) ||
        !upload(blobs.blob_pre_bits, &d_pre_bits) || !upload(blobs.blob_mid, &d_mid) ||
        !upload(blobs.blob_mid_off, &d_mid_off) || !upload(blobs.blob_mid_bits, &d_mid_bits)) {
        return false;
    }
    for (void* p :
         {static_cast<void*>(d_ac_depth), static_cast<void*>(d_ac_bits),
          static_cast<void*>(d_dc_depth), static_cast<void*>(d_dc_bits),
          static_cast<void*>(d_am_depth), static_cast<void*>(d_am_bits), static_cast<void*>(d_pre),
          static_cast<void*>(d_pre_off), static_cast<void*>(d_pre_bits), static_cast<void*>(d_mid),
          static_cast<void*>(d_mid_off), static_cast<void*>(d_mid_bits),
          static_cast<void*>(d_ac_context_map)}) {
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
    if (!d_ac_body || !d_dc_body || !d_ac_sizes || !d_ac_offsets || !d_dc_sizes || !d_dc_offsets) {
        return false;
    }

    std::size_t ac_total{0};
    std::size_t dc_total{0};
    const bool ac_encoded{
        ac_plan.data_driven
            ? ac_encode_groups_runtime_map(
                  ac_device, width, height, d_ac_context_map, d_ac_depth,
                  d_ac_bits, ac_plan.global.depth.size() / AC_HISTOGRAM_SIZE,
                  d_ac_body, ac_capacity, d_ac_sizes, d_ac_offsets, &ac_total)
            : ac_encode_groups(ac_device, width, height, d_ac_depth, d_ac_bits,
                               ac_plan.global.depth.size(), d_ac_body, ac_capacity,
                               d_ac_sizes, d_ac_offsets, &ac_total)};
    if (!ac_encoded) {
        return false;
    }
    if (!dc_encode_groups(dc_device, width, height, quant_field, nullptr, nullptr, nullptr,
                          d_dc_depth, d_dc_bits, d_am_depth, d_am_bits, d_pre, d_pre_off, d_pre_bits,
                          d_mid, d_mid_off, d_mid_bits, d_dc_body, dc_capacity, d_dc_sizes,
                          d_dc_offsets, &dc_total)) {
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
    const std::size_t coeff_bytes{3 * bw * bh * AC_COEFFS_PER_BLOCK * sizeof(std::int16_t) +
                                  3 * bw * bh * sizeof(std::int32_t)};
    entropy.bytes_moved = 2 * coeff_bytes + ac_total + dc_total;

    // Section sizes in codestream order: DcGlobal, DcGroups, AcGlobal, AcGroups.
    const Clock::time_point assembly_head_start{Clock::now()};
    std::vector<std::uint32_t> section_sizes{};
    section_sizes.push_back(static_cast<std::uint32_t>(dc_global.size()));
    for (std::uint32_t s : dc_sizes) {
        section_sizes.push_back(s);
    }
    section_sizes.push_back(static_cast<std::uint32_t>(ac_plan.global.section.size()));
    for (std::uint32_t s : ac_sizes) {
        section_sizes.push_back(s);
    }

    const std::vector<std::uint8_t> head{bitstream::build_codestream_head(
        static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), section_sizes)};
    assembly.cpu_us += us_since(assembly_head_start);

    // Gather the body [DcGlobal | DcGroups | AcGlobal | AcGroups] into one device
    // buffer (byte-aligned sections -> byte concatenation), then a single D2H.
    const std::size_t body_size{dc_global.size() + dc_total +
                                ac_plan.global.section.size() + ac_total};
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
         cudaMemcpy(d_body + at, ac_plan.global.section.data(),
                    ac_plan.global.section.size(), cudaMemcpyHostToDevice) ==
             cudaSuccess) &&
        (at += ac_plan.global.section.size(),
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

bool encode_frame_m3(const std::int16_t* ac_device, const std::int32_t* dc_device,
                     const std::int8_t* acs, const std::int8_t* ytox_map,
                     const std::int8_t* ytob_map, std::size_t width, std::size_t height,
                     const bitstream::QuantParams& qp, const std::int32_t* quant_field,
                     std::vector<std::uint8_t>& out_file,
                     std::vector<StageTiming>* stats, AcClusteringMode clustering) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t num_ac{bitstream::ac_group_count(width, height)};
    const std::size_t num_dc{bitstream::dc_group_count(width, height)};
    if (num_ac <= 1) {
        return false;
    }
    int device{0};
    cudaGetDevice(&device);
    retain_default_mempool(device);

    DeviceScope scope{};
    StageTiming entropy{"entropy", 0, 0.0, 0.0};
    StageTiming assembly{"assembly", 0, 0.0, 0.0};
    AcEntropyPlan ac_plan{};
    if (clustering == AcClusteringMode::DATA_DRIVEN) {
        build_data_driven_ac_plan(ac_device, acs, width, height, num_ac, device,
                                  ac_plan, entropy.gpu_us, entropy.cpu_us);
    }
    const Clock::time_point entropy_gpu_start{Clock::now()};

    std::uint32_t* d_ac_hist{
        ac_plan.ready
            ? nullptr
            : scope.alloc<std::uint32_t>(AC_NUM_CLUSTERS * AC_HISTOGRAM_SIZE)};
    std::uint32_t* d_dc_hist{scope.alloc<std::uint32_t>(num_dc * AC_HISTOGRAM_SIZE)};
    std::uint32_t* d_am_hist{scope.alloc<std::uint32_t>(num_dc * AC_HISTOGRAM_SIZE)};
    if ((!ac_plan.ready && !d_ac_hist) || !d_dc_hist || !d_am_hist) {
        return false;
    }
    if ((!ac_plan.ready &&
         !ac_build_histogram_m3(ac_device, acs, width, height, d_ac_hist)) ||
        !dc_build_histograms(dc_device, width, height, d_dc_hist) ||
        !acmeta_build_histograms(quant_field, acs, ytox_map, ytob_map, width, height, d_am_hist)) {
        return false;
    }
    std::vector<std::uint32_t> ac_hist{};
    if (!ac_plan.ready) {
        ac_hist.assign(AC_NUM_CLUSTERS * AC_HISTOGRAM_SIZE, 0);
    }
    std::vector<std::uint32_t> dc_hist(num_dc * AC_HISTOGRAM_SIZE, 0);
    std::vector<std::uint32_t> am_hist(num_dc * AC_HISTOGRAM_SIZE, 0);
    std::vector<std::int8_t> acs_host(bw * bh, 8);
    if ((!ac_plan.ready &&
         cudaMemcpy(ac_hist.data(), d_ac_hist,
                    ac_hist.size() * sizeof(std::uint32_t),
                    cudaMemcpyDeviceToHost) != cudaSuccess) ||
        cudaMemcpy(dc_hist.data(), d_dc_hist, dc_hist.size() * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(am_hist.data(), d_am_hist, am_hist.size() * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(acs_host.data(), acs, bw * bh, cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    entropy.gpu_us += us_since(entropy_gpu_start);

    // Per-DcGroup first-block counts drive the AcMetadata `count` field.
    const std::size_t xdg{(bw + 255) / 256};
    std::vector<std::size_t> fbc(num_dc, 0);
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

    const Clock::time_point entropy_cpu_start{Clock::now()};
    if (!ac_plan.ready) {
        ac_plan.histograms = std::move(ac_hist);
        ac_plan.global = bitstream::build_ac_global(ac_plan.histograms.data(),
                                                    num_ac);
        ac_plan.ready = true;
    }
    const std::vector<std::uint8_t> dc_global{bitstream::build_dc_global(qp)};
    const bitstream::DcGroupBlobs blobs{bitstream::build_dc_group_blobs(
        width, height, dc_hist.data(), am_hist.data(), fbc.data())};
    entropy.cpu_us += us_since(entropy_cpu_start);

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
    std::uint8_t* d_ac_context_map{nullptr};
    if (!upload(ac_plan.global.depth, &d_ac_depth) ||
        !upload(ac_plan.global.bits, &d_ac_bits) ||
        !upload(ac_plan.context_map, &d_ac_context_map) ||
        !upload(blobs.dc_depth, &d_dc_depth) || !upload(blobs.dc_bits, &d_dc_bits) ||
        !upload(blobs.acmeta_depth, &d_am_depth) || !upload(blobs.acmeta_bits, &d_am_bits) ||
        !upload(blobs.blob_pre, &d_pre) || !upload(blobs.blob_pre_off, &d_pre_off) ||
        !upload(blobs.blob_pre_bits, &d_pre_bits) || !upload(blobs.blob_mid, &d_mid) ||
        !upload(blobs.blob_mid_off, &d_mid_off) || !upload(blobs.blob_mid_bits, &d_mid_bits)) {
        return false;
    }
    for (void* p :
         {static_cast<void*>(d_ac_depth), static_cast<void*>(d_ac_bits),
          static_cast<void*>(d_dc_depth), static_cast<void*>(d_dc_bits),
          static_cast<void*>(d_am_depth), static_cast<void*>(d_am_bits), static_cast<void*>(d_pre),
          static_cast<void*>(d_pre_off), static_cast<void*>(d_pre_bits), static_cast<void*>(d_mid),
          static_cast<void*>(d_mid_off), static_cast<void*>(d_mid_bits),
          static_cast<void*>(d_ac_context_map)}) {
        if (p) {
            scope.track(p);
        }
    }

    const std::size_t ac_capacity{3 * bw * bh * 384 + num_ac * 64 + 4096};
    const std::size_t dc_capacity{48 * bw * bh + blobs.blob_pre.size() + blobs.blob_mid.size() +
                                  num_dc * 64 + 4096};
    std::uint8_t* d_ac_body{scope.alloc<std::uint8_t>(ac_capacity)};
    std::uint8_t* d_dc_body{scope.alloc<std::uint8_t>(dc_capacity)};
    std::uint32_t* d_ac_sizes{scope.alloc<std::uint32_t>(num_ac)};
    std::uint32_t* d_ac_offsets{scope.alloc<std::uint32_t>(num_ac)};
    std::uint32_t* d_dc_sizes{scope.alloc<std::uint32_t>(num_dc)};
    std::uint32_t* d_dc_offsets{scope.alloc<std::uint32_t>(num_dc)};
    if (!d_ac_body || !d_dc_body || !d_ac_sizes || !d_ac_offsets || !d_dc_sizes || !d_dc_offsets) {
        return false;
    }

    std::size_t ac_total{0};
    std::size_t dc_total{0};
    const bool ac_encoded{
        ac_plan.data_driven
            ? ac_encode_groups_m3_runtime_map(
                  ac_device, acs, width, height, d_ac_context_map, d_ac_depth,
                  d_ac_bits, ac_plan.global.depth.size() / AC_HISTOGRAM_SIZE,
                  d_ac_body, ac_capacity, d_ac_sizes, d_ac_offsets, &ac_total)
            : ac_encode_groups_m3(ac_device, acs, width, height, d_ac_depth,
                                  d_ac_bits, d_ac_body, ac_capacity, d_ac_sizes,
                                  d_ac_offsets, &ac_total)};
    if (!ac_encoded) {
        return false;
    }
    if (!dc_encode_groups(dc_device, width, height, quant_field, acs, ytox_map, ytob_map, d_dc_depth,
                          d_dc_bits, d_am_depth, d_am_bits, d_pre, d_pre_off, d_pre_bits, d_mid,
                          d_mid_off, d_mid_bits, d_dc_body, dc_capacity, d_dc_sizes, d_dc_offsets,
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

    // Sections in codestream order: DcGlobal, DcGroups, AcGlobal, AcGroups.
    const Clock::time_point assembly_head_start{Clock::now()};
    std::vector<std::uint32_t> section_sizes{};
    section_sizes.push_back(static_cast<std::uint32_t>(dc_global.size()));
    for (std::uint32_t s : dc_sizes) {
        section_sizes.push_back(s);
    }
    section_sizes.push_back(static_cast<std::uint32_t>(ac_plan.global.section.size()));
    for (std::uint32_t s : ac_sizes) {
        section_sizes.push_back(s);
    }
    const std::vector<std::uint8_t> head{bitstream::build_codestream_head(
        static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height), section_sizes)};
    assembly.cpu_us += us_since(assembly_head_start);

    const std::size_t body_size{dc_global.size() + dc_total +
                                ac_plan.global.section.size() + ac_total};
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
         cudaMemcpy(d_body + at, ac_plan.global.section.data(),
                    ac_plan.global.section.size(), cudaMemcpyHostToDevice) ==
             cudaSuccess) &&
        (at += ac_plan.global.section.size(),
         cudaMemcpy(d_body + at, d_ac_body, ac_total, cudaMemcpyDeviceToDevice) == cudaSuccess)};
    if (!gathered) {
        return false;
    }
    assembly.gpu_us += us_since(assembly_gather_start);

    const std::size_t codestream_size{head.size() + body_size};
    std::vector<std::uint8_t> file{bitstream::container_framing(codestream_size)};
    const std::size_t framing{file.size()};
    file.resize(framing + codestream_size);
    std::copy(head.begin(), head.end(), file.begin() + framing);
    if (cudaMemcpy(file.data() + framing + head.size(), d_body, body_size,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    if (stats != nullptr) {
        stats->push_back(entropy);
        stats->push_back(assembly);
    }
    out_file = std::move(file);
    return true;
}

bitstream::QuantParams quant_params_for_distance(float distance) {
    const QuantCalibration cal{calibrate_quant(distance)};
    return bitstream::QuantParams{cal.global_scale, cal.quant_dc};
}

bool encode_nv12(const std::uint8_t* luma, std::size_t luma_pitch, const std::uint8_t* chroma,
                 std::size_t chroma_pitch, std::size_t width, std::size_t height,
                 std::int32_t device_ordinal, float distance, const bitstream::QuantParams& qp,
                 std::vector<std::uint8_t>& out_file, std::vector<StageTiming>* stats) {
    if (cudaSetDevice(device_ordinal) != cudaSuccess) {
        return false;
    }
    retain_default_mempool(device_ordinal);

    DeviceScope scope{};
    const std::size_t blocks{(width / 8) * (height / 8)};
    std::int16_t* d_ac{scope.alloc<std::int16_t>(3 * blocks * AC_COEFFS_PER_BLOCK)};
    std::int32_t* d_dc{scope.alloc<std::int32_t>(3 * blocks)};
    std::int32_t* d_qf{scope.alloc<std::int32_t>(blocks)};
    if (!d_ac || !d_dc || !d_qf) {
        return false;
    }

    // The chroma pitch2D texture requires a 32-byte-aligned row pitch; re-pitch
    // the plane into an aligned device buffer when the caller's is not.
    constexpr std::size_t CHROMA_PITCH_ALIGNMENT{32};
    const std::uint8_t* chroma_src{chroma};
    std::size_t chroma_src_pitch{chroma_pitch};
    if (chroma_pitch % CHROMA_PITCH_ALIGNMENT != 0) {
        // 512-byte row pitch satisfies any device's texture pitch alignment and
        // keeps the buffer in the stream-ordered pool (freed with cudaFreeAsync).
        const std::size_t aligned_pitch{(width + 511) & ~std::size_t{511}};
        std::uint8_t* aligned{scope.alloc<std::uint8_t>(aligned_pitch * (height / 2))};
        if (!aligned) {
            return false;
        }
        if (cudaMemcpy2D(aligned, aligned_pitch, chroma, chroma_pitch, width, height / 2,
                         cudaMemcpyDeviceToDevice) != cudaSuccess) {
            return false;
        }
        chroma_src = aligned;
        chroma_src_pitch = aligned_pitch;
    }

    const Clock::time_point frontend_start{Clock::now()};
    if (!encode_frontend(luma, luma_pitch, chroma_src, chroma_src_pitch, width, height, distance,
                         d_ac, d_dc, d_qf)) {
        return false;
    }
    if (stats != nullptr) {
        StageTiming frontend{"frontend", 0, us_since(frontend_start), 0.0};
        // Fused front-end DRAM traffic: NV12 read + quantized coefficient write
        // (int16 AC + int32 DC) + quant field write. The XYB and DCT intermediates
        // stay tile-resident.
        frontend.bytes_moved = width * height + width * height / 2 +
                               3 * blocks * AC_COEFFS_PER_BLOCK * sizeof(std::int16_t) +
                               3 * blocks * sizeof(std::int32_t) + blocks * sizeof(std::int32_t);
        stats->push_back(frontend);
    }
    return encode_frame(d_ac, d_dc, width, height, qp, d_qf, out_file, stats);
}

bool encode_nv12_m3(const std::uint8_t* luma, std::size_t luma_pitch, const std::uint8_t* chroma,
                    std::size_t chroma_pitch, std::size_t width, std::size_t height,
                    std::int32_t device_ordinal, float distance, const bitstream::QuantParams& qp,
                    std::vector<std::uint8_t>& out_file, std::vector<StageTiming>* stats) {
    if (cudaSetDevice(device_ordinal) != cudaSuccess) {
        return false;
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
    // The per-block adaptive-quant field is produced by the fused DCT8 front-end
    // (transform-independent); its coefficient outputs are scratch here.
    std::int16_t* d_ac_scratch{scope.alloc<std::int16_t>(3 * blocks * AC_COEFFS_PER_BLOCK)};
    std::int32_t* d_dc_scratch{scope.alloc<std::int32_t>(3 * blocks)};
    if (!d_coeffs || !d_acs || !d_mx || !d_mb || !d_qf || !d_ac16 || !d_dc32 || !d_ac_scratch ||
        !d_dc_scratch) {
        return false;
    }

    constexpr std::size_t CHROMA_PITCH_ALIGNMENT{32};
    const std::uint8_t* chroma_src{chroma};
    std::size_t chroma_src_pitch{chroma_pitch};
    if (chroma_pitch % CHROMA_PITCH_ALIGNMENT != 0) {
        const std::size_t aligned_pitch{(width + 511) & ~std::size_t{511}};
        std::uint8_t* aligned{scope.alloc<std::uint8_t>(aligned_pitch * (height / 2))};
        if (!aligned || cudaMemcpy2D(aligned, aligned_pitch, chroma, chroma_pitch, width, height / 2,
                                     cudaMemcpyDeviceToDevice) != cudaSuccess) {
            return false;
        }
        chroma_src = aligned;
        chroma_src_pitch = aligned_pitch;
    }

    const Clock::time_point frontend_start{Clock::now()};
    if (!encode_frontend(luma, luma_pitch, chroma_src, chroma_src_pitch, width, height, distance,
                         d_ac_scratch, d_dc_scratch, d_qf) ||
        !frontend_transform_m3(luma, luma_pitch, chroma_src, chroma_src_pitch, width, height,
                               distance, d_coeffs, d_acs) ||
        !estimate_cfl_covered(d_coeffs, d_acs, width, height, d_mx, d_mb) ||
        !quantize_residual_m3(d_coeffs, d_acs, d_mx, d_mb, d_qf, width, height, distance, d_ac16,
                              d_dc32)) {
        return false;
    }
    if (stats != nullptr) {
        stats->push_back(StageTiming{"frontend", 0, us_since(frontend_start), 0.0});
    }
    return encode_frame_m3(d_ac16, d_dc32, d_acs, d_mx, d_mb, width, height, qp, d_qf, out_file,
                           stats);
}

}  // namespace cujpegxl
