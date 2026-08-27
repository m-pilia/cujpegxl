// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "prefix_build.h"

#include <cuda_runtime.h>

#include "src/bitstream/bit_sink.h"
#include "src/bitstream/frame_assembly.h"
#include "src/bitstream/serialize_impl.h"
#include "ac_context.h"

namespace cujpegxl {
namespace {

using bitstream::HISTOGRAM_STRIDE;
using bitstream::SpanBitWriter;

constexpr std::size_t DC_GROUP_BLOCKS = 256;

std::size_t ceil_div(std::size_t a, std::size_t b) {
    return (a + b - 1) / b;
}

// Shared-memory staging for the AcGlobal section: the serial bit writer runs
// against shared memory (~tens of cycles per bit-write) rather than global
// (~hundreds), then the section is copied out in parallel. Sized well above the
// real section (~1.4 KB) and the structural worst case, within the Pascal 48 KB
// per-block shared budget for this single-block kernel.
constexpr std::size_t AC_SECTION_SHMEM = 24576;

// Byte slot reserved per cluster for its serialized prefix-code description. A
// 256-symbol code description is a few hundred bits; this bounds it with margin.
constexpr std::size_t CLUSTER_BLOB_STRIDE = 512;

// One block per cluster (one thread): the per-cluster Huffman build and code
// serialization have data-dependent control flow, so distinct blocks keep them on
// independent warps/SMs instead of serializing by SIMT divergence within one warp
// (and one thread per block sidesteps the register-per-block launch limit). Each
// block emits its cluster's code description into its own blob slot, so the
// dominant per-cluster serialization is done in parallel here rather than serially
// in the serialize kernel.
__global__ void ac_build_codes_kernel(const std::uint32_t* ac_hist, std::uint8_t* depth,
                                       std::uint16_t* bits, std::uint8_t* cluster_blob,
                                       std::uint32_t* cluster_nbits) {
    bitstream::detail::build_cluster_code_and_blob(ac_hist, HISTOGRAM_STRIDE, blockIdx.x, depth,
                                                   bits, cluster_blob + blockIdx.x * CLUSTER_BLOB_STRIDE,
                                                   CLUSTER_BLOB_STRIDE, &cluster_nbits[blockIdx.x]);
}

// Serializes the AcGlobal section into shared memory by concatenating the
// precomputed context-map and per-cluster code blobs (no per-cluster Huffman work
// here), then copies it to global in parallel.
__global__ void ac_serialize_kernel(const std::uint32_t* ac_hist, std::size_t num_ac_groups,
                                     const std::uint8_t* ctxmap_blob, std::size_t ctxmap_nbits,
                                     const std::uint8_t* cluster_blob,
                                     const std::uint32_t* cluster_nbits, std::uint8_t* section,
                                     std::uint32_t* section_len) {
    __shared__ std::uint8_t s_section[AC_SECTION_SHMEM];
    __shared__ unsigned s_len;
    if (threadIdx.x == 0) {
        SpanBitWriter w{s_section, AC_SECTION_SHMEM};
        bitstream::detail::serialize_ac_global_prebuilt_blobs(
            w, ac_hist, ctxmap_blob, ctxmap_nbits, AC_NUM_CLUSTERS, num_ac_groups, cluster_blob,
            CLUSTER_BLOB_STRIDE, cluster_nbits);
        s_len = static_cast<unsigned>(w.size_bytes());
        *section_len = s_len;
    }
    __syncthreads();
    for (unsigned i{threadIdx.x}; i < s_len; i += blockDim.x) {
        section[i] = s_section[i];
    }
}

// One thread per Dc group: zero the group's prefix-code rows, then serialize the
// pre/mid header blobs into their fixed strides. Byte-identical to
// build_dc_group_blobs.
__global__ void dc_group_blobs_kernel(std::size_t bw, std::size_t bh, std::size_t xdg,
                                      std::size_t num_groups, const std::uint32_t* dc_hist,
                                      const std::uint32_t* am_hist, const std::size_t* fbc,
                                      std::uint8_t* dc_depth, std::uint16_t* dc_bits,
                                      std::uint8_t* am_depth, std::uint16_t* am_bits,
                                      std::uint8_t* blob_pre, std::uint32_t* blob_pre_off,
                                      std::uint32_t* blob_pre_bits, std::uint8_t* blob_mid,
                                      std::uint32_t* blob_mid_off, std::uint32_t* blob_mid_bits) {
    const std::size_t g{blockIdx.x * blockDim.x + threadIdx.x};
    if (g >= num_groups) {
        return;
    }
    const std::size_t gx{g % xdg};
    const std::size_t gy{g / xdg};
    const std::size_t dgw{
        DC_GROUP_BLOCKS < bw - gx * DC_GROUP_BLOCKS ? DC_GROUP_BLOCKS : bw - gx * DC_GROUP_BLOCKS};
    const std::size_t dgh{
        DC_GROUP_BLOCKS < bh - gy * DC_GROUP_BLOCKS ? DC_GROUP_BLOCKS : bh - gy * DC_GROUP_BLOCKS};
    const std::size_t total_blocks{dgw * dgh};
    const std::size_t count{fbc[g]};

    std::uint8_t* dc_depth_row{dc_depth + g * HISTOGRAM_STRIDE};
    std::uint16_t* dc_bits_row{dc_bits + g * HISTOGRAM_STRIDE};
    std::uint8_t* am_depth_row{am_depth + g * HISTOGRAM_STRIDE};
    std::uint16_t* am_bits_row{am_bits + g * HISTOGRAM_STRIDE};
    for (std::size_t i{0}; i < HISTOGRAM_STRIDE; ++i) {
        dc_depth_row[i] = 0;
        dc_bits_row[i] = 0;
        am_depth_row[i] = 0;
        am_bits_row[i] = 0;
    }

    SpanBitWriter pre{blob_pre + g * DC_BLOB_STRIDE, DC_BLOB_STRIDE};
    bitstream::detail::serialize_dc_pre(pre, dc_hist + g * HISTOGRAM_STRIDE, dc_depth_row,
                                        dc_bits_row);
    blob_pre_off[g] = static_cast<std::uint32_t>(g * DC_BLOB_STRIDE);
    blob_pre_bits[g] = static_cast<std::uint32_t>(pre.bits_written());

    SpanBitWriter mid{blob_mid + g * DC_BLOB_STRIDE, DC_BLOB_STRIDE};
    bitstream::detail::serialize_dc_mid(mid, am_hist + g * HISTOGRAM_STRIDE, total_blocks, count,
                                        am_depth_row, am_bits_row);
    blob_mid_off[g] = static_cast<std::uint32_t>(g * DC_BLOB_STRIDE);
    blob_mid_bits[g] = static_cast<std::uint32_t>(mid.bits_written());
}

}  // namespace

bool build_ac_global_device(const std::uint32_t* d_ac_hist, std::size_t num_ac_groups,
                            const std::uint8_t* d_ctxmap_blob, std::size_t ctxmap_nbits,
                            std::uint8_t* d_depth, std::uint16_t* d_bits, std::uint8_t* d_section,
                            std::size_t section_cap, std::uint32_t* d_section_len) {
    assert(section_cap >= AC_SECTION_SHMEM);
    (void)section_cap;
    std::uint8_t* d_cluster_blob{nullptr};
    std::uint32_t* d_cluster_nbits{nullptr};
    if (cudaMallocAsync(&d_cluster_blob, AC_NUM_CLUSTERS * CLUSTER_BLOB_STRIDE, 0) != cudaSuccess ||
        cudaMallocAsync(&d_cluster_nbits, AC_NUM_CLUSTERS * sizeof(std::uint32_t), 0) !=
            cudaSuccess) {
        return false;
    }
    ac_build_codes_kernel<<<AC_NUM_CLUSTERS, 1>>>(d_ac_hist, d_depth, d_bits, d_cluster_blob,
                                                  d_cluster_nbits);
    ac_serialize_kernel<<<1, 128>>>(d_ac_hist, num_ac_groups, d_ctxmap_blob, ctxmap_nbits,
                                    d_cluster_blob, d_cluster_nbits, d_section, d_section_len);
    const bool ok{cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess};
    cudaFreeAsync(d_cluster_blob, 0);
    cudaFreeAsync(d_cluster_nbits, 0);
    return ok;
}

bool build_dc_group_blobs_device(std::size_t width, std::size_t height,
                                 const std::uint32_t* d_dc_hist, const std::uint32_t* d_am_hist,
                                 const std::size_t* d_fbc, std::uint8_t* d_dc_depth,
                                 std::uint16_t* d_dc_bits, std::uint8_t* d_am_depth,
                                 std::uint16_t* d_am_bits, std::uint8_t* d_blob_pre,
                                 std::uint32_t* d_blob_pre_off, std::uint32_t* d_blob_pre_bits,
                                 std::uint8_t* d_blob_mid, std::uint32_t* d_blob_mid_off,
                                 std::uint32_t* d_blob_mid_bits) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t xdg{ceil_div(bw, DC_GROUP_BLOCKS)};
    const std::size_t ydg{ceil_div(bh, DC_GROUP_BLOCKS)};
    const std::size_t num_groups{xdg * ydg};

    constexpr unsigned threads{64};
    const unsigned blocks{static_cast<unsigned>((num_groups + threads - 1) / threads)};
    dc_group_blobs_kernel<<<blocks, threads>>>(
        bw, bh, xdg, num_groups, d_dc_hist, d_am_hist, d_fbc, d_dc_depth, d_dc_bits, d_am_depth,
        d_am_bits, d_blob_pre, d_blob_pre_off, d_blob_pre_bits, d_blob_mid, d_blob_mid_off,
        d_blob_mid_bits);
    return cudaGetLastError() == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess;
}

}  // namespace cujpegxl
