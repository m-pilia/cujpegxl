// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "dc_entropy_roundtrip.h"

#include <cuda_runtime.h>

#include "entropy.h"

namespace cujpegxl {
namespace {

template <typename T>
bool upload(const std::vector<T>& host, T** device) {
    if (host.empty()) {
        *device = nullptr;
        return true;
    }
    if (cudaMalloc(device, host.size() * sizeof(T)) != cudaSuccess) {
        return false;
    }
    return cudaMemcpy(*device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) ==
           cudaSuccess;
}

// Copies a per-group prefix-code table (length <= AC_HISTOGRAM_SIZE) into slot
// g of a flattened num_groups * AC_HISTOGRAM_SIZE buffer.
template <typename T>
void place_code(std::vector<T>& flat, std::size_t g, const std::vector<T>& code) {
    for (std::size_t i{0}; i < code.size(); ++i) {
        flat[g * AC_HISTOGRAM_SIZE + i] = code[i];
    }
}

}  // namespace

bool dc_encode_device(const std::vector<std::int32_t>& dc, const std::vector<std::int8_t>& acs,
                         const std::vector<std::int8_t>& ytox_map,
                         const std::vector<std::int8_t>& ytob_map,
                         const std::vector<std::int32_t>& quant_field, std::size_t width,
                         std::size_t height, const bitstream::DcReference& ref,
                         DcDeviceResult& out) {
    const std::size_t num_groups{ref.groups.size()};
    const std::size_t code_span{num_groups * AC_HISTOGRAM_SIZE};
    const std::size_t blocks{(width / 8) * (height / 8)};
    const std::size_t capacity{16 * blocks + num_groups * 128 + 8192};

    std::vector<std::uint8_t> dc_depth(code_span, 0);
    std::vector<std::uint16_t> dc_bits(code_span, 0);
    std::vector<std::uint8_t> acmeta_depth(code_span, 0);
    std::vector<std::uint16_t> acmeta_bits(code_span, 0);
    std::vector<std::uint8_t> blob_pre{};
    std::vector<std::uint32_t> blob_pre_off(num_groups, 0);
    std::vector<std::uint32_t> blob_pre_bits(num_groups, 0);
    std::vector<std::uint8_t> blob_mid{};
    std::vector<std::uint32_t> blob_mid_off(num_groups, 0);
    std::vector<std::uint32_t> blob_mid_bits(num_groups, 0);
    for (std::size_t g{0}; g < num_groups; ++g) {
        const bitstream::DcGroupReference& r{ref.groups[g]};
        place_code(dc_depth, g, r.dc_depth);
        place_code(dc_bits, g, r.dc_bits);
        place_code(acmeta_depth, g, r.acmeta_depth);
        place_code(acmeta_bits, g, r.acmeta_bits);
        blob_pre_off[g] = static_cast<std::uint32_t>(blob_pre.size());
        blob_pre_bits[g] = static_cast<std::uint32_t>(r.blob_pre_bits);
        blob_pre.insert(blob_pre.end(), r.blob_pre.begin(), r.blob_pre.end());
        blob_mid_off[g] = static_cast<std::uint32_t>(blob_mid.size());
        blob_mid_bits[g] = static_cast<std::uint32_t>(r.blob_mid_bits);
        blob_mid.insert(blob_mid.end(), r.blob_mid.begin(), r.blob_mid.end());
    }

    std::int32_t* d_dc{nullptr};
    std::int8_t* d_acs{nullptr};
    std::int8_t* d_mx{nullptr};
    std::int8_t* d_mb{nullptr};
    std::int32_t* d_qf{nullptr};
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
    std::uint32_t* d_dc_hist{nullptr};
    std::uint32_t* d_am_hist{nullptr};
    std::uint32_t* d_sizes{nullptr};
    std::uint32_t* d_offsets{nullptr};
    std::uint8_t* d_out{nullptr};

    bool ok{upload(dc, &d_dc) && upload(acs, &d_acs) && upload(ytox_map, &d_mx) &&
            upload(ytob_map, &d_mb) && upload(quant_field, &d_qf) && upload(dc_depth, &d_dc_depth) &&
            upload(dc_bits, &d_dc_bits) && upload(acmeta_depth, &d_am_depth) &&
            upload(acmeta_bits, &d_am_bits) && upload(blob_pre, &d_pre) &&
            upload(blob_pre_off, &d_pre_off) && upload(blob_pre_bits, &d_pre_bits) &&
            upload(blob_mid, &d_mid) && upload(blob_mid_off, &d_mid_off) &&
            upload(blob_mid_bits, &d_mid_bits) &&
            cudaMalloc(&d_dc_hist, code_span * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_am_hist, code_span * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_sizes, num_groups * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_offsets, num_groups * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_out, capacity) == cudaSuccess};

    std::size_t total_bytes{0};
    ok = ok && dc_build_histograms(d_dc, width, height, d_dc_hist);
    ok = ok && acmeta_build_histograms(d_qf, d_acs, d_mx, d_mb, width, height, d_am_hist);
    ok = ok && dc_encode_groups(d_dc, width, height, d_qf, d_acs, d_mx, d_mb, d_dc_depth, d_dc_bits,
                                d_am_depth, d_am_bits, d_pre, d_pre_off, d_pre_bits, d_mid,
                                d_mid_off, d_mid_bits, d_out, capacity, d_sizes, d_offsets,
                                &total_bytes);

    if (ok) {
        std::vector<std::uint32_t> flat_dc(code_span, 0);
        std::vector<std::uint32_t> flat_am(code_span, 0);
        out.group_sizes.assign(num_groups, 0);
        out.group_offsets.assign(num_groups, 0);
        out.stream.assign(total_bytes, 0);
        ok = cudaMemcpy(flat_dc.data(), d_dc_hist, code_span * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(flat_am.data(), d_am_hist, code_span * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.group_sizes.data(), d_sizes, num_groups * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.group_offsets.data(), d_offsets, num_groups * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.stream.data(), d_out, total_bytes, cudaMemcpyDeviceToHost) ==
                 cudaSuccess;
        if (ok) {
            out.histograms.assign(num_groups, {});
            out.acmeta_histograms.assign(num_groups, {});
            for (std::size_t g{0}; g < num_groups; ++g) {
                out.histograms[g].assign(flat_dc.begin() + g * AC_HISTOGRAM_SIZE,
                                         flat_dc.begin() + (g + 1) * AC_HISTOGRAM_SIZE);
                out.acmeta_histograms[g].assign(flat_am.begin() + g * AC_HISTOGRAM_SIZE,
                                                flat_am.begin() + (g + 1) * AC_HISTOGRAM_SIZE);
            }
        }
    }

    cudaFree(d_dc);
    cudaFree(d_acs);
    cudaFree(d_mx);
    cudaFree(d_mb);
    cudaFree(d_qf);
    cudaFree(d_dc_depth);
    cudaFree(d_dc_bits);
    cudaFree(d_am_depth);
    cudaFree(d_am_bits);
    cudaFree(d_pre);
    cudaFree(d_pre_off);
    cudaFree(d_pre_bits);
    cudaFree(d_mid);
    cudaFree(d_mid_off);
    cudaFree(d_mid_bits);
    cudaFree(d_dc_hist);
    cudaFree(d_am_hist);
    cudaFree(d_sizes);
    cudaFree(d_offsets);
    cudaFree(d_out);
    return ok;
}

}  // namespace cujpegxl
