// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "entropy_roundtrip.h"

#include <cuda_runtime.h>

#include "ac_context.h"
#include "entropy.h"

namespace cujpegxl {
namespace {

// The device AC histogram and prefix code span all clusters (cluster*256+symbol).
constexpr std::size_t AC_HIST_SPAN = AC_NUM_CLUSTERS * AC_HISTOGRAM_SIZE;

template <typename T>
bool upload(const std::vector<T>& host, T** device) {
    if (cudaMalloc(device, host.size() * sizeof(T)) != cudaSuccess) {
        return false;
    }
    return cudaMemcpy(*device, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) ==
           cudaSuccess;
}

}  // namespace

static bool ac_encode_device_impl(
    const std::vector<std::int32_t>& q, std::size_t width, std::size_t height,
    const std::vector<std::uint8_t>* context_map, std::size_t num_clusters,
    const std::vector<std::uint8_t>& depth,
    const std::vector<std::uint16_t>& bits, AcDeviceResult& out) {
    const std::size_t num_groups{ac_num_groups(width, height)};
    const std::size_t blocks{(width / 8) * (height / 8)};
    const std::size_t plane{width * height};
    const std::size_t capacity{q.size() * 8 + 4096};
    const std::size_t histogram_span{num_clusters * AC_HISTOGRAM_SIZE};

    // Adapt the combined int32 coefficient layout into the packed int16 AC buffer
    // the device kernels now consume (DC slot elided; coefficient index k in
    // [1, 63] at slot k-1).
    std::vector<std::int16_t> ac(3 * blocks * AC_COEFFS_PER_BLOCK, 0);
    for (std::size_t c{0}; c < 3; ++c) {
        for (std::size_t b{0}; b < blocks; ++b) {
            for (std::size_t kk{1}; kk < 64; ++kk) {
                ac[c * blocks * AC_COEFFS_PER_BLOCK + b * AC_COEFFS_PER_BLOCK + (kk - 1)] =
                    static_cast<std::int16_t>(q[c * plane + b * 64 + kk]);
            }
        }
    }

    std::int16_t* d_ac{nullptr};
    std::uint8_t* d_depth{nullptr};
    std::uint16_t* d_bits{nullptr};
    std::uint32_t* d_hist{nullptr};
    std::uint32_t* d_sizes{nullptr};
    std::uint32_t* d_offsets{nullptr};
    std::uint8_t* d_out{nullptr};
    std::uint8_t* d_context_map{nullptr};
    std::uint32_t* d_context_hist{nullptr};

    bool ok{upload(ac, &d_ac) && upload(depth, &d_depth) && upload(bits, &d_bits) &&
            cudaMalloc(&d_hist, histogram_span * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_sizes, num_groups * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_offsets, num_groups * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_out, capacity) == cudaSuccess};
    if (ok && context_map != nullptr) {
        ok = upload(*context_map, &d_context_map) &&
             cudaMalloc(&d_context_hist,
                        AC_CONTEXT_HISTOGRAM_ENTRIES * sizeof(std::uint32_t)) == cudaSuccess;
    }

    std::size_t total_bytes{0};
    if (ok && context_map == nullptr) {
        ok = ac_build_histogram(d_ac, width, height, d_hist) &&
             ac_encode_groups(d_ac, width, height, d_depth, d_bits, depth.size(),
                              d_out, capacity, d_sizes, d_offsets, &total_bytes);
    } else if (ok) {
        ok = ac_build_context_histograms(d_ac, width, height, d_context_hist) &&
             ac_collapse_context_histograms(d_context_hist, d_context_map,
                                            num_clusters, d_hist) &&
             ac_encode_groups_runtime_map(
                 d_ac, width, height, d_context_map, d_depth, d_bits, num_clusters,
                 d_out, capacity, d_sizes, d_offsets, &total_bytes);
    }

    if (ok) {
        out.histogram.assign(histogram_span, 0);
        out.group_sizes.assign(num_groups, 0);
        out.group_offsets.assign(num_groups, 0);
        out.stream.assign(total_bytes, 0);
        ok = cudaMemcpy(out.histogram.data(), d_hist,
                        histogram_span * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.group_sizes.data(), d_sizes, num_groups * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.group_offsets.data(), d_offsets, num_groups * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.stream.data(), d_out, total_bytes, cudaMemcpyDeviceToHost) ==
                 cudaSuccess;
    }

    cudaFree(d_ac);
    cudaFree(d_depth);
    cudaFree(d_bits);
    cudaFree(d_hist);
    cudaFree(d_sizes);
    cudaFree(d_offsets);
    cudaFree(d_out);
    cudaFree(d_context_map);
    cudaFree(d_context_hist);
    return ok;
}

bool ac_encode_device(const std::vector<std::int32_t>& q, std::size_t width,
                      std::size_t height, const std::vector<std::uint8_t>& depth,
                      const std::vector<std::uint16_t>& bits,
                      AcDeviceResult& out) {
    return ac_encode_device_impl(q, width, height, nullptr, AC_NUM_CLUSTERS,
                                 depth, bits, out);
}

bool ac_encode_device_runtime_map(
    const std::vector<std::int32_t>& q, std::size_t width, std::size_t height,
    const std::vector<std::uint8_t>& context_map, std::size_t num_clusters,
    const std::vector<std::uint8_t>& depth,
    const std::vector<std::uint16_t>& bits, AcDeviceResult& out) {
    if (context_map.size() != AC_NUM_CONTEXTS || num_clusters == 0 ||
        num_clusters > 256 || depth.size() != num_clusters * AC_HISTOGRAM_SIZE ||
        bits.size() != depth.size()) {
        return false;
    }
    for (std::uint8_t cluster : context_map) {
        if (cluster >= num_clusters) {
            return false;
        }
    }
    return ac_encode_device_impl(q, width, height, &context_map, num_clusters,
                                 depth, bits, out);
}

static bool ac_encode_device_ans_impl(
    const std::vector<std::int32_t>& q, std::size_t width, std::size_t height,
    const std::vector<std::uint8_t>* context_map,
    const std::vector<bitstream::AnsEncodingTable>& host_tables,
    std::size_t capacity, AcDeviceResult& out) {
    const std::size_t num_groups{ac_num_groups(width, height)};
    const std::size_t blocks{(width / 8) * (height / 8)};
    const std::size_t plane{width * height};
    const std::size_t num_clusters{host_tables.size()};
    if (num_clusters == 0 || num_clusters > 256) {
        return false;
    }

    std::vector<std::int16_t> ac(3 * blocks * AC_COEFFS_PER_BLOCK, 0);
    for (std::size_t channel{0}; channel < 3; ++channel) {
        for (std::size_t block{0}; block < blocks; ++block) {
            for (std::size_t coefficient{1}; coefficient < 64; ++coefficient) {
                ac[channel * blocks * AC_COEFFS_PER_BLOCK +
                   block * AC_COEFFS_PER_BLOCK + coefficient - 1] =
                    static_cast<std::int16_t>(
                        q[channel * plane + block * 64 + coefficient]);
            }
        }
    }
    std::vector<AcAnsEncodingTable> tables(num_clusters);
    for (std::size_t cluster{0}; cluster < num_clusters; ++cluster) {
        for (std::size_t symbol{0}; symbol < AC_HISTOGRAM_SIZE; ++symbol) {
            tables[cluster].frequencies[symbol] =
                host_tables[cluster].frequencies[symbol];
        }
        for (std::size_t symbol{0}; symbol <= AC_HISTOGRAM_SIZE; ++symbol) {
            tables[cluster].offsets[symbol] = host_tables[cluster].offsets[symbol];
        }
        for (std::size_t value{0}; value < 4096; ++value) {
            tables[cluster].reverse_map[value] =
                host_tables[cluster].reverse_map[value];
        }
    }

    std::int16_t* d_ac{nullptr};
    AcAnsEncodingTable* d_tables{nullptr};
    std::uint8_t* d_context_map{nullptr};
    std::uint32_t* d_sizes{nullptr};
    std::uint32_t* d_offsets{nullptr};
    std::uint8_t* d_out{nullptr};
    bool ok{upload(ac, &d_ac) && upload(tables, &d_tables) &&
            cudaMalloc(&d_sizes, num_groups * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_offsets, num_groups * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_out, capacity) == cudaSuccess};
    if (ok && context_map != nullptr) {
        ok = upload(*context_map, &d_context_map);
    }

    std::size_t total_bytes{0};
    if (ok && context_map == nullptr) {
        ok = ac_encode_groups_ans(d_ac, width, height, d_tables, num_clusters,
                                  d_out, capacity, d_sizes, d_offsets,
                                  &total_bytes);
    } else if (ok) {
        ok = ac_encode_groups_ans_runtime_map(
            d_ac, width, height, d_context_map, d_tables, num_clusters, d_out,
            capacity, d_sizes, d_offsets, &total_bytes);
    }
    if (ok) {
        out.group_sizes.assign(num_groups, 0);
        out.group_offsets.assign(num_groups, 0);
        out.stream.assign(total_bytes, 0);
        ok = cudaMemcpy(out.group_sizes.data(), d_sizes,
                        num_groups * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.group_offsets.data(), d_offsets,
                        num_groups * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.stream.data(), d_out, total_bytes,
                        cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(d_ac);
    cudaFree(d_tables);
    cudaFree(d_context_map);
    cudaFree(d_sizes);
    cudaFree(d_offsets);
    cudaFree(d_out);
    return ok;
}

bool ac_encode_device_ans(
    const std::vector<std::int32_t>& q, std::size_t width, std::size_t height,
    const std::vector<bitstream::AnsEncodingTable>& tables,
    AcDeviceResult& out) {
    return ac_encode_device_ans_impl(q, width, height, nullptr, tables,
                                     q.size() * 8 + 4096, out);
}

bool ac_encode_device_ans_with_capacity(
    const std::vector<std::int32_t>& q, std::size_t width, std::size_t height,
    const std::vector<bitstream::AnsEncodingTable>& tables, std::size_t capacity,
    AcDeviceResult& out) {
    return ac_encode_device_ans_impl(q, width, height, nullptr, tables, capacity, out);
}

bool ac_encode_device_ans_runtime_map(
    const std::vector<std::int32_t>& q, std::size_t width, std::size_t height,
    const std::vector<std::uint8_t>& context_map,
    const std::vector<bitstream::AnsEncodingTable>& tables,
    AcDeviceResult& out) {
    if (context_map.size() != AC_NUM_CONTEXTS) {
        return false;
    }
    for (std::uint8_t cluster : context_map) {
        if (cluster >= tables.size()) {
            return false;
        }
    }
    return ac_encode_device_ans_impl(q, width, height, &context_map, tables,
                                     q.size() * 8 + 4096, out);
}

bool ac_encode_device_m3(const std::vector<std::int32_t>& q, const std::vector<std::int8_t>& acs,
                         std::size_t width, std::size_t height,
                         const std::vector<std::uint8_t>& depth,
                         const std::vector<std::uint16_t>& bits, AcDeviceResult& out) {
    const std::size_t num_groups{ac_num_groups(width, height)};
    const std::size_t blocks{(width / 8) * (height / 8)};
    const std::size_t capacity{q.size() * 8 + 4096};

    // The covered-block layout is consumed directly (int32 -> int16, three planes
    // of blocks*64 slots each).
    std::vector<std::int16_t> ac(q.size());
    for (std::size_t i{0}; i < q.size(); ++i) {
        ac[i] = static_cast<std::int16_t>(q[i]);
    }

    std::int16_t* d_ac{nullptr};
    std::int8_t* d_acs{nullptr};
    std::uint8_t* d_depth{nullptr};
    std::uint16_t* d_bits{nullptr};
    std::uint32_t* d_hist{nullptr};
    std::uint32_t* d_sizes{nullptr};
    std::uint32_t* d_offsets{nullptr};
    std::uint8_t* d_out{nullptr};

    bool ok{upload(ac, &d_ac) && upload(acs, &d_acs) && upload(depth, &d_depth) &&
            upload(bits, &d_bits) &&
            cudaMalloc(&d_hist, AC_HIST_SPAN * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_sizes, num_groups * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_offsets, num_groups * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_out, capacity) == cudaSuccess};
    (void)blocks;

    std::size_t total_bytes{0};
    ok = ok && ac_build_histogram_m3(d_ac, d_acs, width, height, d_hist);
    ok = ok && ac_encode_groups_m3(d_ac, d_acs, width, height, d_depth, d_bits, d_out, capacity,
                                   d_sizes, d_offsets, &total_bytes);

    if (ok) {
        out.histogram.assign(AC_HIST_SPAN, 0);
        out.group_sizes.assign(num_groups, 0);
        out.group_offsets.assign(num_groups, 0);
        out.stream.assign(total_bytes, 0);
        ok = cudaMemcpy(out.histogram.data(), d_hist, AC_HIST_SPAN * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.group_sizes.data(), d_sizes, num_groups * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.group_offsets.data(), d_offsets, num_groups * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.stream.data(), d_out, total_bytes, cudaMemcpyDeviceToHost) ==
                 cudaSuccess;
    }

    cudaFree(d_ac);
    cudaFree(d_acs);
    cudaFree(d_depth);
    cudaFree(d_bits);
    cudaFree(d_hist);
    cudaFree(d_sizes);
    cudaFree(d_offsets);
    cudaFree(d_out);
    return ok;
}

bool ac_encode_device_m3_ans(
    const std::vector<std::int32_t>& q, const std::vector<std::int8_t>& acs,
    std::size_t width, std::size_t height,
    const std::vector<bitstream::AnsEncodingTable>& host_tables,
    AcDeviceResult& out) {
    const std::size_t num_groups{ac_num_groups(width, height)};
    const std::size_t capacity{q.size() * 8 + 4096};
    const std::size_t num_clusters{host_tables.size()};
    if (num_clusters == 0 || num_clusters > 256) {
        return false;
    }
    std::vector<std::int16_t> ac(q.size());
    for (std::size_t i{0}; i < q.size(); ++i) {
        ac[i] = static_cast<std::int16_t>(q[i]);
    }
    std::vector<AcAnsEncodingTable> tables(num_clusters);
    for (std::size_t cluster{0}; cluster < num_clusters; ++cluster) {
        for (std::size_t symbol{0}; symbol < AC_HISTOGRAM_SIZE; ++symbol) {
            tables[cluster].frequencies[symbol] =
                host_tables[cluster].frequencies[symbol];
        }
        for (std::size_t symbol{0}; symbol <= AC_HISTOGRAM_SIZE; ++symbol) {
            tables[cluster].offsets[symbol] = host_tables[cluster].offsets[symbol];
        }
        for (std::size_t value{0}; value < 4096; ++value) {
            tables[cluster].reverse_map[value] =
                host_tables[cluster].reverse_map[value];
        }
    }

    std::int16_t* d_ac{nullptr};
    std::int8_t* d_acs{nullptr};
    AcAnsEncodingTable* d_tables{nullptr};
    std::uint32_t* d_sizes{nullptr};
    std::uint32_t* d_offsets{nullptr};
    std::uint8_t* d_out{nullptr};
    bool ok{upload(ac, &d_ac) && upload(acs, &d_acs) &&
            upload(tables, &d_tables) &&
            cudaMalloc(&d_sizes, num_groups * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_offsets, num_groups * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_out, capacity) == cudaSuccess};
    std::size_t total_bytes{0};
    ok = ok && ac_encode_groups_m3_ans(
                   d_ac, d_acs, width, height, d_tables, num_clusters, d_out,
                   capacity, d_sizes, d_offsets, &total_bytes);
    if (ok) {
        out.group_sizes.assign(num_groups, 0);
        out.group_offsets.assign(num_groups, 0);
        out.stream.assign(total_bytes, 0);
        ok = cudaMemcpy(out.group_sizes.data(), d_sizes,
                        num_groups * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.group_offsets.data(), d_offsets,
                        num_groups * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.stream.data(), d_out, total_bytes,
                        cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(d_ac);
    cudaFree(d_acs);
    cudaFree(d_tables);
    cudaFree(d_sizes);
    cudaFree(d_offsets);
    cudaFree(d_out);
    return ok;
}

bool ac_context_histogram_device(const std::vector<std::int32_t>& q, std::size_t width,
                                 std::size_t height, std::vector<std::uint32_t>& out) {
    const std::size_t blocks{(width / 8) * (height / 8)};
    const std::size_t plane{width * height};
    std::vector<std::int16_t> ac(3 * blocks * AC_COEFFS_PER_BLOCK, 0);
    for (std::size_t c{0}; c < 3; ++c) {
        for (std::size_t b{0}; b < blocks; ++b) {
            for (std::size_t k{1}; k < 64; ++k) {
                ac[c * blocks * AC_COEFFS_PER_BLOCK + b * AC_COEFFS_PER_BLOCK + (k - 1)] =
                    static_cast<std::int16_t>(q[c * plane + b * 64 + k]);
            }
        }
    }

    std::int16_t* d_ac{nullptr};
    std::uint32_t* d_histograms{nullptr};
    bool ok{upload(ac, &d_ac) &&
            cudaMalloc(&d_histograms, AC_CONTEXT_HISTOGRAM_ENTRIES * sizeof(std::uint32_t)) ==
                cudaSuccess};
    ok = ok && ac_build_context_histograms(d_ac, width, height, d_histograms);
    if (ok) {
        out.assign(AC_CONTEXT_HISTOGRAM_ENTRIES, 0);
        ok = cudaMemcpy(out.data(), d_histograms,
                        AC_CONTEXT_HISTOGRAM_ENTRIES * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(d_ac);
    cudaFree(d_histograms);
    return ok;
}

bool ac_context_histogram_device_m3(const std::vector<std::int32_t>& q,
                                    const std::vector<std::int8_t>& acs, std::size_t width,
                                    std::size_t height, std::vector<std::uint32_t>& out) {
    std::vector<std::int16_t> ac(q.size());
    for (std::size_t i{0}; i < q.size(); ++i) {
        ac[i] = static_cast<std::int16_t>(q[i]);
    }

    std::int16_t* d_ac{nullptr};
    std::int8_t* d_acs{nullptr};
    std::uint32_t* d_histograms{nullptr};
    bool ok{upload(ac, &d_ac) && upload(acs, &d_acs) &&
            cudaMalloc(&d_histograms, AC_CONTEXT_HISTOGRAM_ENTRIES * sizeof(std::uint32_t)) ==
                cudaSuccess};
    ok = ok && ac_build_context_histograms_m3(d_ac, d_acs, width, height, d_histograms);
    if (ok) {
        out.assign(AC_CONTEXT_HISTOGRAM_ENTRIES, 0);
        ok = cudaMemcpy(out.data(), d_histograms,
                        AC_CONTEXT_HISTOGRAM_ENTRIES * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(d_ac);
    cudaFree(d_acs);
    cudaFree(d_histograms);
    return ok;
}

}  // namespace cujpegxl
