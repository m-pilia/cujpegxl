// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "entropy_roundtrip.h"

#include <cuda_runtime.h>

#include "entropy.h"

namespace cujpegxl {
namespace {

template <typename T>
bool upload(const std::vector<T>& host, T** device) {
    if (cudaMalloc(device, host.size() * sizeof(T)) != cudaSuccess) {
        return false;
    }
    return cudaMemcpy(*device, host.data(), host.size() * sizeof(T),
                      cudaMemcpyHostToDevice) == cudaSuccess;
}

}  // namespace

bool ac_encode_device(const std::vector<std::int32_t>& q, std::size_t width,
                      std::size_t height, const std::vector<std::uint8_t>& depth,
                      const std::vector<std::uint16_t>& bits, AcDeviceResult& out) {
    const std::size_t num_groups{ac_num_groups(width, height)};
    const std::size_t capacity{q.size() * 8 + 4096};

    std::int32_t* d_q{nullptr};
    std::uint8_t* d_depth{nullptr};
    std::uint16_t* d_bits{nullptr};
    std::uint32_t* d_hist{nullptr};
    std::uint32_t* d_sizes{nullptr};
    std::uint32_t* d_offsets{nullptr};
    std::uint8_t* d_out{nullptr};

    bool ok{upload(q, &d_q) && upload(depth, &d_depth) && upload(bits, &d_bits) &&
            cudaMalloc(&d_hist, AC_HISTOGRAM_SIZE * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_sizes, num_groups * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_offsets, num_groups * sizeof(std::uint32_t)) == cudaSuccess &&
            cudaMalloc(&d_out, capacity) == cudaSuccess};

    std::size_t total_bytes{0};
    ok = ok && ac_build_histogram(d_q, width, height, d_hist);
    ok = ok && ac_encode_groups(d_q, width, height, d_depth, d_bits, depth.size(), d_out,
                                capacity, d_sizes, d_offsets, &total_bytes);

    if (ok) {
        out.histogram.assign(AC_HISTOGRAM_SIZE, 0);
        out.group_sizes.assign(num_groups, 0);
        out.group_offsets.assign(num_groups, 0);
        out.stream.assign(total_bytes, 0);
        ok = cudaMemcpy(out.histogram.data(), d_hist,
                        AC_HISTOGRAM_SIZE * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.group_sizes.data(), d_sizes, num_groups * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.group_offsets.data(), d_offsets,
                        num_groups * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost) == cudaSuccess &&
             cudaMemcpy(out.stream.data(), d_out, total_bytes,
                        cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    cudaFree(d_q);
    cudaFree(d_depth);
    cudaFree(d_bits);
    cudaFree(d_hist);
    cudaFree(d_sizes);
    cudaFree(d_offsets);
    cudaFree(d_out);
    return ok;
}

}  // namespace cujpegxl
