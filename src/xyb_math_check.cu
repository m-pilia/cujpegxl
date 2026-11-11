// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "xyb_math_check.h"

#include <cuda_runtime.h>

#include "xyb_impl.cuh"

namespace cujpegxl {
namespace {

struct CbrtPair {
    __device__ float candidate(float v) const { return cbrt_normal(v); }
    __device__ float reference(float v) const { return cbrtf(v); }
};

struct PowPair {
    float exponent;
    __device__ float candidate(float v) const { return pow_normal(v, exponent); }
    __device__ float reference(float v) const { return __powf(v, exponent); }
};

template <typename Pair>
__global__ void mismatch_kernel(Pair pair, std::uint32_t first_bits, std::uint32_t last_bits,
                                Mismatches* __restrict__ mismatches) {
    const std::uint64_t stride{static_cast<std::uint64_t>(gridDim.x) * blockDim.x};
    for (std::uint64_t bits{first_bits + static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
                            threadIdx.x};
         bits <= last_bits; bits += stride) {
        const float v{__uint_as_float(static_cast<std::uint32_t>(bits))};
        if (__float_as_uint(pair.candidate(v)) != __float_as_uint(pair.reference(v))) {
            atomicAdd(&mismatches->count, 1u);
            atomicMin(&mismatches->first_bits, static_cast<std::uint32_t>(bits));
        }
    }
}

template <typename Pair>
std::optional<Mismatches> count_mismatches(Pair pair, std::uint32_t first_bits,
                                           std::uint32_t last_bits) {
    Mismatches* device{};
    if (cudaMalloc(&device, sizeof(Mismatches)) != cudaSuccess) {
        return std::nullopt;
    }
    Mismatches result{.count = 0, .first_bits = UINT32_MAX};
    bool ok{cudaMemcpy(device, &result, sizeof(result), cudaMemcpyHostToDevice) == cudaSuccess};
    if (ok) {
        mismatch_kernel<<<1024, 256>>>(pair, first_bits, last_bits, device);
        ok = cudaGetLastError() == cudaSuccess &&
             cudaMemcpy(&result, device, sizeof(result), cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    cudaFree(device);
    return ok ? std::optional<Mismatches>{result} : std::nullopt;
}

}  // namespace

std::optional<Mismatches> cbrt_normal_mismatches(std::uint32_t first_bits,
                                                 std::uint32_t last_bits) {
    return count_mismatches(CbrtPair{}, first_bits, last_bits);
}

std::optional<Mismatches> pow_normal_mismatches(float exponent, std::uint32_t first_bits,
                                                std::uint32_t last_bits) {
    return count_mismatches(PowPair{.exponent = exponent}, first_bits, last_bits);
}

}  // namespace cujpegxl
