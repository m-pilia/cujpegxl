// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "transform_select.h"

#include "device_allocation_cache.h"

#include <cuda_runtime.h>

#include "quant_calibration.h"
#include "transform_select_impl.cuh"

namespace cujpegxl {
namespace {

std::size_t region_grid(std::size_t blocks) { return (blocks + 3) / 4; }

// One thread per 32x32 region. The per-region decision is self-contained (no
// cross-thread reduction), so device and host produce byte-identical ACS. This
// is the correctness-first T3 kernel; fusion into the front-end and parallelism
// within a region are deferred to T6/T9.
__global__ void select_transforms_kernel(const float* __restrict__ y, std::size_t width,
                                         std::size_t bw, std::size_t bh, std::size_t rbw,
                                         std::size_t rbh, double qgsf, double lambda,
                                         std::int8_t* __restrict__ acs) {
    const std::size_t region{static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x};
    if (region >= rbw * rbh) {
        return;
    }
    const std::size_t rbx{(region % rbw) * 4};
    const std::size_t rby{(region / rbw) * 4};
    decide_region(y, width, bw, bh, rbx, rby, qgsf, lambda, acs);
}

}  // namespace

bool select_transforms(const float* y, std::size_t width, std::size_t height, float distance,
                       std::int8_t* acs) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t rbw{region_grid(bw)};
    const std::size_t rbh{region_grid(bh)};

    const QuantCalibration cal{calibrate_quant(distance)};
    const double qgsf{static_cast<double>(cal.raw_quant_field) * cal.global_scale_float};

    const unsigned int threads{64};
    const unsigned int blocks{static_cast<unsigned int>((rbw * rbh + threads - 1) / threads)};
    select_transforms_kernel<<<blocks, threads, 0, encoder_stream()>>>(
        y, width, bw, bh, rbw, rbh, qgsf, TRANSFORM_SELECT_LAMBDA, acs);

    const cudaError_t launch{cudaGetLastError()};
    const cudaError_t sync{encoder_stream_synchronize()};
    return launch == cudaSuccess && sync == cudaSuccess;
}

void select_transforms_host(const float* y, std::size_t width, std::size_t height, float distance,
                            std::int8_t* acs) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t rbw{region_grid(bw)};
    const std::size_t rbh{region_grid(bh)};

    const QuantCalibration cal{calibrate_quant(distance)};
    const double qgsf{static_cast<double>(cal.raw_quant_field) * cal.global_scale_float};

    for (std::size_t ry{0}; ry < rbh; ++ry) {
        for (std::size_t rx{0}; rx < rbw; ++rx) {
            decide_region(y, width, bw, bh, rx * 4, ry * 4, qgsf, TRANSFORM_SELECT_LAMBDA, acs);
        }
    }
}

}  // namespace cujpegxl
