// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "quant.h"

#include <mutex>

#include <cuda_runtime.h>

#include "quant_calibration.h"
#include "quant_weights_dct8.h"

namespace cujpegxl {
namespace {

__constant__ float QUANT_WEIGHTS[3][64];
__constant__ float DC_INV_QUANT_D[3];

void init_quant_weights() {
    cudaMemcpyToSymbol(QUANT_WEIGHTS, DCT8_DEQUANT_WEIGHTS, sizeof(DCT8_DEQUANT_WEIGHTS));
    cudaMemcpyToSymbol(DC_INV_QUANT_D, DC_INV_QUANT, sizeof(DC_INV_QUANT));
}

// DC uses the separate DC quantizer (DC_INV_QUANT x global_scale_float x quant_dc);
// AC divides by the DCT8 dequant weight times the per-block AC step
// (inv_global_scale / raw_quant_field), i.e. multiplies by ac_scale / weight
// with ac_scale = raw_quant_field x global_scale_float.
__global__ void quantize_dct8_kernel(const float* __restrict__ coeffs, std::size_t plane,
                                     float ac_scale, float dc_scale,
                                     std::int32_t* __restrict__ q) {
    const std::size_t g{blockIdx.x * blockDim.x + threadIdx.x};
    if (g >= 3 * plane) {
        return;
    }
    const std::size_t c{g / plane};
    const std::size_t k{(g % plane) % 64};
    const float v{coeffs[g]};
    const float scaled{k == 0 ? v * DC_INV_QUANT_D[c] * dc_scale : v * ac_scale / QUANT_WEIGHTS[c][k]};
    q[g] = static_cast<std::int32_t>(rintf(scaled));
}

}  // namespace

bool quantize_dct8(const float* coeffs, std::size_t width, std::size_t height, float distance,
                   std::int32_t* q) {
    static std::once_flag weights_flag;
    std::call_once(weights_flag, init_quant_weights);

    const QuantCalibration cal{calibrate_quant(distance)};
    const float ac_scale{static_cast<float>(cal.raw_quant_field) * cal.global_scale_float};
    const float dc_scale{cal.global_scale_float * static_cast<float>(cal.quant_dc)};

    const std::size_t plane{width * height};
    const unsigned int threads{256};
    const unsigned int blocks{static_cast<unsigned int>((3 * plane + threads - 1) / threads)};
    quantize_dct8_kernel<<<blocks, threads>>>(coeffs, plane, ac_scale, dc_scale, q);

    const cudaError_t launch{cudaGetLastError()};
    const cudaError_t sync{cudaDeviceSynchronize()};
    return launch == cudaSuccess && sync == cudaSuccess;
}

}  // namespace cujpegxl
