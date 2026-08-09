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
__constant__ float Y_TO_B_RATIO_D;

void init_quant_weights() {
    cudaMemcpyToSymbol(QUANT_WEIGHTS, DCT8_DEQUANT_WEIGHTS, sizeof(DCT8_DEQUANT_WEIGHTS));
    cudaMemcpyToSymbol(DC_INV_QUANT_D, DC_INV_QUANT, sizeof(DC_INV_QUANT));
    cudaMemcpyToSymbol(Y_TO_B_RATIO_D, &Y_TO_B_RATIO, sizeof(float));
}

// Quantization factor mapping a coefficient to its rounded integer: DC uses the
// separate DC quantizer (DC_INV_QUANT x global_scale_float x quant_dc); AC
// multiplies by ac_scale / weight with ac_scale = raw_quant_field x
// global_scale_float (the reciprocal of the dequant step).
__device__ inline float quant_factor(std::size_t c, std::size_t k, float ac_scale,
                                     float dc_scale) {
    return k == 0 ? DC_INV_QUANT_D[c] * dc_scale : ac_scale / QUANT_WEIGHTS[c][k];
}

// Phase 1: quantize Y (XYB channel 1). Y has no chroma-from-luma predictor.
__global__ void quantize_y_kernel(const float* __restrict__ coeffs, std::size_t plane,
                                  float ac_scale, float dc_scale,
                                  std::int32_t* __restrict__ q) {
    const std::size_t g{blockIdx.x * blockDim.x + threadIdx.x};
    if (g >= plane) {
        return;
    }
    const std::size_t k{g % 64};
    const float v{coeffs[plane + g]};
    q[plane + g] =
        static_cast<std::int32_t>(rintf(v * quant_factor(1, k, ac_scale, dc_scale)));
}

// Phase 2: quantize X (XYB channel 0) and B (XYB channel 2). B encodes the
// residual after the default Y-to-B chroma-from-luma prediction: the decoder
// adds dequant_y * Y_TO_B_RATIO back to every B coefficient, so we subtract the
// roundtrip Y coefficient (read from the phase-1 output) before quantizing.
__global__ void quantize_xb_kernel(const float* __restrict__ coeffs, std::size_t plane,
                                   float ac_scale, float dc_scale,
                                   std::int32_t* __restrict__ q) {
    const std::size_t g{blockIdx.x * blockDim.x + threadIdx.x};
    if (g >= 2 * plane) {
        return;
    }
    const std::size_t local{g % plane};
    const std::size_t k{local % 64};
    if (g < plane) {
        const float v{coeffs[g]};
        q[g] =
            static_cast<std::int32_t>(rintf(v * quant_factor(0, k, ac_scale, dc_scale)));
    } else {
        const float v{coeffs[2 * plane + local]};
        const float f_y{quant_factor(1, k, ac_scale, dc_scale)};
        const float roundtrip_y{static_cast<float>(q[plane + local]) / f_y};
        const float residual{v - Y_TO_B_RATIO_D * roundtrip_y};
        q[2 * plane + local] = static_cast<std::int32_t>(
            rintf(residual * quant_factor(2, k, ac_scale, dc_scale)));
    }
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
    const unsigned int y_blocks{
        static_cast<unsigned int>((plane + threads - 1) / threads)};
    quantize_y_kernel<<<y_blocks, threads>>>(coeffs, plane, ac_scale, dc_scale, q);
    const unsigned int xb_blocks{
        static_cast<unsigned int>((2 * plane + threads - 1) / threads)};
    quantize_xb_kernel<<<xb_blocks, threads>>>(coeffs, plane, ac_scale, dc_scale, q);

    const cudaError_t launch{cudaGetLastError()};
    const cudaError_t sync{cudaDeviceSynchronize()};
    return launch == cudaSuccess && sync == cudaSuccess;
}

}  // namespace cujpegxl
