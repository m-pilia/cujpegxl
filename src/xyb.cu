// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "xyb.h"

#include <cuda_runtime.h>

namespace cujpegxl {
namespace {

// libjxl opsin absorbance matrix (jxl::cms::kOpsinAbsorbanceMatrix) and bias.
// The intensity-target multiplier is 1.0 for 8-bit SDR (intensity_target=255).
constexpr float M00{0.30f};
constexpr float M02{0.078f};
constexpr float M01{1.0f - M02 - M00};
constexpr float M10{0.23f};
constexpr float M12{0.078f};
constexpr float M11{1.0f - M12 - M10};
constexpr float M20{0.24342268924547819f};
constexpr float M21{0.20476744424496821f};
constexpr float M22{1.0f - M20 - M21};
constexpr float OPSIN_BIAS{0.0037930732552754493f};

// BT.709 full-range Y'CbCr -> R'G'B' (exact inverse of the corpus generator's
// forward matrix; Kr=0.2126, Kb=0.0722).
constexpr float CR_TO_R{1.5748f};
constexpr float CB_TO_G{0.1873242729f};
constexpr float CR_TO_G{0.4681242729f};
constexpr float CB_TO_B{1.8556f};

constexpr float CHROMA_CENTER{128.0f / 255.0f};

__device__ inline float srgb_to_linear(float encoded) {
    return encoded <= 0.04045f ? encoded * (1.0f / 12.92f)
                               : powf((encoded + 0.055f) * (1.0f / 1.055f), 2.4f);
}

__device__ inline float clamp01(float v) {
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

__global__ void nv12_to_xyb_kernel(const std::uint8_t* __restrict__ luma, std::size_t luma_pitch,
                                   cudaTextureObject_t chroma_tex, std::size_t width,
                                   std::size_t height, float* __restrict__ xyb) {
    const std::size_t x{blockIdx.x * blockDim.x + threadIdx.x};
    const std::size_t y{blockIdx.y * blockDim.y + threadIdx.y};
    if (x >= width || y >= height) {
        return;
    }

    const float yv{luma[y * luma_pitch + x] * (1.0f / 255.0f)};
    // Centered bilinear chroma upsample: luma pixel x maps to chroma texel
    // coordinate (x + 0.5) / 2 (texel centers at index + 0.5).
    const float2 c{tex2D<float2>(chroma_tex, (x + 0.5f) * 0.5f, (y + 0.5f) * 0.5f)};
    const float cb{c.x - CHROMA_CENTER};
    const float cr{c.y - CHROMA_CENTER};

    const float r{clamp01(srgb_to_linear(clamp01(yv + CR_TO_R * cr)))};
    const float g{clamp01(srgb_to_linear(clamp01(yv - CB_TO_G * cb - CR_TO_G * cr)))};
    const float b{clamp01(srgb_to_linear(clamp01(yv + CB_TO_B * cb)))};

    float m0{fmaxf(M00 * r + M01 * g + M02 * b + OPSIN_BIAS, 0.0f)};
    float m1{fmaxf(M10 * r + M11 * g + M12 * b + OPSIN_BIAS, 0.0f)};
    float m2{fmaxf(M20 * r + M21 * g + M22 * b + OPSIN_BIAS, 0.0f)};

    const float neg_cbrt_bias{-cbrtf(OPSIN_BIAS)};
    m0 = cbrtf(m0) + neg_cbrt_bias;
    m1 = cbrtf(m1) + neg_cbrt_bias;
    m2 = cbrtf(m2) + neg_cbrt_bias;

    const std::size_t plane{width * height};
    const std::size_t idx{y * width + x};
    xyb[idx] = 0.5f * (m0 - m1);
    xyb[plane + idx] = 0.5f * (m0 + m1);
    xyb[2 * plane + idx] = m2;
}

}  // namespace

bool nv12_to_xyb(const std::uint8_t* luma, std::size_t luma_pitch, const std::uint8_t* chroma,
                 std::size_t chroma_pitch, std::size_t width, std::size_t height, float* xyb) {
    cudaResourceDesc resource{};
    resource.resType = cudaResourceTypePitch2D;
    resource.res.pitch2D.devPtr = const_cast<std::uint8_t*>(chroma);
    resource.res.pitch2D.desc = cudaCreateChannelDesc<uchar2>();
    resource.res.pitch2D.width = width / 2;
    resource.res.pitch2D.height = height / 2;
    resource.res.pitch2D.pitchInBytes = chroma_pitch;

    cudaTextureDesc texture{};
    texture.addressMode[0] = cudaAddressModeClamp;
    texture.addressMode[1] = cudaAddressModeClamp;
    texture.filterMode = cudaFilterModeLinear;
    texture.readMode = cudaReadModeNormalizedFloat;
    texture.normalizedCoords = 0;

    cudaTextureObject_t chroma_tex{};
    if (cudaCreateTextureObject(&chroma_tex, &resource, &texture, nullptr) != cudaSuccess) {
        return false;
    }

    const dim3 block{16, 16};
    const dim3 grid{static_cast<unsigned int>((width + block.x - 1) / block.x),
                    static_cast<unsigned int>((height + block.y - 1) / block.y)};
    nv12_to_xyb_kernel<<<grid, block>>>(luma, luma_pitch, chroma_tex, width, height, xyb);

    const cudaError_t launch{cudaGetLastError()};
    const cudaError_t sync{cudaDeviceSynchronize()};
    cudaDestroyTextureObject(chroma_tex);
    return launch == cudaSuccess && sync == cudaSuccess;
}

}  // namespace cujpegxl
