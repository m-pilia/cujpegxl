// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "xyb.h"

#include "device_allocation_cache.h"

#include <cuda_runtime.h>

#include "xyb_impl.cuh"

namespace cujpegxl {
namespace {

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

    const std::size_t plane{width * height};
    const std::size_t idx{y * width + x};
    nv12_pixel_to_xyb(yv, c.x, c.y, xyb[idx], xyb[plane + idx], xyb[2 * plane + idx]);
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
    nv12_to_xyb_kernel<<<grid, block, 0, encoder_stream()>>>(luma, luma_pitch, chroma_tex, width,
                                                             height, xyb);

    const cudaError_t launch{cudaGetLastError()};
    const cudaError_t sync{encoder_stream_synchronize()};
    cudaDestroyTextureObject(chroma_tex);
    return launch == cudaSuccess && sync == cudaSuccess;
}

}  // namespace cujpegxl
