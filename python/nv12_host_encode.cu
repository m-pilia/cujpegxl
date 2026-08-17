// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "nv12_host_encode.h"

#include <cuda_runtime.h>

#include "src/frame_encoder.h"

namespace cujpegxl::pybind_support {

bool encode_nv12_host(const std::uint8_t* luma, const std::uint8_t* chroma, std::uint32_t width,
                      std::uint32_t height, float distance, std::int32_t device_ordinal,
                      std::vector<std::uint8_t>& out, std::vector<StageTiming>* stats) {
    if (cudaSetDevice(device_ordinal) != cudaSuccess) {
        return false;
    }
    const std::size_t luma_bytes{static_cast<std::size_t>(width) * height};
    const std::size_t chroma_bytes{static_cast<std::size_t>(width) * (height / 2)};

    std::uint8_t* d_luma{nullptr};
    std::uint8_t* d_chroma{nullptr};
    if (cudaMalloc(&d_luma, luma_bytes) != cudaSuccess) {
        return false;
    }
    if (cudaMalloc(&d_chroma, chroma_bytes) != cudaSuccess) {
        cudaFree(d_luma);
        return false;
    }

    bool ok{cudaMemcpy(d_luma, luma, luma_bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(d_chroma, chroma, chroma_bytes, cudaMemcpyHostToDevice) == cudaSuccess};
    if (ok) {
        ok = encode_nv12(d_luma, width, d_chroma, width, width, height, device_ordinal, distance,
                            quant_params_for_distance(distance), out, stats);
    }
    cudaFree(d_luma);
    cudaFree(d_chroma);
    return ok;
}

}  // namespace cujpegxl::pybind_support
