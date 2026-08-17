// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "cfl.h"

#include <cuda_runtime.h>

namespace cujpegxl {
namespace {

// One thread per 64x64 color tile. Each tile's regression is self-contained, so
// device and host produce identical maps. This standalone kernel gathers the tile
// coefficients from the covered-block layout rather than fusing into the front end.
__global__ void estimate_cfl_kernel(const float* __restrict__ x, const float* __restrict__ y,
                                    const float* __restrict__ b, std::size_t num_tiles,
                                    std::size_t coeffs_per_tile, std::int8_t* __restrict__ ytox_map,
                                    std::int8_t* __restrict__ ytob_map) {
    const std::size_t tile{static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x};
    if (tile >= num_tiles) {
        return;
    }
    const std::size_t off{tile * coeffs_per_tile};
    int mx{0};
    int mb{0};
    cfl_estimate(x + off, y + off, b + off, coeffs_per_tile, &mx, &mb);
    ytox_map[tile] = static_cast<std::int8_t>(mx);
    ytob_map[tile] = static_cast<std::int8_t>(mb);
}

}  // namespace

bool estimate_cfl(const float* x, const float* y, const float* b, std::size_t num_tiles,
                  std::size_t coeffs_per_tile, std::int8_t* ytox_map, std::int8_t* ytob_map) {
    const unsigned int threads{128};
    const unsigned int blocks{static_cast<unsigned int>((num_tiles + threads - 1) / threads)};
    estimate_cfl_kernel<<<blocks, threads>>>(x, y, b, num_tiles, coeffs_per_tile, ytox_map,
                                             ytob_map);
    const cudaError_t launch{cudaGetLastError()};
    const cudaError_t sync{cudaDeviceSynchronize()};
    return launch == cudaSuccess && sync == cudaSuccess;
}

void estimate_cfl_host(const float* x, const float* y, const float* b, std::size_t num_tiles,
                       std::size_t coeffs_per_tile, std::int8_t* ytox_map, std::int8_t* ytob_map) {
    for (std::size_t tile{0}; tile < num_tiles; ++tile) {
        const std::size_t off{tile * coeffs_per_tile};
        int mx{0};
        int mb{0};
        cfl_estimate(x + off, y + off, b + off, coeffs_per_tile, &mx, &mb);
        ytox_map[tile] = static_cast<std::int8_t>(mx);
        ytob_map[tile] = static_cast<std::int8_t>(mb);
    }
}

}  // namespace cujpegxl
