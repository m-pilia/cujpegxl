// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_CFL_ESTIMATE_H_
#define CUJPEGXL_SRC_CFL_ESTIMATE_H_

#include <cstddef>
#include <cstdint>

namespace cujpegxl {

// Test-only standalone driver for the shipped `cfl_estimate` core (cfl.h): the
// production encoder estimates CfL fused into the front end
// (estimate_cfl_covered), so this gather-then-regress kernel exists purely to
// exercise device/host parity and determinism of the per-tile regression.
//
// `x`/`y`/`b` hold `num_tiles * coeffs_per_tile` AC coefficients grouped by tile
// (each tile's coefficients contiguous); `ytox_map` and `ytob_map` receive one
// signed-int8 factor per tile. Deterministic. The device entry returns false on a
// CUDA error; the host entry runs the identical per-tile core.
bool estimate_cfl(const float* x, const float* y, const float* b, std::size_t num_tiles,
                  std::size_t coeffs_per_tile, std::int8_t* ytox_map, std::int8_t* ytob_map);
void estimate_cfl_host(const float* x, const float* y, const float* b, std::size_t num_tiles,
                       std::size_t coeffs_per_tile, std::int8_t* ytox_map, std::int8_t* ytob_map);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_CFL_ESTIMATE_H_
