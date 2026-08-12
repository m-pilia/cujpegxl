// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_TRANSFORM_SELECT_H_
#define CUJPEGXL_SRC_TRANSFORM_SELECT_H_

#include <cstddef>
#include <cstdint>

namespace cujpegxl {

// Bounded transform-type selection (T3): a quadtree over the square candidate
// set {8x8, 16x16, 32x32}, driven by a rate/distortion proxy on the XYB Y plane.
// Each 32x32 region (4x4 blocks) either keeps one 32x32 or splits into four
// 16x16 quadrants; each quadrant keeps one 16x16 or splits into four 8x8. The
// larger block is kept unless a split strictly reduces the proxy cost
// (deterministic tie-break). Large candidates are considered only where they fit
// fully within the image; edge regions fall back to smaller sizes.
//
// `y` is a width*height float plane (device). `acs` (device, bw*bh int8, bw =
// width/8, bh = height/8, block raster) receives the per-8x8 decision in the
// vardct_layout.h encoding: a first-block holds its side (8/16/32), every block
// it covers holds ACS_COVERED. `distance` sets the proxy quantizer. width/height
// are multiples of 8. Deterministic. Returns false on a CUDA error.
bool select_transforms(const float* y, std::size_t width, std::size_t height, float distance,
                       std::int8_t* acs);

// Host reference running the identical per-region decision, for validation.
void select_transforms_host(const float* y, std::size_t width, std::size_t height, float distance,
                            std::int8_t* acs);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_TRANSFORM_SELECT_H_
