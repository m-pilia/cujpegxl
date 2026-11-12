// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_WARP_SCAN_CUH_
#define CUJPEGXL_SRC_WARP_SCAN_CUH_

namespace cujpegxl {

// Inclusive prefix sum of `v` over the lanes of a fully active warp.
template <typename T>
__device__ inline T warp_inclusive_sum(T v, unsigned int lane) {
    for (unsigned int d{1}; d < 32; d <<= 1) {
        const T below{__shfl_up_sync(~0u, v, d)};
        if (lane >= d) {
            v += below;
        }
    }
    return v;
}

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_WARP_SCAN_CUH_
