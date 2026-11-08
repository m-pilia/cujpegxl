// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_ROUND_TO_INT_H_
#define CUJPEGXL_SRC_ROUND_TO_INT_H_

#include <cmath>

namespace cujpegxl {

#if defined(__CUDACC__)
#define CUJPEGXL_ROUND_HD __host__ __device__
#else
#define CUJPEGXL_ROUND_HD
#endif

// Round half to even, for x within int range. On the GPU clang compiles this to
// one 32-bit F2I, the same SASS as __float2int_rn; lrintf returns long and
// costs a 64-bit F2I on the slow FP64 pipe.
CUJPEGXL_ROUND_HD inline int round_to_int(float x) {
    return static_cast<int>(rintf(x));
}

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_ROUND_TO_INT_H_
