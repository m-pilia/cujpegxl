// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_COEFF_ORDER_REFERENCE_H_
#define CUJPEGXL_SRC_COEFF_ORDER_REFERENCE_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "coeff_order.h"

namespace cujpegxl {

// Host counterpart of the order statistics of quantize_residual, over the AC
// planes X, Y, B in the covered-block layout of `bw` x `bh` blocks and the
// per-8x8 transform signal `acs` (nullptr = all DCT8). The low-frequency corner
// of every block is skipped.
CoeffOrderStats coeff_order_stats_reference(const std::array<const std::int32_t*, 3>& ac,
                                            const std::int8_t* acs, std::size_t bw, std::size_t bh);

// libjxl's ComputeCoeffOrder and ComputeLehmerCode on the host: a stable sort
// of the natural order and a Fenwick-tree Lehmer code.
CoeffOrders coeff_orders_reference(const CoeffOrderStats& stats);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_COEFF_ORDER_REFERENCE_H_
