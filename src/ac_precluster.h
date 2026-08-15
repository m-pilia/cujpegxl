// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_AC_PRECLUSTER_H_
#define CUJPEGXL_SRC_AC_PRECLUSTER_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "ac_context.h"
#include "entropy.h"

namespace cujpegxl {

inline constexpr std::size_t AC_MAX_PRECLUSTERS = 193;

struct AcPreclusterResult {
    std::array<std::uint16_t, AC_NUM_CONTEXTS> context_map{};
    std::array<std::uint16_t, AC_MAX_PRECLUSTERS> signatures{};
    std::array<std::uint32_t, AC_MAX_PRECLUSTERS * AC_HISTOGRAM_SIZE> histograms{};
    std::size_t num_candidates{0};
};

void ac_precluster(const std::uint32_t* context_histograms, AcPreclusterResult& result);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_AC_PRECLUSTER_H_
