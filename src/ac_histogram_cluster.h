// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_AC_HISTOGRAM_CLUSTER_H_
#define CUJPEGXL_SRC_AC_HISTOGRAM_CLUSTER_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "ac_precluster.h"

namespace cujpegxl {

struct AcClusterConfig {
    std::size_t max_clusters{64};
    std::size_t max_comparisons{20000};
};

struct AcClusterResult {
    std::array<std::uint8_t, AC_NUM_CONTEXTS> context_map{};
    std::array<std::uint32_t, AC_MAX_PRECLUSTERS * AC_HISTOGRAM_SIZE> histograms{};
    std::array<std::uint8_t, AC_MAX_PRECLUSTERS * AC_HISTOGRAM_SIZE> depths{};
    std::array<std::int64_t, AC_MAX_PRECLUSTERS> merge_deltas{};
    std::size_t num_clusters{0};
    std::size_t num_merges{0};
    std::size_t comparisons{0};
    std::uint64_t estimated_bits{0};
};

bool ac_cluster_histograms(const AcPreclusterResult& preclusters, const AcClusterConfig& config,
                           AcClusterResult& result);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_AC_HISTOGRAM_CLUSTER_H_
