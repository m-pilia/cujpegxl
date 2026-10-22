// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_AC_CLUSTERING_REFERENCE_H_
#define CUJPEGXL_SRC_AC_CLUSTERING_REFERENCE_H_

#include <cstdint>

#include "ac_clustering.h"

namespace cujpegxl {

// Sequential host counterpart of ac_cluster_contexts, with the same costs and
// tie-breaks. `context_histograms` holds AC_NUM_CONTEXTS * AC_CONTEXT_ALPHABET
// counts (context * AC_CONTEXT_ALPHABET + symbol).
AcClustering cluster_ac_contexts_reference(const std::uint32_t* context_histograms);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_AC_CLUSTERING_REFERENCE_H_
