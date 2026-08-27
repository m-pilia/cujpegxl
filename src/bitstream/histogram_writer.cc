// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Host binding of the shared, host+device histogram-description serialization
// (serialize_impl.h) to the vector BitWriter.

#include "histogram_writer.h"

#include "serialize_impl.h"

namespace cujpegxl::bitstream {

void write_prefix_histograms(BitWriter& w, const std::uint32_t* histogram, std::size_t length,
                             std::size_t num_contexts, const HybridUintConfig& config,
                             std::uint8_t* depth, std::uint16_t* bits) {
    detail::write_prefix_histograms(w, histogram, length, num_contexts, config, depth, bits);
}

void write_clustered_prefix_histograms(BitWriter& w, const std::uint8_t* context_map,
                                       std::size_t num_contexts, std::size_t num_clusters,
                                       const std::uint32_t* cluster_histograms, std::size_t stride,
                                       const HybridUintConfig& config, std::uint8_t* depth,
                                       std::uint16_t* bits) {
    detail::write_clustered_prefix_histograms(w, context_map, num_contexts, num_clusters,
                                              cluster_histograms, stride, config, depth, bits);
}

}  // namespace cujpegxl::bitstream
