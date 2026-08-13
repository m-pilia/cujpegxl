// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_BITSTREAM_HISTOGRAM_WRITER_H_
#define CUJPEGXL_SRC_BITSTREAM_HISTOGRAM_WRITER_H_

#include <cstddef>
#include <cstdint>

#include "bit_writer.h"
#include "hybrid_uint.h"

namespace cujpegxl::bitstream {

// Emits the JXL prefix-code entropy container's histogram description for a
// pooled `histogram` (`length` symbols, i.e. the alphabet size; `num_contexts`
// contexts all clustered to this single histogram) using `config`, and builds
// the shared prefix code into `depth`/`bits` (each with `length` entries).
// Mirrors libjxl DecodeHistograms for the prefix path (LZ77 disabled). This is
// the single implementation used by both the device encoder's host side and the
// bitstream oracle. Not byte-aligned on exit.
void write_prefix_histograms(BitWriter& w, const std::uint32_t* histogram, std::size_t length,
                             std::size_t num_contexts, const HybridUintConfig& config,
                             std::uint8_t* depth, std::uint16_t* bits);

// Clustered variant: emits the histogram description for `num_clusters`
// prefix-coded histograms selected per context by `context_map` (`num_contexts`
// entries, each a cluster id in [0, num_clusters)), using the JXL simple
// context-map form (so `num_clusters` must be <= 8). `cluster_histograms` and the
// output `depth`/`bits` are laid out as `num_clusters * stride` (row c is cluster
// c); each cluster's alphabet is trailing-trimmed independently. Every cluster id
// in [0, num_clusters) must appear in `context_map`. Mirrors libjxl
// DecodeHistograms (prefix path, LZ77 disabled). Not byte-aligned on exit.
void write_clustered_prefix_histograms(BitWriter& w, const std::uint8_t* context_map,
                                       std::size_t num_contexts, std::size_t num_clusters,
                                       const std::uint32_t* cluster_histograms, std::size_t stride,
                                       const HybridUintConfig& config, std::uint8_t* depth,
                                       std::uint16_t* bits);

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_SRC_BITSTREAM_HISTOGRAM_WRITER_H_
