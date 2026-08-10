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

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_SRC_BITSTREAM_HISTOGRAM_WRITER_H_
