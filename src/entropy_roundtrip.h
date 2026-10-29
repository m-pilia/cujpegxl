// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_ENTROPY_ROUNDTRIP_H_
#define CUJPEGXL_SRC_ENTROPY_ROUNDTRIP_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "ac_clustering.h"
#include "coeff_order.h"

namespace cujpegxl {

// Test-only host-vector wrapper around the device AC token coder: uploads the
// quantized coefficients, their non-zero counts and order statistics (computed
// on the host), runs ac_coeff_orders, ac_build_histogram, ac_token_layout,
// ac_cluster_contexts and ac_pack_tokens (over the groups split into `batches`
// batches), and downloads the results. Used by the GPU conformance test to diff
// against the host bitstream reference.
struct AcDeviceTokens {
    CoeffOrders orders{};
    std::vector<std::uint32_t> context_histograms{};
    AcClustering clustering{};
    std::vector<std::uint32_t> layout{};
    std::vector<std::uint16_t> entries{};
    std::vector<std::uint64_t> raw{};
};

// `q` is the covered-block coefficient layout (three planes of
// (width/8 * height/8) * 64 int32) and `acs` the per-8x8 transform signal.
bool ac_tokens_device(const std::vector<std::int32_t>& q, const std::vector<std::int8_t>& acs,
                      std::size_t width, std::size_t height, std::size_t batches,
                      AcDeviceTokens& out);

bool ac_coeff_orders_device(const CoeffOrderStats& stats, CoeffOrders& out);

// Runs ac_cluster_contexts on `context_histograms` (AC_NUM_CONTEXTS *
// AC_CONTEXT_ALPHABET counts).
bool ac_cluster_contexts_device(const std::vector<std::uint32_t>& context_histograms,
                                AcClustering& out);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_ENTROPY_ROUNDTRIP_H_
