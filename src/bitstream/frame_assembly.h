// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_BITSTREAM_FRAME_ASSEMBLY_H_
#define CUJPEGXL_SRC_BITSTREAM_FRAME_ASSEMBLY_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "bit_writer.h"
#include "src/ac_clustering.h"
#include "src/coeff_order.h"
#include "token_stream.h"

namespace cujpegxl::bitstream {

// Serialized quantizer state for a VarDCT frame (see libjxl Quantizer). The
// per-block quant field is carried separately (a device buffer through the
// entropy path), not here: only these two values are written into DcGlobal.
struct QuantParams {
    std::uint32_t global_scale{1};
    std::uint32_t quant_dc{1};
};

// Width of one device histogram (AC_HISTOGRAM_SIZE in the device coder): the
// hybrid-uint token alphabet upper bound.
constexpr std::size_t HISTOGRAM_STRIDE = 256;

// AC groups and DC groups tiling a width x height image, including partial edge
// groups.
std::size_t ac_group_count(std::size_t width, std::size_t height);
std::size_t dc_group_count(std::size_t width, std::size_t height);

// The byte-aligned DcGlobal section.
std::vector<std::uint8_t> build_dc_global(const QuantParams& qp);

// Builds the byte-aligned AcGlobal section for the coefficient `orders` and the
// AC context clustering `clustering`, and in `code` the matching code of the AC group token streams
// (entries cluster << TOKEN_CLUSTER_SHIFT | symbol). The constructor chooses the
// entropy mode by libjxl's rule: rANS, except prefix codes when the histograms
// hold fewer than 100 tokens or each holds a single symbol. build_cluster(c)
// must then run for every cluster c < num_clusters(), and build_coeff_orders()
// and build_context_map() once, in any order and on any threads (concurrently
// with each other): build_cluster normalizes and describes the cluster's rANS
// histogram, the bulk of the work, build_coeff_orders encodes the coefficient
// orders and build_context_map the context map. finish() assembles the
// section.
class AcGlobalBuilder {
public:
    AcGlobalBuilder(const AcClustering& clustering, const CoeffOrders& orders,
                    std::size_t num_ac_groups, TokenCode& code);

    std::size_t num_clusters() const { return clustering_.num_clusters; }
    void build_cluster(std::size_t c);
    void build_coeff_orders();
    void build_context_map();
    std::vector<std::uint8_t> finish();

private:
    const AcClustering& clustering_;
    const CoeffOrders& orders_;
    std::size_t num_ac_groups_;
    TokenCode& code_;
    std::size_t log_alpha_size_{5};
    std::vector<BitWriter> cluster_bits_;
    BitWriter coeff_orders_bits_;
    BitWriter context_map_bits_;
};

// The codestream head: signature + SizeHeader + ImageMetadata + FrameHeader +
// multi-entry TOC for `section_sizes` (in codestream order: DcGlobal, DcGroups,
// AcGlobal, AcGroups). Byte-aligned; the body sections follow directly.
std::vector<std::uint8_t> build_codestream_head(std::uint32_t width, std::uint32_t height,
                                                const std::vector<std::uint32_t>& section_sizes);

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_SRC_BITSTREAM_FRAME_ASSEMBLY_H_
