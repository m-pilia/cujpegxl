// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "prefix_build_reference.h"

#include <algorithm>

#include "bit_writer.h"
#include "frame_assembly.h"
#include "histogram_writer.h"
#include "hybrid_uint.h"
#include "serialize_impl.h"
#include "src/vardct_layout.h"

namespace cujpegxl::bitstream {

DcGroupBlobs build_dc_group_blobs(std::size_t width, std::size_t height,
                                  const std::uint32_t* dc_histograms,
                                  const std::uint32_t* acmeta_histograms,
                                  const std::size_t* first_block_counts) {
    const std::size_t bw{width / 8};
    const std::size_t bh{height / 8};
    const std::size_t xdg{(bw + DC_GROUP_BLOCKS - 1) / DC_GROUP_BLOCKS};
    const std::size_t num_groups{dc_group_count(width, height)};

    DcGroupBlobs out{};
    out.dc_depth.assign(num_groups * HISTOGRAM_STRIDE, 0);
    out.dc_bits.assign(num_groups * HISTOGRAM_STRIDE, 0);
    out.acmeta_depth.assign(num_groups * HISTOGRAM_STRIDE, 0);
    out.acmeta_bits.assign(num_groups * HISTOGRAM_STRIDE, 0);
    out.blob_pre_off.assign(num_groups, 0);
    out.blob_pre_bits.assign(num_groups, 0);
    out.blob_mid_off.assign(num_groups, 0);
    out.blob_mid_bits.assign(num_groups, 0);

    for (std::size_t g{0}; g < num_groups; ++g) {
        const std::size_t gx{g % xdg};
        const std::size_t gy{g / xdg};
        const std::size_t dgw{std::min(DC_GROUP_BLOCKS, bw - gx * DC_GROUP_BLOCKS)};
        const std::size_t dgh{std::min(DC_GROUP_BLOCKS, bh - gy * DC_GROUP_BLOCKS)};
        const std::size_t total_blocks{dgw * dgh};
        const std::size_t count{first_block_counts == nullptr ? total_blocks
                                                              : first_block_counts[g]};

        BitWriter pre{};
        detail::serialize_dc_pre(pre, dc_histograms + g * HISTOGRAM_STRIDE,
                                 out.dc_depth.data() + g * HISTOGRAM_STRIDE,
                                 out.dc_bits.data() + g * HISTOGRAM_STRIDE);
        out.blob_pre_off[g] = static_cast<std::uint32_t>(out.blob_pre.size());
        out.blob_pre_bits[g] = static_cast<std::uint32_t>(pre.bits_written());
        const std::vector<std::uint8_t>& pre_bytes{pre.bytes()};
        out.blob_pre.insert(out.blob_pre.end(), pre_bytes.begin(), pre_bytes.end());

        BitWriter mid{};
        detail::serialize_dc_mid(mid, acmeta_histograms + g * HISTOGRAM_STRIDE, total_blocks, count,
                                 out.acmeta_depth.data() + g * HISTOGRAM_STRIDE,
                                 out.acmeta_bits.data() + g * HISTOGRAM_STRIDE);
        out.blob_mid_off[g] = static_cast<std::uint32_t>(out.blob_mid.size());
        out.blob_mid_bits[g] = static_cast<std::uint32_t>(mid.bits_written());
        const std::vector<std::uint8_t>& mid_bytes{mid.bytes()};
        out.blob_mid.insert(out.blob_mid.end(), mid_bytes.begin(), mid_bytes.end());
    }
    return out;
}

}  // namespace cujpegxl::bitstream
