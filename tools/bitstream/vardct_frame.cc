// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "vardct_frame.h"

#include <algorithm>
#include <cassert>
#include <utility>

#include "src/ac_clustering.h"
#include "src/ac_clustering_reference.h"
#include "src/ac_context.h"
#include "src/bitstream/bit_writer.h"
#include "src/bitstream/codestream.h"
#include "src/bitstream/field_coder.h"
#include "src/bitstream/frame_assembly.h"
#include "src/bitstream/histogram_writer.h"
#include "src/bitstream/hybrid_uint.h"
#include "src/bitstream/token_stream.h"
#include "src/bitstream/token_stream_reference.h"
#include "src/coeff_order.h"
#include "src/coeff_order_reference.h"
#include "src/vardct_layout.h"

#include "modular.h"

namespace cujpegxl::bitstream {
namespace {

constexpr U32Enc GLOBAL_SCALE_ENC{
    {BitsOffset(11, 1), BitsOffset(11, 2049), BitsOffset(12, 4097), BitsOffset(16, 8193)}};
constexpr U32Enc QUANT_DC_ENC{{Val(16), BitsOffset(5, 1), BitsOffset(8, 1), BitsOffset(16, 1)}};
constexpr U32Enc ORDER_ENC{{Val(0x5F), Val(0x13), Val(0), Bits(13)}};

constexpr ModularTree DC_TREE{.predictor = DC_PREDICTOR_GRADIENT};

// The all-zero EPF channel (index 3) takes the first leaf, so its samples cost
// no bits; YtoX, YtoB and ACS+QF share the second.
constexpr ModularTree ACMETA_TREE{.predictor = DC_PREDICTOR_ZERO, .split_channel = 2};
constexpr std::size_t ACMETA_DATA_LEAF{1};

std::size_t ceil_div(std::size_t a, std::size_t b) {
    return (a + b - 1) / b;
}

std::size_t ceil_log2_nonzero(std::size_t x) {
    if (x <= 1) {
        return 0;
    }
    std::size_t bits{0};
    std::size_t v{x - 1};
    while (v) {
        v >>= 1;
        ++bits;
    }
    return bits;
}

// libjxl CoeffOrderAndLut for a 1x1 (DCT8) block: builds the natural scan order.
std::array<std::uint32_t, 64> compute_dct8_natural_order() {
    constexpr std::size_t dim{8};
    std::array<std::uint32_t, 64> out{};
    std::size_t cur{1};
    for (std::size_t i{0}; i < dim; ++i) {
        for (std::size_t j{0}; j <= i; ++j) {
            std::size_t x{j};
            std::size_t y{i - j};
            if (i & 1) {
                std::swap(x, y);
            }
            const std::size_t val{(x < 1 && y < 1) ? 0 : cur++};
            out[val] = static_cast<std::uint32_t>(y * dim + x);
        }
    }
    for (std::size_t ip{dim - 1}; ip > 0; --ip) {
        const std::size_t i{ip - 1};
        for (std::size_t j{0}; j <= i; ++j) {
            std::size_t x{dim - 1 - (i - j)};
            std::size_t y{dim - 1 - j};
            if (i & 1) {
                std::swap(x, y);
            }
            out[cur++] = static_cast<std::uint32_t>(y * dim + x);
        }
    }
    return out;
}

// Extracts the DC block sub-rectangle [bx0, bx0+dgw) x [by0, by0+dgh) from the
// full-image DC planes into physical modular channels Y, X, B (DequantDC reads
// channel[1]=X, channel[0]=Y, channel[2]=B), while fc.dc is X, Y, B.
std::vector<ModularChannel> make_vardct_dc_channels(const FrameCoefficients& fc, std::size_t bw,
                                                    std::size_t bx0, std::size_t by0,
                                                    std::size_t dgw, std::size_t dgh) {
    const int xyb_for_physical[3]{1, 0, 2};
    std::vector<ModularChannel> channels{};
    for (int p{0}; p < 3; ++p) {
        const std::vector<std::int32_t>& plane{fc.dc[xyb_for_physical[p]]};
        std::vector<std::int32_t> sub(dgw * dgh);
        for (std::size_t y{0}; y < dgh; ++y) {
            for (std::size_t x{0}; x < dgw; ++x) {
                sub[y * dgw + x] = plane[(by0 + y) * bw + (bx0 + x)];
            }
        }
        channels.push_back({dgw, dgh, std::move(sub)});
    }
    return channels;
}

// libjxl AcStrategyType raw value for a square block side (DCT=0, DCT16X16=4,
// DCT32X32=5).
std::int32_t raw_strategy(int side) {
    return side == 16 ? 4 : (side == 32 ? 5 : 0);
}

// The block's transform side; an empty acs signals an all-DCT8 frame.
int block_side(const FrameCoefficients& fc, std::size_t blk) {
    return fc.acs.empty() ? 8 : fc.acs[blk];
}

CoeffOrders frame_coeff_orders(const FrameCoefficients& fc, std::size_t bw, std::size_t bh) {
    return coeff_orders_reference(
        coeff_order_stats_reference({fc.ac[0].data(), fc.ac[1].data(), fc.ac[2].data()},
                                    fc.acs.empty() ? nullptr : fc.acs.data(), bw, bh));
}

// AcMetadata channels for a DC group covering dgw x dgh blocks at block origin
// (bx0, by0): YtoX/YtoB (per 64x64 color tile, from the CfL maps), ACS+QF (one
// column per first-block in DC-group raster order: row 0 = raw strategy, row 1 =
// quant field - 1), EPF (all zero). The ACS+QF width equals the number of
// first-blocks, which the decoder reads as the AcMetadata `count`.
std::vector<ModularChannel> make_ac_metadata_channels(const FrameCoefficients& fc, std::size_t bw,
                                                      std::size_t bx0, std::size_t by0,
                                                      std::size_t dgw, std::size_t dgh) {
    const std::size_t cw{ceil_div(dgw, 8)};
    const std::size_t ch{ceil_div(dgh, 8)};

    ModularChannel ytox{cw, ch, std::vector<std::int32_t>(cw * ch, 0)};
    ModularChannel ytob{cw, ch, std::vector<std::int32_t>(cw * ch, 0)};
    if (!fc.ytox_map.empty()) {
        const std::size_t cmw{ceil_div(bw, 8)};
        for (std::size_t y{0}; y < ch; ++y) {
            for (std::size_t x{0}; x < cw; ++x) {
                const std::size_t src{(by0 / 8 + y) * cmw + (bx0 / 8 + x)};
                ytox.pixels[y * cw + x] = fc.ytox_map[src];
                ytob.pixels[y * cw + x] = fc.ytob_map[src];
            }
        }
    }

    std::vector<std::int32_t> acs_row{};
    std::vector<std::int32_t> qf_row{};
    for (std::size_t by{0}; by < dgh; ++by) {
        for (std::size_t bx{0}; bx < dgw; ++bx) {
            const std::size_t blk{(by0 + by) * bw + (bx0 + bx)};
            const int side{block_side(fc, blk)};
            if (side == ACS_COVERED) {
                continue;
            }
            acs_row.push_back(raw_strategy(side));
            qf_row.push_back(static_cast<std::int32_t>(fc.raw_quant_field) - 1);
        }
    }
    const std::size_t count{acs_row.size()};
    std::vector<std::int32_t> acs_qf(count * 2);
    for (std::size_t i{0}; i < count; ++i) {
        acs_qf[i] = acs_row[i];
        acs_qf[count + i] = qf_row[i];
    }
    ModularChannel acs_qf_ch{count, 2, std::move(acs_qf)};

    ModularChannel epf{dgw, dgh, std::vector<std::int32_t>(dgw * dgh, 0)};
    return {std::move(ytox), std::move(ytob), std::move(acs_qf_ch), std::move(epf)};
}

struct AcTok {
    std::uint32_t context;
    std::uint32_t symbol;
    std::uint32_t nbits;
    std::uint32_t bits;
};

std::size_t log2_covered(int side) {
    return side == 16 ? 2 : (side == 32 ? 4 : 0);
}

// Tokenizes one AC group in the coefficient `orders` with libjxl's real
// per-token contexts, appending the (context, symbol, extra-bits) tuples to
// `out` and accumulating the per-context symbol histograms. The context math
// mirrors dec_group.cc DecodeACVarBlock: a group-local normalized non-zero grid
// drives the non-zero-count prediction, and each coefficient's zero-density
// context follows the scan.
void tokenize_ac_group_clustered(const FrameCoefficients& fc, const CoeffOrders& orders,
                                 std::size_t bw, std::size_t bx0, std::size_t by0, std::size_t gbw,
                                 std::size_t gbh, std::vector<std::uint32_t>& context_hist,
                                 std::vector<AcTok>& out) {
    const int channel_order[3]{1, 0, 2};  // physical planes Y, X, B
    const HybridUintConfig config{};

    // Pass 1: group-local normalized non-zero grid per channel, covered positions
    // filled with the covering first-block's value.
    std::array<std::vector<std::int32_t>, 3> nz_norm{};
    for (int p{0}; p < 3; ++p) {
        nz_norm[p].assign(gbw * gbh, 0);
    }
    for (std::size_t sby{0}; sby < gbh; ++sby) {
        for (std::size_t sbx{0}; sbx < gbw; ++sbx) {
            const int side{block_side(fc, (by0 + sby) * bw + (bx0 + sbx))};
            if (side == ACS_COVERED) {
                continue;
            }
            const std::size_t cx{covered_blocks_side(side)};
            const std::size_t cb{cx * cx};
            const std::size_t sz{static_cast<std::size_t>(side) * side};
            const std::size_t l2{log2_covered(side)};
            for (int c : channel_order) {
                const std::uint16_t* const order{orders.order + coeff_order_offset(side, c)};
                std::int32_t nz{0};
                for (std::size_t k{cb}; k < sz; ++k) {
                    if (fc.ac[c][covered_plane_slot(side, bx0 + sbx, by0 + sby, bw, order[k])] !=
                        0) {
                        ++nz;
                    }
                }
                const std::int32_t norm{static_cast<std::int32_t>((nz + cb - 1) >> l2)};
                for (std::size_t dy{0}; dy < cx; ++dy) {
                    for (std::size_t dx{0}; dx < cx; ++dx) {
                        nz_norm[c][(sby + dy) * gbw + (sbx + dx)] = norm;
                    }
                }
            }
        }
    }

    const auto emit = [&](std::uint32_t value, std::uint32_t context) {
        std::uint32_t symbol{}, nbits{}, bits{};
        config.encode(value, symbol, nbits, bits);
        ++context_hist[context * AC_CONTEXT_ALPHABET + symbol];
        out.push_back({context, symbol, nbits, bits});
    };

    // Pass 2: contexts + tokens in decoder order.
    for (std::size_t sby{0}; sby < gbh; ++sby) {
        for (std::size_t sbx{0}; sbx < gbw; ++sbx) {
            const std::size_t gbx{bx0 + sbx};
            const std::size_t gby{by0 + sby};
            const int side{block_side(fc, gby * bw + gbx)};
            if (side == ACS_COVERED) {
                continue;
            }
            const std::size_t cb{covered_blocks_side(side) * covered_blocks_side(side)};
            const std::size_t sz{static_cast<std::size_t>(side) * side};
            const std::uint32_t l2{static_cast<std::uint32_t>(log2_covered(side))};
            const int ord{ac_strategy_order(side)};
            for (int c : channel_order) {
                const std::uint16_t* const order{orders.order + coeff_order_offset(side, c)};
                std::uint32_t nzeros{0};
                for (std::size_t k{cb}; k < sz; ++k) {
                    if (fc.ac[c][covered_plane_slot(side, gbx, gby, bw, order[k])] != 0) {
                        ++nzeros;
                    }
                }
                const int block_ctx{ac_block_context(c, ord)};
                const bool has_left{sbx > 0};
                const bool has_top{sby > 0};
                const std::int32_t topv{has_top ? nz_norm[c][(sby - 1) * gbw + sbx] : 0};
                const std::int32_t leftv{has_left ? nz_norm[c][sby * gbw + (sbx - 1)] : 0};
                const std::uint32_t predicted{ac_predict_nonzeros(has_left, has_top, topv, leftv)};
                emit(nzeros, ac_nonzero_context(predicted, block_ctx));

                const std::uint32_t histo_off{ac_zero_density_offset(block_ctx)};
                std::uint32_t remaining{nzeros};
                std::uint32_t prev{nzeros > sz / 16 ? 0u : 1u};
                for (std::size_t k{cb}; k < sz && remaining > 0; ++k) {
                    const std::int32_t v{
                        fc.ac[c][covered_plane_slot(side, gbx, gby, bw, order[k])]};
                    const std::uint32_t ctx{
                        histo_off +
                        ac_zero_density_context(remaining, static_cast<std::uint32_t>(k),
                                                static_cast<std::uint32_t>(cb), l2, prev)};
                    emit(pack_signed(v), ctx);
                    prev = v != 0 ? 1u : 0u;
                    remaining -= prev;
                }
            }
        }
    }
}

// Emits a run of AC tokens with the per-cluster prefix codes.
void emit_ac_tokens(BitWriter& w, const std::vector<AcTok>& toks, std::size_t begin,
                    std::size_t end, const AcClustering& clustering,
                    const std::vector<std::uint8_t>& depth,
                    const std::vector<std::uint16_t>& bits) {
    for (std::size_t i{begin}; i < end; ++i) {
        const AcTok& t{toks[i]};
        const std::size_t idx{clustering.context_map[t.context] * AC_CONTEXT_ALPHABET + t.symbol};
        w.write(depth[idx], bits[idx]);
        if (t.nbits) {
            w.write(t.nbits, t.bits);
        }
    }
}

// Tokenizes every AC group in raster order, returning each group's [begin, end)
// range into `toks`.
std::vector<std::pair<std::size_t, std::size_t>> tokenize_all_ac_groups(
    const FrameCoefficients& fc, const CoeffOrders& orders, std::size_t bw, std::size_t bh,
    std::vector<std::uint32_t>& context_hist, std::vector<AcTok>& toks) {
    const std::size_t xg{ceil_div(bw, AC_GROUP_BLOCKS)};
    const std::size_t yg{ceil_div(bh, AC_GROUP_BLOCKS)};
    std::vector<std::pair<std::size_t, std::size_t>> ranges(xg * yg);
    for (std::size_t g{0}; g < ranges.size(); ++g) {
        const std::size_t bx0{(g % xg) * AC_GROUP_BLOCKS};
        const std::size_t by0{(g / xg) * AC_GROUP_BLOCKS};
        const std::size_t gbw{std::min(AC_GROUP_BLOCKS, bw - bx0)};
        const std::size_t gbh{std::min(AC_GROUP_BLOCKS, bh - by0)};
        const std::size_t begin{toks.size()};
        tokenize_ac_group_clustered(fc, orders, bw, bx0, by0, gbw, gbh, context_hist, toks);
        ranges[g] = {begin, toks.size()};
    }
    return ranges;
}

// Writes the clustered AC histograms (context map + per-cluster prefix codes)
// and fills the per-cluster `depth`/`bits` used by emit_ac_tokens.
void write_ac_histograms(BitWriter& w, const AcClustering& clustering,
                         std::vector<std::uint8_t>& depth, std::vector<std::uint16_t>& bits) {
    depth.assign(clustering.num_clusters * AC_CONTEXT_ALPHABET, 0);
    bits.assign(clustering.num_clusters * AC_CONTEXT_ALPHABET, 0);
    write_clustered_prefix_histograms(
        w, clustering.context_map, AC_NUM_CONTEXTS, clustering.num_clusters, clustering.histograms,
        AC_CONTEXT_ALPHABET, HybridUintConfig{}, depth.data(), bits.data());
}

void write_dc_global(BitWriter& w, const FrameCoefficients& fc) {
    write_bool(w, true);  // DequantMatrices::DecodeDC all_default
    write_u32(w, GLOBAL_SCALE_ENC, fc.global_scale);
    write_u32(w, QUANT_DC_ENC, fc.quant_dc);  // Quantizer::Decode
    write_bool(w, true);                      // DecodeBlockCtxMap is_default
    write_bool(w, true);                      // ColorCorrelation::DecodeDC all_default
    write_bool(w, false);                     // modular global has_tree = 0 (no channels)
}

// One DcGroup section covering dgw x dgh blocks at block origin (bx0, by0).
void write_dc_group(BitWriter& w, const FrameCoefficients& fc, std::size_t bw, std::size_t bx0,
                    std::size_t by0, std::size_t dgw, std::size_t dgh) {
    write_bits(w, 2, 0);  // DecodeVarDCTDC extra_precision = 0
    write_modular_image(w, make_vardct_dc_channels(fc, bw, bx0, by0, dgw, dgh), DC_TREE);
    // DecodeGroup(ModularDC): VarDCT full image has no channels -> nothing.
    const std::vector<ModularChannel> meta{make_ac_metadata_channels(fc, bw, bx0, by0, dgw, dgh)};
    const std::size_t count{meta[2].w};  // ACS+QF width = number of first-blocks
    // AcMetadata count - 1, width fixed by the block-count upper bound.
    write_bits(w, ceil_log2_nonzero(dgw * dgh), static_cast<std::uint32_t>(count - 1));
    write_modular_image(w, meta, ACMETA_TREE);
}

std::vector<std::uint8_t> write_single_group_codestream(const FrameCoefficients& fc, std::size_t bw,
                                                        std::size_t bh) {
    BitWriter body{};
    write_dc_global(body, fc);
    write_dc_group(body, fc, bw, 0, 0, bw, bh);
    // AcGlobal.
    write_bool(body, true);  // DequantMatrices::Decode all_default
    // num_histograms: CeilLog2Nonzero(num_groups == 1) == 0 bits.
    write_u32(body, ORDER_ENC, 0);  // used_orders = 0 (natural order)
    // Without statistics every position has the same key: the natural orders.
    const CoeffOrders natural{coeff_orders_reference(CoeffOrderStats{})};
    std::vector<std::uint32_t> context_hist(AC_NUM_CONTEXTS * AC_CONTEXT_ALPHABET, 0);
    std::vector<AcTok> toks{};
    tokenize_ac_group_clustered(fc, natural, bw, 0, 0, bw, bh, context_hist, toks);
    const AcClustering clustering{cluster_ac_contexts_reference(context_hist.data())};
    std::vector<std::uint8_t> depth{};
    std::vector<std::uint16_t> bits{};
    write_ac_histograms(body, clustering, depth, bits);
    emit_ac_tokens(body, toks, 0, toks.size(), clustering, depth, bits);
    body.zero_pad_to_byte();

    BitWriter w{};
    write_codestream_headers(w, static_cast<std::uint32_t>(fc.width),
                             static_cast<std::uint32_t>(fc.height));
    write_frame_header(w);
    write_toc_single_section(w, body.size_bytes());
    w.append_aligned(body);
    return w.bytes();
}

// The AC groups' token streams as the device packs them and their sections,
// coded with the AcGlobal code of the frame's context clustering.
AcReference build_ac_reference(const FrameCoefficients& fc, std::size_t bw, std::size_t bh) {
    AcReference ref{};
    ref.orders = frame_coeff_orders(fc, bw, bh);
    ref.context_histograms.assign(AC_NUM_CONTEXTS * AC_CONTEXT_ALPHABET, 0);
    std::vector<AcTok> toks{};
    const std::vector<std::pair<std::size_t, std::size_t>> ranges{
        tokenize_all_ac_groups(fc, ref.orders, bw, bh, ref.context_histograms, toks)};
    ref.clustering = cluster_ac_contexts_reference(ref.context_histograms.data());
    for (const auto& range : ranges) {
        std::vector<std::uint16_t> entries{};
        std::vector<std::uint64_t> raw(1, 0);
        std::uint64_t raw_bits{0};
        for (std::size_t i{range.first}; i < range.second; ++i) {
            const AcTok& t{toks[i]};
            entries.push_back(static_cast<std::uint16_t>(
                ref.clustering.context_map[t.context] << TOKEN_CLUSTER_SHIFT | t.symbol));
            for (std::uint32_t b{0}; b < t.nbits; ++b) {
                if (raw_bits / 64 + 1 >= raw.size()) {
                    raw.push_back(0);
                }
                raw[raw_bits / 64] |= static_cast<std::uint64_t>((t.bits >> b) & 1)
                                      << (raw_bits % 64);
                ++raw_bits;
            }
        }
        ref.group_entries.push_back(std::move(entries));
        ref.group_raw.push_back(std::move(raw));
        ref.group_raw_bits.push_back(raw_bits);
    }
    TokenCode code{};
    AcGlobalBuilder builder{ref.clustering, ref.orders, ranges.size(), code};
    for (std::size_t c{0}; c < builder.num_clusters(); ++c) {
        builder.build_cluster(c);
    }
    builder.build_coeff_orders();
    builder.build_context_map();
    ref.ac_global = builder.finish();
    for (std::size_t g{0}; g < ranges.size(); ++g) {
        const TokenStream stream{ref.group_entries[g].data(), ref.group_entries[g].size(),
                                 ref.group_raw[g].data(), ref.group_raw_bits[g]};
        ref.group_streams.push_back(code_token_stream_reference(code, stream));
    }
    return ref;
}

std::vector<std::uint8_t> write_multi_group_codestream(const FrameCoefficients& fc, std::size_t bw,
                                                       std::size_t bh) {
    const std::size_t xdg{ceil_div(bw, DC_GROUP_BLOCKS)};
    const std::size_t ydg{ceil_div(bh, DC_GROUP_BLOCKS)};
    const std::size_t num_dc_groups{xdg * ydg};

    // Sections in codestream (TOC id) order: DcGlobal, DcGroups, AcGlobal,
    // AcGroups.
    std::vector<std::vector<std::uint8_t>> sections{};

    BitWriter dc_global{};
    write_dc_global(dc_global, fc);
    dc_global.zero_pad_to_byte();
    sections.push_back(dc_global.bytes());

    for (std::size_t i{0}; i < num_dc_groups; ++i) {
        const std::size_t gx{i % xdg};
        const std::size_t gy{i / xdg};
        const std::size_t bx0{gx * DC_GROUP_BLOCKS};
        const std::size_t by0{gy * DC_GROUP_BLOCKS};
        const std::size_t dgw{std::min(DC_GROUP_BLOCKS, bw - bx0)};
        const std::size_t dgh{std::min(DC_GROUP_BLOCKS, bh - by0)};
        BitWriter dc_group{};
        write_dc_group(dc_group, fc, bw, bx0, by0, dgw, dgh);
        dc_group.zero_pad_to_byte();
        sections.push_back(dc_group.bytes());
    }

    AcReference ac{build_ac_reference(fc, bw, bh)};
    sections.push_back(std::move(ac.ac_global));
    for (std::vector<std::uint8_t>& group : ac.group_streams) {
        sections.push_back(std::move(group));
    }

    std::vector<std::uint32_t> section_sizes{};
    section_sizes.reserve(sections.size());
    for (const std::vector<std::uint8_t>& section : sections) {
        section_sizes.push_back(static_cast<std::uint32_t>(section.size()));
    }

    BitWriter w{};
    write_codestream_headers(w, static_cast<std::uint32_t>(fc.width),
                             static_cast<std::uint32_t>(fc.height));
    write_frame_header(w);
    write_toc_multi_section(w, section_sizes);
    std::vector<std::uint8_t> out{w.bytes()};
    for (const std::vector<std::uint8_t>& section : sections) {
        out.insert(out.end(), section.begin(), section.end());
    }
    return out;
}

template <typename T>
std::vector<T> leaf_row(const std::vector<T>& rows, std::size_t leaf) {
    return std::vector<T>(rows.begin() + leaf * HISTOGRAM_STRIDE,
                          rows.begin() + (leaf + 1) * HISTOGRAM_STRIDE);
}

}  // namespace

const std::array<std::uint32_t, 64>& dct8_natural_order() {
    static const std::array<std::uint32_t, 64> ORDER{compute_dct8_natural_order()};
    return ORDER;
}

AcReference reference_ac_encode(const FrameCoefficients& fc) {
    assert(fc.width % 8 == 0 && fc.height % 8 == 0);
    return build_ac_reference(fc, fc.width / 8, fc.height / 8);
}

DcReference reference_dc_encode(const FrameCoefficients& fc) {
    assert(fc.width % 8 == 0 && fc.height % 8 == 0);
    const std::size_t bw{fc.width / 8};
    const std::size_t bh{fc.height / 8};
    const std::size_t xdg{ceil_div(bw, DC_GROUP_BLOCKS)};
    const std::size_t ydg{ceil_div(bh, DC_GROUP_BLOCKS)};

    DcReference ref{};
    ref.groups.resize(xdg * ydg);
    for (std::size_t i{0}; i < xdg * ydg; ++i) {
        const std::size_t gx{i % xdg};
        const std::size_t gy{i / xdg};
        const std::size_t bx0{gx * DC_GROUP_BLOCKS};
        const std::size_t by0{gy * DC_GROUP_BLOCKS};
        const std::size_t dgw{std::min(DC_GROUP_BLOCKS, bw - bx0)};
        const std::size_t dgh{std::min(DC_GROUP_BLOCKS, bh - by0)};

        DcGroupReference& g{ref.groups[i]};
        g.bx0 = bx0;
        g.by0 = by0;
        g.dgw = dgw;
        g.dgh = dgh;

        const std::vector<ModularChannel> dc_ch{
            make_vardct_dc_channels(fc, bw, bx0, by0, dgw, dgh)};
        const std::vector<ModularChannel> meta_ch{
            make_ac_metadata_channels(fc, bw, bx0, by0, dgw, dgh)};

        BitWriter pre{};
        write_bits(pre, 2, 0);  // DecodeVarDCTDC extra_precision = 0
        const ModularData dc_data{write_modular_header(pre, dc_ch, DC_TREE)};
        g.blob_pre = pre.bytes();
        g.blob_pre_bits = pre.bits_written();
        g.dc_depth = leaf_row(dc_data.depth, 0);
        g.dc_bits = leaf_row(dc_data.bits, 0);
        g.dc_histogram = leaf_row(dc_data.histograms, 0);

        BitWriter mid{};
        write_bits(mid, ceil_log2_nonzero(dgw * dgh),
                   static_cast<std::uint32_t>(meta_ch[2].w - 1));  // AcMetadata count - 1
        const ModularData meta_data{write_modular_header(mid, meta_ch, ACMETA_TREE)};
        g.blob_mid = mid.bytes();
        g.blob_mid_bits = mid.bits_written();
        g.acmeta_depth = leaf_row(meta_data.depth, ACMETA_DATA_LEAF);
        g.acmeta_bits = leaf_row(meta_data.bits, ACMETA_DATA_LEAF);
        g.acmeta_histogram = leaf_row(meta_data.histograms, ACMETA_DATA_LEAF);

        BitWriter sec{};
        write_dc_group(sec, fc, bw, bx0, by0, dgw, dgh);
        sec.zero_pad_to_byte();
        g.section = sec.bytes();
    }
    return ref;
}

std::vector<std::uint8_t> write_vardct_codestream(const FrameCoefficients& fc) {
    assert(fc.width % 8 == 0 && fc.height % 8 == 0);
    const std::size_t bw{fc.width / 8};
    const std::size_t bh{fc.height / 8};

    // NumTocEntries(num_groups, .., 1) == 1 only when there is a single AC
    // group; then the whole frame is one combined section.
    if (bw <= AC_GROUP_BLOCKS && bh <= AC_GROUP_BLOCKS) {
        return write_single_group_codestream(fc, bw, bh);
    }
    return write_multi_group_codestream(fc, bw, bh);
}

}  // namespace cujpegxl::bitstream
