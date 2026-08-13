// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "vardct_frame.h"

#include <algorithm>
#include <cassert>
#include <utility>

#include "src/bitstream/bit_writer.h"
#include "src/bitstream/codestream.h"
#include "src/bitstream/field_coder.h"
#include "src/bitstream/hybrid_uint.h"
#include "src/coeff_order.h"
#include "src/vardct_layout.h"

#include "entropy_encoder.h"
#include "modular.h"

namespace cujpegxl::bitstream {
namespace {

// libjxl default BlockCtxMap: 15 block contexts, 37 non-zero buckets, 458
// zero-density contexts. The AC entropy container is sized to the resulting
// context count so the decoder reads a (single-cluster) context map.
constexpr std::size_t NUM_BLOCK_CTX = 15;
constexpr std::size_t NON_ZERO_BUCKETS = 37;
constexpr std::size_t ZERO_DENSITY_CONTEXT_COUNT = 458;
constexpr std::size_t NUM_AC_CONTEXTS =
    NUM_BLOCK_CTX * (NON_ZERO_BUCKETS + ZERO_DENSITY_CONTEXT_COUNT);

// Group geometry in blocks: AC groups are 256px (32 blocks) square, DC groups
// are 2048px (256 blocks) square (libjxl kGroupDim with group_size_shift = 1).
constexpr std::size_t AC_GROUP_BLOCKS = 32;
constexpr std::size_t DC_GROUP_BLOCKS = 256;

constexpr U32Enc GLOBAL_SCALE_ENC{
    {BitsOffset(11, 1), BitsOffset(11, 2049), BitsOffset(12, 4097), BitsOffset(16, 8193)}};
constexpr U32Enc QUANT_DC_ENC{{Val(16), BitsOffset(5, 1), BitsOffset(8, 1), BitsOffset(16, 1)}};
constexpr U32Enc ORDER_ENC{{Val(0x5F), Val(0x13), Val(0), Bits(13)}};

std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

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
std::vector<ModularChannel> make_vardct_dc_channels(const FrameCoefficients& fc,
                                                    std::size_t bw, std::size_t bx0,
                                                    std::size_t by0, std::size_t dgw,
                                                    std::size_t dgh) {
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
std::int32_t raw_strategy(int side) { return side == 16 ? 4 : (side == 32 ? 5 : 0); }

// The block's transform side; an empty acs signals an all-DCT8 frame.
int block_side(const FrameCoefficients& fc, std::size_t blk) {
    return fc.acs.empty() ? 8 : fc.acs[blk];
}

const std::vector<std::uint32_t>& natural_order_for(int side) {
    static const std::vector<std::uint32_t> o8{natural_coeff_order(8)};
    static const std::vector<std::uint32_t> o16{natural_coeff_order(16)};
    static const std::vector<std::uint32_t> o32{natural_coeff_order(32)};
    return side == 16 ? o16 : (side == 32 ? o32 : o8);
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

// Tokenizes the AC coefficients of one AC group into `ac`: for each first-block
// in group raster order (covered blocks skipped), channels Y, X, B, the nonzero
// count over the block's AC coefficients (its size - covered_blocks scan
// positions, starting past the LLF) followed by those coefficients in the
// block's natural scan order up to the last nonzero. Returns the token count.
std::size_t tokenize_ac_group(const FrameCoefficients& fc, std::size_t bw, std::size_t bx0,
                              std::size_t by0, std::size_t gbw, std::size_t gbh, EntropyEncoder& ac) {
    const int channel_order[3]{1, 0, 2};  // Y, X, B (decoder LoadBlock order)
    const std::size_t before{ac.num_tokens()};
    for (std::size_t by{0}; by < gbh; ++by) {
        for (std::size_t bx{0}; bx < gbw; ++bx) {
            const std::size_t gbx{bx0 + bx};
            const std::size_t gby{by0 + by};
            const int side{block_side(fc, gby * bw + gbx)};
            if (side == ACS_COVERED) {
                continue;
            }
            const std::size_t covered{covered_blocks_side(side) * covered_blocks_side(side)};
            const std::size_t size{static_cast<std::size_t>(side) * side};
            const std::vector<std::uint32_t>& order{natural_order_for(side)};
            for (int c : channel_order) {
                std::uint32_t nzeros{0};
                for (std::size_t k{covered}; k < size; ++k) {
                    if (fc.ac[c][covered_plane_slot(side, gbx, gby, bw, order[k])] != 0) {
                        ++nzeros;
                    }
                }
                ac.add_token(0, nzeros);
                std::uint32_t remaining{nzeros};
                for (std::size_t k{covered}; k < size && remaining > 0; ++k) {
                    const std::int32_t v{fc.ac[c][covered_plane_slot(side, gbx, gby, bw, order[k])]};
                    ac.add_token(0, pack_signed(v));
                    if (v != 0) {
                        --remaining;
                    }
                }
            }
        }
    }
    return ac.num_tokens() - before;
}

// Tokenizes every AC group in raster order into `ac`, returning each group's
// [begin, end) token range. Shared by the codestream writer and the device
// reference so both tokenize identically.
std::vector<std::pair<std::size_t, std::size_t>> tokenize_all_ac_groups(
    const FrameCoefficients& fc, std::size_t bw, std::size_t bh, EntropyEncoder& ac) {
    const std::size_t xg{ceil_div(bw, AC_GROUP_BLOCKS)};
    const std::size_t yg{ceil_div(bh, AC_GROUP_BLOCKS)};
    std::vector<std::pair<std::size_t, std::size_t>> ranges(xg * yg);
    for (std::size_t g{0}; g < xg * yg; ++g) {
        const std::size_t gx{g % xg};
        const std::size_t gy{g / xg};
        const std::size_t bx0{gx * AC_GROUP_BLOCKS};
        const std::size_t by0{gy * AC_GROUP_BLOCKS};
        const std::size_t gbw{std::min(AC_GROUP_BLOCKS, bw - bx0)};
        const std::size_t gbh{std::min(AC_GROUP_BLOCKS, bh - by0)};
        const std::size_t begin{ac.num_tokens()};
        tokenize_ac_group(fc, bw, bx0, by0, gbw, gbh, ac);
        ranges[g] = {begin, ac.num_tokens()};
    }
    return ranges;
}

void write_dc_global(BitWriter& w, const FrameCoefficients& fc) {
    write_bool(w, true);                     // DequantMatrices::DecodeDC all_default
    write_u32(w, GLOBAL_SCALE_ENC, fc.global_scale);
    write_u32(w, QUANT_DC_ENC, fc.quant_dc);  // Quantizer::Decode
    write_bool(w, true);                     // DecodeBlockCtxMap is_default
    write_bool(w, true);                     // ColorCorrelation::DecodeDC all_default
    write_bool(w, false);                    // modular global has_tree = 0 (no channels)
}

// One DcGroup section covering dgw x dgh blocks at block origin (bx0, by0).
void write_dc_group(BitWriter& w, const FrameCoefficients& fc, std::size_t bw,
                    std::size_t bx0, std::size_t by0, std::size_t dgw, std::size_t dgh) {
    write_bits(w, 2, 0);  // DecodeVarDCTDC extra_precision = 0
    write_modular_image(w, make_vardct_dc_channels(fc, bw, bx0, by0, dgw, dgh),
                        DC_PREDICTOR_GRADIENT);
    // DecodeGroup(ModularDC): VarDCT full image has no channels -> nothing.
    const std::vector<ModularChannel> meta{make_ac_metadata_channels(fc, bw, bx0, by0, dgw, dgh)};
    const std::size_t count{meta[2].w};  // ACS+QF width = number of first-blocks
    // AcMetadata count - 1, width fixed by the block-count upper bound.
    write_bits(w, static_cast<std::size_t>(ceil_log2_nonzero(dgw * dgh)),
               static_cast<std::uint32_t>(count - 1));
    write_modular_image(w, meta);
}

std::vector<std::uint8_t> write_single_group_codestream(const FrameCoefficients& fc,
                                                        std::size_t bw, std::size_t bh) {
    EntropyEncoder ac{NUM_AC_CONTEXTS};
    tokenize_ac_group(fc, bw, 0, 0, bw, bh, ac);

    BitWriter body{};
    write_dc_global(body, fc);
    write_dc_group(body, fc, bw, 0, 0, bw, bh);
    // AcGlobal.
    write_bool(body, true);         // DequantMatrices::Decode all_default
    // num_histograms: CeilLog2Nonzero(num_groups == 1) == 0 bits.
    write_u32(body, ORDER_ENC, 0);  // used_orders = 0 (natural order)
    ac.write_histograms(body);
    // AcGroup.
    ac.write_tokens(body);
    body.zero_pad_to_byte();

    BitWriter w{};
    write_codestream_headers(w, static_cast<std::uint32_t>(fc.width),
                             static_cast<std::uint32_t>(fc.height));
    write_frame_header(w);
    write_toc_single_section(w, body.size_bytes());
    w.append_aligned(body);
    return w.bytes();
}

std::vector<std::uint8_t> write_multi_group_codestream(const FrameCoefficients& fc,
                                                       std::size_t bw, std::size_t bh) {
    const std::size_t xdg{ceil_div(bw, DC_GROUP_BLOCKS)};
    const std::size_t ydg{ceil_div(bh, DC_GROUP_BLOCKS)};
    const std::size_t num_dc_groups{xdg * ydg};

    // Pool all AC groups into one container (one shared prefix code) and record
    // each group's token range so it can be emitted in its own AcGroup section.
    EntropyEncoder ac{NUM_AC_CONTEXTS};
    const std::vector<std::pair<std::size_t, std::size_t>> group_token_ranges{
        tokenize_all_ac_groups(fc, bw, bh, ac)};
    const std::size_t num_groups{group_token_ranges.size()};

    // Sections in codestream (TOC id) order: DcGlobal, DcGroups, AcGlobal,
    // AcGroups.
    std::vector<BitWriter> sections{};

    BitWriter dc_global{};
    write_dc_global(dc_global, fc);
    dc_global.zero_pad_to_byte();
    sections.push_back(std::move(dc_global));

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
        sections.push_back(std::move(dc_group));
    }

    BitWriter ac_global{};
    write_bool(ac_global, true);  // DequantMatrices::Decode all_default
    write_bits(ac_global, ceil_log2_nonzero(num_groups), 0);  // num_histograms - 1
    write_u32(ac_global, ORDER_ENC, 0);                       // used_orders = 0
    ac.write_histograms(ac_global);
    ac_global.zero_pad_to_byte();
    sections.push_back(std::move(ac_global));

    for (std::size_t g{0}; g < num_groups; ++g) {
        BitWriter ac_group{};
        ac.write_tokens_range(ac_group, group_token_ranges[g].first,
                              group_token_ranges[g].second);
        ac_group.zero_pad_to_byte();
        sections.push_back(std::move(ac_group));
    }

    std::vector<std::uint32_t> section_sizes{};
    section_sizes.reserve(sections.size());
    for (const BitWriter& s : sections) {
        section_sizes.push_back(static_cast<std::uint32_t>(s.size_bytes()));
    }

    BitWriter w{};
    write_codestream_headers(w, static_cast<std::uint32_t>(fc.width),
                             static_cast<std::uint32_t>(fc.height));
    write_frame_header(w);
    write_toc_multi_section(w, section_sizes);
    for (const BitWriter& s : sections) {
        w.append_aligned(s);
    }
    return w.bytes();
}

}  // namespace

const std::array<std::uint32_t, 64>& dct8_natural_order() {
    static const std::array<std::uint32_t, 64> ORDER{compute_dct8_natural_order()};
    return ORDER;
}

AcReference reference_ac_encode(const FrameCoefficients& fc) {
    assert(fc.width % 8 == 0 && fc.height % 8 == 0);
    const std::size_t bw{fc.width / 8};
    const std::size_t bh{fc.height / 8};

    EntropyEncoder ac{NUM_AC_CONTEXTS};
    const std::vector<std::pair<std::size_t, std::size_t>> ranges{
        tokenize_all_ac_groups(fc, bw, bh, ac)};

    AcReference ref{};
    ref.histogram = ac.histogram();

    BitWriter histograms{};
    ac.write_histograms(histograms);  // builds the shared prefix code
    ref.depth = ac.code_depth();
    ref.bits = ac.code_bits();

    ref.group_streams.reserve(ranges.size());
    for (const auto& r : ranges) {
        BitWriter group{};
        ac.write_tokens_range(group, r.first, r.second);
        group.zero_pad_to_byte();
        ref.group_streams.push_back(group.bytes());
    }
    return ref;
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
        EntropyEncoder dc_data{1};
        write_modular_header(pre, dc_ch, dc_data, DC_PREDICTOR_GRADIENT);
        g.blob_pre = pre.bytes();
        g.blob_pre_bits = pre.bits_written();
        g.dc_depth = dc_data.code_depth();
        g.dc_bits = dc_data.code_bits();
        g.dc_histogram = dc_data.histogram();

        BitWriter mid{};
        write_bits(mid, static_cast<std::size_t>(ceil_log2_nonzero(dgw * dgh)),
                   static_cast<std::uint32_t>(meta_ch[2].w - 1));  // AcMetadata count - 1
        EntropyEncoder meta_data{1};
        write_modular_header(mid, meta_ch, meta_data);
        g.blob_mid = mid.bytes();
        g.blob_mid_bits = mid.bits_written();
        g.acmeta_depth = meta_data.code_depth();
        g.acmeta_bits = meta_data.code_bits();
        g.acmeta_histogram = meta_data.histogram();

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
