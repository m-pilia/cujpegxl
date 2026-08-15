// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "vardct_frame.h"

#include <algorithm>
#include <cassert>
#include <utility>

#include "src/ac_context.h"
#include "src/bitstream/ans_histogram.h"
#include "src/bitstream/bit_writer.h"
#include "src/bitstream/codestream.h"
#include "src/bitstream/field_coder.h"
#include "src/bitstream/histogram_writer.h"
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

// --- Clustered AC entropy (real per-token contexts + 8-cluster histograms) ----

constexpr std::size_t AC_STRIDE = 256;

struct AcTok {
    std::uint8_t cluster;
    std::uint32_t context;
    std::uint32_t symbol;
    std::uint32_t nbits;
    std::uint32_t bits;
};

std::size_t log2_covered(int side) { return side == 16 ? 2 : (side == 32 ? 4 : 0); }

// Tokenizes one AC group with libjxl's real per-token contexts, appending the
// (cluster, symbol, extra-bits) tuples to `out` and accumulating the per-cluster
// symbol histograms. The context math mirrors dec_group.cc DecodeACVarBlock: a
// group-local normalized non-zero grid drives the non-zero-count prediction, and
// each coefficient's zero-density context follows the scan.
void tokenize_ac_group_clustered(const FrameCoefficients& fc, std::size_t bw, std::size_t bx0,
                                 std::size_t by0, std::size_t gbw, std::size_t gbh,
                                 std::vector<std::uint32_t>& cluster_hist,
                                 std::vector<AcTok>& out,
                                 const std::uint8_t* context_map = nullptr) {
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
            const std::vector<std::uint32_t>& order{natural_order_for(side)};
            for (int c : channel_order) {
                std::int32_t nz{0};
                for (std::size_t k{cb}; k < sz; ++k) {
                    if (fc.ac[c][covered_plane_slot(side, bx0 + sbx, by0 + sby, bw, order[k])] != 0) {
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
        const std::uint8_t cluster{
            context_map == nullptr
                ? static_cast<std::uint8_t>(ac_cluster(context))
                : context_map[context]};
        std::uint32_t symbol{}, nbits{}, bits{};
        config.encode(value, symbol, nbits, bits);
        ++cluster_hist[static_cast<std::size_t>(cluster) * AC_STRIDE + symbol];
        out.push_back({cluster, context, symbol, nbits, bits});
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
            const std::vector<std::uint32_t>& order{natural_order_for(side)};
            for (int c : channel_order) {
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
                const std::uint32_t predicted{
                    ac_predict_nonzeros(has_left, has_top, topv, leftv)};
                emit(nzeros, ac_nonzero_context(predicted, block_ctx));

                const std::uint32_t histo_off{ac_zero_density_offset(block_ctx)};
                std::uint32_t remaining{nzeros};
                std::uint32_t prev{nzeros > sz / 16 ? 0u : 1u};
                for (std::size_t k{cb}; k < sz && remaining > 0; ++k) {
                    const std::int32_t v{
                        fc.ac[c][covered_plane_slot(side, gbx, gby, bw, order[k])]};
                    const std::uint32_t ctx{
                        histo_off + ac_zero_density_context(remaining, static_cast<std::uint32_t>(k),
                                                            static_cast<std::uint32_t>(cb), l2, prev)};
                    emit(pack_signed(v), ctx);
                    prev = v != 0 ? 1u : 0u;
                    remaining -= prev;
                }
            }
        }
    }
}

// The fixed context map: ac_cluster evaluated over the whole AC context space.
std::vector<std::uint8_t> ac_context_map() {
    std::vector<std::uint8_t> map(AC_NUM_CONTEXTS);
    for (std::size_t i{0}; i < static_cast<std::size_t>(AC_NUM_CONTEXTS); ++i) {
        map[i] = static_cast<std::uint8_t>(ac_cluster(static_cast<std::uint32_t>(i)));
    }
    return map;
}

// Emits a run of clustered AC tokens with the per-cluster prefix codes.
void emit_ac_tokens(BitWriter& w, const std::vector<AcTok>& toks, std::size_t begin, std::size_t end,
                    const std::vector<std::uint8_t>& depth, const std::vector<std::uint16_t>& bits) {
    for (std::size_t i{begin}; i < end; ++i) {
        const AcTok& t{toks[i]};
        const std::size_t idx{static_cast<std::size_t>(t.cluster) * AC_STRIDE + t.symbol};
        w.write(depth[idx], bits[idx]);
        if (t.nbits) {
            w.write(t.nbits, t.bits);
        }
    }
}

void emit_ac_ans_tokens(BitWriter& w, const std::vector<AcTok>& tokens,
                        std::size_t begin, std::size_t end,
                        const std::vector<AnsEncodingTable>& tables) {
    std::vector<std::uint32_t> renormalization(end - begin, 0);
    std::uint32_t state{ANS_INITIAL_STATE};
    for (std::size_t i{end}; i > begin; --i) {
        const AcTok& token{tokens[i - 1]};
        const AnsStateTransition transition{
            ans_put_symbol(state, tables[token.cluster], token.symbol)};
        state = transition.state;
        renormalization[i - begin - 1] =
            transition.renormalized ? 0x10000u | transition.renormalization_bits : 0;
    }
    w.write(32, state);
    for (std::size_t i{begin}; i < end; ++i) {
        const AcTok& token{tokens[i]};
        const std::uint32_t renorm{renormalization[i - begin]};
        if ((renorm & 0x10000u) != 0) {
            w.write(16, renorm & 0xffffu);
        }
        if (token.nbits != 0) {
            w.write(token.nbits, token.bits);
        }
    }
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
                                                        std::size_t bw, std::size_t bh,
                                                        bool clustered_ac) {
    BitWriter body{};
    write_dc_global(body, fc);
    write_dc_group(body, fc, bw, 0, 0, bw, bh);
    // AcGlobal.
    write_bool(body, true);         // DequantMatrices::Decode all_default
    // num_histograms: CeilLog2Nonzero(num_groups == 1) == 0 bits.
    write_u32(body, ORDER_ENC, 0);  // used_orders = 0 (natural order)
    if (clustered_ac) {
        std::vector<std::uint32_t> cluster_hist(AC_NUM_CLUSTERS * AC_STRIDE, 0);
        std::vector<AcTok> toks{};
        tokenize_ac_group_clustered(fc, bw, 0, 0, bw, bh, cluster_hist, toks);
        const std::vector<std::uint8_t> cmap{ac_context_map()};
        std::vector<std::uint8_t> depth(AC_NUM_CLUSTERS * AC_STRIDE, 0);
        std::vector<std::uint16_t> bits(AC_NUM_CLUSTERS * AC_STRIDE, 0);
        write_clustered_prefix_histograms(body, cmap.data(), AC_NUM_CONTEXTS, AC_NUM_CLUSTERS,
                                          cluster_hist.data(), AC_STRIDE, HybridUintConfig{},
                                          depth.data(), bits.data());
        emit_ac_tokens(body, toks, 0, toks.size(), depth, bits);
    } else {
        EntropyEncoder ac{NUM_AC_CONTEXTS};
        tokenize_ac_group(fc, bw, 0, 0, bw, bh, ac);
        ac.write_histograms(body);
        ac.write_tokens(body);  // AcGroup
    }
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
                                                       std::size_t bw, std::size_t bh,
                                                       bool clustered_ac) {
    const std::size_t xdg{ceil_div(bw, DC_GROUP_BLOCKS)};
    const std::size_t ydg{ceil_div(bh, DC_GROUP_BLOCKS)};
    const std::size_t num_dc_groups{xdg * ydg};
    const std::size_t xg{ceil_div(bw, AC_GROUP_BLOCKS)};
    const std::size_t yg{ceil_div(bh, AC_GROUP_BLOCKS)};
    const std::size_t num_groups{xg * yg};

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

    std::vector<BitWriter> ac_groups{};
    if (clustered_ac) {
        std::vector<std::uint32_t> cluster_hist(AC_NUM_CLUSTERS * AC_STRIDE, 0);
        std::vector<AcTok> toks{};
        std::vector<std::pair<std::size_t, std::size_t>> ranges(num_groups);
        for (std::size_t g{0}; g < num_groups; ++g) {
            const std::size_t bx0{(g % xg) * AC_GROUP_BLOCKS};
            const std::size_t by0{(g / xg) * AC_GROUP_BLOCKS};
            const std::size_t gbw{std::min(AC_GROUP_BLOCKS, bw - bx0)};
            const std::size_t gbh{std::min(AC_GROUP_BLOCKS, bh - by0)};
            const std::size_t begin{toks.size()};
            tokenize_ac_group_clustered(fc, bw, bx0, by0, gbw, gbh, cluster_hist, toks);
            ranges[g] = {begin, toks.size()};
        }
        const std::vector<std::uint8_t> cmap{ac_context_map()};
        std::vector<std::uint8_t> depth(AC_NUM_CLUSTERS * AC_STRIDE, 0);
        std::vector<std::uint16_t> bits(AC_NUM_CLUSTERS * AC_STRIDE, 0);
        write_clustered_prefix_histograms(ac_global, cmap.data(), AC_NUM_CONTEXTS, AC_NUM_CLUSTERS,
                                          cluster_hist.data(), AC_STRIDE, HybridUintConfig{},
                                          depth.data(), bits.data());
        for (std::size_t g{0}; g < num_groups; ++g) {
            BitWriter ac_group{};
            emit_ac_tokens(ac_group, toks, ranges[g].first, ranges[g].second, depth, bits);
            ac_group.zero_pad_to_byte();
            ac_groups.push_back(std::move(ac_group));
        }
    } else {
        EntropyEncoder ac{NUM_AC_CONTEXTS};
        const std::vector<std::pair<std::size_t, std::size_t>> group_token_ranges{
            tokenize_all_ac_groups(fc, bw, bh, ac)};
        ac.write_histograms(ac_global);
        for (std::size_t g{0}; g < num_groups; ++g) {
            BitWriter ac_group{};
            ac.write_tokens_range(ac_group, group_token_ranges[g].first,
                                  group_token_ranges[g].second);
            ac_group.zero_pad_to_byte();
            ac_groups.push_back(std::move(ac_group));
        }
    }
    ac_global.zero_pad_to_byte();
    sections.push_back(std::move(ac_global));
    for (BitWriter& ac_group : ac_groups) {
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
    return reference_ac_encode(fc, ac_context_map(), AC_NUM_CLUSTERS);
}

AcReference reference_ac_encode(const FrameCoefficients& fc,
                                const std::vector<std::uint8_t>& context_map,
                                std::size_t num_clusters) {
    assert(fc.width % 8 == 0 && fc.height % 8 == 0);
    assert(context_map.size() == AC_NUM_CONTEXTS);
    assert(num_clusters > 0 && num_clusters <= 256);
    for (std::uint8_t cluster : context_map) {
        assert(cluster < num_clusters);
    }
    const std::size_t bw{fc.width / 8};
    const std::size_t bh{fc.height / 8};
    const std::size_t xg{ceil_div(bw, AC_GROUP_BLOCKS)};
    const std::size_t yg{ceil_div(bh, AC_GROUP_BLOCKS)};
    const std::size_t num_groups{xg * yg};

    std::vector<std::uint32_t> cluster_hist(num_clusters * AC_STRIDE, 0);
    std::vector<AcTok> toks{};
    std::vector<std::pair<std::size_t, std::size_t>> ranges(num_groups);
    for (std::size_t g{0}; g < num_groups; ++g) {
        const std::size_t bx0{(g % xg) * AC_GROUP_BLOCKS};
        const std::size_t by0{(g / xg) * AC_GROUP_BLOCKS};
        const std::size_t gbw{std::min(AC_GROUP_BLOCKS, bw - bx0)};
        const std::size_t gbh{std::min(AC_GROUP_BLOCKS, bh - by0)};
        const std::size_t begin{toks.size()};
        tokenize_ac_group_clustered(fc, bw, bx0, by0, gbw, gbh, cluster_hist, toks,
                                    context_map.data());
        ranges[g] = {begin, toks.size()};
    }

    AcReference ref{};
    ref.histogram = cluster_hist;

    ref.depth.assign(num_clusters * AC_STRIDE, 0);
    ref.bits.assign(num_clusters * AC_STRIDE, 0);
    BitWriter histograms{};
    write_clustered_prefix_histograms(histograms, context_map.data(), AC_NUM_CONTEXTS,
                                      num_clusters, cluster_hist.data(), AC_STRIDE,
                                      HybridUintConfig{},
                                      ref.depth.data(), ref.bits.data());

    ref.group_streams.reserve(num_groups);
    for (const auto& r : ranges) {
        BitWriter group{};
        emit_ac_tokens(group, toks, r.first, r.second, ref.depth, ref.bits);
        group.zero_pad_to_byte();
        ref.group_streams.push_back(group.bytes());
    }
    return ref;
}

AcAnsReference reference_ac_ans_encode(const FrameCoefficients& fc) {
    return reference_ac_ans_encode(fc, ac_context_map(), AC_NUM_CLUSTERS);
}

AcAnsReference reference_ac_ans_encode(
    const FrameCoefficients& fc, const std::vector<std::uint8_t>& context_map,
    std::size_t num_clusters) {
    assert(fc.width % 8 == 0 && fc.height % 8 == 0);
    assert(context_map.size() == AC_NUM_CONTEXTS);
    const std::size_t bw{fc.width / 8};
    const std::size_t bh{fc.height / 8};
    const std::size_t xg{ceil_div(bw, AC_GROUP_BLOCKS)};
    const std::size_t yg{ceil_div(bh, AC_GROUP_BLOCKS)};
    const std::size_t num_groups{xg * yg};

    std::vector<std::uint32_t> histograms(num_clusters * AC_STRIDE, 0);
    std::vector<AcTok> tokens{};
    std::vector<std::pair<std::size_t, std::size_t>> ranges(num_groups);
    for (std::size_t group{0}; group < num_groups; ++group) {
        const std::size_t bx0{(group % xg) * AC_GROUP_BLOCKS};
        const std::size_t by0{(group / xg) * AC_GROUP_BLOCKS};
        const std::size_t gbw{std::min(AC_GROUP_BLOCKS, bw - bx0)};
        const std::size_t gbh{std::min(AC_GROUP_BLOCKS, bh - by0)};
        const std::size_t begin{tokens.size()};
        tokenize_ac_group_clustered(fc, bw, bx0, by0, gbw, gbh, histograms,
                                    tokens, context_map.data());
        ranges[group] = {begin, tokens.size()};
    }

    AcAnsReference reference{};
    reference.histogram = histograms;
    reference.tables.resize(num_clusters);
    for (std::size_t cluster{0}; cluster < num_clusters; ++cluster) {
        AnsDistribution distribution{};
        build_ans_distribution(histograms.data() + cluster * AC_STRIDE,
                               AC_STRIDE, distribution);
        build_ans_encoding_table(distribution, reference.tables[cluster]);
    }
    reference.group_streams.reserve(num_groups);
    for (const auto& range : ranges) {
        BitWriter group{};
        emit_ac_ans_tokens(group, tokens, range.first, range.second,
                           reference.tables);
        group.zero_pad_to_byte();
        reference.group_streams.push_back(group.bytes());
    }
    return reference;
}

std::vector<std::uint32_t> reference_ac_context_histogram(const FrameCoefficients& fc) {
    assert(fc.width % 8 == 0 && fc.height % 8 == 0);
    const std::size_t bw{fc.width / 8};
    const std::size_t bh{fc.height / 8};
    const std::size_t xg{ceil_div(bw, AC_GROUP_BLOCKS)};
    const std::size_t yg{ceil_div(bh, AC_GROUP_BLOCKS)};

    std::vector<std::uint32_t> cluster_hist(AC_NUM_CLUSTERS * AC_STRIDE, 0);
    std::vector<AcTok> tokens{};
    for (std::size_t gy{0}; gy < yg; ++gy) {
        for (std::size_t gx{0}; gx < xg; ++gx) {
            const std::size_t bx0{gx * AC_GROUP_BLOCKS};
            const std::size_t by0{gy * AC_GROUP_BLOCKS};
            const std::size_t gbw{std::min(AC_GROUP_BLOCKS, bw - bx0)};
            const std::size_t gbh{std::min(AC_GROUP_BLOCKS, bh - by0)};
            tokenize_ac_group_clustered(fc, bw, bx0, by0, gbw, gbh, cluster_hist, tokens);
        }
    }

    std::vector<std::uint32_t> histograms(AC_NUM_CONTEXTS * AC_STRIDE, 0);
    for (const AcTok& token : tokens) {
        ++histograms[static_cast<std::size_t>(token.context) * AC_STRIDE + token.symbol];
    }
    return histograms;
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

std::vector<std::uint8_t> write_vardct_codestream(const FrameCoefficients& fc, bool clustered_ac) {
    assert(fc.width % 8 == 0 && fc.height % 8 == 0);
    const std::size_t bw{fc.width / 8};
    const std::size_t bh{fc.height / 8};

    // NumTocEntries(num_groups, .., 1) == 1 only when there is a single AC
    // group; then the whole frame is one combined section.
    if (bw <= AC_GROUP_BLOCKS && bh <= AC_GROUP_BLOCKS) {
        return write_single_group_codestream(fc, bw, bh, clustered_ac);
    }
    return write_multi_group_codestream(fc, bw, bh, clustered_ac);
}

}  // namespace cujpegxl::bitstream
