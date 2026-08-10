// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "vardct_frame.h"

#include <cassert>
#include <utility>

#include "bit_writer.h"
#include "codestream.h"
#include "entropy_encoder.h"
#include "field_coder.h"
#include "hybrid_uint.h"
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

constexpr U32Enc GLOBAL_SCALE_ENC{
    {BitsOffset(11, 1), BitsOffset(11, 2049), BitsOffset(12, 4097), BitsOffset(16, 8193)}};
constexpr U32Enc QUANT_DC_ENC{{Val(16), BitsOffset(5, 1), BitsOffset(8, 1), BitsOffset(16, 1)}};
constexpr U32Enc ORDER_ENC{{Val(0x5F), Val(0x13), Val(0), Bits(13)}};

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

std::vector<ModularChannel> make_vardct_dc_channels(const FrameCoefficients& fc,
                                                    std::size_t bw, std::size_t bh) {
    // Physical modular channels are Y, X, B (DequantDC reads channel[1]=X,
    // channel[0]=Y, channel[2]=B), while fc.dc is X, Y, B.
    const int xyb_for_physical[3]{1, 0, 2};
    std::vector<ModularChannel> channels{};
    for (int p{0}; p < 3; ++p) {
        channels.push_back({bw, bh, fc.dc[xyb_for_physical[p]]});
    }
    return channels;
}

std::vector<ModularChannel> make_ac_metadata_channels(const FrameCoefficients& fc,
                                                      std::size_t bw, std::size_t bh) {
    const std::size_t cw{(bw + 7) / 8};
    const std::size_t ch{(bh + 7) / 8};
    const std::size_t count{bw * bh};

    ModularChannel ytox{cw, ch, std::vector<std::int32_t>(cw * ch, 0)};
    ModularChannel ytob{cw, ch, std::vector<std::int32_t>(cw * ch, 0)};

    // ACS + QF: width `count`, height 2. Row 0 = raw AC strategy (0 = DCT8),
    // row 1 = quant field encoded as raw_quant_field - 1.
    ModularChannel acs_qf{count, 2, std::vector<std::int32_t>(count * 2, 0)};
    for (std::size_t i{0}; i < count; ++i) {
        acs_qf.pixels[count + i] = static_cast<std::int32_t>(fc.raw_quant_field) - 1;
    }

    ModularChannel epf{bw, bh, std::vector<std::int32_t>(bw * bh, 0)};
    return {std::move(ytox), std::move(ytob), std::move(acs_qf), std::move(epf)};
}

void tokenize_ac(const FrameCoefficients& fc, std::size_t bw, std::size_t bh,
                 EntropyEncoder& ac) {
    const std::array<std::uint32_t, 64>& order{dct8_natural_order()};
    const int channel_order[3]{1, 0, 2};  // Y, X, B (decoder LoadBlock order)
    for (std::size_t by{0}; by < bh; ++by) {
        for (std::size_t bx{0}; bx < bw; ++bx) {
            for (int c : channel_order) {
                const std::int32_t* blk{&fc.ac[c][(by * bw + bx) * 64]};
                std::uint32_t nzeros{0};
                for (std::size_t k{1}; k < 64; ++k) {
                    if (blk[order[k]] != 0) {
                        ++nzeros;
                    }
                }
                ac.add_token(0, nzeros);
                std::uint32_t remaining{nzeros};
                for (std::size_t k{1}; k < 64 && remaining > 0; ++k) {
                    const std::int32_t v{blk[order[k]]};
                    ac.add_token(0, pack_signed(v));
                    if (v != 0) {
                        --remaining;
                    }
                }
            }
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

void write_dc_group(BitWriter& w, const FrameCoefficients& fc, std::size_t bw,
                    std::size_t bh) {
    write_bits(w, 2, 0);  // DecodeVarDCTDC extra_precision = 0
    write_modular_image(w, make_vardct_dc_channels(fc, bw, bh));
    // DecodeGroup(ModularDC): VarDCT full image has no channels -> nothing.
    write_bits(w, static_cast<std::size_t>(ceil_log2_nonzero(bw * bh)),
               static_cast<std::uint32_t>(bw * bh - 1));  // AcMetadata count - 1
    write_modular_image(w, make_ac_metadata_channels(fc, bw, bh));
}

}  // namespace

const std::array<std::uint32_t, 64>& dct8_natural_order() {
    static const std::array<std::uint32_t, 64> ORDER{compute_dct8_natural_order()};
    return ORDER;
}

std::vector<std::uint8_t> write_vardct_codestream(const FrameCoefficients& fc) {
    assert(fc.width % 8 == 0 && fc.height % 8 == 0);
    const std::size_t bw{fc.width / 8};
    const std::size_t bh{fc.height / 8};

    EntropyEncoder ac{NUM_AC_CONTEXTS};
    tokenize_ac(fc, bw, bh, ac);

    BitWriter body{};
    write_dc_global(body, fc);
    write_dc_group(body, fc, bw, bh);
    // AcGlobal.
    write_bool(body, true);        // DequantMatrices::Decode all_default
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

}  // namespace cujpegxl::bitstream
