// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "codestream.h"

#include "field_coder.h"

namespace cujpegxl::bitstream {
namespace {

// SizeHeader dimension U32 encoding shared by xsize and ysize (libjxl
// headers.cc): BitsOffset(9,1) / (13,1) / (18,1) / (30,1).
constexpr U32Enc DIM_ENC{{BitsOffset(9, 1), BitsOffset(13, 1), BitsOffset(18, 1),
                          BitsOffset(30, 1)}};

// TOC size distribution (libjxl toc.h TOC_DIST).
constexpr U32Enc TOC_DIST{
    {Bits(10), BitsOffset(14, 1024), BitsOffset(22, 17408), BitsOffset(30, 4211712)}};

void write_size_header(BitWriter& w, std::uint32_t xsize, std::uint32_t ysize) {
    const bool small{xsize % 8 == 0 && ysize % 8 == 0 && xsize <= 256 && ysize <= 256};
    write_bool(w, small);
    if (small) {
        write_bits(w, 5, ysize / 8 - 1);
    } else {
        write_u32(w, DIM_ENC, ysize);
    }
    // ratio_ = 0: always send xsize explicitly (valid for any dimensions).
    write_bits(w, 3, 0);
    if (small) {
        write_bits(w, 5, xsize / 8 - 1);
    } else {
        write_u32(w, DIM_ENC, xsize);
    }
}

}  // namespace

void write_codestream_headers(BitWriter& w, std::uint32_t xsize, std::uint32_t ysize) {
    w.write(8, 0xFF);
    w.write(8, 0x0A);
    write_size_header(w, xsize, ysize);
    // ImageMetadata all_default: 8-bit int samples, xyb_encoded, enumerated
    // sRGB, no extra channels, default tone mapping -> the whole bundle is one
    // "all default" bit.
    write_bool(w, true);
    // CustomTransformData (transform_data) always follows ImageMetadata in the
    // codestream; the default opsin inverse matrix and upsampling weights are
    // one "all default" bit.
    write_bool(w, true);
}

void write_frame_header(BitWriter& w) {
    write_bool(w, false);                                          // all_default = false
    write_u32(w, U32Enc{{Val(0), Val(1), Val(2), Val(3)}}, 0);     // frame_type = kRegularFrame
    write_bool(w, false);                                          // is_modular = false -> kVarDCT
    write_u64(w, 0);                                               // flags
    // xyb_encoded forces color_transform = kXYB (nothing written).
    write_u32(w, U32Enc{{Val(1), Val(2), Val(4), Val(8)}}, 1);     // upsampling = 1
    // encoding == kVarDCT && color_transform == kXYB:
    write_bits(w, 3, 2);                                           // x_qm_scale = 2 (neutral)
    write_bits(w, 3, 2);                                           // b_qm_scale = 2 (neutral)
    // passes (frame_type != kReferenceOnly):
    write_u32(w, U32Enc{{Val(1), Val(2), Val(3), BitsOffset(3, 4)}}, 1);  // num_passes = 1
    // frame_type != kDCFrame:
    write_bool(w, false);                                         // custom_size_or_origin = false
    // frame_type == kRegularFrame -> blending_info:
    write_u32(w, U32Enc{{Val(0), Val(1), Val(2), BitsOffset(2, 3)}}, 0);  // mode = kReplace
    // no extra channels, mode == kReplace, not partial -> no alpha/clamp/source.
    // no animation.
    write_bool(w, true);                                          // is_last = true
    // is_last -> no save_as_reference; CanBeReferenced() == false -> no
    // save_before_color_transform.
    write_u32(w, U32Enc{{Val(0), Bits(4), BitsOffset(5, 16), BitsOffset(10, 48)}}, 0);  // name len 0
    write_bool(w, true);                                          // loop_filter all_default = true
    write_u64(w, 0);                                              // extensions = 0
}

void write_toc_single_section(BitWriter& w, std::size_t section_size_bytes) {
    write_bool(w, false);  // no permutation
    w.zero_pad_to_byte();
    write_u32(w, TOC_DIST, static_cast<std::uint32_t>(section_size_bytes));
    w.zero_pad_to_byte();
}

}  // namespace cujpegxl::bitstream
