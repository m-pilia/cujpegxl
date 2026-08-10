// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_TOOLS_BITSTREAM_HYBRID_UINT_H_
#define CUJPEGXL_TOOLS_BITSTREAM_HYBRID_UINT_H_

#include <cassert>
#include <cstdint>

namespace cujpegxl::bitstream {

// libjxl hybrid-uint split as used by the entropy coders: a value is split into
// a `token` (prefix-coded symbol) plus `nbits` raw trailing bits. Mirrors
// jxl::HybridUintConfig::Encode. The JXL default configuration is (4, 2, 0).
struct HybridUintConfig {
    std::uint32_t split_exponent{4};
    std::uint32_t msb_in_token{2};
    std::uint32_t lsb_in_token{0};

    std::uint32_t split_token() const { return 1u << split_exponent; }

    void encode(std::uint32_t value, std::uint32_t& token, std::uint32_t& nbits,
                std::uint32_t& bits) const {
        const std::uint32_t split{split_token()};
        if (value < split) {
            token = value;
            nbits = 0;
            bits = 0;
            return;
        }
        std::uint32_t n{31u};
        while ((value & (1u << n)) == 0) {
            --n;
        }
        const std::uint32_t m{value - (1u << n)};
        token = split + ((n - split_exponent) << (msb_in_token + lsb_in_token)) +
                ((m >> (n - msb_in_token)) << lsb_in_token) +
                (m & ((1u << lsb_in_token) - 1));
        nbits = n - msb_in_token - lsb_in_token;
        bits = (value >> lsb_in_token) & ((1u << nbits) - 1);
    }
};

// libjxl PackSigned: maps signed values to unsigned so small magnitudes get
// small tokens (0,-1,1,-2,2,... -> 0,1,2,3,4,...).
inline std::uint32_t pack_signed(std::int32_t value) {
    return (static_cast<std::uint32_t>(value) << 1) ^
           static_cast<std::uint32_t>(value >> 31);
}

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_TOOLS_BITSTREAM_HYBRID_UINT_H_
