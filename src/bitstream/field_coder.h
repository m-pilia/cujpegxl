// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_BITSTREAM_FIELD_CODER_H_
#define CUJPEGXL_SRC_BITSTREAM_FIELD_CODER_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "bit_writer.h"

namespace cujpegxl::bitstream {

// One of the four distributions selectable by a 2-bit U32 selector, matching
// libjxl's U32Distr (direct value or offset + extra bits).
struct Distr {
    bool direct{false};
    std::uint32_t value_or_offset{0};
    std::uint32_t extra_bits{0};
};

constexpr Distr Val(std::uint32_t v) { return Distr{true, v, 0}; }
constexpr Distr BitsOffset(std::uint32_t bits, std::uint32_t offset) {
    return Distr{false, offset, bits};
}
constexpr Distr Bits(std::uint32_t bits) { return BitsOffset(bits, 0); }

struct U32Enc {
    Distr d[4];
};

inline void write_bool(BitWriter& w, bool value) { w.write(1, value ? 1 : 0); }

inline void write_bits(BitWriter& w, std::size_t bits, std::uint32_t value) {
    w.write(bits, value);
}

inline void write_u32(BitWriter& w, const U32Enc& enc, std::uint32_t value) {
    std::uint32_t best_selector{0};
    std::size_t best_bits{64};
    for (std::uint32_t s{0}; s < 4; ++s) {
        const Distr& d{enc.d[s]};
        if (d.direct) {
            if (d.value_or_offset == value) {
                best_selector = s;
                best_bits = 2;
                break;
            }
            continue;
        }
        const std::uint32_t offset{d.value_or_offset};
        if (value < offset || value >= offset + (1ULL << d.extra_bits)) {
            continue;
        }
        if (2 + d.extra_bits < best_bits) {
            best_selector = s;
            best_bits = 2 + d.extra_bits;
        }
    }
    assert(best_bits != 64);
    w.write(2, best_selector);
    const Distr& chosen{enc.d[best_selector]};
    if (!chosen.direct) {
        w.write(chosen.extra_bits, value - chosen.value_or_offset);
    }
}

// libjxl U64Coder: 2 bits for 0, 6 bits for 1..16, 10 bits for 17..272, then
// 12-bit chunks each preceded by a continuation bit.
inline void write_u64(BitWriter& w, std::uint64_t value) {
    if (value == 0) {
        w.write(2, 0);
        return;
    }
    if (value <= 16) {
        w.write(2, 1);
        w.write(4, static_cast<std::uint32_t>(value - 1));
        return;
    }
    if (value <= 272) {
        w.write(2, 2);
        w.write(8, static_cast<std::uint32_t>(value - 17));
        return;
    }
    w.write(2, 3);
    value -= 273;
    w.write(12, static_cast<std::uint32_t>(value & 0xFFF));
    value >>= 12;
    for (int i{0}; i < 8; ++i) {
        if (value == 0) {
            w.write(1, 0);
            return;
        }
        w.write(1, 1);
        const std::size_t chunk{i < 7 ? 8u : 4u};
        w.write(chunk, static_cast<std::uint32_t>(value & ((1u << chunk) - 1)));
        value >>= chunk;
    }
}

inline std::uint16_t float_to_half(float value) {
    std::uint32_t bits{};
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign{(bits >> 16) & 0x8000u};
    const std::int32_t exp{static_cast<std::int32_t>((bits >> 23) & 0xFF) - 127 + 15};
    const std::uint32_t mant{bits & 0x7FFFFFu};
    if (value == 0.0f) {
        return static_cast<std::uint16_t>(sign);
    }
    assert(exp > 0 && exp < 31);
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) |
                                      (mant >> 13));
}

inline void write_f16(BitWriter& w, float value) { w.write(16, float_to_half(value)); }

inline void write_enum(BitWriter& w, std::uint32_t value) {
    write_u32(w, U32Enc{{Val(0), Val(1), BitsOffset(4, 2), BitsOffset(6, 18)}}, value);
}

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_SRC_BITSTREAM_FIELD_CODER_H_
