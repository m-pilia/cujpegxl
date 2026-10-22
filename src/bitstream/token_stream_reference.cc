// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "token_stream_reference.h"

#include "ans.h"
#include "bit_writer.h"

namespace cujpegxl::bitstream {

std::vector<std::uint8_t> code_token_stream_reference(const TokenCode& code,
                                                      const TokenStream& stream) {
    const auto raw_value = [&](std::uint64_t pos, std::uint32_t n) {
        std::uint32_t v{0};
        for (std::uint32_t b{0}; b < n; ++b) {
            v |= static_cast<std::uint32_t>((stream.raw[(pos + b) / 64] >> ((pos + b) % 64)) & 1)
                 << b;
        }
        return v;
    };

    BitWriter w{};
    std::uint64_t raw_pos{0};
    if (code.use_prefix_code) {
        for (std::size_t i{0}; i < stream.num_tokens; ++i) {
            const PrefixEntryCode& c{code.prefix[stream.entries[i]]};
            w.write(c.depth, c.bits);
            w.write(c.raw_bits, raw_value(raw_pos, c.raw_bits));
            raw_pos += c.raw_bits;
        }
        w.zero_pad_to_byte();
        return w.bytes();
    }

    std::vector<std::uint32_t> chunks(stream.num_tokens, 0);
    std::vector<bool> renormalized(stream.num_tokens, false);
    std::uint32_t state{ANS_INITIAL_STATE};
    for (std::size_t i{stream.num_tokens}; i > 0; --i) {
        const AnsEntryCode& c{code.ans[stream.entries[i - 1]]};
        if ((state >> (32 - ANS_LOG_TAB_SIZE)) >= c.freq) {
            chunks[i - 1] = state & 0xffff;
            renormalized[i - 1] = true;
            state >>= 16;
        }
        state = ((state / c.freq) << ANS_LOG_TAB_SIZE) + code.reverse_map[c.slot + state % c.freq];
    }
    w.write(32, state);
    for (std::size_t i{0}; i < stream.num_tokens; ++i) {
        const AnsEntryCode& c{code.ans[stream.entries[i]]};
        if (renormalized[i]) {
            w.write(16, chunks[i]);
        }
        w.write(c.raw_bits, raw_value(raw_pos, c.raw_bits));
        raw_pos += c.raw_bits;
    }
    w.zero_pad_to_byte();
    return w.bytes();
}

}  // namespace cujpegxl::bitstream
