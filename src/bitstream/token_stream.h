// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_BITSTREAM_TOKEN_STREAM_H_
#define CUJPEGXL_SRC_BITSTREAM_TOKEN_STREAM_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cujpegxl::bitstream {

// A token is an entry `cluster << TOKEN_CLUSTER_SHIFT | symbol` plus the raw
// bits of its hybrid-uint value, whose count follows from the symbol.
constexpr std::uint32_t TOKEN_CLUSTER_SHIFT{8};

// rANS code of one entry: its symbol's frequency, the 64-bit reciprocal of
// libjxl's PutSymbol and the index of its first slot in TokenCode::reverse_map.
struct AnsEntryCode {
    std::uint64_t reciprocal{0};
    std::uint32_t slot{0};
    std::uint16_t freq{0};
    std::uint16_t raw_bits{0};
};

struct PrefixEntryCode {
    std::uint16_t bits{0};
    std::uint8_t depth{0};
    std::uint8_t raw_bits{0};
};

// The entropy code of a set of token streams sharing one histogram set, in the
// mode signalled for it: `ans` and `reverse_map` (the clusters' slot maps of
// ANS_TAB_SIZE entries each) for rANS, `prefix` for prefix codes. The entry
// tables are indexed by token entry.
struct TokenCode {
    bool use_prefix_code{false};
    std::vector<AnsEntryCode> ans;
    std::vector<std::uint16_t> reverse_map;
    std::vector<PrefixEntryCode> prefix;
};

// The tokens of one stream: `num_tokens` entries and their raw bits, packed
// LSB-first in `raw` (`raw_bits` bits; the word after the last one holding raw
// bits must be readable).
struct TokenStream {
    const std::uint16_t* entries{nullptr};
    std::size_t num_tokens{0};
    const std::uint64_t* raw{nullptr};
    std::uint64_t raw_bits{0};
};

// A coded stream, LSB-first: its first `head_bits` bits are the low bits of
// `head`, the `num_words` 64-bit `words` follow. Coding runs backwards from the
// end of the stream's word slot, so the stream starts at an arbitrary bit.
struct CodedStream {
    const std::uint64_t* words{nullptr};
    std::size_t num_words{0};
    std::uint64_t head{0};
    std::uint32_t head_bits{0};

    std::size_t size_bytes() const { return (head_bits + 64 * num_words + 7) / 8; }
};

// Words a slot needs to code `stream` in either mode.
std::size_t coded_stream_capacity_words(const TokenStream& stream);

// Shared work list of a set of streams. Callers claim streams in the sequence
// `order` (stream indices), its k-th entry once `ready` exceeds k, so that the
// streams can be released as they arrive in memory and in any order (written
// before the `ready` store that releases them); claiming stops once `cancelled`
// is set. Stream i is coded backwards into the word slot ending at
// `slot_words + slot_end[i]`, which holds coded_stream_capacity_words of it.
struct StreamQueue {
    const TokenStream* streams{nullptr};
    std::size_t num_streams{0};
    const std::size_t* order{nullptr};
    std::uint64_t* slot_words{nullptr};
    const std::size_t* slot_end{nullptr};
    CodedStream* coded{nullptr};
    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> ready{0};
    std::atomic<bool> cancelled{false};
};

// Codes streams claimed from `queue` until none are left, several at a time
// interleaved so that the serial rANS state chains overlap. Each stream is
// coded independently, so the output does not depend on how many callers share
// the queue.
void code_token_streams(const TokenCode& code, StreamQueue& queue);

// Writes the coded stream's size_bytes() bytes to `dst`, zero-padding its last
// byte.
void place_coded_stream(const CodedStream& stream, std::uint8_t* dst);

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_SRC_BITSTREAM_TOKEN_STREAM_H_
