// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_BITSTREAM_BIT_WRITER_H_
#define CUJPEGXL_SRC_BITSTREAM_BIT_WRITER_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cujpegxl::bitstream {

// LSB-first bit writer matching libjxl's BitWriter byte/bit ordering: bits are
// written into increasing byte addresses, and within a byte least-significant
// bit first.
class BitWriter {
  public:
    void write(std::size_t n_bits, std::uint64_t bits) {
        assert(n_bits <= 56);
        assert(n_bits == 64 || (bits >> n_bits) == 0);
        while (n_bits > 0) {
            if (bit_pos_ == 0) {
                bytes_.push_back(0);
            }
            const std::size_t take{n_bits < (8 - bit_pos_) ? n_bits : (8 - bit_pos_)};
            const std::uint8_t mask{static_cast<std::uint8_t>((1u << take) - 1)};
            bytes_.back() |= static_cast<std::uint8_t>((bits & mask) << bit_pos_);
            bits >>= take;
            n_bits -= take;
            bit_pos_ = (bit_pos_ + take) & 7;
        }
    }

    void zero_pad_to_byte() {
        if (bit_pos_ != 0) {
            bit_pos_ = 0;
        }
    }

    std::size_t bits_written() const { return bytes_.empty() ? 0 : (bytes_.size() - 1) * 8 + (bit_pos_ == 0 ? 8 : bit_pos_); }

    bool byte_aligned() const { return bit_pos_ == 0; }

    // Appends another byte-aligned writer's bytes. Both must be byte aligned.
    void append_aligned(const BitWriter& other) {
        assert(byte_aligned());
        assert(other.byte_aligned());
        bytes_.insert(bytes_.end(), other.bytes_.begin(), other.bytes_.end());
    }

    const std::vector<std::uint8_t>& bytes() const { return bytes_; }
    std::size_t size_bytes() const { return bytes_.size(); }

  private:
    std::vector<std::uint8_t> bytes_{};
    std::size_t bit_pos_{0};
};

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_SRC_BITSTREAM_BIT_WRITER_H_
