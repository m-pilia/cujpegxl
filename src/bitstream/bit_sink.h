// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_BITSTREAM_BIT_SINK_H_
#define CUJPEGXL_SRC_BITSTREAM_BIT_SINK_H_

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace cujpegxl::bitstream {

#if defined(__CUDACC__)
#define CUJPEGXL_BS_HD __host__ __device__
#else
#define CUJPEGXL_BS_HD
#endif

// LSB-first bit sink over a caller-owned fixed-capacity buffer, byte/bit ordering
// identical to BitWriter (host vector writer). Usable from host and device so the
// prefix-code serialization runs unchanged in the on-device build kernels.
class SpanBitWriter {
public:
    CUJPEGXL_BS_HD SpanBitWriter(std::uint8_t* buffer, std::size_t capacity)
        : buf_{buffer}, cap_{capacity} {}

    CUJPEGXL_BS_HD void write(std::size_t n_bits, std::uint64_t bits) {
        assert(n_bits <= 56);
        assert(n_bits == 64 || (bits >> n_bits) == 0);
        while (n_bits > 0) {
            if (bit_pos_ == 0) {
                assert(byte_len_ < cap_);
                buf_[byte_len_] = 0;
                ++byte_len_;
            }
            const std::size_t take{n_bits < (8 - bit_pos_) ? n_bits : (8 - bit_pos_)};
            const std::uint8_t mask{static_cast<std::uint8_t>((1u << take) - 1)};
            buf_[byte_len_ - 1] |= static_cast<std::uint8_t>((bits & mask) << bit_pos_);
            bits >>= take;
            n_bits -= take;
            bit_pos_ = (bit_pos_ + take) & 7;
        }
    }

    CUJPEGXL_BS_HD void zero_pad_to_byte() { bit_pos_ = 0; }

    CUJPEGXL_BS_HD std::size_t bits_written() const {
        return byte_len_ == 0 ? 0 : (byte_len_ - 1) * 8 + (bit_pos_ == 0 ? 8 : bit_pos_);
    }

    CUJPEGXL_BS_HD bool byte_aligned() const { return bit_pos_ == 0; }
    CUJPEGXL_BS_HD std::size_t size_bytes() const { return byte_len_; }
    CUJPEGXL_BS_HD std::uint8_t byte_at(std::size_t i) const { return buf_[i]; }

private:
    std::uint8_t* buf_{nullptr};
    [[maybe_unused]] std::size_t cap_{0};
    std::size_t byte_len_{0};
    std::size_t bit_pos_{0};
};

// Bit sink that only counts, discarding the bytes. Used to size the candidate
// context-map encodings before the winner is emitted, so no scratch buffer is
// needed for the comparison. bits_written() equals the total number of bits
// passed to write(), matching BitWriter/SpanBitWriter.
class CountingBitWriter {
public:
    CUJPEGXL_BS_HD void write(std::size_t n_bits, std::uint64_t /*bits*/) { total_ += n_bits; }
    CUJPEGXL_BS_HD void zero_pad_to_byte() { total_ = (total_ + 7) & ~std::size_t{7}; }
    CUJPEGXL_BS_HD std::size_t bits_written() const { return total_; }
    CUJPEGXL_BS_HD bool byte_aligned() const { return (total_ & 7) == 0; }

private:
    std::size_t total_{0};
};

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_SRC_BITSTREAM_BIT_SINK_H_
