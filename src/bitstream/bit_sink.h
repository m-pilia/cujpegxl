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

// Bit sink that only counts, discarding the bytes. Used to size the candidate
// context-map encodings before the winner is emitted, so no scratch buffer is
// needed for the comparison. bits_written() equals the total number of bits
// passed to write(), matching BitWriter.
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
