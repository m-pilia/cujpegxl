// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "container.h"

namespace cujpegxl::bitstream {

const std::array<std::uint8_t, 32> CONTAINER_HEADER = {
    0,   0,   0,   0x0c, 'J', 'X', 'L', ' ', 0x0d, 0x0a, 0x87, 0x0a, 0,   0,   0,   0x14,
    'f', 't', 'y', 'p',  'j', 'x', 'l', ' ', 0,    0,    0,    0,    'j', 'x', 'l', ' '};

namespace {

constexpr std::size_t SMALL_BOX_HEADER_SIZE{8};
constexpr std::size_t LARGE_BOX_HEADER_SIZE{16};

// A 32-bit box size field cannot represent a box of >= 2^32 bytes, so once the
// content plus the small header would reach that, the extended 64-bit size is
// required. Mirrors libjxl's kLargeBoxContentSizeThreshold.
constexpr std::size_t LARGE_BOX_CONTENT_THRESHOLD{0x100000000ull - SMALL_BOX_HEADER_SIZE};

constexpr std::array<std::uint8_t, 4> JXLC_TYPE{'j', 'x', 'l', 'c'};

}  // namespace

std::size_t jxlc_box_header_size(std::size_t codestream_size) {
    return codestream_size >= LARGE_BOX_CONTENT_THRESHOLD ? LARGE_BOX_HEADER_SIZE
                                                          : SMALL_BOX_HEADER_SIZE;
}

std::size_t container_size(std::size_t codestream_size) {
    return CONTAINER_HEADER.size() + jxlc_box_header_size(codestream_size) + codestream_size;
}

std::vector<std::uint8_t> container_framing(std::size_t codestream_size) {
    const std::size_t header{jxlc_box_header_size(codestream_size)};
    std::vector<std::uint8_t> framing{};
    framing.reserve(CONTAINER_HEADER.size() + header);
    framing.insert(framing.end(), CONTAINER_HEADER.begin(), CONTAINER_HEADER.end());

    const bool large{header == LARGE_BOX_HEADER_SIZE};
    const std::uint64_t box_size{static_cast<std::uint64_t>(codestream_size) + header};
    const std::uint32_t size_field{large ? 1u : static_cast<std::uint32_t>(box_size)};
    for (std::size_t i{0}; i < 4; ++i) {
        framing.push_back(static_cast<std::uint8_t>(size_field >> (8 * (3 - i))));
    }
    framing.insert(framing.end(), JXLC_TYPE.begin(), JXLC_TYPE.end());
    if (large) {
        for (std::size_t i{0}; i < 8; ++i) {
            framing.push_back(static_cast<std::uint8_t>(box_size >> (8 * (7 - i))));
        }
    }
    return framing;
}

std::vector<std::uint8_t> write_container(const std::vector<std::uint8_t>& codestream) {
    std::vector<std::uint8_t> out{container_framing(codestream.size())};
    out.insert(out.end(), codestream.begin(), codestream.end());
    return out;
}

}  // namespace cujpegxl::bitstream
