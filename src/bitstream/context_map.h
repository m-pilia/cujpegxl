// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_BITSTREAM_CONTEXT_MAP_H_
#define CUJPEGXL_SRC_BITSTREAM_CONTEXT_MAP_H_

#include <cstddef>
#include <cstdint>

namespace cujpegxl::bitstream {

void move_to_front_transform(const std::uint8_t* input, std::size_t size, std::uint8_t* output);

}  // namespace cujpegxl::bitstream

#endif  // CUJPEGXL_SRC_BITSTREAM_CONTEXT_MAP_H_
