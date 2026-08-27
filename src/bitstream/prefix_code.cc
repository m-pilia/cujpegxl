// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Host binding of the shared, host+device Huffman-tree construction and storage
// (serialize_impl.h) to the vector BitWriter. The algorithm itself lives in the
// header so the on-device build kernels run the identical, bit-exact code.

#include "prefix_code.h"

#include "serialize_impl.h"

namespace cujpegxl::bitstream {

void build_and_store_huffman_tree(const std::uint32_t* histogram, std::size_t length,
                                  std::uint8_t* depth, std::uint16_t* bits, BitWriter& w) {
    detail::build_and_store_huffman_tree(histogram, length, depth, bits, w);
}

}  // namespace cujpegxl::bitstream
