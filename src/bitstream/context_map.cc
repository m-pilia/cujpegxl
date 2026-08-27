// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Host binding of the shared, host+device context-map serialization
// (serialize_impl.h) to the vector BitWriter.

#include "context_map.h"

#include "serialize_impl.h"

namespace cujpegxl::bitstream {

void move_to_front_transform(const std::uint8_t* input, std::size_t size, std::uint8_t* output) {
    assert(size == 0 || (input != nullptr && output != nullptr));
    std::uint8_t symbols[256];
    detail::mtf_reset(symbols);
    for (std::size_t i{0}; i < size; ++i) {
        output[i] = detail::mtf_apply(symbols, input[i]);
    }
}

void write_complex_prefix_context_map(BitWriter& w, const std::uint8_t* context_map,
                                      std::size_t num_contexts, bool use_mtf) {
    detail::write_complex_prefix_context_map(w, context_map, num_contexts, use_mtf);
}

ContextMapEncoding write_best_prefix_context_map(BitWriter& w, const std::uint8_t* context_map,
                                                 std::size_t num_contexts,
                                                 std::size_t num_clusters) {
    return detail::write_best_prefix_context_map(w, context_map, num_contexts, num_clusters);
}

}  // namespace cujpegxl::bitstream
