// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "context_map.h"

#include <array>
#include <cassert>
#include <functional>

#include "histogram_writer.h"
#include "hybrid_uint.h"

namespace cujpegxl::bitstream {
namespace {

constexpr std::size_t CONTEXT_MAP_ALPHABET_SIZE = 16;
constexpr std::size_t MAX_CONTEXT_MAP_SIZE = 7425;

std::size_t ceil_log2(std::size_t value) {
    if (value <= 1) {
        return 0;
    }
    std::size_t bits{0};
    --value;
    while (value != 0) {
        value >>= 1;
        ++bits;
    }
    return bits;
}

void append_writer_bits(BitWriter& destination, const BitWriter& source) {
    std::size_t remaining{source.bits_written()};
    std::size_t byte{0};
    while (remaining != 0) {
        const std::size_t take{remaining < 8 ? remaining : 8};
        destination.write(take, source.bytes()[byte]);
        remaining -= take;
        ++byte;
    }
}

void write_simple_context_map(BitWriter& w, const std::uint8_t* context_map,
                              std::size_t num_contexts, std::size_t num_clusters) {
    assert(num_clusters >= 1 && num_clusters <= 8);
    const std::size_t bits_per_entry{ceil_log2(num_clusters)};
    w.write(1, 1);
    w.write(2, bits_per_entry);
    for (std::size_t i{0}; i < num_contexts; ++i) {
        assert(context_map[i] < num_clusters);
        if (bits_per_entry != 0) {
            w.write(bits_per_entry, context_map[i]);
        }
    }
}

template <typename Emit>
void for_each_context_map_symbol(const std::uint8_t* context_map, std::size_t num_contexts,
                                 bool use_mtf, Emit emit) {
    if (!use_mtf) {
        for (std::size_t i{0}; i < num_contexts; ++i) {
            emit(context_map[i]);
        }
        return;
    }

    std::array<std::uint8_t, 256> symbols{};
    for (std::size_t i{0}; i < symbols.size(); ++i) {
        symbols[i] = static_cast<std::uint8_t>(i);
    }
    for (std::size_t i{0}; i < num_contexts; ++i) {
        std::size_t rank{0};
        while (symbols[rank] != context_map[i]) {
            ++rank;
        }
        emit(static_cast<std::uint8_t>(rank));

        const std::uint8_t value{symbols[rank]};
        while (rank > 0) {
            symbols[rank] = symbols[rank - 1];
            --rank;
        }
        symbols[0] = value;
    }
}

}  // namespace

void move_to_front_transform(const std::uint8_t* input, std::size_t size, std::uint8_t* output) {
    assert(size == 0 || (input != nullptr && output != nullptr));
    std::size_t position{0};
    for_each_context_map_symbol(input, size, true,
                                [&](std::uint8_t value) { output[position++] = value; });
}

void write_complex_prefix_context_map(BitWriter& w, const std::uint8_t* context_map,
                                      std::size_t num_contexts, bool use_mtf) {
    assert(context_map != nullptr);
    assert(num_contexts > 0);

    const HybridUintConfig config{2, 0, 1};
    std::array<std::uint32_t, CONTEXT_MAP_ALPHABET_SIZE> histogram{};
    for_each_context_map_symbol(context_map, num_contexts, use_mtf, [&](std::uint8_t value) {
        std::uint32_t token{}, nbits{}, bits{};
        config.encode(value, token, nbits, bits);
        assert(token < histogram.size());
        ++histogram[token];
    });

    std::size_t alphabet_size{1};
    for (std::size_t i{0}; i < histogram.size(); ++i) {
        if (histogram[i] != 0) {
            alphabet_size = i + 1;
        }
    }

    w.write(1, 0);
    w.write(1, use_mtf ? 1 : 0);

    std::array<std::uint8_t, CONTEXT_MAP_ALPHABET_SIZE> depth{};
    std::array<std::uint16_t, CONTEXT_MAP_ALPHABET_SIZE> bits{};
    write_prefix_histograms(w, histogram.data(), alphabet_size, 1, config, depth.data(),
                            bits.data());

    for_each_context_map_symbol(context_map, num_contexts, use_mtf, [&](std::uint8_t value) {
        std::uint32_t token{}, nbits{}, extra_bits{};
        config.encode(value, token, nbits, extra_bits);
        w.write(depth[token], bits[token]);
        if (nbits != 0) {
            w.write(nbits, extra_bits);
        }
    });
}

ContextMapEncoding write_best_prefix_context_map(BitWriter& w, const std::uint8_t* context_map,
                                                 std::size_t num_contexts,
                                                 std::size_t num_clusters) {
    assert(context_map != nullptr);
    assert(num_contexts > 0 && num_contexts <= MAX_CONTEXT_MAP_SIZE);
    assert(num_clusters >= 1 && num_clusters <= 256);
    for (std::size_t i{0}; i < num_contexts; ++i) {
        assert(context_map[i] < num_clusters);
    }

    BitWriter complex{};
    write_complex_prefix_context_map(complex, context_map, num_contexts, false);
    BitWriter complex_mtf{};
    write_complex_prefix_context_map(complex_mtf, context_map, num_contexts, true);

    std::reference_wrapper<const BitWriter> selected{complex};
    ContextMapEncoding encoding{ContextMapEncoding::COMPLEX};
    if (complex_mtf.bits_written() < selected.get().bits_written()) {
        selected = std::cref(complex_mtf);
        encoding = ContextMapEncoding::COMPLEX_MTF;
    }

    BitWriter simple{};
    if (num_clusters <= 8) {
        write_simple_context_map(simple, context_map, num_contexts, num_clusters);
        if (simple.bits_written() <= selected.get().bits_written()) {
            selected = std::cref(simple);
            encoding = ContextMapEncoding::SIMPLE;
        }
    }

    append_writer_bits(w, selected.get());
    return encoding;
}

}  // namespace cujpegxl::bitstream
