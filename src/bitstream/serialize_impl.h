// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_BITSTREAM_SERIALIZE_IMPL_H_
#define CUJPEGXL_SRC_BITSTREAM_SERIALIZE_IMPL_H_

// Writer-templated, host+device implementation of the JXL prefix-code entropy
// container serialization: length-limited Huffman construction, the histogram
// description, the context map, and the VarDCT DC / AcMetadata modular headers.
// It is the single source of truth shared by the host bitstream library (via the
// thin wrappers in prefix_code.cc / histogram_writer.cc / context_map.cc /
// frame_assembly.cc, instantiated with the vector BitWriter) and the on-device
// build kernels (instantiated with SpanBitWriter). All arithmetic is integer, so
// the emitted bytes are bit-identical across the two instantiations.
//
// Every dynamic container of the host originals is replaced by a bounded stack
// buffer (alphabets are <= HISTOGRAM_STRIDE and context maps <= AC_NUM_CONTEXTS),
// std::stable_sort by a stable insertion sort, and the recursive depth walk by an
// explicit stack, so the code compiles and runs unchanged on the device.

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "bit_sink.h"
#include "context_map.h"
#include "field_coder.h"
#include "frame_assembly.h"
#include "hybrid_uint.h"
#include "src/ac_context.h"
#include "src/dc_predict.h"

namespace cujpegxl::bitstream::detail {

constexpr std::size_t MAX_ALPHABET = HISTOGRAM_STRIDE;
constexpr int CODE_LENGTH_CODES = 18;
constexpr std::size_t NUM_TREE_CONTEXTS = 6;
constexpr std::size_t CONTEXT_MAP_ALPHABET_SIZE = 16;

CUJPEGXL_BS_HD inline std::uint32_t max_u32(std::uint32_t a, std::uint32_t b) {
    return a < b ? b : a;
}

CUJPEGXL_BS_HD inline std::size_t ceil_log2_nonzero(std::size_t x) {
    if (x <= 1) {
        return 0;
    }
    std::size_t bits{0};
    std::size_t v{x - 1};
    while (v) {
        v >>= 1;
        ++bits;
    }
    return bits;
}

// Index of the last non-zero symbol plus one, or 1 if empty; the trailing-trimmed
// alphabet size the host EntropyEncoder emits.
CUJPEGXL_BS_HD inline std::size_t alphabet_size(const std::uint32_t* histogram,
                                                std::size_t stride) {
    std::size_t last{0};
    for (std::size_t s{0}; s < stride; ++s) {
        if (histogram[s]) {
            last = s + 1;
        }
    }
    return last == 0 ? 1 : last;
}

// --- Huffman tree construction (libjxl enc_huffman_tree.cc) ------------------

struct HuffmanTree {
    std::uint32_t total_count;
    std::int16_t index_left;
    std::int16_t index_right_or_value;
};

// Assign leaf depths by an explicit-stack DFS (iterative form of the recursive
// libjxl SetDepth), so the walk stays within the device stack for the deep,
// pre-limit trees the doubling loop discards.
CUJPEGXL_BS_HD inline void set_depth(int root, const HuffmanTree* pool, std::uint8_t* depth) {
    struct Item {
        int node;
        std::uint8_t level;
    };
    Item stack[2 * MAX_ALPHABET + 4];
    std::size_t sp{0};
    stack[sp++] = {root, 0};
    while (sp != 0) {
        const Item it{stack[--sp]};
        const HuffmanTree& node{pool[it.node]};
        if (node.index_left >= 0) {
            stack[sp++] = {node.index_left, static_cast<std::uint8_t>(it.level + 1)};
            stack[sp++] = {node.index_right_or_value, static_cast<std::uint8_t>(it.level + 1)};
        } else {
            depth[node.index_right_or_value] = it.level;
        }
    }
}

CUJPEGXL_BS_HD inline void create_huffman_tree(const std::uint32_t* data, std::size_t length,
                                               int tree_limit, std::uint8_t* depth) {
    for (std::uint32_t count_limit{1};; count_limit *= 2) {
        HuffmanTree tree[2 * MAX_ALPHABET + 4];
        std::size_t size{0};

        for (std::size_t i{length}; i != 0;) {
            --i;
            if (data[i]) {
                const std::uint32_t count{max_u32(data[i], count_limit - 1)};
                tree[size++] = {count, -1, static_cast<std::int16_t>(i)};
            }
        }

        const std::size_t n{size};
        if (n == 1) {
            depth[tree[0].index_right_or_value] = 1;
            break;
        }

        // Stable ascending sort by count (matches std::stable_sort: equal counts
        // keep their input order, so identical bytes come out).
        for (std::size_t i{1}; i < n; ++i) {
            const HuffmanTree key{tree[i]};
            std::size_t j{i};
            while (j > 0 && tree[j - 1].total_count > key.total_count) {
                tree[j] = tree[j - 1];
                --j;
            }
            tree[j] = key;
        }

        const HuffmanTree sentinel{0xFFFFFFFFu, -1, -1};
        tree[size++] = sentinel;
        tree[size++] = sentinel;

        std::size_t i{0};
        std::size_t j{n + 1};
        for (std::size_t k{n - 1}; k != 0; --k) {
            std::size_t left{};
            std::size_t right{};
            if (tree[i].total_count <= tree[j].total_count) {
                left = i;
                ++i;
            } else {
                left = j;
                ++j;
            }
            if (tree[i].total_count <= tree[j].total_count) {
                right = i;
                ++i;
            } else {
                right = j;
                ++j;
            }

            const std::size_t j_end{size - 1};
            tree[j_end].total_count = tree[left].total_count + tree[right].total_count;
            tree[j_end].index_left = static_cast<std::int16_t>(left);
            tree[j_end].index_right_or_value = static_cast<std::int16_t>(right);
            tree[size++] = sentinel;
        }
        set_depth(static_cast<int>(2 * n - 1), tree, depth);

        std::uint8_t max_depth{0};
        for (std::size_t d{0}; d < length; ++d) {
            if (depth[d] > max_depth) {
                max_depth = depth[d];
            }
        }
        if (max_depth <= tree_limit) {
            break;
        }
    }
}

CUJPEGXL_BS_HD inline std::uint16_t reverse_bits(int num_bits, std::uint16_t bits) {
    const std::size_t LUT[16]{0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe,
                              0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf};
    std::size_t retval{LUT[bits & 0xf]};
    for (int i{4}; i < num_bits; i += 4) {
        retval <<= 4;
        bits = static_cast<std::uint16_t>(bits >> 4);
        retval |= LUT[bits & 0xf];
    }
    retval >>= (-num_bits & 0x3);
    return static_cast<std::uint16_t>(retval);
}

CUJPEGXL_BS_HD inline void convert_bit_depths_to_symbols(const std::uint8_t* depth, std::size_t len,
                                                         std::uint16_t* bits) {
    constexpr int max_bits{16};
    std::uint16_t bl_count[max_bits]{};
    for (std::size_t i{0}; i < len; ++i) {
        ++bl_count[depth[i]];
    }
    bl_count[0] = 0;
    std::uint16_t next_code[max_bits]{};
    int code{0};
    for (std::size_t i{1}; i < max_bits; ++i) {
        code = (code + bl_count[i - 1]) << 1;
        next_code[i] = static_cast<std::uint16_t>(code);
    }
    for (std::size_t i{0}; i < len; ++i) {
        if (depth[i]) {
            bits[i] = reverse_bits(depth[i], next_code[depth[i]]++);
        }
    }
}

CUJPEGXL_BS_HD inline void reverse(std::uint8_t* v, std::size_t start, std::size_t end) {
    --end;
    while (start < end) {
        const std::uint8_t t{v[start]};
        v[start] = v[end];
        v[end] = t;
        ++start;
        --end;
    }
}

CUJPEGXL_BS_HD inline void write_repetitions(std::uint8_t previous_value, std::uint8_t value,
                                             std::size_t repetitions, std::size_t& tree_size,
                                             std::uint8_t* tree, std::uint8_t* extra_bits) {
    if (previous_value != value) {
        tree[tree_size] = value;
        extra_bits[tree_size] = 0;
        ++tree_size;
        --repetitions;
    }
    if (repetitions == 7) {
        tree[tree_size] = value;
        extra_bits[tree_size] = 0;
        ++tree_size;
        --repetitions;
    }
    if (repetitions < 3) {
        for (std::size_t i{0}; i < repetitions; ++i) {
            tree[tree_size] = value;
            extra_bits[tree_size] = 0;
            ++tree_size;
        }
    } else {
        repetitions -= 3;
        const std::size_t start{tree_size};
        while (true) {
            tree[tree_size] = 16;
            extra_bits[tree_size] = repetitions & 0x3;
            ++tree_size;
            repetitions >>= 2;
            if (repetitions == 0) {
                break;
            }
            --repetitions;
        }
        reverse(tree, start, tree_size);
        reverse(extra_bits, start, tree_size);
    }
}

CUJPEGXL_BS_HD inline void write_repetitions_zeros(std::size_t repetitions, std::size_t& tree_size,
                                                   std::uint8_t* tree, std::uint8_t* extra_bits) {
    if (repetitions == 11) {
        tree[tree_size] = 0;
        extra_bits[tree_size] = 0;
        ++tree_size;
        --repetitions;
    }
    if (repetitions < 3) {
        for (std::size_t i{0}; i < repetitions; ++i) {
            tree[tree_size] = 0;
            extra_bits[tree_size] = 0;
            ++tree_size;
        }
    } else {
        repetitions -= 3;
        const std::size_t start{tree_size};
        while (true) {
            tree[tree_size] = 17;
            extra_bits[tree_size] = repetitions & 0x7;
            ++tree_size;
            repetitions >>= 3;
            if (repetitions == 0) {
                break;
            }
            --repetitions;
        }
        reverse(tree, start, tree_size);
        reverse(extra_bits, start, tree_size);
    }
}

CUJPEGXL_BS_HD inline void write_huffman_tree(const std::uint8_t* depth, std::size_t length,
                                              std::size_t& tree_size, std::uint8_t* tree,
                                              std::uint8_t* extra_bits) {
    std::uint8_t previous_value{8};

    std::size_t new_length{length};
    for (std::size_t i{0}; i < length; ++i) {
        if (depth[length - i - 1] == 0) {
            --new_length;
        } else {
            break;
        }
    }

    for (std::size_t i{0}; i < new_length;) {
        const std::uint8_t value{depth[i]};
        const std::size_t reps{1};
        if (value == 0) {
            write_repetitions_zeros(reps, tree_size, tree, extra_bits);
        } else {
            write_repetitions(previous_value, value, reps, tree_size, tree, extra_bits);
            previous_value = value;
        }
        i += reps;
    }
}

template <typename W>
CUJPEGXL_BS_HD void store_huffman_tree_of_huffman_tree(int num_codes,
                                                       const std::uint8_t* code_length_bitdepth,
                                                       W& w) {
    const std::uint8_t STORAGE_ORDER[CODE_LENGTH_CODES]{1, 2, 3, 4,  0,  5,  17, 6,  16,
                                                        7, 8, 9, 10, 11, 12, 13, 14, 15};
    const std::uint8_t SYMBOLS[6]{0, 7, 3, 2, 1, 15};
    const std::uint8_t BIT_LENGTHS[6]{2, 4, 3, 2, 2, 4};

    std::size_t codes_to_store{CODE_LENGTH_CODES};
    if (num_codes > 1) {
        for (; codes_to_store > 0; --codes_to_store) {
            if (code_length_bitdepth[STORAGE_ORDER[codes_to_store - 1]] != 0) {
                break;
            }
        }
    }
    std::size_t skip_some{0};
    if (code_length_bitdepth[STORAGE_ORDER[0]] == 0 &&
        code_length_bitdepth[STORAGE_ORDER[1]] == 0) {
        skip_some = 2;
        if (code_length_bitdepth[STORAGE_ORDER[2]] == 0) {
            skip_some = 3;
        }
    }
    w.write(2, skip_some);
    for (std::size_t i{skip_some}; i < codes_to_store; ++i) {
        const std::size_t l{code_length_bitdepth[STORAGE_ORDER[i]]};
        w.write(BIT_LENGTHS[l], SYMBOLS[l]);
    }
}

template <typename W>
CUJPEGXL_BS_HD void store_huffman_tree_to_bit_mask(
    std::size_t huffman_tree_size, const std::uint8_t* huffman_tree,
    const std::uint8_t* huffman_tree_extra_bits, const std::uint8_t* code_length_bitdepth,
    const std::uint16_t* code_length_bitdepth_symbols, W& w) {
    for (std::size_t i{0}; i < huffman_tree_size; ++i) {
        const std::size_t ix{huffman_tree[i]};
        w.write(code_length_bitdepth[ix], code_length_bitdepth_symbols[ix]);
        if (ix == 16) {
            w.write(2, huffman_tree_extra_bits[i]);
        } else if (ix == 17) {
            w.write(3, huffman_tree_extra_bits[i]);
        }
    }
}

template <typename W>
CUJPEGXL_BS_HD void store_simple_huffman_tree(const std::uint8_t* depths, std::size_t symbols[4],
                                              std::size_t num_symbols, std::size_t max_bits, W& w) {
    w.write(2, 1);
    w.write(2, num_symbols - 1);

    for (std::size_t i{0}; i < num_symbols; ++i) {
        for (std::size_t j{i + 1}; j < num_symbols; ++j) {
            if (depths[symbols[j]] < depths[symbols[i]]) {
                const std::size_t t{symbols[j]};
                symbols[j] = symbols[i];
                symbols[i] = t;
            }
        }
    }

    if (num_symbols == 2) {
        w.write(max_bits, symbols[0]);
        w.write(max_bits, symbols[1]);
    } else if (num_symbols == 3) {
        w.write(max_bits, symbols[0]);
        w.write(max_bits, symbols[1]);
        w.write(max_bits, symbols[2]);
    } else {
        w.write(max_bits, symbols[0]);
        w.write(max_bits, symbols[1]);
        w.write(max_bits, symbols[2]);
        w.write(max_bits, symbols[3]);
        w.write(1, depths[symbols[0]] == 1 ? 1 : 0);
    }
}

template <typename W>
CUJPEGXL_BS_HD void store_huffman_tree(const std::uint8_t* depths, std::size_t num, W& w) {
    std::uint8_t huffman_tree[2 * MAX_ALPHABET];
    std::uint8_t huffman_tree_extra_bits[2 * MAX_ALPHABET];
    std::size_t huffman_tree_size{0};
    write_huffman_tree(depths, num, huffman_tree_size, huffman_tree, huffman_tree_extra_bits);

    std::uint32_t huffman_tree_histogram[CODE_LENGTH_CODES]{};
    for (std::size_t i{0}; i < huffman_tree_size; ++i) {
        ++huffman_tree_histogram[huffman_tree[i]];
    }

    int num_codes{0};
    int code{0};
    for (int i{0}; i < CODE_LENGTH_CODES; ++i) {
        if (huffman_tree_histogram[i]) {
            if (num_codes == 0) {
                code = i;
                num_codes = 1;
            } else if (num_codes == 1) {
                num_codes = 2;
                break;
            }
        }
    }

    std::uint8_t code_length_bitdepth[CODE_LENGTH_CODES]{};
    std::uint16_t code_length_bitdepth_symbols[CODE_LENGTH_CODES]{};
    create_huffman_tree(huffman_tree_histogram, CODE_LENGTH_CODES, 5, code_length_bitdepth);
    convert_bit_depths_to_symbols(code_length_bitdepth, CODE_LENGTH_CODES,
                                  code_length_bitdepth_symbols);

    store_huffman_tree_of_huffman_tree(num_codes, code_length_bitdepth, w);

    if (num_codes == 1) {
        code_length_bitdepth[code] = 0;
    }

    store_huffman_tree_to_bit_mask(huffman_tree_size, huffman_tree, huffman_tree_extra_bits,
                                   code_length_bitdepth, code_length_bitdepth_symbols, w);
}

// Counts the used symbols (capped at the 5 the simple-form decision needs) and
// records the first four indices and the max_bits field width.
CUJPEGXL_BS_HD inline void scan_histogram(const std::uint32_t* histogram, std::size_t length,
                                          std::size_t& count, std::size_t s4[4],
                                          std::size_t& max_bits) {
    count = 0;
    s4[0] = s4[1] = s4[2] = s4[3] = 0;
    for (std::size_t i{0}; i < length; ++i) {
        if (histogram[i]) {
            if (count < 4) {
                s4[count] = i;
            } else if (count > 4) {
                break;
            }
            ++count;
        }
    }
    std::size_t max_bits_counter{length - 1};
    max_bits = 0;
    while (max_bits_counter) {
        max_bits_counter >>= 1;
        ++max_bits;
    }
}

// Builds only the prefix code (depth/bits) for `histogram`, writing nothing. This
// is the compute half of build_and_store_huffman_tree, split out so the device
// can build all cluster codes in parallel before the serial bit-writing pass.
CUJPEGXL_BS_HD inline void build_huffman_code(const std::uint32_t* histogram, std::size_t length,
                                              std::uint8_t* depth, std::uint16_t* bits) {
    for (std::size_t i{0}; i < length; ++i) {
        depth[i] = 0;
        bits[i] = 0;
    }
    std::size_t count{}, s4[4]{}, max_bits{};
    scan_histogram(histogram, length, count, s4, max_bits);
    if (count <= 1) {
        return;
    }
    create_huffman_tree(histogram, length, 15, depth);
    convert_bit_depths_to_symbols(depth, length, bits);
}

// Serializes the prefix-code description for `histogram` given the already-built
// `depth`; the write half of build_and_store_huffman_tree.
template <typename W>
CUJPEGXL_BS_HD void store_huffman_code(const std::uint32_t* histogram, std::size_t length,
                                       const std::uint8_t* depth, W& w) {
    std::size_t count{}, s4[4]{}, max_bits{};
    scan_histogram(histogram, length, count, s4, max_bits);
    if (count <= 1) {
        w.write(4, 1);
        w.write(max_bits, s4[0]);
        return;
    }
    if (count <= 4) {
        store_simple_huffman_tree(depth, s4, count, max_bits, w);
    } else {
        store_huffman_tree(depth, length, w);
    }
}

template <typename W>
CUJPEGXL_BS_HD void build_and_store_huffman_tree(const std::uint32_t* histogram, std::size_t length,
                                                 std::uint8_t* depth, std::uint16_t* bits, W& w) {
    build_huffman_code(histogram, length, depth, bits);
    store_huffman_code(histogram, length, depth, w);
}

// --- histogram description (libjxl DecodeHistograms, prefix path) ------------

template <typename W>
CUJPEGXL_BS_HD void encode_uint_config(W& w, const HybridUintConfig& config,
                                       std::size_t log_alpha_size) {
    w.write(ceil_log2_nonzero(log_alpha_size + 1), config.split_exponent);
    if (config.split_exponent == log_alpha_size) {
        return;
    }
    w.write(ceil_log2_nonzero(config.split_exponent + 1), config.msb_in_token);
    w.write(ceil_log2_nonzero(config.split_exponent - config.msb_in_token + 1),
            config.lsb_in_token);
}

template <typename W>
CUJPEGXL_BS_HD void store_var_len_uint16(W& w, std::size_t n) {
    assert(n <= 65535);
    if (n == 0) {
        w.write(1, 0);
        return;
    }
    w.write(1, 1);
    std::size_t nbits{0};
    std::size_t v{n};
    while (v > 1) {
        v >>= 1;
        ++nbits;
    }
    w.write(4, nbits);
    w.write(nbits, n - (1ull << nbits));
}

constexpr std::size_t PREFIX_MAX_BITS = 15;

template <typename W>
CUJPEGXL_BS_HD void write_prefix_histograms(W& w, const std::uint32_t* histogram,
                                            std::size_t length, std::size_t num_contexts,
                                            const HybridUintConfig& config, std::uint8_t* depth,
                                            std::uint16_t* bits) {
    w.write(1, 0);  // lz77.enabled = false

    if (num_contexts > 1) {
        w.write(1, 1);  // is_simple
        w.write(2, 0);  // bits_per_entry = 0 -> all contexts map to histogram 0
    }

    w.write(1, 1);  // use_prefix_code
    encode_uint_config(w, config, PREFIX_MAX_BITS);

    store_var_len_uint16(w, length - 1);

    for (std::size_t i{0}; i < length; ++i) {
        depth[i] = 0;
        bits[i] = 0;
    }
    if (length > 1) {
        build_and_store_huffman_tree(histogram, length, depth, bits, w);
    }
}

// --- context map (libjxl EncodeContextMap, prefix path) ---------------------

// Returns the rank of `value` before the move-to-front update, then updates.
CUJPEGXL_BS_HD inline std::uint8_t mtf_apply(std::uint8_t* symbols, std::uint8_t value) {
    std::size_t rank{0};
    while (symbols[rank] != value) {
        ++rank;
    }
    const std::uint8_t out{static_cast<std::uint8_t>(rank)};
    const std::uint8_t v{symbols[rank]};
    while (rank > 0) {
        symbols[rank] = symbols[rank - 1];
        --rank;
    }
    symbols[0] = v;
    return out;
}

CUJPEGXL_BS_HD inline void mtf_reset(std::uint8_t* symbols) {
    for (std::size_t i{0}; i < 256; ++i) {
        symbols[i] = static_cast<std::uint8_t>(i);
    }
}

template <typename W>
CUJPEGXL_BS_HD void write_simple_context_map(W& w, const std::uint8_t* context_map,
                                             std::size_t num_contexts, std::size_t num_clusters) {
    assert(num_clusters >= 1 && num_clusters <= 8);
    const std::size_t bits_per_entry{ceil_log2_nonzero(num_clusters)};
    w.write(1, 1);
    w.write(2, bits_per_entry);
    for (std::size_t i{0}; i < num_contexts; ++i) {
        assert(context_map[i] < num_clusters);
        if (bits_per_entry != 0) {
            w.write(bits_per_entry, context_map[i]);
        }
    }
}

template <typename W>
CUJPEGXL_BS_HD void write_complex_prefix_context_map(W& w, const std::uint8_t* context_map,
                                                     std::size_t num_contexts, bool use_mtf) {
    const HybridUintConfig config{2, 0, 1};
    std::uint32_t histogram[CONTEXT_MAP_ALPHABET_SIZE]{};

    std::uint8_t symbols[256];
    mtf_reset(symbols);
    for (std::size_t i{0}; i < num_contexts; ++i) {
        const std::uint8_t value{use_mtf ? mtf_apply(symbols, context_map[i]) : context_map[i]};
        std::uint32_t token{}, nbits{}, bits{};
        config.encode(value, token, nbits, bits);
        assert(token < CONTEXT_MAP_ALPHABET_SIZE);
        ++histogram[token];
    }

    std::size_t alpha{1};
    for (std::size_t i{0}; i < CONTEXT_MAP_ALPHABET_SIZE; ++i) {
        if (histogram[i] != 0) {
            alpha = i + 1;
        }
    }

    w.write(1, 0);
    w.write(1, use_mtf ? 1 : 0);

    std::uint8_t depth[CONTEXT_MAP_ALPHABET_SIZE]{};
    std::uint16_t bits[CONTEXT_MAP_ALPHABET_SIZE]{};
    write_prefix_histograms(w, histogram, alpha, 1, config, depth, bits);

    mtf_reset(symbols);
    for (std::size_t i{0}; i < num_contexts; ++i) {
        const std::uint8_t value{use_mtf ? mtf_apply(symbols, context_map[i]) : context_map[i]};
        std::uint32_t token{}, nbits{}, extra_bits{};
        config.encode(value, token, nbits, extra_bits);
        w.write(depth[token], bits[token]);
        if (nbits != 0) {
            w.write(nbits, extra_bits);
        }
    }
}

// Emits the smallest of the legal context-map encodings, choosing by encoded
// size (measured with a CountingBitWriter so no scratch buffer is needed) and
// writing the winner straight into `w`. Byte-identical to the host original,
// which built the candidates into separate writers and appended the shortest.
template <typename W>
CUJPEGXL_BS_HD ContextMapEncoding write_best_prefix_context_map(W& w,
                                                                const std::uint8_t* context_map,
                                                                std::size_t num_contexts,
                                                                std::size_t num_clusters) {
    CountingBitWriter c_complex{};
    write_complex_prefix_context_map(c_complex, context_map, num_contexts, false);
    CountingBitWriter c_mtf{};
    write_complex_prefix_context_map(c_mtf, context_map, num_contexts, true);

    std::size_t best_bits{c_complex.bits_written()};
    ContextMapEncoding encoding{ContextMapEncoding::COMPLEX};
    if (c_mtf.bits_written() < best_bits) {
        best_bits = c_mtf.bits_written();
        encoding = ContextMapEncoding::COMPLEX_MTF;
    }
    if (num_clusters <= 8) {
        CountingBitWriter c_simple{};
        write_simple_context_map(c_simple, context_map, num_contexts, num_clusters);
        if (c_simple.bits_written() <= best_bits) {
            encoding = ContextMapEncoding::SIMPLE;
        }
    }

    switch (encoding) {
        case ContextMapEncoding::SIMPLE:
            write_simple_context_map(w, context_map, num_contexts, num_clusters);
            break;
        case ContextMapEncoding::COMPLEX:
            write_complex_prefix_context_map(w, context_map, num_contexts, false);
            break;
        case ContextMapEncoding::COMPLEX_MTF:
            write_complex_prefix_context_map(w, context_map, num_contexts, true);
            break;
    }
    return encoding;
}

// Builds the per-cluster prefix codes (depth/bits) without writing. Independent
// across clusters, so the device builds them in parallel (one thread per cluster)
// before the serial serialize pass.
CUJPEGXL_BS_HD inline void build_cluster_code(const std::uint32_t* cluster_histograms,
                                              std::size_t stride, std::size_t cluster,
                                              std::uint8_t* depth, std::uint16_t* bits) {
    const std::uint32_t* h{cluster_histograms + cluster * stride};
    const std::size_t a{alphabet_size(h, stride)};
    std::uint8_t* d{depth + cluster * stride};
    std::uint16_t* b{bits + cluster * stride};
    for (std::size_t i{0}; i < stride; ++i) {
        d[i] = 0;
        b[i] = 0;
    }
    if (a > 1) {
        build_huffman_code(h, a, d, b);
    }
}

// Appends `nbits` LSB-first bits taken from the byte buffer `blob` (as produced
// by a byte writer) into `w`.
template <typename W>
CUJPEGXL_BS_HD void append_bits(W& w, const std::uint8_t* blob, std::size_t nbits) {
    std::size_t byte{0};
    while (nbits != 0) {
        const std::size_t take{nbits < 8 ? nbits : 8};
        w.write(take, blob[byte]);
        nbits -= take;
        ++byte;
    }
}

// Builds cluster `cluster`'s prefix code (depth/bits) and, if its alphabet has
// more than one symbol, serializes the code description into `blob` (bit length
// into `nbits`). Runs in the parallel build kernel (one block per cluster) so the
// per-cluster store — the dominant serial cost when done single-threaded — is
// spread across blocks; the serialize pass then only concatenates the blobs.
CUJPEGXL_BS_HD inline void build_cluster_code_and_blob(const std::uint32_t* cluster_histograms,
                                                       std::size_t stride, std::size_t cluster,
                                                       std::uint8_t* depth, std::uint16_t* bits,
                                                       std::uint8_t* blob, std::size_t blob_cap,
                                                       std::uint32_t* nbits) {
    build_cluster_code(cluster_histograms, stride, cluster, depth, bits);
    const std::uint32_t* h{cluster_histograms + cluster * stride};
    const std::size_t a{alphabet_size(h, stride)};
    if (a > 1) {
        SpanBitWriter w{blob, blob_cap};
        store_huffman_code(h, a, depth + cluster * stride, w);
        *nbits = static_cast<std::uint32_t>(w.bits_written());
    } else {
        *nbits = 0;
    }
}

// use_prefix_code flag, per-cluster uint configs, alphabet sizes, and the
// per-cluster prefix-code descriptions from precomputed per-cluster blobs
// (`cluster_blob` / `cluster_nbits`, one `blob_stride`-byte slot per cluster).
template <typename W>
CUJPEGXL_BS_HD void write_cluster_codes_tail_prebuilt(
    W& w, std::size_t num_clusters, const std::uint32_t* cluster_histograms, std::size_t stride,
    const HybridUintConfig& config, const std::uint8_t* cluster_blob, std::size_t blob_stride,
    const std::uint32_t* cluster_nbits) {
    w.write(1, 1);  // use_prefix_code
    for (std::size_t c{0}; c < num_clusters; ++c) {
        encode_uint_config(w, config, PREFIX_MAX_BITS);
    }
    std::size_t alpha[256]{};
    for (std::size_t c{0}; c < num_clusters; ++c) {
        alpha[c] = alphabet_size(cluster_histograms + c * stride, stride);
        store_var_len_uint16(w, alpha[c] - 1);
    }
    for (std::size_t c{0}; c < num_clusters; ++c) {
        if (alpha[c] > 1) {
            append_bits(w, cluster_blob + c * blob_stride, cluster_nbits[c]);
        }
    }
}

// use_prefix_code flag, per-cluster uint configs, alphabet sizes, and the
// per-cluster prefix-code descriptions (from the prebuilt `depth`). This is the
// data-dependent tail of the clustered histogram description, shared by the host
// serialize and the device serialize (which prepends a precomputed context map).
template <typename W>
CUJPEGXL_BS_HD void write_cluster_codes_tail(W& w, std::size_t num_clusters,
                                             const std::uint32_t* cluster_histograms,
                                             std::size_t stride, const HybridUintConfig& config,
                                             const std::uint8_t* depth) {
    w.write(1, 1);  // use_prefix_code
    for (std::size_t c{0}; c < num_clusters; ++c) {
        encode_uint_config(w, config, PREFIX_MAX_BITS);
    }

    std::size_t alpha[256]{};
    for (std::size_t c{0}; c < num_clusters; ++c) {
        const std::size_t a{alphabet_size(cluster_histograms + c * stride, stride)};
        alpha[c] = a;
        store_var_len_uint16(w, a - 1);
    }

    for (std::size_t c{0}; c < num_clusters; ++c) {
        if (alpha[c] > 1) {
            store_huffman_code(cluster_histograms + c * stride, alpha[c], depth + c * stride, w);
        }
    }
}

// Serializes the clustered histogram description assuming the per-cluster codes
// in `depth` are already built (via build_cluster_code). The compute-free write
// pass; byte-identical to write_clustered_prefix_histograms.
template <typename W>
CUJPEGXL_BS_HD void write_clustered_prefix_histograms_serialize(
    W& w, const std::uint8_t* context_map, std::size_t num_contexts, std::size_t num_clusters,
    const std::uint32_t* cluster_histograms, std::size_t stride, const HybridUintConfig& config,
    const std::uint8_t* depth) {
    assert(num_clusters >= 1 && num_clusters <= 256);

    w.write(1, 0);  // lz77.enabled = false

    if (num_contexts > 1) {
        write_best_prefix_context_map(w, context_map, num_contexts, num_clusters);
    }

    write_cluster_codes_tail(w, num_clusters, cluster_histograms, stride, config, depth);
}

template <typename W>
CUJPEGXL_BS_HD void write_clustered_prefix_histograms(
    W& w, const std::uint8_t* context_map, std::size_t num_contexts, std::size_t num_clusters,
    const std::uint32_t* cluster_histograms, std::size_t stride, const HybridUintConfig& config,
    std::uint8_t* depth, std::uint16_t* bits) {
    for (std::size_t c{0}; c < num_clusters; ++c) {
        build_cluster_code(cluster_histograms, stride, c, depth, bits);
    }
    write_clustered_prefix_histograms_serialize(w, context_map, num_contexts, num_clusters,
                                                cluster_histograms, stride, config, depth);
}

// --- VarDCT DC / AcMetadata modular headers ---------------------------------

// GroupHeader + single-leaf MA tree (with `predictor`) + the data histogram
// description, filling depth/bits with the data prefix code. Mirrors the oracle
// modular.cc write_modular_header (which then emits the data tokens).
template <typename W>
CUJPEGXL_BS_HD void write_modular_header(W& w, const std::uint32_t* data_hist, std::size_t data_len,
                                         int predictor, std::uint8_t* depth, std::uint16_t* bits) {
    w.write(1, 0);  // use_global_tree = false
    w.write(1, 1);  // weighted::Header all_default = true
    w.write(2, 0);  // num_transforms = 0

    const std::uint32_t tree_tokens[NUM_TREE_CONTEXTS - 1]{0, static_cast<std::uint32_t>(predictor),
                                                           0, 0, 0};
    std::uint32_t tree_hist[16]{};
    std::size_t tree_len{1};
    for (std::uint32_t t : tree_tokens) {
        ++tree_hist[t];
        if (static_cast<std::size_t>(t) + 1 > tree_len) {
            tree_len = static_cast<std::size_t>(t) + 1;
        }
    }
    std::uint8_t tree_depth[16]{};
    std::uint16_t tree_bits[16]{};
    write_prefix_histograms(w, tree_hist, tree_len, NUM_TREE_CONTEXTS, HybridUintConfig{},
                            tree_depth, tree_bits);
    for (std::uint32_t t : tree_tokens) {
        w.write(tree_depth[t], tree_bits[t]);
    }

    for (std::size_t i{0}; i < data_len; ++i) {
        depth[i] = 0;
        bits[i] = 0;
    }
    write_prefix_histograms(w, data_hist, data_len, 1, HybridUintConfig{}, depth, bits);
}

// Leading AcGlobal fields: DequantMatrices all_default, num_histograms,
// used_orders.
template <typename W>
CUJPEGXL_BS_HD void serialize_ac_global_head(W& w, std::size_t num_ac_groups) {
    constexpr U32Enc order_enc{{Val(0x5F), Val(0x13), Val(0), Bits(13)}};
    write_bool(w, true);                                 // DequantMatrices::Decode all_default
    write_bits(w, ceil_log2_nonzero(num_ac_groups), 0);  // num_histograms - 1
    write_u32(w, order_enc, 0);                          // used_orders = 0 (natural)
}

// Serialize-only AcGlobal from precomputed context-map and per-cluster code blobs
// (the per-cluster code descriptions built in parallel by build_cluster_code_and_blob).
// The serialize pass therefore only concatenates blobs, with no per-cluster
// Huffman work. Byte-identical to serializing the clustered histograms from
// scratch with write_clustered_prefix_histograms.
template <typename W>
CUJPEGXL_BS_HD void serialize_ac_global_prebuilt_blobs(
    W& w, const std::uint32_t* ac_histograms, const std::uint8_t* ctxmap_blob,
    std::size_t ctxmap_nbits, std::size_t num_clusters, std::size_t num_ac_groups,
    const std::uint8_t* cluster_blob, std::size_t blob_stride, const std::uint32_t* cluster_nbits) {
    serialize_ac_global_head(w, num_ac_groups);
    w.write(1, 0);  // lz77.enabled = false
    append_bits(w, ctxmap_blob, ctxmap_nbits);
    write_cluster_codes_tail_prebuilt(w, num_clusters, ac_histograms, HISTOGRAM_STRIDE,
                                      HybridUintConfig{}, cluster_blob, blob_stride, cluster_nbits);
    w.zero_pad_to_byte();
}

// DcGroup pre-blob: extra_precision + the VarDCTDC modular header (Gradient).
template <typename W>
CUJPEGXL_BS_HD void serialize_dc_pre(W& w, const std::uint32_t* dc_hist, std::uint8_t* dc_depth,
                                     std::uint16_t* dc_bits) {
    write_bits(w, 2, 0);  // DecodeVarDCTDC extra_precision = 0
    write_modular_header(w, dc_hist, alphabet_size(dc_hist, HISTOGRAM_STRIDE),
                         DC_PREDICTOR_GRADIENT, dc_depth, dc_bits);
}

// DcGroup mid-blob: AcMetadata `count` field + the AcMetadata modular header
// (Zero predictor). `total_blocks` sizes the count field, `count` is the group's
// first-block count.
template <typename W>
CUJPEGXL_BS_HD void serialize_dc_mid(W& w, const std::uint32_t* meta_hist, std::size_t total_blocks,
                                     std::size_t count, std::uint8_t* meta_depth,
                                     std::uint16_t* meta_bits) {
    write_bits(w, ceil_log2_nonzero(total_blocks), static_cast<std::uint32_t>(count - 1));
    write_modular_header(w, meta_hist, alphabet_size(meta_hist, HISTOGRAM_STRIDE),
                         DC_PREDICTOR_ZERO, meta_depth, meta_bits);
}

}  // namespace cujpegxl::bitstream::detail

#endif  // CUJPEGXL_SRC_BITSTREAM_SERIALIZE_IMPL_H_
