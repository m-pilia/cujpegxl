// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_BITSTREAM_SERIALIZE_IMPL_H_
#define CUJPEGXL_SRC_BITSTREAM_SERIALIZE_IMPL_H_

// Writer-templated implementation of the JXL prefix-code entropy container
// serialization: length-limited Huffman construction, the histogram
// description, the context map, and the VarDCT DC / AcMetadata modular headers.
// The host bitstream library instantiates it with the vector BitWriter (via the
// thin wrappers in prefix_code.cc / histogram_writer.cc / context_map.cc /
// frame_assembly.cc). The field-level writers are host+device: the on-device
// prefix build (prefix_build.cu) emits its header fields through them, while
// building the Huffman codes warp-cooperatively.
//
// Containers are bounded stack buffers (alphabets are <= HISTOGRAM_STRIDE and
// context maps <= AC_NUM_CONTEXTS), std::stable_sort is a stable insertion sort,
// and the recursive depth walk uses an explicit stack.

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
// libjxl SetDepth), which bounds the walk for the deep, pre-limit trees the
// doubling loop discards.
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

// Builds only the prefix code (depth/bits) for `histogram`, writing nothing: the
// compute half of build_and_store_huffman_tree.
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

// The fields of a single-histogram prefix description that precede its code.
template <typename W>
CUJPEGXL_BS_HD void write_prefix_histogram_head(W& w, std::size_t length, std::size_t num_contexts,
                                                const HybridUintConfig& config) {
    w.write(1, 0);  // lz77.enabled = false

    if (num_contexts > 1) {
        w.write(1, 1);  // is_simple
        w.write(2, 0);  // bits_per_entry = 0 -> all contexts map to histogram 0
    }

    w.write(1, 1);  // use_prefix_code
    encode_uint_config(w, config, PREFIX_MAX_BITS);

    store_var_len_uint16(w, length - 1);
}

template <typename W>
CUJPEGXL_BS_HD void write_prefix_histograms(W& w, const std::uint32_t* histogram,
                                            std::size_t length, std::size_t num_contexts,
                                            const HybridUintConfig& config, std::uint8_t* depth,
                                            std::uint16_t* bits) {
    write_prefix_histogram_head(w, length, num_contexts, config);

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

// Counts of the context-map values as written plainly and after the
// move-to-front transform, from one pass over the map.
CUJPEGXL_BS_HD inline void count_context_map_values(const std::uint8_t* context_map,
                                                    std::size_t num_contexts,
                                                    std::uint32_t* plain_counts,
                                                    std::uint32_t* mtf_counts) {
    for (std::size_t v{0}; v < 256; ++v) {
        plain_counts[v] = 0;
        mtf_counts[v] = 0;
    }
    std::uint8_t symbols[256];
    mtf_reset(symbols);
    for (std::size_t i{0}; i < num_contexts; ++i) {
        ++plain_counts[context_map[i]];
        ++mtf_counts[mtf_apply(symbols, context_map[i])];
    }
}

constexpr HybridUintConfig CONTEXT_MAP_UINT_CONFIG{2, 0, 1};

// The prefix code of a complex context map and its encoded size in bits.
struct ContextMapCode {
    std::uint32_t histogram[CONTEXT_MAP_ALPHABET_SIZE];
    std::size_t alphabet_size;
    std::uint8_t depth[CONTEXT_MAP_ALPHABET_SIZE];
    std::uint16_t bits[CONTEXT_MAP_ALPHABET_SIZE];
    std::size_t size_bits;
};

CUJPEGXL_BS_HD inline ContextMapCode build_context_map_code(const std::uint32_t* value_counts) {
    ContextMapCode code{};
    std::size_t raw_bits{0};
    for (std::uint32_t v{0}; v < 256; ++v) {
        if (value_counts[v] == 0) {
            continue;
        }
        std::uint32_t token{}, nbits{}, extra_bits{};
        CONTEXT_MAP_UINT_CONFIG.encode(v, token, nbits, extra_bits);
        assert(token < CONTEXT_MAP_ALPHABET_SIZE);
        code.histogram[token] += value_counts[v];
        raw_bits += static_cast<std::size_t>(value_counts[v]) * nbits;
    }
    code.alphabet_size = 1;
    for (std::size_t t{0}; t < CONTEXT_MAP_ALPHABET_SIZE; ++t) {
        if (code.histogram[t] != 0) {
            code.alphabet_size = t + 1;
        }
    }
    CountingBitWriter head{};
    head.write(2, 0);  // is_simple, use_mtf
    write_prefix_histograms(head, code.histogram, code.alphabet_size, 1, CONTEXT_MAP_UINT_CONFIG,
                            code.depth, code.bits);
    code.size_bits = head.bits_written() + raw_bits;
    for (std::size_t t{0}; t < code.alphabet_size; ++t) {
        code.size_bits += static_cast<std::size_t>(code.histogram[t]) * code.depth[t];
    }
    return code;
}

// Writes a complex context map with `code`, built from the counts of the values
// as written (with the move-to-front transform if `use_mtf`). The entries are
// gathered into 64-bit chunks before reaching `w`.
template <typename W>
CUJPEGXL_BS_HD void write_complex_prefix_context_map(W& w, const std::uint8_t* context_map,
                                                     std::size_t num_contexts, bool use_mtf,
                                                     const ContextMapCode& code) {
    w.write(1, 0);
    w.write(1, use_mtf ? 1 : 0);
    write_prefix_histogram_head(w, code.alphabet_size, 1, CONTEXT_MAP_UINT_CONFIG);
    if (code.alphabet_size > 1) {
        store_huffman_code(code.histogram, code.alphabet_size, code.depth, w);
    }

    constexpr std::size_t CHUNK_BITS{56};
    std::uint64_t chunk{0};
    std::size_t chunk_bits{0};
    const auto put = [&](std::size_t n, std::uint64_t value) {
        if (chunk_bits + n > CHUNK_BITS) {
            w.write(chunk_bits, chunk);
            chunk = 0;
            chunk_bits = 0;
        }
        chunk |= value << chunk_bits;
        chunk_bits += n;
    };
    std::uint8_t symbols[256];
    mtf_reset(symbols);
    for (std::size_t i{0}; i < num_contexts; ++i) {
        const std::uint8_t value{use_mtf ? mtf_apply(symbols, context_map[i]) : context_map[i]};
        std::uint32_t token{}, nbits{}, extra_bits{};
        CONTEXT_MAP_UINT_CONFIG.encode(value, token, nbits, extra_bits);
        put(code.depth[token], code.bits[token]);
        put(nbits, extra_bits);
    }
    if (chunk_bits != 0) {
        w.write(chunk_bits, chunk);
    }
}

template <typename W>
CUJPEGXL_BS_HD void write_complex_prefix_context_map(W& w, const std::uint8_t* context_map,
                                                     std::size_t num_contexts, bool use_mtf) {
    std::uint32_t plain_counts[256];
    std::uint32_t mtf_counts[256];
    count_context_map_values(context_map, num_contexts, plain_counts, mtf_counts);
    write_complex_prefix_context_map(w, context_map, num_contexts, use_mtf,
                                     build_context_map_code(use_mtf ? mtf_counts : plain_counts));
}

// Emits the smallest of the legal context-map encodings. The sizes of the
// complex encodings follow from the value counts, so only the chosen encoding
// is written.
template <typename W>
CUJPEGXL_BS_HD ContextMapEncoding write_best_prefix_context_map(W& w,
                                                                const std::uint8_t* context_map,
                                                                std::size_t num_contexts,
                                                                std::size_t num_clusters) {
    std::uint32_t plain_counts[256];
    std::uint32_t mtf_counts[256];
    count_context_map_values(context_map, num_contexts, plain_counts, mtf_counts);
    const ContextMapCode complex{build_context_map_code(plain_counts)};
    const ContextMapCode mtf{build_context_map_code(mtf_counts)};

    std::size_t best_bits{complex.size_bits};
    ContextMapEncoding encoding{ContextMapEncoding::COMPLEX};
    if (mtf.size_bits < best_bits) {
        best_bits = mtf.size_bits;
        encoding = ContextMapEncoding::COMPLEX_MTF;
    }
    if (num_clusters <= 8) {
        const std::size_t simple_bits{3 + num_contexts * ceil_log2_nonzero(num_clusters)};
        if (simple_bits <= best_bits) {
            encoding = ContextMapEncoding::SIMPLE;
        }
    }

    switch (encoding) {
        case ContextMapEncoding::SIMPLE:
            write_simple_context_map(w, context_map, num_contexts, num_clusters);
            break;
        case ContextMapEncoding::COMPLEX:
            write_complex_prefix_context_map(w, context_map, num_contexts, false, complex);
            break;
        case ContextMapEncoding::COMPLEX_MTF:
            write_complex_prefix_context_map(w, context_map, num_contexts, true, mtf);
            break;
    }
    return encoding;
}

// Builds cluster `cluster`'s prefix code (depth/bits) without writing.
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

// use_prefix_code flag, per-cluster uint configs, alphabet sizes, and the
// per-cluster prefix-code descriptions (from the prebuilt `depth`): the
// data-dependent tail of the clustered histogram description.
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

// MA tree tokens in libjxl DecodeTree order. All NUM_TREE_CONTEXTS contexts
// share one histogram, so only the token values are kept.
constexpr std::size_t MAX_TREE_TOKENS = 12;
constexpr std::size_t TREE_ALPHABET = 16;

struct TreeTokens {
    std::uint32_t values[MAX_TREE_TOKENS];
    std::size_t count;
};

// Property 0 (leaf), `predictor`, offset 0, multiplier 1 (log 0, bits 0).
CUJPEGXL_BS_HD inline void append_leaf(TreeTokens& tree, int predictor) {
    const std::uint32_t leaf[]{0, static_cast<std::uint32_t>(predictor), 0, 0, 0};
    for (std::uint32_t t : leaf) {
        assert(tree.count < MAX_TREE_TOKENS);
        tree.values[tree.count] = t;
        ++tree.count;
    }
}

CUJPEGXL_BS_HD inline TreeTokens single_leaf_tree(int predictor) {
    TreeTokens tree{{}, 0};
    append_leaf(tree, predictor);
    return tree;
}

// The AcMetadata channels are YtoX, YtoB, ACS+QF and EPF. The root splits on the
// channel index (property 0, token 1): libjxl takes the first child, decoded
// first, when the property exceeds the split value, so the all-zero EPF channel
// gets the first leaf (data context 0) and the other channels the second
// (context 1). Both leaves use the Zero predictor.
constexpr std::int32_t ACMETA_EPF_CHANNEL = 3;

CUJPEGXL_BS_HD inline TreeTokens acmeta_tree() {
    TreeTokens tree{{1, pack_signed(ACMETA_EPF_CHANNEL - 1)}, 2};
    append_leaf(tree, DC_PREDICTOR_ZERO);
    append_leaf(tree, DC_PREDICTOR_ZERO);
    return tree;
}

template <typename W>
CUJPEGXL_BS_HD void write_group_header(W& w) {
    w.write(1, 0);  // use_global_tree = false
    w.write(1, 1);  // weighted::Header all_default = true
    w.write(2, 0);  // num_transforms = 0
}

// GroupHeader + the MA tree. Mirrors the oracle modular.cc.
template <typename W>
CUJPEGXL_BS_HD void write_modular_tree(W& w, const TreeTokens& tree) {
    write_group_header(w);
    std::uint32_t tree_hist[TREE_ALPHABET]{};
    std::size_t tree_len{1};
    for (std::size_t i{0}; i < tree.count; ++i) {
        const std::uint32_t t{tree.values[i]};
        assert(t < TREE_ALPHABET);
        ++tree_hist[t];
        if (static_cast<std::size_t>(t) + 1 > tree_len) {
            tree_len = static_cast<std::size_t>(t) + 1;
        }
    }
    std::uint8_t tree_depth[TREE_ALPHABET]{};
    std::uint16_t tree_bits[TREE_ALPHABET]{};
    write_prefix_histograms(w, tree_hist, tree_len, NUM_TREE_CONTEXTS, HybridUintConfig{},
                            tree_depth, tree_bits);
    for (std::size_t i{0}; i < tree.count; ++i) {
        w.write(tree_depth[tree.values[i]], tree_bits[tree.values[i]]);
    }
}

// GroupHeader + single-leaf MA tree (with `predictor`) + the data histogram
// description, filling depth/bits with the data prefix code.
template <typename W>
CUJPEGXL_BS_HD void write_modular_header(W& w, const std::uint32_t* data_hist, std::size_t data_len,
                                         int predictor, std::uint8_t* depth, std::uint16_t* bits) {
    write_modular_tree(w, single_leaf_tree(predictor));
    write_prefix_histograms(w, data_hist, data_len, 1, HybridUintConfig{}, depth, bits);
}

// The data histogram description of acmeta_tree() up to the prefix code of its
// second leaf (`data_len` symbols): context map {0, 1}, then the codes' uint
// configs and alphabet sizes. The EPF leaf only codes symbol 0, so its
// alphabet size is 1 and it has no code.
template <typename W>
CUJPEGXL_BS_HD void write_acmeta_histograms_head(W& w, std::size_t data_len) {
    constexpr std::size_t NUM_LEAVES{2};
    const std::uint8_t context_map[NUM_LEAVES]{0, 1};
    w.write(1, 0);  // lz77.enabled = false
    write_simple_context_map(w, context_map, NUM_LEAVES, NUM_LEAVES);
    w.write(1, 1);  // use_prefix_code
    for (std::size_t l{0}; l < NUM_LEAVES; ++l) {
        encode_uint_config(w, HybridUintConfig{}, PREFIX_MAX_BITS);
    }
    store_var_len_uint16(w, 0);
    store_var_len_uint16(w, data_len - 1);
}

// GroupHeader + acmeta_tree() + its data histogram description, filling
// depth/bits with the prefix code of the non-EPF leaf over `data_hist`.
template <typename W>
CUJPEGXL_BS_HD void write_acmeta_modular_header(W& w, const std::uint32_t* data_hist,
                                                std::size_t data_len, std::uint8_t* depth,
                                                std::uint16_t* bits) {
    write_modular_tree(w, acmeta_tree());
    write_acmeta_histograms_head(w, data_len);
    for (std::size_t i{0}; i < data_len; ++i) {
        depth[i] = 0;
        bits[i] = 0;
    }
    if (data_len > 1) {
        build_and_store_huffman_tree(data_hist, data_len, depth, bits, w);
    }
}

// Leading AcGlobal fields: DequantMatrices all_default, num_histograms.
template <typename W>
CUJPEGXL_BS_HD void serialize_ac_global_head(W& w, std::size_t num_ac_groups) {
    write_bool(w, true);                                 // DequantMatrices::Decode all_default
    write_bits(w, ceil_log2_nonzero(num_ac_groups), 0);  // num_histograms - 1
}

// DcGroup pre-blob: extra_precision + the VarDCTDC modular header (Gradient).
template <typename W>
CUJPEGXL_BS_HD void serialize_dc_pre(W& w, const std::uint32_t* dc_hist, std::uint8_t* dc_depth,
                                     std::uint16_t* dc_bits) {
    write_bits(w, 2, 0);  // DecodeVarDCTDC extra_precision = 0
    write_modular_header(w, dc_hist, alphabet_size(dc_hist, HISTOGRAM_STRIDE),
                         DC_PREDICTOR_GRADIENT, dc_depth, dc_bits);
}

// DcGroup mid-blob: AcMetadata `count` field + the AcMetadata modular header.
// `total_blocks` sizes the count field, `count` is the group's first-block
// count, `meta_hist` the histogram of the non-EPF channels.
template <typename W>
CUJPEGXL_BS_HD void serialize_dc_mid(W& w, const std::uint32_t* meta_hist, std::size_t total_blocks,
                                     std::size_t count, std::uint8_t* meta_depth,
                                     std::uint16_t* meta_bits) {
    write_bits(w, ceil_log2_nonzero(total_blocks), static_cast<std::uint32_t>(count - 1));
    write_acmeta_modular_header(w, meta_hist, alphabet_size(meta_hist, HISTOGRAM_STRIDE),
                                meta_depth, meta_bits);
}

}  // namespace cujpegxl::bitstream::detail

#endif  // CUJPEGXL_SRC_BITSTREAM_SERIALIZE_IMPL_H_
