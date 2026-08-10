// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Port of libjxl's Huffman-tree construction and storage (enc_huffman_tree.cc,
// enc_huffman.cc) restricted to what the JXL prefix-code header needs. Kept
// byte-for-byte compatible with HuffmanDecodingData::ReadFromBitStream.

#include "prefix_code.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

namespace cujpegxl::bitstream {
namespace {

constexpr int CODE_LENGTH_CODES = 18;

struct HuffmanTree {
    std::uint32_t total_count;
    std::int16_t index_left;
    std::int16_t index_right_or_value;
};

void set_depth(int p, std::vector<HuffmanTree>& pool, std::uint8_t* depth,
               std::uint8_t level) {
    const HuffmanTree& node{pool[p]};
    if (node.index_left >= 0) {
        set_depth(node.index_left, pool, depth, level + 1);
        set_depth(node.index_right_or_value, pool, depth, level + 1);
    } else {
        depth[node.index_right_or_value] = level;
    }
}

void create_huffman_tree(const std::uint32_t* data, std::size_t length, int tree_limit,
                         std::uint8_t* depth) {
    for (std::uint32_t count_limit{1};; count_limit *= 2) {
        std::vector<HuffmanTree> tree{};
        tree.reserve(2 * length + 1);

        for (std::size_t i{length}; i != 0;) {
            --i;
            if (data[i]) {
                const std::uint32_t count{std::max(data[i], count_limit - 1)};
                tree.push_back({count, -1, static_cast<std::int16_t>(i)});
            }
        }

        const std::size_t n{tree.size()};
        if (n == 1) {
            depth[tree[0].index_right_or_value] = 1;
            break;
        }

        std::stable_sort(tree.begin(), tree.end(),
                         [](const HuffmanTree& a, const HuffmanTree& b) {
                             return a.total_count < b.total_count;
                         });

        const HuffmanTree sentinel{std::numeric_limits<std::uint32_t>::max(), -1, -1};
        tree.push_back(sentinel);
        tree.push_back(sentinel);

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

            const std::size_t j_end{tree.size() - 1};
            tree[j_end].total_count = tree[left].total_count + tree[right].total_count;
            tree[j_end].index_left = static_cast<std::int16_t>(left);
            tree[j_end].index_right_or_value = static_cast<std::int16_t>(right);
            tree.push_back(sentinel);
        }
        set_depth(static_cast<int>(2 * n - 1), tree, depth, 0);

        if (*std::max_element(&depth[0], &depth[length]) <= tree_limit) {
            break;
        }
    }
}

std::uint16_t reverse_bits(int num_bits, std::uint16_t bits) {
    static const std::size_t LUT[16]{0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe,
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

void convert_bit_depths_to_symbols(const std::uint8_t* depth, std::size_t len,
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

void reverse(std::uint8_t* v, std::size_t start, std::size_t end) {
    --end;
    while (start < end) {
        std::swap(v[start], v[end]);
        ++start;
        --end;
    }
}

void write_repetitions(std::uint8_t previous_value, std::uint8_t value,
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

void write_repetitions_zeros(std::size_t repetitions, std::size_t& tree_size,
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

void write_huffman_tree(const std::uint8_t* depth, std::size_t length,
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
        std::size_t reps{1};
        // RLE is only worthwhile for long codes (length > 50 in libjxl); our
        // alphabets are small enough that the non-RLE path is always taken.
        if (value == 0) {
            write_repetitions_zeros(reps, tree_size, tree, extra_bits);
        } else {
            write_repetitions(previous_value, value, reps, tree_size, tree, extra_bits);
            previous_value = value;
        }
        i += reps;
    }
}

void store_huffman_tree_of_huffman_tree(int num_codes,
                                        const std::uint8_t* code_length_bitdepth,
                                        BitWriter& w) {
    static const std::uint8_t STORAGE_ORDER[CODE_LENGTH_CODES]{
        1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    static const std::uint8_t SYMBOLS[6]{0, 7, 3, 2, 1, 15};
    static const std::uint8_t BIT_LENGTHS[6]{2, 4, 3, 2, 2, 4};

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

void store_huffman_tree_to_bit_mask(std::size_t huffman_tree_size,
                                    const std::uint8_t* huffman_tree,
                                    const std::uint8_t* huffman_tree_extra_bits,
                                    const std::uint8_t* code_length_bitdepth,
                                    const std::uint16_t* code_length_bitdepth_symbols,
                                    BitWriter& w) {
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

void store_simple_huffman_tree(const std::uint8_t* depths, std::size_t symbols[4],
                               std::size_t num_symbols, std::size_t max_bits,
                               BitWriter& w) {
    w.write(2, 1);
    w.write(2, num_symbols - 1);

    for (std::size_t i{0}; i < num_symbols; ++i) {
        for (std::size_t j{i + 1}; j < num_symbols; ++j) {
            if (depths[symbols[j]] < depths[symbols[i]]) {
                std::swap(symbols[j], symbols[i]);
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

void store_huffman_tree(const std::uint8_t* depths, std::size_t num, BitWriter& w) {
    std::vector<std::uint8_t> arena(2 * num);
    std::uint8_t* huffman_tree{arena.data()};
    std::uint8_t* huffman_tree_extra_bits{arena.data() + num};
    std::size_t huffman_tree_size{0};
    write_huffman_tree(depths, num, huffman_tree_size, huffman_tree,
                       huffman_tree_extra_bits);

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

    store_huffman_tree_to_bit_mask(huffman_tree_size, huffman_tree,
                                   huffman_tree_extra_bits, code_length_bitdepth,
                                   code_length_bitdepth_symbols, w);
}

}  // namespace

void build_and_store_huffman_tree(const std::uint32_t* histogram, std::size_t length,
                                  std::uint8_t* depth, std::uint16_t* bits, BitWriter& w) {
    std::memset(depth, 0, length);
    std::memset(bits, 0, length * sizeof(*bits));

    std::size_t count{0};
    std::size_t s4[4]{0};
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
    std::size_t max_bits{0};
    while (max_bits_counter) {
        max_bits_counter >>= 1;
        ++max_bits;
    }

    if (count <= 1) {
        w.write(4, 1);
        w.write(max_bits, s4[0]);
        return;
    }

    create_huffman_tree(histogram, length, 15, depth);
    convert_bit_depths_to_symbols(depth, length, bits);

    if (count <= 4) {
        store_simple_huffman_tree(depth, s4, count, max_bits, w);
    } else {
        store_huffman_tree(depth, length, w);
    }
}

}  // namespace cujpegxl::bitstream
