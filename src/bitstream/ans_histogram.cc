// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "ans_histogram.h"

#include <algorithm>
#include <array>
#include <cassert>

#include "context_map.h"

namespace cujpegxl::bitstream {
namespace {

constexpr std::size_t ANS_LOG_TABLE_SIZE = 12;
constexpr std::size_t LOG_ALPHA_SIZE = 8;
constexpr std::array<std::uint8_t, ANS_LOG_TABLE_SIZE + 2> LOG_COUNT_DEPTH{5, 4, 4, 4, 4, 4, 3,
                                                                           3, 3, 3, 3, 6, 7, 7};
constexpr std::array<std::uint8_t, ANS_LOG_TABLE_SIZE + 2> LOG_COUNT_BITS{17, 11, 15, 3, 9,  7, 4,
                                                                          2,  5,  6,  0, 33, 1, 65};

std::size_t floor_log2(std::size_t value) {
    assert(value != 0);
    std::size_t result{0};
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

std::size_t ceil_log2(std::size_t value) {
    if (value <= 1) {
        return 0;
    }
    return floor_log2(value - 1) + 1;
}

void store_var_len_uint8(BitWriter& w, std::size_t value) {
    assert(value <= 255);
    if (value == 0) {
        w.write(1, 0);
        return;
    }
    const std::size_t nbits{floor_log2(value)};
    w.write(1, 1);
    w.write(3, nbits);
    w.write(nbits, value - (1u << nbits));
}

void encode_uint_config(BitWriter& w, const HybridUintConfig& config) {
    w.write(ceil_log2(LOG_ALPHA_SIZE + 1), config.split_exponent);
    if (config.split_exponent == LOG_ALPHA_SIZE) {
        return;
    }
    w.write(ceil_log2(config.split_exponent + 1), config.msb_in_token);
    w.write(ceil_log2(config.split_exponent - config.msb_in_token + 1), config.lsb_in_token);
}

void normalize_counts(const std::uint32_t* histogram, std::size_t alphabet_size,
                      std::array<std::uint16_t, ANS_ALPHABET_SIZE>& counts) {
    assert(histogram != nullptr);
    assert(alphabet_size >= 1 && alphabet_size <= ANS_ALPHABET_SIZE);

    std::uint64_t total{0};
    std::size_t nonzero{0};
    std::size_t only_symbol{0};
    for (std::size_t i{0}; i < alphabet_size; ++i) {
        total += histogram[i];
        if (histogram[i] != 0) {
            ++nonzero;
            only_symbol = i;
        }
    }
    if (nonzero == 0) {
        counts[0] = ANS_TABLE_SIZE;
        return;
    }
    if (nonzero == 1) {
        counts[only_symbol] = ANS_TABLE_SIZE;
        return;
    }

    std::array<std::uint64_t, ANS_ALPHABET_SIZE> remainders{};
    std::size_t sum{0};
    for (std::size_t i{0}; i < alphabet_size; ++i) {
        if (histogram[i] == 0) {
            continue;
        }
        const std::uint64_t scaled{static_cast<std::uint64_t>(histogram[i]) * ANS_TABLE_SIZE};
        counts[i] = static_cast<std::uint16_t>(scaled / total);
        remainders[i] = scaled % total;
        if (counts[i] == 0) {
            counts[i] = 1;
            remainders[i] = 0;
        }
        sum += counts[i];
    }

    while (sum < ANS_TABLE_SIZE) {
        std::size_t best{0};
        for (std::size_t i{1}; i < alphabet_size; ++i) {
            if (remainders[i] > remainders[best]) {
                best = i;
            }
        }
        ++counts[best];
        remainders[best] = 0;
        ++sum;
    }
    while (sum > ANS_TABLE_SIZE) {
        std::size_t best{ANS_ALPHABET_SIZE};
        for (std::size_t i{0}; i < alphabet_size; ++i) {
            if (counts[i] <= 1) {
                continue;
            }
            if (best == ANS_ALPHABET_SIZE || remainders[i] < remainders[best] ||
                (remainders[i] == remainders[best] && counts[i] > counts[best])) {
                best = i;
            }
        }
        assert(best != ANS_ALPHABET_SIZE);
        --counts[best];
        --sum;
    }
}

void build_aliases(AnsDistribution& distribution) {
    constexpr std::size_t ENTRY_SIZE = ANS_TABLE_SIZE / ANS_ALPHABET_SIZE;
    std::array<std::uint16_t, ANS_ALPHABET_SIZE> cutoffs{distribution.counts};
    std::array<std::uint8_t, ANS_ALPHABET_SIZE> underfull{};
    std::array<std::uint8_t, ANS_ALPHABET_SIZE> overfull{};
    std::size_t num_underfull{0};
    std::size_t num_overfull{0};

    std::size_t single_symbol{ANS_ALPHABET_SIZE};
    for (std::size_t i{0}; i < ANS_ALPHABET_SIZE; ++i) {
        if (cutoffs[i] == ANS_TABLE_SIZE) {
            single_symbol = i;
            break;
        }
    }
    if (single_symbol != ANS_ALPHABET_SIZE) {
        for (std::size_t i{0}; i < ANS_ALPHABET_SIZE; ++i) {
            distribution.aliases[i] = {
                .cutoff = 0,
                .right_value = static_cast<std::uint8_t>(single_symbol),
                .frequency = 0,
                .right_offset = static_cast<std::uint16_t>(ENTRY_SIZE * i),
                .right_frequency_xor = ANS_TABLE_SIZE,
            };
        }
        return;
    }

    for (std::size_t i{0}; i < ANS_ALPHABET_SIZE; ++i) {
        if (cutoffs[i] < ENTRY_SIZE) {
            underfull[num_underfull] = static_cast<std::uint8_t>(i);
            ++num_underfull;
        } else if (cutoffs[i] > ENTRY_SIZE) {
            overfull[num_overfull] = static_cast<std::uint8_t>(i);
            ++num_overfull;
        }
    }
    while (num_overfull != 0) {
        --num_overfull;
        const std::size_t over{overfull[num_overfull]};
        assert(num_underfull != 0);
        --num_underfull;
        const std::size_t under{underfull[num_underfull]};
        cutoffs[over] -= ENTRY_SIZE - cutoffs[under];
        distribution.aliases[under].right_value = static_cast<std::uint8_t>(over);
        distribution.aliases[under].right_offset = cutoffs[over];
        if (cutoffs[over] < ENTRY_SIZE) {
            underfull[num_underfull] = static_cast<std::uint8_t>(over);
            ++num_underfull;
        } else if (cutoffs[over] > ENTRY_SIZE) {
            overfull[num_overfull] = static_cast<std::uint8_t>(over);
            ++num_overfull;
        }
    }

    for (std::size_t i{0}; i < ANS_ALPHABET_SIZE; ++i) {
        AnsAliasEntry& entry{distribution.aliases[i]};
        if (cutoffs[i] == ENTRY_SIZE) {
            entry.right_value = static_cast<std::uint8_t>(i);
            entry.right_offset = 0;
            entry.cutoff = 0;
        } else {
            entry.right_offset -= cutoffs[i];
            entry.cutoff = static_cast<std::uint8_t>(cutoffs[i]);
        }
        entry.frequency = distribution.counts[i];
        entry.right_frequency_xor = distribution.counts[entry.right_value] ^ distribution.counts[i];
    }
}

void write_counts(BitWriter& w, const AnsDistribution& distribution, std::size_t alphabet_size) {
    std::array<std::size_t, 2> symbols{};
    std::size_t num_symbols{0};
    for (std::size_t i{0}; i < alphabet_size; ++i) {
        if (distribution.counts[i] != 0) {
            if (num_symbols < symbols.size()) {
                symbols[num_symbols] = i;
            }
            ++num_symbols;
        }
    }
    if (num_symbols <= 2) {
        w.write(1, 1);
        w.write(1, num_symbols == 2 ? 1 : 0);
        const std::size_t stored_symbols{num_symbols == 0 ? 1 : num_symbols};
        for (std::size_t i{0}; i < stored_symbols; ++i) {
            store_var_len_uint8(w, symbols[i]);
        }
        if (num_symbols == 2) {
            w.write(ANS_LOG_TABLE_SIZE, distribution.counts[symbols[0]]);
        }
        return;
    }

    std::size_t omit_pos{0};
    std::size_t largest_log{0};
    std::size_t length{0};
    std::array<std::size_t, ANS_ALPHABET_SIZE> logcounts{};
    for (std::size_t i{0}; i < alphabet_size; ++i) {
        if (distribution.counts[i] != 0) {
            logcounts[i] = floor_log2(distribution.counts[i]) + 1;
            length = i + 1;
            if (logcounts[i] > largest_log) {
                largest_log = logcounts[i];
                omit_pos = i;
            }
        }
    }
    std::size_t omit_log{0};
    for (std::size_t i{0}; i < length; ++i) {
        if (i == omit_pos || logcounts[i] == 0) {
            continue;
        }
        omit_log =
            i < omit_pos ? std::max(omit_log, logcounts[i] + 1) : std::max(omit_log, logcounts[i]);
    }
    logcounts[omit_pos] = omit_log;

    w.write(1, 0);
    w.write(1, 0);
    w.write(3, 7);
    w.write(3, 5);
    store_var_len_uint8(w, length - 3);
    for (std::size_t i{0}; i < length; ++i) {
        w.write(LOG_COUNT_DEPTH[logcounts[i]], LOG_COUNT_BITS[logcounts[i]]);
    }
    for (std::size_t i{0}; i < length; ++i) {
        if (i == omit_pos || logcounts[i] <= 1) {
            continue;
        }
        const std::size_t bitcount{logcounts[i] - 1};
        w.write(bitcount, distribution.counts[i] - (1u << bitcount));
    }
}

}  // namespace

void build_ans_distribution(const std::uint32_t* histogram, std::size_t alphabet_size,
                            AnsDistribution& distribution) {
    distribution = {};
    normalize_counts(histogram, alphabet_size, distribution.counts);
    build_aliases(distribution);
}

void write_clustered_ans_histograms(BitWriter& w, const std::uint8_t* context_map,
                                    std::size_t num_contexts, std::size_t num_clusters,
                                    const std::uint32_t* cluster_histograms, std::size_t stride,
                                    const HybridUintConfig& config,
                                    AnsDistribution* distributions) {
    assert(context_map != nullptr && cluster_histograms != nullptr && distributions != nullptr);
    assert(num_contexts >= 1);
    assert(num_clusters >= 1 && num_clusters <= 256);
    assert(stride >= 1 && stride <= ANS_ALPHABET_SIZE);

    w.write(1, 0);
    if (num_contexts > 1) {
        write_best_prefix_context_map(w, context_map, num_contexts, num_clusters);
    }
    w.write(1, 0);
    w.write(2, LOG_ALPHA_SIZE - 5);
    for (std::size_t c{0}; c < num_clusters; ++c) {
        encode_uint_config(w, config);
    }
    for (std::size_t c{0}; c < num_clusters; ++c) {
        std::size_t alphabet_size{1};
        for (std::size_t i{0}; i < stride; ++i) {
            if (cluster_histograms[c * stride + i] != 0) {
                alphabet_size = i + 1;
            }
        }
        build_ans_distribution(cluster_histograms + c * stride, alphabet_size, distributions[c]);
        write_counts(w, distributions[c], alphabet_size);
    }
}

}  // namespace cujpegxl::bitstream
