// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Port of libjxl's ANS histogram normalization and signalling (enc_ans.cc,
// ans_common.cc). The normalization method (flat, or the counts' precision
// shift) is chosen by float estimates of the description and data bits, whose
// rounding depends on the compiler's floating-point contraction; any choice is
// a valid description of its normalized counts.

#include "ans.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>

#include "bit_sink.h"

namespace cujpegxl::bitstream {
namespace {

constexpr std::int32_t TABLE_SIZE{static_cast<std::int32_t>(ANS_TAB_SIZE)};
constexpr std::int32_t MAX_SYMBOLS_FOR_SMALL_CODE{4};
constexpr std::uint32_t MIN_RLE_REPS{4};

// Static prefix code of the log-counts; the last symbol starts an RLE run.
constexpr std::array<std::uint8_t, ANS_LOG_TAB_SIZE + 2> LOG_COUNT_BIT_LENGTHS{5, 4, 4, 4, 4, 4, 3,
                                                                               3, 3, 3, 3, 6, 7, 7};
constexpr std::array<std::uint8_t, ANS_LOG_TAB_SIZE + 2> LOG_COUNT_SYMBOLS{
    17, 11, 15, 3, 9, 7, 4, 2, 5, 6, 0, 33, 1, 65};

using Counts = std::array<std::int32_t, ANS_MAX_ALPHABET_SIZE>;

std::int32_t floor_log2(std::uint32_t x) {
    assert(x != 0);
    return 31 - __builtin_clz(x);
}

// libjxl FastLog2f: a (2,2) rational polynomial of the mantissa after range
// reduction to [-1/3, 1/3].
float fast_log2(float x) {
    constexpr float P0{-1.8503833400518310E-06f};
    constexpr float P1{1.4287160470083755E+00f};
    constexpr float P2{7.4245873327820566E-01f};
    constexpr float Q0{9.9032814277590719E-01f};
    constexpr float Q1{1.0096718572241148E+00f};
    constexpr float Q2{1.7409343003366853E-01f};
    std::int32_t x_bits{};
    std::memcpy(&x_bits, &x, sizeof(x_bits));
    const std::int32_t exp_shifted{(x_bits - 0x3f2aaaab) >> 23};
    const std::int32_t mantissa_bits{
        x_bits - static_cast<std::int32_t>(static_cast<std::uint32_t>(exp_shifted) << 23)};
    float mantissa{};
    std::memcpy(&mantissa, &mantissa_bits, sizeof(mantissa));
    const float m{mantissa - 1.0f};
    const float yp{(P2 * m + P1) * m + P0};
    const float yq{(Q2 * m + Q1) * m + Q0};
    return yp / yq + static_cast<float>(exp_shifted);
}

// Bits of a count of log2 `logcount` kept by the description at `shift`.
std::int32_t population_count_precision(std::int32_t logcount, std::uint32_t shift) {
    const std::int32_t r{
        std::min(logcount, static_cast<std::int32_t>(shift) -
                               ((static_cast<std::int32_t>(ANS_LOG_TAB_SIZE) - logcount) >> 1))};
    return r < 0 ? 0 : r;
}

// Spacing of the counts representable at `shift` around `count`.
std::int32_t smallest_increment(std::uint32_t count, std::uint32_t shift) {
    const std::int32_t bits{count == 0 ? -1 : floor_log2(count)};
    const std::int32_t drop_bits{bits - population_count_precision(bits, shift)};
    return drop_bits < 0 ? 1 : (1 << drop_bits);
}

template <bool MINIMIZE_ERROR_OF_SUM>
bool rebalance_histogram(const float* targets, std::int32_t max_symbol, std::uint32_t shift,
                         std::int32_t& omit_pos, Counts& counts) {
    std::int32_t sum{0};
    float sum_nonrounded{0.0f};
    std::int32_t remainder_pos{0};
    std::int32_t remainder_log{-1};
    for (std::int32_t n{0}; n < max_symbol; ++n) {
        if (targets[n] > 0 && targets[n] < 1.0f) {
            counts[n] = 1;
            sum_nonrounded += targets[n];
            sum += counts[n];
        }
    }
    const float discount_ratio{static_cast<float>(TABLE_SIZE - sum) /
                               (static_cast<float>(TABLE_SIZE) - sum_nonrounded)};
    if (!(discount_ratio > 0) || !(discount_ratio <= 1.0f)) {
        return false;
    }
    for (std::int32_t n{0}; n < max_symbol; ++n) {
        if (targets[n] >= 1.0f) {
            sum_nonrounded += targets[n];
            counts[n] = static_cast<std::int32_t>(targets[n] * discount_ratio);
            if (counts[n] == 0) {
                counts[n] = 1;
            }
            if (counts[n] == TABLE_SIZE) {
                counts[n] = TABLE_SIZE - 1;
            }
            // Round to a multiple of the representable increment, to the
            // nearest one (or, when minimizing the error of the sum, to the one
            // that keeps the sum closest to the unrounded sum).
            const std::int32_t inc{
                smallest_increment(static_cast<std::uint32_t>(counts[n]), shift)};
            counts[n] -= counts[n] & (inc - 1);
            const std::int32_t target{MINIMIZE_ERROR_OF_SUM
                                          ? (static_cast<std::int32_t>(sum_nonrounded) - sum)
                                          : static_cast<std::int32_t>(targets[n])};
            if (counts[n] == 0 || (target >= counts[n] + inc / 2 && counts[n] + inc < TABLE_SIZE)) {
                counts[n] += inc;
            }
            sum += counts[n];
            const std::int32_t count_log{floor_log2(static_cast<std::uint32_t>(counts[n]))};
            if (count_log > remainder_log) {
                remainder_pos = n;
                remainder_log = count_log;
            }
        }
    }
    counts[remainder_pos] -= sum - TABLE_SIZE;
    omit_pos = remainder_pos;
    return counts[remainder_pos] > 0;
}

struct Normalized {
    std::int32_t omit_pos{0};
    std::int32_t num_symbols{0};
    std::array<std::int32_t, MAX_SYMBOLS_FOR_SMALL_CODE> symbols{};
};

bool normalize_counts(Counts& counts, std::int32_t length, std::uint32_t shift, Normalized& out) {
    std::uint64_t total{0};
    std::int32_t max_symbol{0};
    std::int32_t symbol_count{0};
    for (std::int32_t n{0}; n < length; ++n) {
        total += static_cast<std::uint64_t>(counts[n]);
        if (counts[n] > 0) {
            if (symbol_count < MAX_SYMBOLS_FOR_SMALL_CODE) {
                out.symbols[symbol_count] = n;
            }
            ++symbol_count;
            max_symbol = n + 1;
        }
    }
    out.num_symbols = symbol_count;
    if (symbol_count == 0) {
        return true;
    }
    if (symbol_count == 1) {
        counts[out.symbols[0]] = TABLE_SIZE;
        return true;
    }
    const float norm{1.f * static_cast<float>(TABLE_SIZE) / static_cast<float>(total)};
    std::array<float, ANS_MAX_ALPHABET_SIZE> targets{};
    for (std::int32_t n{0}; n < max_symbol; ++n) {
        targets[n] = norm * static_cast<float>(counts[n]);
    }
    return rebalance_histogram<false>(targets.data(), max_symbol, shift, out.omit_pos, counts) ||
           rebalance_histogram<true>(targets.data(), max_symbol, shift, out.omit_pos, counts);
}

template <typename W>
void store_var_len_uint8(W& w, std::uint32_t n) {
    assert(n <= 255);
    if (n == 0) {
        w.write(1, 0);
        return;
    }
    w.write(1, 1);
    const auto nbits{static_cast<std::uint32_t>(floor_log2(n))};
    w.write(3, nbits);
    w.write(nbits, n - (1u << nbits));
}

template <typename W>
void encode_counts(const Counts& counts, std::int32_t alphabet_size, const Normalized& norm,
                   std::uint32_t shift, W& w) {
    if (norm.num_symbols <= 2) {
        w.write(1, 1);  // small tree
        if (norm.num_symbols == 0) {
            w.write(1, 0);
            store_var_len_uint8(w, 0);
        } else {
            w.write(1, static_cast<std::uint64_t>(norm.num_symbols - 1));
            for (std::int32_t i{0}; i < norm.num_symbols; ++i) {
                store_var_len_uint8(w, static_cast<std::uint32_t>(norm.symbols[i]));
            }
        }
        if (norm.num_symbols == 2) {
            w.write(ANS_LOG_TAB_SIZE, static_cast<std::uint64_t>(counts[norm.symbols[0]]));
        }
        return;
    }
    w.write(1, 0);  // not a small tree
    w.write(1, 0);  // not flat

    const std::int32_t omit_pos{norm.omit_pos};
    // same[i]: length of the run of equal counts starting at i (set at a run's
    // first index only). The omitted count never joins a run, as it may take a
    // different number of log2 bits.
    std::array<std::uint32_t, ANS_MAX_ALPHABET_SIZE> same{};
    std::int32_t last{0};
    for (std::int32_t i{1}; i < alphabet_size; ++i) {
        if (counts[i] != counts[last] || i + 1 == alphabet_size || (i - last) >= 255 ||
            i == omit_pos || i == omit_pos + 1) {
            same[last] = static_cast<std::uint32_t>(i - last);
            last = i + 1;
        }
    }

    std::int32_t length{0};
    std::array<std::int32_t, ANS_MAX_ALPHABET_SIZE> logcounts{};
    std::int32_t omit_log{0};
    for (std::int32_t i{0}; i < alphabet_size; ++i) {
        if (i == omit_pos) {
            length = i + 1;
        } else if (counts[i] > 0) {
            logcounts[i] = floor_log2(static_cast<std::uint32_t>(counts[i])) + 1;
            length = i + 1;
            omit_log = i < omit_pos ? std::max(omit_log, logcounts[i] + 1)
                                    : std::max(omit_log, logcounts[i]);
        }
    }
    logcounts[omit_pos] = omit_log;

    // Elias-gamma-like code of the shift, without the terminating 0 when the
    // unary part reaches its upper bound.
    const std::int32_t upper_bound_log{floor_log2(ANS_LOG_TAB_SIZE + 1)};
    const std::int32_t log{floor_log2(shift + 1)};
    w.write(static_cast<std::size_t>(log), (1u << log) - 1);
    if (log != upper_bound_log) {
        w.write(1, 0);
    }
    w.write(static_cast<std::size_t>(log), ((1u << log) - 1) & (shift + 1));

    store_var_len_uint8(w, static_cast<std::uint32_t>(length - 3));

    constexpr std::size_t RLE_SYMBOL{ANS_LOG_TAB_SIZE + 1};
    for (std::int32_t i{0}; i < length; ++i) {
        if (i > 0 && same[i - 1] > MIN_RLE_REPS) {
            w.write(LOG_COUNT_BIT_LENGTHS[RLE_SYMBOL], LOG_COUNT_SYMBOLS[RLE_SYMBOL]);
            store_var_len_uint8(w, same[i - 1] - MIN_RLE_REPS - 1);
            i += static_cast<std::int32_t>(same[i - 1]) - 2;
            continue;
        }
        w.write(LOG_COUNT_BIT_LENGTHS[logcounts[i]], LOG_COUNT_SYMBOLS[logcounts[i]]);
    }
    for (std::int32_t i{0}; i < length; ++i) {
        if (i > 0 && same[i - 1] > MIN_RLE_REPS) {
            i += static_cast<std::int32_t>(same[i - 1]) - 2;
            continue;
        }
        if (logcounts[i] > 1 && i != omit_pos) {
            const std::int32_t bitcount{population_count_precision(logcounts[i] - 1, shift)};
            const std::int32_t drop_bits{logcounts[i] - 1 - bitcount};
            assert((counts[i] & ((1 << drop_bits) - 1)) == 0);
            w.write(static_cast<std::size_t>(bitcount),
                    static_cast<std::uint64_t>((counts[i] >> drop_bits) - (1 << bitcount)));
        }
    }
}

template <typename W>
void encode_flat_histogram(std::int32_t alphabet_size, W& w) {
    w.write(1, 0);  // not a small tree
    w.write(1, 1);  // flat
    store_var_len_uint8(w, static_cast<std::uint32_t>(alphabet_size - 1));
}

float estimate_data_bits(const std::uint32_t* histogram, const Counts& counts,
                         std::int32_t length) {
    float sum{0.0f};
    for (std::int32_t i{0}; i < length; ++i) {
        if (histogram[i] > 0) {
            sum += static_cast<float>(histogram[i]) *
                   std::max(0.0f, static_cast<float>(ANS_LOG_TAB_SIZE) -
                                      fast_log2(static_cast<float>(counts[i])));
        }
    }
    return sum;
}

float estimate_data_bits_flat(const std::uint32_t* histogram, std::int32_t length) {
    const float flat_bits{std::max(fast_log2(static_cast<float>(length)), 0.0f)};
    float total_histogram{0};
    for (std::int32_t i{0}; i < length; ++i) {
        total_histogram += static_cast<float>(histogram[i]);
    }
    return total_histogram * flat_bits;
}

// Method 0 is the flat histogram, method m > 0 the counts at shift m - 1. A
// shift whose normalization fails is skipped (libjxl aborts the encode).
float histogram_and_data_cost(const std::uint32_t* histogram, std::int32_t alphabet_size,
                              std::uint32_t method) {
    if (method == 0) {
        return static_cast<float>(ANS_LOG_TAB_SIZE + 2) +
               estimate_data_bits_flat(histogram, alphabet_size);
    }
    const std::uint32_t shift{method - 1};
    Counts counts{};
    for (std::int32_t i{0}; i < alphabet_size; ++i) {
        counts[i] = static_cast<std::int32_t>(histogram[i]);
    }
    Normalized norm{};
    if (!normalize_counts(counts, alphabet_size, shift, norm)) {
        return std::numeric_limits<float>::max();
    }
    CountingBitWriter size{};
    encode_counts(counts, alphabet_size, norm, shift, size);
    return static_cast<float>(size.bits_written()) +
           estimate_data_bits(histogram, counts, alphabet_size);
}

// kApproximate: the flat histogram and every other shift.
std::uint32_t best_method(const std::uint32_t* histogram, std::int32_t alphabet_size) {
    std::uint32_t method{0};
    float best_cost{histogram_and_data_cost(histogram, alphabet_size, 0)};
    for (std::uint32_t shift{0}; shift <= ANS_LOG_TAB_SIZE; shift += 2) {
        const float cost{histogram_and_data_cost(histogram, alphabet_size, shift + 1)};
        if (cost < best_cost) {
            method = shift + 1;
            best_cost = cost;
        }
    }
    return method;
}

// libjxl InitAliasTable: every entry of `entry_size` decoder slots holds its
// own symbol below `cutoff` and `right_value` from there on, the latter's
// occurrences starting at `offsets1` + the slot position.
struct AliasEntry {
    std::uint32_t cutoff{0};
    std::uint32_t right_value{0};
    std::uint32_t offsets1{0};
};

void build_reverse_map(const Counts& counts, std::int32_t alphabet_size, std::size_t log_alpha_size,
                       AnsHistogram& out) {
    const std::size_t table_size{std::size_t{1} << log_alpha_size};
    const std::uint32_t log_entry_size{ANS_LOG_TAB_SIZE -
                                       static_cast<std::uint32_t>(log_alpha_size)};
    const std::uint32_t entry_size{1u << log_entry_size};
    std::size_t length{static_cast<std::size_t>(alphabet_size)};
    while (length > 0 && counts[length - 1] == 0) {
        --length;
    }
    assert(length > 0 && length <= table_size);

    std::array<AliasEntry, ANS_MAX_ALPHABET_SIZE> entries{};
    std::size_t single_symbol{ANS_MAX_ALPHABET_SIZE};
    for (std::size_t s{0}; s < length; ++s) {
        if (counts[s] == TABLE_SIZE) {
            single_symbol = s;
        }
    }
    if (single_symbol != ANS_MAX_ALPHABET_SIZE) {
        for (std::size_t i{0}; i < table_size; ++i) {
            entries[i] = AliasEntry{0, static_cast<std::uint32_t>(single_symbol),
                                    static_cast<std::uint32_t>(entry_size * i)};
        }
    } else {
        std::array<std::uint32_t, ANS_MAX_ALPHABET_SIZE> cutoffs{};
        std::array<std::uint32_t, ANS_MAX_ALPHABET_SIZE> underfull{};
        std::array<std::uint32_t, ANS_MAX_ALPHABET_SIZE> overfull{};
        std::size_t num_underfull{0};
        std::size_t num_overfull{0};
        for (std::size_t i{0}; i < table_size; ++i) {
            cutoffs[i] = i < length ? static_cast<std::uint32_t>(counts[i]) : 0;
            if (cutoffs[i] > entry_size) {
                overfull[num_overfull] = static_cast<std::uint32_t>(i);
                ++num_overfull;
            } else if (cutoffs[i] < entry_size) {
                underfull[num_underfull] = static_cast<std::uint32_t>(i);
                ++num_underfull;
            }
        }
        while (num_overfull > 0) {
            --num_overfull;
            const std::uint32_t over{overfull[num_overfull]};
            assert(num_underfull > 0);
            --num_underfull;
            const std::uint32_t under{underfull[num_underfull]};
            cutoffs[over] -= entry_size - cutoffs[under];
            entries[under].right_value = over;
            entries[under].offsets1 = cutoffs[over];
            if (cutoffs[over] < entry_size) {
                underfull[num_underfull] = over;
                ++num_underfull;
            } else if (cutoffs[over] > entry_size) {
                overfull[num_overfull] = over;
                ++num_overfull;
            }
        }
        for (std::size_t i{0}; i < table_size; ++i) {
            if (cutoffs[i] == entry_size) {
                entries[i] = AliasEntry{0, static_cast<std::uint32_t>(i), 0};
            } else {
                entries[i].offsets1 -= cutoffs[i];
                entries[i].cutoff = cutoffs[i];
            }
        }
    }

    out.freq.fill(0);
    out.offset.fill(0);
    std::uint32_t offset{0};
    for (std::size_t s{0}; s < length; ++s) {
        out.freq[s] = static_cast<std::uint16_t>(counts[s]);
        out.offset[s] = static_cast<std::uint16_t>(offset);
        offset += static_cast<std::uint32_t>(counts[s]);
    }
    for (std::uint32_t value{0}; value < ANS_TAB_SIZE; ++value) {
        const AliasEntry& e{entries[value >> log_entry_size]};
        const std::uint32_t pos{value & (entry_size - 1)};
        const bool greater{pos >= e.cutoff};
        const std::uint32_t symbol{greater ? e.right_value : value >> log_entry_size};
        const std::uint32_t slot{(greater ? e.offsets1 : 0) + pos};
        assert(slot < out.freq[symbol]);
        out.reverse_map[out.offset[symbol] + slot] = static_cast<std::uint16_t>(value);
    }
}

}  // namespace

void write_ans_histogram(BitWriter& w, const std::uint32_t* histogram, std::size_t alphabet_size,
                         std::size_t log_alpha_size, AnsHistogram& out) {
    assert(alphabet_size >= 1 && alphabet_size <= ANS_MAX_ALPHABET_SIZE);
    assert(log_alpha_size >= 5 && log_alpha_size <= 8);
    assert(alphabet_size <= (std::size_t{1} << log_alpha_size));
    const auto length{static_cast<std::int32_t>(alphabet_size)};
    const std::uint32_t method{best_method(histogram, length)};

    Counts counts{};
    std::uint64_t sum{0};
    for (std::int32_t i{0}; i < length; ++i) {
        counts[i] = static_cast<std::int32_t>(histogram[i]);
        sum += histogram[i];
    }
    if (sum == 0) {
        counts[0] = TABLE_SIZE;
    }
    if (method == 0) {
        // libjxl CreateFlatHistogram: the larger counts first.
        for (std::int32_t i{0}; i < length; ++i) {
            counts[i] = TABLE_SIZE / length + (i < TABLE_SIZE % length ? 1 : 0);
        }
        build_reverse_map(counts, length, log_alpha_size, out);
        encode_flat_histogram(length, w);
        return;
    }
    Normalized norm{};
    const bool normalized{normalize_counts(counts, length, method - 1, norm)};
    assert(normalized);
    (void)normalized;
    build_reverse_map(counts, length, log_alpha_size, out);
    encode_counts(counts, length, norm, method - 1, w);
}

}  // namespace cujpegxl::bitstream
