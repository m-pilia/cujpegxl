// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

// Decodes write_ans_histogram's descriptions with libjxl's DecodeHistograms:
// the decoder must read exactly the written bits and build an alias table whose
// symbol frequencies are the normalized ones and whose slots are the slot maps
// the rANS encoder uses.

#include "ans.h"

#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

#include <jxl/memory_manager.h>

#include <gtest/gtest.h>

// Before libjxl: its ans_params.h defines macros named like our constants.
#include "bit_writer.h"
#include "hybrid_uint.h"
#include "serialize_impl.h"

#include "lib/jxl/ans_common.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"

namespace cujpegxl::bitstream {
namespace {

void* test_alloc(void*, size_t size) {
    return std::malloc(size);
}
void test_free(void*, void* p) {
    std::free(p);
}

std::size_t log_alpha_size_for(std::size_t alphabet_size) {
    std::size_t log{5};
    while ((std::size_t{1} << log) < alphabet_size) {
        ++log;
    }
    return log;
}

void check_decodes(const std::vector<std::uint32_t>& histogram) {
    const std::size_t alphabet_size{histogram.size()};
    const std::size_t log_alpha_size{log_alpha_size_for(alphabet_size)};

    // A single-context rANS entropy header around the histogram description.
    BitWriter w{};
    w.write(1, 0);  // lz77.enabled
    w.write(1, 0);  // use_prefix_code
    w.write(2, log_alpha_size - 5);
    detail::encode_uint_config(w, HybridUintConfig{}, log_alpha_size);
    AnsHistogram table{};
    write_ans_histogram(w, histogram.data(), alphabet_size, log_alpha_size, table);
    const std::size_t written_bits{w.bits_written()};
    w.zero_pad_to_byte();

    JxlMemoryManager mm{nullptr, &test_alloc, &test_free};
    jxl::BitReader br{jxl::Bytes(w.bytes().data(), w.bytes().size())};
    jxl::ANSCode code{};
    std::vector<std::uint8_t> map{};
    ASSERT_TRUE(jxl::DecodeHistograms(&mm, &br, 1, &code, &map));
    EXPECT_EQ(br.TotalBitsConsumed(), written_bits);
    ASSERT_TRUE(br.Close());
    ASSERT_FALSE(code.use_prefix_code);
    ASSERT_EQ(code.log_alpha_size, log_alpha_size);

    std::uint32_t total{0};
    for (std::size_t s{0}; s < alphabet_size; ++s) {
        if (histogram[s] != 0) {
            EXPECT_GT(table.freq[s], 0u) << "symbol " << s << " is not codable";
        }
        total += table.freq[s];
    }
    EXPECT_EQ(total, ANS_TAB_SIZE);

    const auto* aliases{code.alias_tables.address<jxl::AliasTable::Entry>()};
    const std::size_t log_entry_size{ANS_LOG_TAB_SIZE - log_alpha_size};
    for (std::uint32_t value{0}; value < ANS_TAB_SIZE; ++value) {
        const jxl::AliasTable::Symbol symbol{jxl::AliasTable::Lookup(
            aliases, value, log_entry_size, (std::size_t{1} << log_entry_size) - 1)};
        ASSERT_LT(symbol.value, alphabet_size) << "slot " << value;
        ASSERT_EQ(symbol.freq, table.freq[symbol.value]) << "slot " << value;
        ASSERT_LT(symbol.offset, symbol.freq) << "slot " << value;
        ASSERT_EQ(table.reverse_map[table.offset[symbol.value] + symbol.offset], value)
            << "slot " << value;
    }
}

TEST(Ans, EmptyHistogram) {
    check_decodes({0});
}

TEST(Ans, SingleSymbol) {
    check_decodes({0, 0, 0, 12345});
}

TEST(Ans, TwoSymbols) {
    check_decodes({7, 0, 1000});
    check_decodes({1, 1});
}

TEST(Ans, FlatHistogram) {
    check_decodes(std::vector<std::uint32_t>(64, 1000));
    check_decodes(std::vector<std::uint32_t>(37, 3));
}

TEST(Ans, LongRuns) {
    std::vector<std::uint32_t> h(64, 5);
    h[0] = 1000000;
    h[40] = 77;
    check_decodes(h);
}

TEST(Ans, RandomHistograms) {
    std::mt19937 rng{1234};
    for (int iter{0}; iter < 3000; ++iter) {
        const std::size_t alphabet_size{1 + rng() % (iter % 3 == 0 ? 256 : 64)};
        const std::uint32_t scale{1u << (rng() % 24)};
        const double decay{0.5 + 0.5 * static_cast<double>(rng() % 1000) / 1000.0};
        std::vector<std::uint32_t> h(alphabet_size, 0);
        double weight{static_cast<double>(scale)};
        for (std::size_t s{0}; s < alphabet_size; ++s) {
            if (rng() % 4 != 0) {
                h[s] = static_cast<std::uint32_t>(weight * static_cast<double>(1 + rng() % 8) / 8);
            }
            weight *= decay;
        }
        SCOPED_TRACE(iter);
        check_decodes(h);
    }
}

}  // namespace
}  // namespace cujpegxl::bitstream
