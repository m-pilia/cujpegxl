// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "ac_histogram_exchange.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include "entropy.h"

namespace cujpegxl {
namespace {

std::vector<std::uint32_t> exchange_contents(AcHistogramExchangeMode mode) {
    AcHistogramExchange exchange{};
    EXPECT_TRUE(exchange.initialize(mode));
    EXPECT_EQ(exchange.ownership(), AcHistogramOwnership::GPU_WRITABLE);

    cudaStream_t stream{nullptr};
    EXPECT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    EXPECT_EQ(cudaMemsetAsync(exchange.gpu_data(), 0x5a,
                              AC_CONTEXT_HISTOGRAM_ENTRIES * sizeof(std::uint32_t), stream),
              cudaSuccess);
    EXPECT_TRUE(exchange.release_to_cpu(stream));
    EXPECT_EQ(exchange.ownership(), AcHistogramOwnership::GPU_PENDING);
    EXPECT_TRUE(exchange.acquire_for_cpu());
    EXPECT_EQ(exchange.ownership(), AcHistogramOwnership::CPU_READABLE);

    const std::uint32_t* data{exchange.cpu_data()};
    std::vector<std::uint32_t> result(data, data + AC_CONTEXT_HISTOGRAM_ENTRIES);
    exchange.release_to_gpu();
    EXPECT_EQ(exchange.ownership(), AcHistogramOwnership::GPU_WRITABLE);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    return result;
}

TEST(AcHistogramExchange, StagedContentsAndOwnership) {
    const std::vector<std::uint32_t> values{
        exchange_contents(AcHistogramExchangeMode::STAGED)};
    ASSERT_EQ(values.size(), AC_CONTEXT_HISTOGRAM_ENTRIES);
    for (std::uint32_t value : values) {
        ASSERT_EQ(value, 0x5a5a5a5au);
    }
}

TEST(AcHistogramExchange, MappedMatchesStaged) {
    if (!AcHistogramExchange::mapped_supported(0)) {
        GTEST_SKIP() << "mapped pinned memory is unsupported";
    }
    EXPECT_EQ(exchange_contents(AcHistogramExchangeMode::MAPPED),
              exchange_contents(AcHistogramExchangeMode::STAGED));
}

TEST(AcHistogramExchange, AutoUsesStagedOnDiscreteGpu) {
    cudaDeviceProp properties{};
    ASSERT_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);
    AcHistogramExchange exchange{};
    ASSERT_TRUE(exchange.initialize(AcHistogramExchangeMode::AUTO));
    EXPECT_EQ(exchange.mode(), properties.integrated != 0 && properties.canMapHostMemory != 0
                                   ? AcHistogramExchangeMode::MAPPED
                                   : AcHistogramExchangeMode::STAGED);
}

}  // namespace
}  // namespace cujpegxl
