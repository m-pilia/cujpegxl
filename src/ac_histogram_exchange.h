// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_AC_HISTOGRAM_EXCHANGE_H_
#define CUJPEGXL_SRC_AC_HISTOGRAM_EXCHANGE_H_

#include <cstdint>

#include <cuda_runtime_api.h>

namespace cujpegxl {

enum class AcHistogramExchangeMode : std::int32_t {
    AUTO = 0,
    STAGED = 1,
    MAPPED = 2,
};

enum class AcHistogramOwnership : std::int32_t {
    UNINITIALIZED = 0,
    GPU_WRITABLE = 1,
    GPU_PENDING = 2,
    CPU_READABLE = 3,
};

class AcHistogramExchange {
public:
    AcHistogramExchange() = default;
    ~AcHistogramExchange();

    AcHistogramExchange(const AcHistogramExchange&) = delete;
    AcHistogramExchange& operator=(const AcHistogramExchange&) = delete;

    bool initialize(AcHistogramExchangeMode mode, std::int32_t device_ordinal = 0);
    void shutdown();

    std::uint32_t* gpu_data();
    bool release_to_cpu(cudaStream_t stream);
    bool acquire_for_cpu();
    const std::uint32_t* cpu_data() const;
    void release_to_gpu();

    AcHistogramExchangeMode mode() const { return mode_; }
    AcHistogramOwnership ownership() const { return ownership_; }

    static bool mapped_supported(std::int32_t device_ordinal);

private:
    AcHistogramExchangeMode mode_{AcHistogramExchangeMode::STAGED};
    AcHistogramOwnership ownership_{AcHistogramOwnership::UNINITIALIZED};
    std::uint32_t* device_data_{nullptr};
    std::uint32_t* host_data_{nullptr};
    cudaEvent_t ready_event_{nullptr};
};

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_AC_HISTOGRAM_EXCHANGE_H_
