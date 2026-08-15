// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include "ac_histogram_exchange.h"

#include <cassert>
#include <cstddef>

#include <cuda_runtime.h>

#include "entropy.h"

namespace cujpegxl {
namespace {

constexpr std::size_t HISTOGRAM_BYTES =
    AC_CONTEXT_HISTOGRAM_ENTRIES * sizeof(std::uint32_t);

}  // namespace

AcHistogramExchange::~AcHistogramExchange() { shutdown(); }

bool AcHistogramExchange::mapped_supported(std::int32_t device_ordinal) {
    cudaDeviceProp properties{};
    return cudaGetDeviceProperties(&properties, device_ordinal) == cudaSuccess &&
           properties.canMapHostMemory != 0;
}

bool AcHistogramExchange::initialize(AcHistogramExchangeMode requested,
                                     std::int32_t device_ordinal) {
    assert(ownership_ == AcHistogramOwnership::UNINITIALIZED);
    if (cudaSetDevice(device_ordinal) != cudaSuccess) {
        return false;
    }

    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device_ordinal) != cudaSuccess) {
        return false;
    }
    mode_ = requested;
    if (requested == AcHistogramExchangeMode::AUTO) {
        mode_ = properties.integrated != 0 && properties.canMapHostMemory != 0
                    ? AcHistogramExchangeMode::MAPPED
                    : AcHistogramExchangeMode::STAGED;
    }
    if (mode_ == AcHistogramExchangeMode::MAPPED && properties.canMapHostMemory == 0) {
        return false;
    }

    if (cudaEventCreateWithFlags(&ready_event_, cudaEventDisableTiming) != cudaSuccess) {
        shutdown();
        return false;
    }
    if (mode_ == AcHistogramExchangeMode::MAPPED) {
        if (cudaHostAlloc(reinterpret_cast<void**>(&host_data_), HISTOGRAM_BYTES,
                          cudaHostAllocMapped) != cudaSuccess ||
            cudaHostGetDevicePointer(reinterpret_cast<void**>(&device_data_), host_data_, 0) !=
                cudaSuccess) {
            shutdown();
            return false;
        }
    } else if (cudaHostAlloc(reinterpret_cast<void**>(&host_data_), HISTOGRAM_BYTES,
                             cudaHostAllocDefault) != cudaSuccess ||
               cudaMalloc(&device_data_, HISTOGRAM_BYTES) != cudaSuccess) {
        shutdown();
        return false;
    }

    ownership_ = AcHistogramOwnership::GPU_WRITABLE;
    return true;
}

void AcHistogramExchange::shutdown() {
    if (ownership_ == AcHistogramOwnership::GPU_PENDING && ready_event_ != nullptr) {
        cudaEventSynchronize(ready_event_);
    }
    if (mode_ == AcHistogramExchangeMode::STAGED && device_data_ != nullptr) {
        cudaFree(device_data_);
    }
    if (host_data_ != nullptr) {
        cudaFreeHost(host_data_);
    }
    if (ready_event_ != nullptr) {
        cudaEventDestroy(ready_event_);
    }
    device_data_ = nullptr;
    host_data_ = nullptr;
    ready_event_ = nullptr;
    ownership_ = AcHistogramOwnership::UNINITIALIZED;
}

std::uint32_t* AcHistogramExchange::gpu_data() {
    assert(ownership_ == AcHistogramOwnership::GPU_WRITABLE);
    return device_data_;
}

bool AcHistogramExchange::release_to_cpu(cudaStream_t stream) {
    assert(ownership_ == AcHistogramOwnership::GPU_WRITABLE);
    if (mode_ == AcHistogramExchangeMode::STAGED &&
        cudaMemcpyAsync(host_data_, device_data_, HISTOGRAM_BYTES, cudaMemcpyDeviceToHost,
                        stream) != cudaSuccess) {
        return false;
    }
    if (cudaEventRecord(ready_event_, stream) != cudaSuccess) {
        cudaStreamSynchronize(stream);
        return false;
    }
    ownership_ = AcHistogramOwnership::GPU_PENDING;
    return true;
}

bool AcHistogramExchange::acquire_for_cpu() {
    assert(ownership_ == AcHistogramOwnership::GPU_PENDING);
    if (cudaEventSynchronize(ready_event_) != cudaSuccess) {
        return false;
    }
    ownership_ = AcHistogramOwnership::CPU_READABLE;
    return true;
}

const std::uint32_t* AcHistogramExchange::cpu_data() const {
    assert(ownership_ == AcHistogramOwnership::CPU_READABLE);
    return host_data_;
}

void AcHistogramExchange::release_to_gpu() {
    assert(ownership_ == AcHistogramOwnership::CPU_READABLE);
    ownership_ = AcHistogramOwnership::GPU_WRITABLE;
}

}  // namespace cujpegxl
