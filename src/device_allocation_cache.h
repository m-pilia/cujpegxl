// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_DEVICE_ALLOCATION_CACHE_H_
#define CUJPEGXL_SRC_DEVICE_ALLOCATION_CACHE_H_

#include <cuda_runtime.h>

#include <array>
#include <cstddef>

namespace cujpegxl {

class DeviceAllocationCache {
public:
    DeviceAllocationCache() {
        cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&completion_event_, cudaEventDisableTiming);
    }
    ~DeviceAllocationCache() {
        cudaStreamSynchronize(stream_);
        for (std::size_t i{0}; i < allocation_count_; ++i) {
            cudaFree(allocations_[i].pointer);
        }
        cudaEventDestroy(completion_event_);
        cudaStreamDestroy(stream_);
    }

    DeviceAllocationCache(const DeviceAllocationCache&) = delete;
    DeviceAllocationCache& operator=(const DeviceAllocationCache&) = delete;

    cudaError_t allocate(void** pointer, std::size_t bytes) {
        Allocation* replacement{nullptr};
        for (std::size_t i{0}; i < allocation_count_; ++i) {
            Allocation& allocation{allocations_[i]};
            if (!allocation.in_use && allocation.capacity >= bytes) {
                allocation.in_use = true;
                *pointer = allocation.pointer;
                return cudaSuccess;
            }
            if (!allocation.in_use &&
                (replacement == nullptr || allocation.capacity < replacement->capacity)) {
                replacement = &allocation;
            }
        }
        void* storage{nullptr};
        const cudaError_t status{cudaMalloc(&storage, bytes)};
        if (status != cudaSuccess) {
            return status;
        }
        if (replacement != nullptr) {
            cudaFree(replacement->pointer);
            *replacement = {storage, bytes, true};
            *pointer = storage;
            return cudaSuccess;
        }
        if (allocation_count_ == allocations_.size()) {
            cudaFree(storage);
            return cudaErrorMemoryAllocation;
        }
        allocations_[allocation_count_] = {storage, bytes, true};
        ++allocation_count_;
        *pointer = storage;
        return cudaSuccess;
    }

    cudaError_t release(void* pointer) {
        if (pointer == nullptr) {
            return cudaSuccess;
        }
        for (std::size_t i{0}; i < allocation_count_; ++i) {
            Allocation& allocation{allocations_[i]};
            if (allocation.pointer == pointer) {
                allocation.in_use = false;
                return cudaSuccess;
            }
        }
        return cudaErrorInvalidDevicePointer;
    }

    std::size_t allocation_count() const { return allocation_count_; }
    cudaStream_t stream() const { return stream_; }
    cudaError_t synchronize() {
        const cudaError_t record_status{cudaEventRecord(completion_event_, stream_)};
        return record_status == cudaSuccess ? cudaEventSynchronize(completion_event_)
                                            : record_status;
    }

private:
    struct Allocation {
        void* pointer;
        std::size_t capacity;
        bool in_use;
    };
    std::array<Allocation, 64> allocations_{};
    std::size_t allocation_count_{0};
    cudaStream_t stream_{nullptr};
    cudaEvent_t completion_event_{nullptr};
};

inline thread_local DeviceAllocationCache* current_device_allocation_cache{nullptr};
inline thread_local cudaStream_t current_encoder_stream{nullptr};

class ScopedDeviceAllocationCache {
public:
    explicit ScopedDeviceAllocationCache(DeviceAllocationCache& cache)
        : previous_cache_{current_device_allocation_cache},
          previous_stream_{current_encoder_stream} {
        current_device_allocation_cache = &cache;
        current_encoder_stream = cache.stream();
    }
    ~ScopedDeviceAllocationCache() {
        current_device_allocation_cache = previous_cache_;
        current_encoder_stream = previous_stream_;
    }

    ScopedDeviceAllocationCache(const ScopedDeviceAllocationCache&) = delete;
    ScopedDeviceAllocationCache& operator=(const ScopedDeviceAllocationCache&) = delete;

private:
    DeviceAllocationCache* previous_cache_;
    cudaStream_t previous_stream_;
};

inline cudaStream_t encoder_stream() {
    return current_encoder_stream;
}

inline std::size_t retained_device_allocation_count() {
    return current_device_allocation_cache == nullptr
               ? 0
               : current_device_allocation_cache->allocation_count();
}

inline cudaError_t encoder_stream_synchronize() {
    return current_device_allocation_cache == nullptr
               ? cudaStreamSynchronize(encoder_stream())
               : current_device_allocation_cache->synchronize();
}

inline cudaError_t encoder_memset(void* pointer, int value, std::size_t bytes) {
    return cudaMemsetAsync(pointer, value, bytes, encoder_stream());
}

inline cudaError_t encoder_memcpy(void* destination, const void* source, std::size_t bytes,
                                  cudaMemcpyKind kind) {
    const cudaError_t status{cudaMemcpyAsync(destination, source, bytes, kind, encoder_stream())};
    return status == cudaSuccess ? encoder_stream_synchronize() : status;
}

inline cudaError_t encoder_memcpy_2d(void* destination, std::size_t destination_pitch,
                                     const void* source, std::size_t source_pitch,
                                     std::size_t width, std::size_t height, cudaMemcpyKind kind) {
    const cudaError_t status{cudaMemcpy2DAsync(destination, destination_pitch, source, source_pitch,
                                               width, height, kind, encoder_stream())};
    return status == cudaSuccess ? encoder_stream_synchronize() : status;
}

inline cudaError_t cached_device_allocate(void** pointer, std::size_t bytes,
                                          cudaStream_t stream = nullptr) {
    if (current_device_allocation_cache != nullptr) {
        return current_device_allocation_cache->allocate(pointer, bytes);
    }
    return cudaMallocAsync(pointer, bytes, stream);
}

template <typename T>
cudaError_t cached_device_allocate(T** pointer, std::size_t bytes, cudaStream_t stream = nullptr) {
    return cached_device_allocate(reinterpret_cast<void**>(pointer), bytes, stream);
}

inline cudaError_t cached_device_release(void* pointer, cudaStream_t stream = nullptr) {
    if (current_device_allocation_cache != nullptr) {
        return current_device_allocation_cache->release(pointer);
    }
    return cudaFreeAsync(pointer, stream);
}

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_DEVICE_ALLOCATION_CACHE_H_
