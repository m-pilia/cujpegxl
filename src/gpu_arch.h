// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_GPU_ARCH_H_
#define CUJPEGXL_SRC_GPU_ARCH_H_

#include <cstdint>

namespace cujpegxl {

// The architectures with their own kernel tuning; all others use DEFAULT.
enum class GpuArch : std::uint8_t { DEFAULT, SM87 };

// `virtual_arch` is the compute capability the running device code was
// compiled for, as major * 10 + minor: cudaFuncAttributes::ptxVersion on the
// host, __CUDA_ARCH__ / 10 on the device. It differs from the device's compute
// capability when the driver JIT-compiles an older PTX image.
constexpr GpuArch gpu_arch(std::int32_t virtual_arch) {
    return virtual_arch == 87 ? GpuArch::SM87 : GpuArch::DEFAULT;
}

// For device code only: host code takes the GpuArch of the running device code
// (query_device_support) instead, and sees DEFAULT here.
#ifdef __CUDA_ARCH__
inline constexpr GpuArch COMPILED_ARCH{gpu_arch(__CUDA_ARCH__ / 10)};
#else
inline constexpr GpuArch COMPILED_ARCH{GpuArch::DEFAULT};
#endif

// A kernel's per-architecture constants. The device reads its entry with
// COMPILED_ARCH and the host with the running code's GpuArch, so both sides
// agree on the values.
template <typename T>
struct PerArch {
    T default_entry;
    T sm87;

    constexpr const T& operator[](GpuArch arch) const {
        return arch == GpuArch::SM87 ? sm87 : default_entry;
    }
};

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_GPU_ARCH_H_
