// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_DEVICE_SUPPORT_H_
#define CUJPEGXL_SRC_DEVICE_SUPPORT_H_

#include <cstdint>

namespace cujpegxl {

enum class DeviceSupport : std::uint8_t { SUPPORTED, NO_DEVICE, UNSUPPORTED };

// UNSUPPORTED when the library holds no code image that can run on the device
// (neither a compatible binary nor PTX it can JIT). Makes `ordinal` the
// current device of the calling thread when it exists.
DeviceSupport query_device_support(std::int32_t ordinal);

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_DEVICE_SUPPORT_H_
