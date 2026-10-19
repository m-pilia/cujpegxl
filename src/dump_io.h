// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#ifndef CUJPEGXL_SRC_DUMP_IO_H_
#define CUJPEGXL_SRC_DUMP_IO_H_

#include <cstddef>
#include <cstdio>
#include <vector>

namespace cujpegxl {

// Reads exactly `count` raw T values from `path`.
template <typename T>
bool read_raw(const char* path, std::size_t count, std::vector<T>* out) {
    out->resize(count);
    std::FILE* file{std::fopen(path, "rb")};
    if (file == nullptr) {
        return false;
    }
    const std::size_t got{std::fread(out->data(), sizeof(T), count, file)};
    std::fclose(file);
    return got == count;
}

template <typename T>
bool write_raw(const char* path, const T* data, std::size_t count) {
    std::FILE* file{std::fopen(path, "wb")};
    if (file == nullptr) {
        return false;
    }
    const std::size_t put{std::fwrite(data, sizeof(T), count, file)};
    std::fclose(file);
    return put == count;
}

}  // namespace cujpegxl

#endif  // CUJPEGXL_SRC_DUMP_IO_H_
