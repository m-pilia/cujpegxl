// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include <cstdio>

#if !defined(__linux__) || !defined(__x86_64__)
#error "The code generator must compile for Linux x86_64"
#endif

int main() {
    return std::printf("inline constexpr std::size_t GENERATOR_POINTER_BYTES{%zu};\n",
                       sizeof(void*)) < 0;
}
