// Copyright (c) 2026 Martino Pilia
// SPDX-License-Identifier: BSD-3-Clause

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <vector>

#include "toolchains/tests/generated.h"

static_assert(GENERATOR_POINTER_BYTES == 8);

int main() {
    std::vector<std::uint32_t> values{1, 2};
    std::thread worker{[&values] { values.push_back(3); }};
    worker.join();
    try {
        static_cast<void>(values.at(values.size()));
        return 1;
    } catch (const std::out_of_range&) {
        return values.size() != 3;
    }
}
