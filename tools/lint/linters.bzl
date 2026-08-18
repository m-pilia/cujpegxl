# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

load("@aspect_rules_lint//lint:clang_tidy.bzl", "lint_clang_tidy_aspect")

clang_tidy = lint_clang_tidy_aspect(
    binary = Label("@llvm_toolchain//:clang-tidy"),
    configs = [Label("//:.clang-tidy")],
)
