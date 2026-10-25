# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

def _jetson_transition_impl(settings, attr):
    return {
        "//command_line_option:features": [
            feature
            for feature in settings["//command_line_option:features"]
            if feature not in ["asan", "ubsan", "tsan"]
        ],
        "//command_line_option:platforms": ["//platforms:jetson_orin_nano"],
        "@rules_cuda//cuda:archs": "sm_87",
    }

_jetson_transition = transition(
    implementation = _jetson_transition_impl,
    inputs = ["//command_line_option:features"],
    outputs = [
        "//command_line_option:features",
        "//command_line_option:platforms",
        "@rules_cuda//cuda:archs",
    ],
)

def _jetson_binary_impl(ctx):
    executable = ctx.attr.binary[0][DefaultInfo].files_to_run.executable
    return [DefaultInfo(files = depset([executable]))]

jetson_binary = rule(
    implementation = _jetson_binary_impl,
    attrs = {
        "binary": attr.label(
            cfg = _jetson_transition,
            executable = True,
            mandatory = True,
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)
