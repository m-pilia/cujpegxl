# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

_RELEASES = {
    "jp62": {
        "//command_line_option:platforms": ["//platforms:jetson_orin_nano_jp62"],
        "@rules_cuda//cuda:aarch64": "aarch64",
        "@rules_cuda//cuda:version": "12.6.3",
    },
    "jp72": {
        "//command_line_option:platforms": ["//platforms:jetson_orin_nano_jp72"],
        "@rules_cuda//cuda:aarch64": "sbsa",
        "@rules_cuda//cuda:version": "13.2.1",
    },
}

def _jetson_transition_impl(settings, attr):
    return dict(
        _RELEASES[attr.release],
        **{
            "//command_line_option:features": [
                feature
                for feature in settings["//command_line_option:features"]
                if feature not in ["asan", "ubsan", "tsan"]
            ],
            "@rules_cuda//cuda:archs": "sm_87",
        }
    )

_jetson_transition = transition(
    implementation = _jetson_transition_impl,
    inputs = ["//command_line_option:features"],
    outputs = [
        "//command_line_option:features",
        "//command_line_option:platforms",
        "@rules_cuda//cuda:aarch64",
        "@rules_cuda//cuda:archs",
        "@rules_cuda//cuda:version",
    ],
)

def _jetson_binary_impl(ctx):
    executable = ctx.attr.binary[0][DefaultInfo].files_to_run.executable
    return [DefaultInfo(files = depset([executable]))]

# Rule to simplify implementation  of transition-based tests
jetson_binary = rule(
    implementation = _jetson_binary_impl,
    attrs = {
        "binary": attr.label(
            cfg = _jetson_transition,
            executable = True,
            mandatory = True,
        ),
        "release": attr.string(
            mandatory = True,
            values = _RELEASES.keys(),
        ),
        "_allowlist_function_transition": attr.label(
            default = "@bazel_tools//tools/allowlists/function_transition_allowlist",
        ),
    },
)
