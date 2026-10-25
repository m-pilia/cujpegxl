# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

def _oci_sysroot_impl(ctx):
    if ctx.os.name != "linux" or ctx.os.arch not in ["amd64", "x86_64"]:
        fail("Sysroot preparation requires a Linux x86_64 host")
    image = ctx.path(ctx.attr.image).dirname
    python = ctx.path(ctx.attr.python)
    umoci = ctx.path(ctx.attr.umoci)
    prepare = ctx.path(ctx.attr._prepare)
    for input in [ctx.attr.image, ctx.attr.python, ctx.attr.umoci, ctx.attr._prepare]:
        ctx.watch(input)
    ctx.report_progress("Extracting and validating the Jetson system sysroot")
    result = ctx.execute(
        [str(python), "-I", str(prepare), "--image", str(image), "--umoci", str(umoci)] + ctx.attr.packages,
        timeout = 1800,
        quiet = False,
    )
    if result.return_code:
        fail("Sysroot preparation failed:\n" + result.stderr)
    ctx.file("BUILD.bazel", """\
package(default_visibility = ["//visibility:public"])

exports_files(["abi.json"])
""")
    ctx.file("sysroot/BUILD.bazel", """\
package(default_visibility = ["//visibility:public"])

exports_files(["SOURCES.md"])

filegroup(
    name = "sysroot",
    srcs = glob(["**"], exclude = ["BUILD.bazel"], allow_empty = False),
)
""")
    return ctx.repo_metadata(reproducible = True)

oci_sysroot = repository_rule(
    implementation = _oci_sysroot_impl,
    attrs = {
        "image": attr.label(mandatory = True, allow_single_file = True),
        "packages": attr.string_list(mandatory = True),
        "python": attr.label(mandatory = True, allow_single_file = True),
        "umoci": attr.label(mandatory = True, allow_single_file = True),
        "_prepare": attr.label(default = ":prepare_sysroot.py", allow_single_file = True),
    },
)
