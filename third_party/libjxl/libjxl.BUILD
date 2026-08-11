# Copyright (c) 2026 Martino Pilia
# SPDX-License-Identifier: BSD-3-Clause

# Testonly validation oracle: pinned libjxl v0.11.2 compiled from source against
# the system highway/brotli/lcms2 so the dumper can reach internal encoder
# symbols (ToXYB, forward DCT) that the system shared library does not export.

load("@rules_cc//cc:defs.bzl", "cc_library")
load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
)

cmake(
    name = "jxl_build",
    build_args = ["-j"],
    cache_entries = {
        "BUILD_SHARED_LIBS": "OFF",
        "BUILD_TESTING": "OFF",
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_POSITION_INDEPENDENT_CODE": "ON",
        # System highway/brotli/lcms2 are found directly; do not run libjxl's
        # submodule/testdata provisioning script.
        "PROVISION_DEPENDENCIES": "OFF",
        "JPEGXL_FORCE_SYSTEM_HWY": "ON",
        "JPEGXL_FORCE_SYSTEM_BROTLI": "ON",
        "JPEGXL_FORCE_SYSTEM_LCMS2": "ON",
        "JPEGXL_ENABLE_SKCMS": "OFF",
        "JPEGXL_ENABLE_TOOLS": "OFF",
        "JPEGXL_ENABLE_JNI": "OFF",
        "JPEGXL_ENABLE_MANPAGES": "OFF",
        "JPEGXL_ENABLE_BENCHMARK": "OFF",
        "JPEGXL_ENABLE_EXAMPLES": "OFF",
        "JPEGXL_ENABLE_DOXYGEN": "OFF",
        "JPEGXL_ENABLE_PLUGINS": "OFF",
        "JPEGXL_ENABLE_DEVTOOLS": "OFF",
        "JPEGXL_ENABLE_OPENEXR": "OFF",
        "JPEGXL_ENABLE_TRANSCODE_JPEG": "OFF",
        "JPEGXL_ENABLE_SJPEG": "OFF",
        "JPEGXL_ENABLE_JPEGLI": "OFF",
        "JPEGXL_BUNDLE_LIBPNG": "OFF",
        "JPEGXL_WARNINGS_AS_ERRORS": "OFF",
    },
    lib_source = ":all_srcs",
    out_static_libs = [
        "libjxl.a",
        "libjxl_cms.a",
        "libjxl_threads.a",
    ],
    tags = ["no-sandbox"],
)

# Internal (non-installed) libjxl headers, rooted so that libjxl's own
# repo-root-relative includes (e.g. "lib/jxl/enc_xyb.h") resolve.
#
# NDEBUG must match the Release jxl_build: JXL_IS_DEBUG_BUILD toggles an extra
# virtual (Fields::Name) into every Fields-derived vtable, so a consumer that
# compiles these headers without NDEBUG gets an ABI-incompatible vtable layout
# from libjxl.a and corrupts encoder state. Propagated to all dependents.
cc_library(
    name = "internal_headers",
    hdrs = glob(
        ["lib/**/*.h"],
        allow_empty = False,
    ),
    defines = ["NDEBUG"],
    includes = ["."],
)

cc_library(
    name = "jxl",
    linkopts = [
        "-lhwy",
        "-lbrotlienc",
        "-lbrotlidec",
        "-lbrotlicommon",
        "-llcms2",
        "-lm",
    ],
    deps = [
        ":internal_headers",
        ":jxl_build",
    ],
)

# SSIMULACRA2 reference implementation. Lives under tools/ in the libjxl tree
# and is not compiled into libjxl.a, so the sources are compiled here against
# the already-built static libjxl and its internal headers. gauss_blur.cc uses
# highway's per-target compilation; it resolves to the compile-time target when
# built as a single translation unit.
cc_library(
    name = "ssimulacra2",
    testonly = True,
    srcs = [
        "tools/gauss_blur.cc",
        "tools/no_memory_manager.cc",
        "tools/ssimulacra2.cc",
    ],
    hdrs = glob(["tools/*.h"]),
    includes = ["."],
    visibility = ["//visibility:public"],
    deps = [
        ":internal_headers",
        ":jxl",
    ],
)
