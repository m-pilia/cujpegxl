# Jetson system sysroot

The `jetson_sysroot` repository extracts system development files from the ARM64
`stereolabs/zed:5.2-runtime-l4t-r36.4` image. `MODULE.bazel` pins its platform
manifest to
`sha256:6d1a76e8f9b9c5e1b2430b392bbb61cfe58814044be3fd77aae0577a92ddaa41`.

```sh
bazelisk build @jetson_sysroot//sysroot:sysroot
bazelisk test //toolchains/sysroot:prepare_sysroot_test
```

Fetching uses `rules_oci` and Bazel's downloader. A checksum-pinned umoci 0.6.0
unpacks the image with rootless OCI layer and whiteout handling. No Docker daemon,
container execution, ARM64 emulation, or system package installation is required.
Preparation runs on Linux x86_64 using the existing pinned Python runtime.

The initial fetch downloads approximately 5.72 GB of compressed image layers and
temporarily unpacks the full filesystem. The resulting sysroot is approximately
89 MiB. Image fetching and preparation are lazy: native targets do not depend on
this repository.

Only headers and libraries owned by the explicitly selected system packages are
materialized. Symbolic links are resolved within the image's filesystem and their
file contents copied, so no output link can resolve against the worker's root.
CUDA and unrelated image libraries are excluded. Inside the sysroot, package
copyright files are retained under `usr/share/doc/<package>/copyright`, along
with the image's standard license texts under `usr/share/common-licenses/`.
These paths preserve the references in the original copyright notices.

`@jetson_sysroot//sysroot:SOURCES.md` links each binary package to the exact
Ubuntu source package version on Launchpad, whose download section provides
the source archives, Ubuntu patches and packaging, and `.dsc` checksums.
Source names and versions come from the pinned image's dpkg database, including
explicit source versions that differ from binary versions. No additional network
access is needed to generate these links. When redistributing the sysroot,
retain this documentation and ensure corresponding sources remain available;
the generated links do not archive the sources.

Preparation fails on missing packages or files, wrong architectures, escaping or
broken links, missing startup objects/runtime libraries, or unresolved linker
script inputs. Absolute library paths in linker scripts remain target paths,
resolved by the linker through its sysroot. An incomplete source image requires
review before adding packages or changing its source.

## Validated baseline

The image reports Ubuntu 22.04.3 LTS. Its selected ARM64 packages are:

| Package | Version |
| --- | --- |
| libc6, libc6-dev | 2.35-0ubuntu3.8 |
| libgcc-11-dev, libstdc++-11-dev | 11.4.0-1ubuntu1~22.04 |
| libgcc-s1, libstdc++6 | 12.3.0-1ubuntu1~22.04 |
| linux-libc-dev | 5.15.0-124.134 |

The libc runtime contains symbol versions through `GLIBC_2.35`; libstdc++
contains versions through `GLIBCXX_3.4.30` and `CXXABI_1.3.13`. These describe the
sysroot's available runtime baseline, not the minimum requirements of every
binary built against it. Inspect each final binary's version requirements when
assessing deployment compatibility.

`@jetson_sysroot//:abi.json` records the source manifest, OS release, selected
binary and source package versions, validated ELF count, and runtime
symbol-version maxima.
`@jetson_sysroot//sysroot:sysroot` declares the headers and libraries as Bazel
inputs; its package directory is the sysroot root for subsequent C++ toolchain
configuration.
