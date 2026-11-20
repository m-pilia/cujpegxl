# Jetson system sysroots

Each supported JetPack release has a sysroot repository that extracts system
development files from a pinned ARM64 `stereolabs/zed` image. `MODULE.bazel`
pins each image's platform manifest:

| Repository | Image | Platform manifest |
| --- | --- | --- |
| `jetson_jp62_sysroot` | `stereolabs/zed:5.2-runtime-l4t-r36.4` | `sha256:6d1a76e8f9b9c5e1b2430b392bbb61cfe58814044be3fd77aae0577a92ddaa41` |
| `jetson_jp72_sysroot` | `stereolabs/zed:5.6-runtime-l4t-r39.2` | `sha256:47b1885ed7c38b86672be4436f18a01b58e001e396e8078031b5927ed866f3ed` |

```sh
bazelisk build @jetson_jp62_sysroot//sysroot:sysroot @jetson_jp72_sysroot//sysroot:sysroot
bazelisk test //toolchains/sysroot:prepare_sysroot_test
```

Fetching uses `rules_oci` and Bazel's downloader. A checksum-pinned umoci 0.6.0
unpacks the image with rootless OCI layer and whiteout handling. No Docker daemon,
container execution, ARM64 emulation, or system package installation is required.
Preparation runs on Linux x86_64 using the existing pinned Python runtime.

The initial fetch downloads approximately 5.72 GB (JetPack 6.2) or 11.38 GB
(JetPack 7.2) of compressed image layers and temporarily unpacks the full
filesystem. The resulting sysroots are approximately 89 MiB and 123 MiB. Image
fetching and preparation are lazy: native targets do not depend on these
repositories, and each Jetson configuration fetches only its own.

Only headers and libraries owned by the explicitly selected system packages are
materialized. Symbolic links are resolved within the image's filesystem and their
file contents copied, so no output link can resolve against the worker's root.
On merged-`/usr` images, packages own their libraries under `/usr/lib` while
linker scripts reference them through `/lib`; such linker script inputs are
copied to their `/lib` paths as well. CUDA and unrelated image libraries are
excluded. Inside the sysroot, package copyright files are retained under
`usr/share/doc/<package>/copyright`, along with the image's standard license
texts under `usr/share/common-licenses/`. These paths preserve the references in
the original copyright notices.

`@jetson_jp<release>_sysroot//sysroot:SOURCES.md` links each binary package to
the exact Ubuntu source package version on Launchpad, whose download section
provides the source archives, Ubuntu patches and packaging, and `.dsc` checksums.
Source names and versions come from the pinned image's dpkg database, including
explicit source versions that differ from binary versions. No additional network
access is needed to generate these links. When redistributing a sysroot,
retain this documentation and ensure corresponding sources remain available;
the generated links do not archive the sources.

The GCC version whose headers, startup objects and libraries are validated is
taken from the selected `libstdc++-<version>-dev` package. Preparation fails on
missing packages or files, wrong architectures, escaping or broken links,
missing startup objects/runtime libraries, or unresolved linker script inputs.
Absolute library paths in linker scripts remain target paths, resolved by the
linker through its sysroot. An incomplete source image requires review before
adding packages or changing its source.

## Validated baselines

### JetPack 6.2

The image reports Ubuntu 22.04.3 LTS. Its selected ARM64 packages are:

| Package | Version |
| --- | --- |
| libc6, libc6-dev | 2.35-0ubuntu3.8 |
| libgcc-11-dev, libstdc++-11-dev | 11.4.0-1ubuntu1~22.04 |
| libgcc-s1, libstdc++6 | 12.3.0-1ubuntu1~22.04 |
| linux-libc-dev | 5.15.0-124.134 |

The libc runtime contains symbol versions through `GLIBC_2.35`; libstdc++
contains versions through `GLIBCXX_3.4.30` and `CXXABI_1.3.13`.

### JetPack 7.2

The image reports Ubuntu 24.04.5 LTS. Its selected ARM64 packages are:

| Package | Version |
| --- | --- |
| libc6, libc6-dev | 2.39-0ubuntu8.12 |
| libgcc-13-dev, libstdc++-13-dev | 13.3.0-6ubuntu2~24.04.1 |
| libgcc-s1, libstdc++6 | 14.2.0-4ubuntu2~24.04.1 |
| linux-libc-dev | 6.8.0-149.149 |

The libc runtime contains symbol versions through `GLIBC_2.39`; libstdc++
contains versions through `GLIBCXX_3.4.33` and `CXXABI_1.3.15`.

These maxima describe each sysroot's available runtime baseline, not the minimum
requirements of every binary built against it. Inspect each final binary's
version requirements when assessing deployment compatibility.

`@jetson_jp<release>_sysroot//:abi.json` records the source manifest, OS
release, selected binary and source package versions, validated ELF count, and
runtime symbol-version maxima.
`@jetson_jp<release>_sysroot//sysroot:sysroot` declares the headers and
libraries as Bazel inputs; its package directory is the sysroot root for the
release's C++ toolchain.
