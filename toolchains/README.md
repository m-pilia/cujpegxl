# Build toolchains

The root module registers C++ toolchains from the pinned Clang 23.1.6
distribution for three target platforms. All of them execute their compiler and
linker on Linux x86_64.

| Target platform | Config | CPU | System and C++ libraries |
| --- | --- | --- | --- |
| `//platforms:linux_x86_64` | (default) | x86_64 | Existing host libstdc++ setup |
| `//platforms:jetson_orin_nano_jp62` | `jetson_jp62` | Cortex-A78AE, AArch64 | JetPack 6.2 sysroot (Ubuntu 22.04), GCC 11 headers, dynamic libstdc++ |
| `//platforms:jetson_orin_nano_jp72` | `jetson_jp72` | Cortex-A78AE, AArch64 | JetPack 7.2 sysroot (Ubuntu 24.04), GCC 13 headers, dynamic libstdc++ |

Linux x86_64 is the default host and target platform. Each Jetson toolchain is
constrained by its release's `//platforms:jetson_jp62` or
`//platforms:jetson_jp72` value as well as Linux and AArch64, so its sysroot
and CPU settings do not apply to the other release or to arbitrary ARM64
platforms. The JetPack 7.2 toolchain is a second `toolchains_llvm` instance on
the same distribution, because sysroot and C++ standard library are configured
per target. The sysroot is declared as an input to both compilation and linking;
target headers and libraries are selected from it rather than from the worker's
system directories.

Select a Jetson platform with its config, e.g.:

```sh
bazelisk build --config=jetson_jp72 //toolchains/tests:cpp_toolchain_test
```

The check builds and executes an x86_64 C++ code generator, then compiles a test
using the generated header, threads, exceptions, and standard containers for the
selected target. The ARM64 executable can be run on a Jetson device with the
matching JetPack release. Run the same check natively with:

```sh
bazelisk test //toolchains/tests:cpp_toolchain_test
```

Build-time tools use the x86_64 execution platform even when their consumers
target Jetson.

## CUDA

The root module declares two CUDA releases behind the common `@cuda` labels, with
the manifests and component archives pinned by checksum:

| Release | Used by | Target redistribution |
| --- | --- | --- |
| 12.6.3 | Native, JetPack 6.2 | `linux-x86_64`, Jetson `linux-aarch64` |
| 13.2.1 | JetPack 7.2 | `linux-sbsa` |

CUDA 13 no longer supports `sm_61`, so native builds stay on 12.6.3. NVIDIA
publishes no `linux-aarch64` redistribution from CUDA 13 on; JetPack 7.2 builds
use the `linux-sbsa` target components.

`--@rules_cuda//cuda:version` selects the release. Without it, rules_cuda would
use the highest declared release, so `.bazelrc` sets 12.6.3 for all builds and
`--config=jetson_jp72` overrides it with 13.2.1. On AArch64,
`--@rules_cuda//cuda:aarch64` chooses between `linux-aarch64` (`jetson_jp62`)
and `linux-sbsa` (`jetson_jp72`). Existing `@cuda` library labels select the
components of the configured release and target platform.

CUDA code is compiled by the registered Clang 23.1.6 C++ compiler with the
release's x86_64 CUDA executables. Jetson builds use ARM64 runtime libraries and
target headers along with the Jetson C++ toolchain's CPU settings and sysroot.
The root module owns this selection; downstream Bazel consumers still supply
their own C++ and CUDA toolchains.

```sh
bazelisk build --config=jetson_jp62 \
  //packaging:install_tree //tools/benchmark:encode_bench \
  //toolchains/tests:cuda_toolchain_test
bazelisk build --config=jetson_jp72 \
  //packaging:install_tree //tools/benchmark:encode_bench \
  //toolchains/tests:cuda_toolchain_test
```

The CUDA check consumes the x86_64-generated header, compiles a kernel and C++
standard-library code, and links the selected CUDA runtime. Its runtime check
allocates device memory, launches the kernel, and verifies the copied result;
it requires a GPU and should execute on the platform it was built for.

Both Jetson configs select their platform, CUDA release and redistribution, and
`sm_87` for project kernels.
The install tree retains its relocatable layout and static cudart linkage.
The Jetson benchmark links cudart, nvJPEG, and its culibos support library
statically and uses the NVTX 3 headers for profiling. It can be copied as a single
executable without Bazel runfiles or CUDA/NVTX shared libraries. The device must
provide the system C/C++ libraries and NVIDIA driver. NVIDIA ships no CUDA 13
nvJPEG for Tegra, so the JetPack 7.2 benchmark links the `linux-sbsa` nvJPEG,
which is not supported on the device; its nvJPEG comparison is not run there.

Native builds continue to select x86_64 libraries and the configured
`sm_61;sm_87` GPU architectures, preserving their shared nvJPEG and NVTX linkage.

Check each Jetson benchmark's architecture, absence of CUDA/NVTX shared-library
dependencies, and absence of RPATH/RUNPATH on an x86_64 worker without a GPU:

```sh
bazelisk test //tools/benchmark:jetson_jp62_linkage_test //tools/benchmark:jetson_jp72_linkage_test
```

Each test cross-builds its benchmark dependency for its release and inspects the
ELF with the pinned LLVM reader without executing the benchmark.
The Jetson dependencies disable the `asan`, `ubsan`, and `tsan` features because
the compiler distribution does not include AArch64 sanitizer runtimes. Other
targets retain their configured sanitizer features.
