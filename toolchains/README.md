# C++ toolchains

The root module registers two C++ toolchains from the pinned Clang 19.1.0
distribution. Both execute their compiler and linker on Linux x86_64.

| Target platform | CPU | System and C++ libraries |
| --- | --- | --- |
| `//platforms:linux_x86_64` | x86_64 | Existing host libstdc++ setup |
| `//platforms:jetson_orin_nano` | Cortex-A78AE, AArch64 | Pinned Jetson sysroot, GCC 11 headers, dynamic libstdc++ |

Linux x86_64 is the default host and target platform. The Jetson toolchain is
constrained by `//platforms:jetson_linux` as well as Linux and AArch64, so its
sysroot and CPU settings do not apply to arbitrary ARM64 platforms. The sysroot
is declared as an input to both compilation and linking; target headers and
libraries are selected from it rather than from the worker's system directories.

Select the Jetson platform with:

```sh
bazelisk build --platforms=//platforms:jetson_orin_nano //toolchains/tests:cpp_toolchain_test
```

The check builds and executes an x86_64 C++ code generator, then compiles a test
using the generated header, threads, exceptions, and standard containers for the
selected target. The ARM64 executable can be run on Jetson. Run the same check
natively with:

```sh
bazelisk test //toolchains/tests:cpp_toolchain_test
```

Build-time tools use the x86_64 execution platform even when their consumers
target Jetson. CUDA compilation additionally requires a matching CUDA target
toolchain; the C++ configuration alone does not provide Jetson CUDA libraries.
