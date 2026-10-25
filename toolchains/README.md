# Build toolchains

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
target Jetson.

## CUDA

The root module selects CUDA 12.6.3 redistributions for Linux x86_64 and Jetson
`linux-aarch64`, with the manifest and component archives pinned by checksum.
`linux-sbsa` is a different distribution and is not used for Jetson.

Both CUDA toolchains use the registered Clang 19.1.0 C++ compiler and x86_64 CUDA
executables. The Jetson toolchain uses ARM64 runtime libraries and target headers
along with the Jetson C++ toolchain's CPU settings and sysroot. Existing `@cuda`
library labels select the appropriate concrete toolkit for the target platform.
The root module owns this selection; downstream Bazel consumers still supply
their own C++ and CUDA toolchains.

```sh
bazelisk build --platforms=//platforms:jetson_orin_nano \
  --@rules_cuda//cuda:archs=sm_87 \
  //packaging:install_tree //tools/benchmark:encode_bench \
  //toolchains/tests:cuda_toolchain_test
```

The CUDA check consumes the x86_64-generated header, compiles a kernel and C++
standard-library code, and links the selected CUDA runtime. Its runtime check
allocates device memory, launches the kernel, and verifies the copied result;
it requires a GPU and should execute on the platform it was built for.

The install tree retains static cudart linkage. The benchmark currently also
requires ARM64 `libnvjpeg.so.12` and `libnvToolsExt.so.1`; retain its runfiles or
provide those libraries when deploying it. Native builds continue to select
x86_64 libraries and the configured `sm_61;sm_87` GPU architectures.
