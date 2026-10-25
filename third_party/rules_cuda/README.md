# CUDA cross-compilation support

`cross-compilation.patch` extends the pinned `rules_cuda` 0.3.0 module extension
and toolkit repository rules without changing application rule interfaces.

- `cuda.redist_json(platform = ...)` selects a NVIDIA redistribution platform.
  Component repository names are scoped by the redistribution name to prevent
  native and target archives with identical component versions from colliding.
- Concrete `cuda.toolkit` declarations select `redist` for target components and
  optionally `exec_redist` for compiler executables. Both must have the same CUDA
  release version. Toolchain execution and target constraints are explicit.
- `cuda.toolkit(target_toolkits = ..., default_toolkit = ...)` provides a common
  library interface using target `config_setting` labels. Its concrete toolkits
  must have the same version and component set. The default toolkit supplies the
  standard automatically registered toolchain aliases; additional concrete
  toolchains are registered by the root module.

The common `@cuda` interface keeps runtime libraries and header dependencies
consistent throughout the graph. Concrete toolchain compiler inputs refer to
their own repositories, so an execution transition cannot accidentally select
native CUDA runtime data for a cross-compilation action. Compiler tools and the
compiler root use the execution redistribution; CPU libraries and target headers
use the target redistribution. Existing native NVCC constraint defaults are
preserved when callers omit explicit constraints.

The project configures x86_64 and Jetson toolchains from the same release using
these general settings. `//toolchains/tests:cuda_toolchain_test` exercises a host
code generator, target C++ headers, a CUDA kernel, and runtime linking. Build it
for both supported platforms; GPU execution requires the corresponding hardware
and driver.
