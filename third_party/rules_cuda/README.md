# rules_cuda patches

The pinned `rules_cuda` 0.4.0 selects CUDA redistributions by target platform,
build-host platform and toolkit version on its own: `cuda.redist_json(platforms
= ...)` declares the platforms, `@cuda` labels resolve to the target platform's
components, and compiler executables resolve to the execution platform's.
`--@rules_cuda//cuda:version` chooses among the declared releases. On AArch64,
`--@rules_cuda//cuda:aarch64` chooses between `linux-sbsa` and the Jetson
`linux-aarch64` redistribution.

Three patches cover what the release does not, applied in this order:

- `nvjpeg-static-deps.patch`: `@cuda//:nvjpeg_static` declares its cudart and
  culibos dependencies, so consumers of the static library receive its complete
  link requirements through the existing `@cuda` interface.
- `clang-cuda-root.patch`: a backport of the upstream fix for
  [bazel-contrib/rules_cuda#532](https://github.com/bazel-contrib/rules_cuda/issues/532).
  With redistribution components, `--cuda-path` points at the nvcc component,
  which lacks `include/cuda.h`. Clang reads the toolkit version only from that
  header, so it assumes the newest CUDA it knows and emits PTX that an older
  `ptxas` rejects. The patch assembles a CUDA root at build time from the
  execution platform's `ptxas`, `fatbinary` and libdevice and the target's
  `cuda.h`, and passes it to Clang. Delete it once a rules_cuda release
  contains the fix.
- `mixed-version-culibos.patch`: CUDA 12 ships `libculibos.a` in cudart, CUDA 13
  in its own culibos component. With both releases declared, the `@cuda` facade
  aliases `culibos_a` to the culibos component only, which is empty under
  CUDA 12. The patch makes `@cuda//:culibos_a` depend on both owners, so it
  resolves for either release.

`examples/bazel_consumer` is a root module of its own, so it repeats the
override; its patch files link to these.

`//toolchains/tests:cuda_toolchain_test` exercises a host code generator, target
C++ headers, a CUDA kernel, and runtime linking. Build it natively and for both
Jetson releases; GPU execution requires the corresponding hardware and driver.
