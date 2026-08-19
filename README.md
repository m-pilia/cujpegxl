# cujpegxl - CUDA-accelerated JPEG XL encoder

[![CI](https://github.com/m-pilia/cujpegxl/actions/workflows/ci.yaml/badge.svg)](https://github.com/m-pilia/cujpegxl/actions/workflows/ci.yaml)

This project is a CUDA port of a JPEG XL encoder, implementing a GPU- and real-time-friendly subset of the JPEG XL algorithm.

## Goals

* Conformance. The encoded image must be readable by any standard-conforming decoder.
* Real-time performance with 4K resolution inputs.

## Status

The project is still in an early development stage, and further iterations are required to improve compression quality, runtime performance, feature support, and general robustness.

- Current quality/size ratio is somewhat halfway between JPEG and the reference JPEG XL. Generally better than JPEG, especially at medium compression settings (5-15% better compression at comparable quality), but inferior to a full JPEG XL implementation (about 40-50% higher bits per pixel at comparable quality).
- Performance is currently limited (it achieves sustained 15 fps at 4K resolution on a GTX 1080Ti). The future ambition would be to sustain 15 fps at 4K resolution using a limited slice (ideally <10%) of an embedded/constrained device (e.g. a Jetson Orin Nano).

## Usage

See examples for usage in Bazel ([examples/bazel_consumer](https://github.com/m-pilia/cujpegxl/tree/master/examples/bazel_consumer)) and CMake ([examples/cmake_consumer](https://github.com/m-pilia/cujpegxl/tree/master/examples/cmake_consumer)) projects.

Python bindings are provided in [//python:pycujpegxl](https://github.com/m-pilia/cujpegxl/blob/master/python/BUILD.bazel). Currently they take host buffers as input and are meant primarily for evaluation, not for high-performance applications.

## Features

Quality and compression rate are lower compared to the reference libjxl implementation. This is expected, as cujpegxl only implements a subset of the JPEG XL algorithm, excluding some parts of it (in particular, some GPU-unfriendly steps are intentionally omitted for performance reasons).

Notable exclusions include:
- Entropy coding is prefix-only. [ANS coding](https://en.wikipedia.org/wiki/Asymmetric_numeral_systems) is not implemented.
- LZ77 pre-pass is omitted.
- No non-square, <8 and >32 transforms.
- No iterative Butteraugli rate control.
- No optimized coefficient orders.

Some numerical approximations are made for performance reasons, trading a bit of quality for speed:
- Huffman codes limited to 15 bit.
- Transform selection uses a simpler heuristic.
- Fuzzy erosion is local (clamped to a 32px tile) as opposed to global.
- Fixed-size 32-cluster map, with one histogram per group, as opposed to the full greedy cost-driven clustering of 7425 contexts performed by libjxl.
- DCT coefficients stored as FP16.
- Uniform field quant calibration.
- Fixed-strength inverse Gaborish pre-sharpening.
- Simplified CfL estimation.

Some features are not implemented:
- Modular mode and lossless compression.
- Multi-pass encoding.
- Reference frames.
- Input pixel formats other than YUV NV12.
- ICC colour profiles.

## Evaluation tools

A small development set of freely licensed images is bundled in [//data:frames](https://github.com/m-pilia/cujpegxl/tree/master/data). See [data/LICENSE.txt](https://github.com/m-pilia/cujpegxl/blob/master/data/LICENSE.txt) for credits and respective licenses.

The `//eval_dataset` package implements tooling to generate a reproducible evaluation dataset using high-quality and free-license images, while `//tools/quality_benchmark` implements comparative compression quality benchmarking, see [eval_dataset/README.md](https://github.com/m-pilia/cujpegxl/blob/master/eval_dataset/README.md) for information.

Evaluation results using a 100 image subset of the evaluation dataset at 1080p resolution (`bazelisk run //eval_dataset:prepare_dataset -- --resolution 1080p --limit 100 ...`):

![ssimulacra2](doc/2026-08-19_ssimulacra2.png)

## Code checks

To run tests and various code checks:

```sh
# Including GPU tests
bazelisk test //...

# Excluding GPU tests
bazelisk test //... --test_tag_filters=-gpu

# Sanitizers
bazelisk test --config=asan //... --test_tag_filters=-gpu,-no-sanitizers
bazelisk test --config=ubsan //... --test_tag_filters=-gpu,-no-sanitizers
bazelisk test --config=tsan //... --test_tag_filters=-gpu,-no-sanitizers

# Code style
bazelisk build --config=clang-tidy //...
bazelisk run //tools/format:format.check

# Fix formatting
bazelisk run //tools/format
```

## Credits

The implementation is a derivative work of libjxl, see the License section below for details.

The source code of this project is almost entirely LLM-generated, using a mix of GLM-5.3, GLM-5.2, GPT-5.6 Sol, and Claude Opus 4.8.

## License

The cujpegxl software is made available under a [BSD-3-Clause license](https://raw.githubusercontent.com/m-pilia/cujpegxl/c369abf5f4436c51bab95635b6ac3ca4ad2a7244/LICENSE?token=GHSAT0AAAAAAD7IWYDCTWYTLNESHJQFQP5S2UFY2GA).

This software is a [derivative work](https://en.wikipedia.org/wiki/Derivative_work) of [libjxl](https://github.com/libjxl/libjxl), the reference JPEG XL implementation, which is made available by its authors [under the same license type](https://github.com/libjxl/libjxl/blob/bdde644b94c125a15e532b2572b96306371a7d4e/LICENSE).

For additional patent rights, see the [PATENTS file in the libjxl source codebase](https://raw.githubusercontent.com/libjxl/libjxl/bdde644b94c125a15e532b2572b96306371a7d4e/PATENTS).
