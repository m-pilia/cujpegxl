#!/usr/bin/env bash

set -euo pipefail

source_dir=$(cd "$(dirname "$1")" && pwd)
output_path="$PWD/$2"
cuda_arch="$3"
build_dir="$(dirname "$output_path")/gjxl_cmake"

cmake -S "$source_dir" -B "$build_dir" -G "Unix Makefiles" \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_CUDA_COMPILER=/opt/cuda/bin/nvcc \
    -DCMAKE_CUDA_HOST_COMPILER="/usr/bin/g++-14" \
    -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="-Wno-error=sign-compare -march=native" \
    -DCMAKE_CUDA_FLAGS="-Xcompiler=-march=native" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DGJXL_ENABLE_CUDA=ON \
    -DGJXL_ENABLE_METAL=OFF \
    -DGJXL_BUILD_TESTS=OFF \
    -DGJXL_BUILD_BENCHMARKS=OFF \
    -DGJXL_ENABLE_LIBJXL_REFERENCE=OFF

cmake --build "$build_dir" --target gjxl_c --parallel 4

/usr/bin/g++ -shared -Wl,-z,defs -Wl,--whole-archive \
    "$build_dir/libgjxl.a" \
    "$build_dir/libgjxl_codestream.a" \
    "$build_dir/libgjxl_cuda.a" \
    "$build_dir/libgjxl_gpu_ops.a" \
    "$build_dir/libgjxl_gpu_butteraugli.a" \
    "$build_dir/libgjxl_gpu.a" \
    "$build_dir/libgjxl_codec.a" \
    -Wl,--no-whole-archive \
    -L/opt/cuda/targets/x86_64-linux/lib -lcudart -pthread \
    -Wl,-rpath,/opt/cuda/targets/x86_64-linux/lib \
    -o "$output_path"

rm -rf "$build_dir"
