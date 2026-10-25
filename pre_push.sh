#!/usr/bin/env bash

set -euo pipefail

function echo_and_run() {
    echo
    echo "\$" "$@"
    echo
    "$@"
}

echo_and_run bazelisk run //tools/format:format.check
echo_and_run bazelisk build //...
echo_and_run bazelisk build --config=jetson //packaging:install_tree //tools/benchmark:encode_bench
echo_and_run bazelisk test //...
echo_and_run bazelisk test --config=asan //... --test_tag_filters=-gpu,-no-sanitizers
echo_and_run bazelisk test --config=ubsan //... --test_tag_filters=-gpu,-no-sanitizers
echo_and_run bazelisk test --config=tsan //... --test_tag_filters=-gpu,-no-sanitizers
echo_and_run bazelisk build --config=clang-tidy //...
