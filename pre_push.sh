#!/usr/bin/env bash

set -euo pipefail

bazelisk build //...
bazelisk test //...
bazelisk test --config=asan //... --test_tag_filters=-gpu,-no-sanitizers
bazelisk test --config=ubsan //... --test_tag_filters=-gpu,-no-sanitizers
bazelisk test --config=tsan //... --test_tag_filters=-gpu,-no-sanitizers
bazelisk build --config=clang-tidy //...
bazelisk run //tools/format
