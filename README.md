# Code checks

```
run: bazelisk test --config=<asan,ubsan,tsan> //... --test_tag_filters=-gpu,-no-sanitizers
bazelisk build --config=clang-tidy //...
bazelisk run //tools/format:format.check
bazelisk run //tools/format
```
