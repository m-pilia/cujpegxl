# Code checks

```
run: bazelisk test --config=<asan,ubsan,tsan> //...
bazelisk build --config=clang-tidy //...
bazelisk run //tools/format:format.check
bazelisk run //tools/format
```
