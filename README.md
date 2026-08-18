# Code checks

```
bazelisk build --config=clang-tidy //...
bazelisk run //tools/format:format.check
bazelisk run //tools/format
```
