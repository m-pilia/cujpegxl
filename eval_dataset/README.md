# Eval dataset

A ~1000-image corpus of Wikimedia Commons featured pictures
(≥ 3840×2160, ≤ 8000×8000 raster originals) for quality benchmarking. Only
metadata is committed (`sources.json`); pixel data is fetched on demand and
never enters the repository.

Pipeline (pass absolute paths: under `bazelisk run` the working directory is
the runfiles tree, not the workspace):

```bash
# 1. Query the Commons API and (re)generate the metadata snapshot.
bazelisk run //eval_dataset:query -- --num-images 1000 \
    --output $PWD/eval_dataset/sources.json

# 2. Download originals, build 4K masters, emit the rung's NV12 corpus.
#    Downloads are serial and paced (~30 s/file, so a full 1000-image run
#    takes hours); every stage is resumable via the sha-verified cache.
bazelisk run //eval_dataset:prepare_dataset -- --resolution 1080p \
    --sources $PWD/eval_dataset/sources.json \
    --cache-dir $PWD/eval_dataset/cache \
    --masters-dir $PWD/eval_dataset/masters \
    --out-dir $PWD/eval_dataset/out

# 3. Benchmark it.
bazelisk run //tools/quality_benchmark:quality_benchmark -- \
    --distances 0.5 0.7 1.0 1.3 1.6 2.0 2.5 3.0 4.0 \
    --qualities 20 30 40 50 60 70 75 80 85 90 95 100 \
    --data-dir $PWD/eval_dataset/masters \
    --resolution 1080p \
    --output $PWD/tmp/sweep_1080p_eval_dataset.json

# 4. Compare the codecs on the sweep results.
bazelisk run //tools/quality_benchmark:result_analysis -- \
    --input $PWD/tmp/sweep_1080p_eval_dataset.json \
    --output-dir $PWD/tmp
```

Layout after a run:

- `cache/` — original downloads, kept for resume (~8 GB for 1000 images).
- `masters/` — 4K PNG masters (~10 GB); drop-in equivalent of `//data`,
  usable as any tool's `--data-dir` of source PNGs (e.g. `//tools/corpus`,
  `//tools/quality_benchmark`).
- `out/` — `<name>_<rung>.nv12` frames plus `corpus_manifest.json`, following
  the `tools/corpus` conventions (BT.709, full range, 4:2:0), and
  `ATTRIBUTIONS.txt` (CC-BY license obligations travel with the data).

All three directories are gitignored.
