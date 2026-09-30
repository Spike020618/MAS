# Movie Consensus Benchmark

This benchmark is built from the official ReDial train/test files and a frozen
MovieLens 25M catalog. ReDial is CC BY 4.0. MovieLens use is governed by the
license distributed with the dataset.

- Development cases come only from ReDial train.
- Formal cases come only from ReDial test.
- Binary relevance requires the seeker-side labels `suggested=1` and `liked=1`.
- Dialogue context is truncated before the first mention of a relevant movie.
- Every movie label must map to a MovieLens `movieId`.
- Ambiguous title mappings are written to `mapping_review.jsonl` and never
  silently accepted.
- Formal test cases must not be removed after inspecting method performance.

Generated manifests include catalog and split SHA-256 values. Generated JSONL
files are intentionally excluded from Git; the scripts and manifest document
their provenance.
