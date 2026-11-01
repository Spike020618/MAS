# Conservative recommendation research inputs

This is a separate research workspace. It does not modify the running recommender
or frozen legacy results. No script here calls a model or uses the network.

From the repository root:

```sh
python3 -m experiments.recommendation_v2.prepare
python3 -m experiments.recommendation_v2.audit --output experiments/results/recommendation_v2/audit_verified
python3 -m experiments.recommendation_v2.verify
```

Preparation is deterministic. Existing identical files are reused; changed
outputs require a new version directory. Audit components checkpoint only after
completion. Completed components check source/code dependencies; retrieval
checkpoints also verify candidate artifacts. Failed integrity checkpoints cannot
be resumed as successful ones. An interrupted component must be rerun; this is
component-level, not per-rating-row resume.

Serving input files and offline label files are separate. At a fixed cutoff the
input builder never reads dialogue-wide liking/seen annotations. Prefix movie
mentions are neutral events, not inferred positive anchors or seen exclusions.
Even explicitly expressed preferences remain in text without a certified parser.

Case selection still uses retrospective positive labels to locate the target
cutoff. This is disclosed in the manifest. It is not a prospective deployment
request sample, and the legacy test is explicitly diagnostic, not fresh.

Conversation-hash train/dev splitting does not imply worker isolation. Overlap
and the bounded scope of the legacy usage ledger are reported. MovieLens user
identities must not be joined to ReDial workers.

The BM25 coverage diagnostic indexes only movie titles and genres. It has no
co-occurrence, popularity, preference parsing, constraint parsing or LLM. It is
not a replacement production retriever, a new method, a historical-catalog
backtest, or an effect estimate for removing the legacy labels: the retriever
also differs from the old experiment.

The MovieLens inventory scans every rating and verifies source hashes, rating
validity, sorted key order and adjacent duplicates. It does not yet construct a
personalized train/validation/test protocol or establish an external utility
metric. Whole-snapshot tag features and global averages are not automatically
safe for temporal evaluation.
