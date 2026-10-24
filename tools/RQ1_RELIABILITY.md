# RQ1 reliability recomputation

This standard-library script consumes local evidence snapshots and the paper source. It never starts training, requests an LLM, or evaluates TEST.

```text
python tools/recompute_rq1_reliability.py --legacy LEGACY_CHECKOUT --paper PAPER_CHECKOUT --bert-observed ADOPTED_OBSERVED_JSON --out OUTPUT_DIRECTORY
```

The inputs are the existing seven-task search summary, adopted BERT source-repeat policy, equal-API data, original table provenance, current TeX main table, and adopted BERT/ML1M observation snapshot. Every input receives a SHA-256 in the output. The adopted BERT repeats are read from the paper policy, not selected by performance. Excluded source records remain listed.

The output separates search-level summary recomputation from independently verified fit provenance. Search summary rows do not establish checkpoint/source/evaluator identity. BERT links are preserved from the hash-matched adopted snapshot, with historical TEST exposure and selection-rule differences visible. Checkpoint-only evaluation is not called refitting. Unknown charges remain unknown.

The TeX parser targets the current fixed four-model, four-dataset, seven-method table and fails if the expected 224 metric cells are not present. Historical embedded JSON is compared with current TeX, not silently substituted for it.

Sample standard deviation uses search-level endpoint means (ddof=1), never individual fits pooled as independent searches. Missing tasks and unverified chain links remain explicit gaps. The tool supplies descriptive evidence only; it does not certify statistical superiority or causal validity.
