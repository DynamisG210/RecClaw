# RQ1 reliability recomputation

This standard-library script consumes local evidence snapshots and the paper source. It never starts training, requests an LLM, or evaluates TEST.

```text
python tools/recompute_rq1_reliability.py --legacy LEGACY_CHECKOUT --paper PAPER_CHECKOUT --bert-observed ADOPTED_OBSERVED_JSON --out OUTPUT_DIRECTORY
```

The inputs are the existing seven-task search summary, adopted BERT source-repeat policy, equal-API data, original table provenance, current TeX main table, and adopted BERT/ML1M observation snapshot. Every input receives a SHA-256 in the output. The adopted BERT repeats are read from the paper policy, not selected by performance. Excluded source records remain listed.

The output separates search-level summary recomputation from independently verified fit provenance. Search summary rows do not establish checkpoint/source/evaluator identity. BERT links are preserved from the hash-matched adopted snapshot, with historical TEST exposure and selection-rule differences visible. Checkpoint-only evaluation is not called refitting. Unknown charges remain unknown.

The TeX parser targets the current fixed four-model, four-dataset, seven-method table and fails if the expected 224 metric cells are not present. Historical embedded JSON is compared with current TeX, not silently substituted for it.

Sample standard deviation uses search-level endpoint means (ddof=1), never individual fits pooled as independent searches. Missing tasks and unverified chain links remain explicit gaps. The tool supplies descriptive evidence only; it does not certify statistical superiority or causal validity.

## Retained checkpoint chain verification

```text
python tools/verify_rq1_checkpoint_chains.py --material NATIVE_REVIEW_MATERIAL --observed ADOPTED_OBSERVED_JSON --summary RQ1_RECOMPUTED_JSON --out NATIVE_CHAINS_JSON
```

This checks retained candidate and checkpoint bytes against the original selection freeze, TEST configuration/result and adopted observation. It reports every check individually, including search seed and checkpoint-only execution. Ancillary files outside the archive remain explicitly unverified by this local command; live hashes must be compared with their original recorded hashes separately. Mutable continuation ledgers must not substitute for the original freeze.

## Historical archive reconciliation

```text
python tools/reconcile_rq1_frozen_archives.py --comparison REMOTE_ANCILLARY_JSON --archives ARCHIVE_PARENT --out HISTORICAL_RECONCILIATION_JSON
python tools/recompute_morec_native_test_variation.py --snapshot SIX_SYSTEM_REPORT --ml1m-v3 ADOPTED_V3_REPORT --native HASH_NAMED_TEST_FILES --out MOREC_VARIATION_JSON
```

Archive reconciliation reads matching members without executing serialized campaign states. It checks preserved evaluation/event prefixes, selected algorithms and existing GPU jobs, and reports the original and continued cost snapshots separately. The MoRec tool verifies native TEST result bytes and recomputes search-level mean/sample SD. The optional explicit v3 report replaces only the adopted ML1M RO cohort; omit it to reproduce the historical cohort. Source and checkpoint fields in these TEST results are identity assertions, not independent byte checks of those artifacts.
