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

## MoRec frozen binding

```text
python tools/verify_morec_frozen_binding.py --binding BINDING_DIRECTORY --review ADOPTED_V3_REVIEW --native-tests HASH_NAMED_TEST_FILES --out MOREC_V3_NATIVE_CHAIN_JSON
```

The binding directory contains the original all3_frozen_inputs.json, final_plan.json, FIT_PLAN_1/2/3.json, and REMOTE_PINS.json from a read-only remote SHA-256 collection. The tool checks candidate-file hashes against the original fit plan, actual checkpoint hashes, search/train/evaluation seeds, and native TEST linkage. It does not run training or TEST. Remote pins record their collection time and expected/actual hashes.

## Frozen-plan remote byte check

```text
python tools/verify_morec_plan_remote.py --plan FINAL_PLAN_JSON --snapshot SIX_SYSTEM_REPORT --native-tests HASH_NAMED_TEST_FILES --host SSH_HOST --out REMOTE_CHAIN_JSON
```

This runs only remote sha256sum on paths pinned by the frozen plan. The output distinguishes missing/changed files and keeps root fallback/cached results in scope. Current adopted cohort composition is applied separately; historical ML1M RO results remain available and must not be pooled with its adopted v3 replacement.

## Search quality and resource dispersion

```text
python tools/analyze_rq1_search_variation.py --searches SEARCH_SUMMARY --protocol COHORT_PROTOCOL --out VARIATION_JSON
```

Computes mean, sample SD, range and coefficient of variation across source searches for quality and reference resource usage. Verifies unique search identities and GPU unit conversion. Comparisons use each task's highest-mean comparator; counts above that mean are descriptive, not paired wins or significance tests. Job time and root allocation remain separate.

## Source-search versus refit variation for transfers

```text
python tools/recompute_bert_transfer_variation.py --snapshot BERT_TRANSFER_RECORDS --host SSH_HOST --out TRANSFER_VARIATION_JSON
```

Reads the exact recorded TEST files through read-only SSH and verifies byte hashes and parsed content. Groups by dataset, method and source search repeat before computing three-refit means; reports the SD across source-search means separately from mean within-source refit SD. Transferred winners are not relabeled as new target searches.

## MoRec Electronics transfer

```text
python tools/recompute_morec_el_transfer.py --plan TEST_PLAN_JSON --analysis TEST_ANALYSIS_JSON --host SSH_HOST --out MOREC_EL_TRANSFER_JSON
```

Verifies retained TEST bytes, metrics, training-result bytes and checkpoint bytes for all 21 method/seed slots (including Seed), then computes three-refit means and SDs. This cohort transfers one selected program per method; target-search count is zero, not three.

NGCF native variation: `python tools/recompute_ngcf_native_variation.py --snapshot <ngcf_final_native.json> --table <RQ1_main_table.tex> --host <ssh-host> --out <result.json>` verifies retained remote TEST bytes and computes between-source-search and within-source-refit sample SD separately. Uses the adopted replacement RO1 with original RO2/RO3; CDs is source-program transfer. Recorded checkpoint hashes are not a checkpoint-byte verification.

Supplied SASRec cost variation: `python tools/recompute_sasrec_supplied_cost_variation.py --costs <search_costs.csv> --reference <paper_eight_task_plot_inputs.json> --out <result.json>`. Reports sample SD/range/CV for six methods and checks all twelve supplied means. Does not infer native TEST identity, missing tokens, total invoice cost or per-search quality from method means.

NGCF transfer comparators: `python tools/recompute_ngcf_transfer_comparators.py --results <CURRENT_RESULTS.json> --sources <SOURCES.json> --table <RQ1_main_table.tex> --host <ssh-host> --out <result.json>`. Verifies five comparator methods on CDs/Office against frozen source identities, native TEST bytes and table metrics. Reports source-search and within-refit SD separately. The replaced RO cohort is handled separately by the native-variation tool.

Use `--include-original-office-ro` only to reproduce the original Office RO cohort explicitly retained by the historical Office comparison. This adds nine native results; CDS RO remains excluded. Do not silently apply the CDS/ML1M RO1 replacement to Office.

NGCF version sensitivity: `python tools/compare_ngcf_cohort_versions.py --evidence <ngcf_transfer-directory> --out <result.json>` computes original/replacement cohorts, retains Electronics protocol mismatch, and extracts timestamped source decisions and recovery limitations. This does not adopt either version or infer an unrecorded pre-TEST rule.
