# Task files and evaluation protocol

For SASRec/CDS, supply `cds.train.inter` and `cds.dev.inter` with these tab-separated columns:

```text
user_id:token	item_id_list:token_seq	item_id:token
u1	i1 i2	i3
```

Each TRAIN row is a chronological history prefix and its next-item target, in consecutive order within each user. Truncate prefixes to the last `MAX_ITEM_LIST_LENGTH` events (default 50). DEV has one row per evaluated user, whose prefix is the last TRAIN history and whose target is unseen in TRAIN. Neither DEV targets nor TEST events may enter TRAIN prefixes. `cds.user` and `cds.item` list the user/item catalog as described below. Catalog metadata may include IDs; it contains no future interaction labels.

The sequential binding uses `metric_binding_version: "sequential.fixed100.ndcg5.hr5.v1"`. Its editable defaults are in `inputs/binding.json`: two attention layers, two heads, hidden size 64, CE loss and length 50; Adam, learning rate 0.001, batch 256, at most 100 epochs and patience 10. Freeze any edits before baseline and search. Mapping, paired seeds, candidate-pool format and baseline receipts follow the shared contract below. Construct the pool from mapped IDs: one DEV target and 99 distinct negatives, excluding all TRAIN history/targets and padding. Sequence scoring uses each user's prefix, aligned to the pool even if its user order differs from the input file.

The graph adapter accepts four tab-separated RecBole files named `<dataset>.train.inter`, `<dataset>.dev.inter`, `<dataset>.user`, and `<dataset>.item`. Interaction columns are `user_id:token` and `item_id:token`; side files contain the corresponding ID column. In the binding, `data_files` maps `train`, `dev`, `user`, `item` to `{ "path": "...", "sha256": "..." }`. Asset paths resolve against the binding file.

The graph binding also supplies (the common data, mapping and baseline requirements also apply to SASRec):

- `status: "FROZEN"`, `root_source_sha256`, `catalog_mapping_sha256`, and `model_config`. Model-only fields are `embedding_size`, `n_layers`, `alpha`, `beta`, `gamma`, `reg_weight`, `hidden_size_list`, `message_dropout`.
- `training`: `epochs`, `patience`, `eval_step`, `batch_size`, `learning_rate`, `weight_decay`, fixed before both baseline and search.
- `metric_binding_version: "graph.fixed100.ndcg5.hr5.se-style.v1"` and `evaluation`: `primary_metric: "NDCG@5"`, `secondary_metric: "Hit@5"`, `metrics` including `NDCG` and `Hit`, `topk: [5, 10]`, `batch_size`, `candidate_universe: "FIXED_100_UNIFORM"`, `seed: 72026`.
- `dev_candidate_pool`: path and byte SHA256 of an NPZ containing `users` and `[number_of_users, 100]` `candidates`. Use mapped IDs, one DEV positive and 99 fixed negatives per user, with no padding or seen TRAIN positives.

The `assets` command generates the mapping and candidate pool. Mapping JSON has `user_id` and `item_id` token arrays including padding at index zero. Its identity is SHA256 of `json.dumps(mapping, sort_keys=True).encode()`, not the formatted file bytes. The `baseline` command writes `root_calibration.json` with `status`, the native worker `result`, and `artifacts.native_evidence` (termination, physical cost, unresolved GPU time). It also writes `root_binding.json`, `root_gpu_ledger.json`, and `facts.json`. Keep these files together with the baseline output directory.

Task facts include `task_id`, a `task` description, `root_source_sha256`, `data_sha256`, `catalog_mapping_sha256`, `original_model_config`, `training`, `evaluation`, `selected_root_metrics`, `root_reference_cost`, and `simulation_context`. The latter uses the adapter's existing `CONTEXT_RULE` in `native/task/research_line/graph_native_inputs.py`, which defines TRAIN-only fact availability. Literature is an object with a nonempty `documents` array. The packaged worker and shared baseline validator define the executable file contract.

The main method profile retains model routing, paired seed schedules (`41000 + repeat` for search and `51000 + repeat` for training, repeats 1-3), budget and evaluation protocol. These are method inputs, not host settings. The prepared run records these settings and its source identities. Custom tasks are labeled `USER_RESEARCH`.

`prepare` copies the released runtime into a new directory and binds its source hashes, interpreter, installed dependency identities and task inputs. `inspect` assembles the actual entry under a read-only audit boundary: no provider request, worker launch or resource claim. `run` performs MechSpace generation, research selection, source qualification, training, evaluation and feedback under the common native ledgers. A completed run is read idempotently; unresolved dispatches remain subject to native reconciliation, even with `--resume`.

The retained protocol has three paired repeat identities, up to 36 research rounds, 12 effective candidate evaluations and 36 training admissions. MechSpace and search share a budget of 7 reference USD under the original fixed accounting rates. Reference prices are experimental accounting inputs, not current provider price quotations. Failures, negative/no-op results, paid bytes and unknown usage stay in the run's accounting. Search receives TRAIN/DEV evidence; held-out TEST is not a tuning input or a command in this search release.

Outputs are under `new-run/task/research_line/runs/<run_id>/`: MechSpace artifacts, committed alternatives and decisions, source-linked evidence, campaign state, candidate source, learning curves, checkpoints, validation metrics and API/GPU ledgers. The terminal result is `RESULT.json`, or `PAUSED.json` when reconciliation is required.

