# RecClaw / RecRSI

**Mechanism-Guided Discovery of Recommendation Algorithms**

RecRSI takes a recommendation model, a frozen task environment and a finite experiment budget. It constructs mechanism hypotheses, implements candidate algorithms, trains and evaluates them, and uses the observations to guide subsequent research.

The final research path is:

1. **MechSpace**: source-grounded Genesis and Critic calls followed by deterministic assembly of mechanism objects, implementation requirements and composition relations.
2. **Research Line**: Mechanism Composer, Lineage Refiner, Falsification Designer and Frontier Architect construct hypotheses and complete candidate programs. The graph adapter supports explicit candidate-local mechanism definitions and measured construction parents. Two committed programs are compared by an independent selector; selection preserves their exact source bytes.
3. **Mechanism Feedback**: qualification, executed training signals, validation outcomes and source-linked contrasts update the next research context. Construction-parent, original-root and current-frontier comparisons remain distinct. Negative results and unresolved explanations are retained. The search policy is explicit and frozen; this release does not claim online policy learning.

## Distribution

This repository contains executable source, runtime prompts/schemas, installation metadata, focused tests and a task configuration example. Experiment logs, paid responses, credentials, datasets, checkpoints, paper drafts and coordination documents are external inputs or generated outputs.

The repository supplies the RecRSI main method in the RecClaw project. Users supply their task assets and host environment. The `recclaw` and `recrsi` commands are aliases.

The supplied adapter is **`recbole-graph-fixed100-v1`**: a single-file pairwise RecBole model, user/item interactions and fixed-100 validation. Your dataset, initial model and training recipe can be supplied through a frozen binding. The historical NGCF/Electronics configuration remains supported via `examples/legacy-task.json`. Sequential, multimodal or other evaluation environments need a corresponding adapter; changing a JSON name does not implement one.

## Install

Production execution requires Linux x86-64, Python 3.10–3.12, a CUDA-capable PyTorch environment, an NVIDIA GPU with capacity for the 9 GiB worker allocation, and four assigned CPU cores. The native worker boundary uses Linux namespaces, Landlock and an unprivileged worker identity. The operator must provision the required `sudo -n` launcher access, readable runtime/data paths and NVIDIA driver-library directory. Generated programs run inside that boundary.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install '.[train,test]'
recrsi --help
```

The training extra pins PyTorch, NumPy, SciPy and the RecBole source revision used by the worker. Set `recbole_root` to the installed `site-packages` directory containing `recbole/`, or to a checkout of the pinned RecBole revision. A separate controller environment is supported through `worker_python`.

## Prepare and run

Create your project outside the source checkout:

```bash
recclaw init --output /path/to/my-research
# Edit task.json and environment.json; supply your task assets.
recclaw check --config /path/to/my-research/task.json
recclaw prepare --config /path/to/my-research/task.json --output /path/to/new-run
recclaw inspect /path/to/new-run
recclaw run /path/to/new-run
# After reconciling an interruption, retain the original directory:
recclaw run /path/to/new-run --resume
```

`init` creates templates without overwriting an existing directory. `check` verifies input hashes, baseline identity and protocol consistency without API calls, GPU claims, user-code execution or worker launches. It does not probe host isolation privileges or provider availability. `prepare` also checks worker dependencies; `inspect` assembles the actual controller without execution.

**Task configuration** (`examples/task.json`) defines the adapter, run ID, repeat and scientific inputs. Paths resolve relative to this file:

- `binding`: your frozen task identity, model/training parameters, evaluation protocol, data hashes, ID mapping and fixed candidate pool. Custom bindings explicitly declare `adapter: "recbole-graph-fixed100-v1"` and use their own `task_id`, `dataset`, and model `line` names (letters, digits, underscores or hyphens).
- `root_source`: a complete `candidate.py` exposing `FreshCandidateModel(config, train_dataset)`, `calculate_loss(pairwise Interaction)`, `predict` and `full_sort_predict`. Only TRAIN interactions enter the model. The trusted worker owns optimization, evaluation and checkpoint selection.
- `root_binding`, `root_calibration`, `root_gpu_ledger`, `root_mapping`: the exact baseline binding, completed paired-seed baseline result, actual cost ledger and ID mapping from your environment. Preparation neither fabricates nor refits a baseline.
- `facts`, `literature`: TRAIN-only task context and literature cards. Facts must match the source, data, mapping, model/training/evaluation settings and actual baseline metrics. No held-out TEST observations belong in research context.

**Environment configuration** (`examples/environment.json`) is referenced by `environment` in the task config. Paths resolve relative to this separate environment file, which can serve multiple tasks:

- `worker_python`, `recbole_root`: your training interpreter and installed RecBole location. Virtualenv symlinks retain their selected environment.
- `driver_root`, `resource_slots`, `cpus`, `gpu_uuid`: NVIDIA libraries and host allocation. Assign four distinct CPU cores. All runs sharing a host must use the same canonical resource-lock directory. Physical stages select available GPUs through the allocator and record the actual UUID.
- `api_config`: a private file outside Git and worker-readable roots, with `base_url: https://YOUR_ENDPOINT/v1` and `api_key: YOUR_KEY` on separate lines. The endpoint must support the frozen `gpt-5.6-terra` and `gpt-5.6-luna` model identities through Responses and chat-completions. There is no implicit provider fallback.

Paths support `${VARIABLE}` and `~`; undefined variables fail before preparation. For example, `"api_config": "${RECRSI_PRIVATE}/endpoint2.api-config"`. Unknown fields fail instead of being ignored. `TASK.json` records task and environment configuration digests independently; resume uses the prepared configuration.

### Environment input contract

The graph adapter accepts four tab-separated RecBole files named `<dataset>.train.inter`, `<dataset>.dev.inter`, `<dataset>.user`, and `<dataset>.item`. Interaction columns are `user_id:token` and `item_id:token`; side files contain the corresponding ID column. In the binding, `data_files` maps `train`, `dev`, `user`, `item` to `{ "path": "...", "sha256": "..." }`. Asset paths resolve against the binding file.

The binding also supplies:

- `status: "FROZEN"`, `root_source_sha256`, `catalog_mapping_sha256`, and `model_config`. Model-only fields are `embedding_size`, `n_layers`, `alpha`, `beta`, `gamma`, `reg_weight`, `hidden_size_list`, `message_dropout`.
- `training`: `epochs`, `patience`, `eval_step`, `batch_size`, `learning_rate`, `weight_decay`, fixed before both baseline and search.
- `metric_binding_version: "graph.fixed100.ndcg5.hr5.se-style.v1"` and `evaluation`: `primary_metric: "NDCG@5"`, `secondary_metric: "Hit@5"`, `metrics` including `NDCG` and `Hit`, `topk: [5, 10]`, `batch_size`, `candidate_universe: "FIXED_100_UNIFORM"`, `seed: 72026`.
- `dev_candidate_pool`: path and byte SHA256 of an NPZ containing `users` and `[number_of_users, 100]` `candidates`. Use mapped IDs, one DEV positive and 99 fixed negatives per user, with no padding or seen TRAIN positives.

Mapping JSON has `user_id` and `item_id` token arrays including padding at index zero. Its identity is SHA256 of `json.dumps(mapping, sort_keys=True).encode()`, not the formatted file bytes. The baseline envelope is `{ "status": "SUCCESS", "result": ... }` containing the native worker result, including the exact baseline binding digest; preserve its actual resource ledger.

Task facts include `task_id`, a `task` description, `root_source_sha256`, `data_sha256`, `catalog_mapping_sha256`, `original_model_config`, `training`, `evaluation`, `selected_root_metrics`, `root_reference_cost`, and `simulation_context`. The latter uses the adapter's existing `CONTEXT_RULE` in `native/task/research_line/graph_native_inputs.py`, which defines TRAIN-only fact availability. Literature is an object with a nonempty `documents` array. The packaged worker is the executable environment reference. `check` validates consistency, not scientific quality or authenticity of externally supplied measurements.

The main method profile retains model routing, paired seed schedules (`41000 + repeat` for search and `51000 + repeat` for training, repeats 1-3), budget and evaluation protocol. These are method inputs, not host settings. A different protocol requires an explicit implementation/version. Custom tasks use `USER_RESEARCH` and `USER_CONFIGURED_SOFTWARE_RUN` metadata and do not inherit the paper's accepted results.

`prepare` copies the released runtime into a new directory and binds its source hashes, interpreter, installed dependency identities and task inputs. `inspect` assembles the actual entry under a read-only audit boundary: no provider request, worker launch or resource claim. `run` performs MechSpace generation, research selection, source qualification, training, evaluation and feedback under the common native ledgers. A completed run is read idempotently; unresolved dispatches remain subject to native reconciliation, even with `--resume`.

The retained protocol has three paired repeat identities, up to 36 research rounds, 12 effective candidate evaluations and 36 training admissions. MechSpace and search share a budget of 7 reference USD under the original fixed accounting rates. Reference prices are experimental accounting inputs, not current provider price quotations. Failures, negative/no-op results, paid bytes and unknown usage stay in the run's accounting. Search receives TRAIN/DEV evidence; held-out TEST is not a tuning input or a command in this search release.

Outputs are under `new-run/task/research_line/runs/<run_id>/`: MechSpace artifacts, committed alternatives and decisions, source-linked evidence, campaign state, candidate source, learning curves, checkpoints, validation metrics and API/GPU ledgers. The terminal result is `RESULT.json`, or `PAUSED.json` when reconciliation is required.

## Verification and interpretation

```bash
python -m pytest tests -q
```

The CPU integration fixtures perform real optimization, fixed-100 scoring and exact checkpoint reload on both the legacy identity and a custom dataset/model identity. Configuration tests cover relative path bases, environment variables, mismatched baseline/data/protocol identities and unknown fields. Other checks cover immutable program selection, actual-parent contrasts and read-only inspection. Installation and the native controller/worker bootstrap are checked separately on Linux. These checks establish software behavior; they are not recommendation-quality measurements. A new paid-provider/CUDA research campaign is not part of the packaging test.

Scientific claims must use the accepted common-protocol evidence for each environment. Mechanism prose and static source differences are hypotheses or observations, not component-wise causal proof. Shared-host timings are not exclusive GPU timings, and the accelerated evaluator's validated metric agreement does not imply universal bitwise score equality. This repository makes no universal-best or cross-version pooled superiority claim.
