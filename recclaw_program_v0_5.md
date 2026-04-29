# RecClaw Program

## Scope

- Current task: general recommendation on `ml-1m`
- Current stage: `RecClaw v0.5`
- Current role: method-level extension stage between config-only experimentation and future agent-driven improvement

## Fixed Baselines

- `BPR`
- `LightGCN`

## Metrics

- Primary metric: `NDCG@10`
- Observation metrics: `Recall@10`, `MRR@10`, `Hit@10`, `Precision@10`, `ItemCoverage@10`

## Allowed Changes

- config files under `configs/`
- scripts under `scripts/`
- notes under `notes/`
- local extension code under `recclaw_ext/`

Scripts define the execution interface and should be changed only by the runner
owner. Method-entry refactors should stay in `configs/`, `notes/`, and
`recclaw_ext/`.

## Disallowed Changes In v0.5

- RecBole core source files
- evaluation protocol
- data split definition

## Candidate Entry Contract

Every candidate in `configs/candidate_registry.yaml` must declare:

- `runner_type`: one of `config_only`, `model`, or `posthoc`
- `wired`: whether the current code path actually executes the candidate logic
- `entrypoint`: the import path for the executable class or RecBole baseline
- `consumes`: config keys that are actually read by the executable path

`wired: true` should only be used when the entrypoint exists and the candidate
is reachable by the current execution flow.

## Local Extension Layout

`recclaw_ext/` has two primary entry areas:

- `recclaw_ext/models/`: all training-time candidates, including loss and sampler ideas
- `recclaw_ext/posthoc/`: trained-score adjustment candidates after model scoring

Internal helpers may live under:

- `recclaw_ext/models/_losses.py`
- `recclaw_ext/models/_samplers.py`

Top-level `losses/`, `samplers/`, `features/`, and `rerank/` packages are not
part of the v0.5 layout.

## Candidate Types

- `config_only`: RecBole-native config changes, such as embedding size or layer count
- `model`: local model classes that override training-time behavior
- `posthoc`: score adjustment after a trained model produces recommendations

Loss and sampler changes should be implemented as `model` candidates, not as
standalone candidate entrypoints.

## Per-Run Output Requirements

Every experiment round must produce:

- log
- metrics
- elapsed time
- status
- notes

## Decision Rule

- Keep or discard remains mainly based on `NDCG@10`.
- Observation metrics are used for analysis, not as the primary decision rule.
- If a candidate run fails, mark `crash`.

Baseline and candidate budgets must be comparable. A smoke candidate should be
compared with a smoke baseline, and a full-budget candidate should be compared
with a full-budget baseline.
