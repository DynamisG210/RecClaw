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

## Disallowed Changes In v0.5

- RecBole core source files
- evaluation protocol
- data split definition

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
