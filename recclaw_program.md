# RecClaw Program

## Scope

- Current task: general recommendation on `ml-1m`
- Current stage: `RecClaw v0`
- Current role: automatic experiment assistant

## Fixed Baselines

- `BPR`
- `LightGCN`

## Metrics

- Primary metric: `NDCG@10`
- Secondary metric: `Recall@10`

## Allowed Changes

- config files under `configs/`
- scripts under `scripts/`

## Disallowed Changes In v0

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

- If candidate `NDCG@10` is higher than baseline `NDCG@10`, mark `keep`.
- Otherwise, mark `discard`.
- If the candidate run fails, mark `crash`.