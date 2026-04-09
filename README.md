# RecClaw v0

RecClaw v0 is a minimal experiment assistant for recommendation research.
It does not try to be a full autonomous research agent yet. The current
goal is much narrower and easier to validate:

1. Fix one RecBole task environment.
2. Run a baseline.
3. Collect and record metrics.
4. Run one candidate config change.
5. Compare baseline vs candidate and return `keep`, `discard`, or `crash`.

## Why RecBole And RecClaw Stay Side By Side

RecClaw is designed to sit next to RecBole, not inside it:

```text
~/projects/
├── RecBole/
└── RecClaw/
```

This keeps the experiment assistant separate from the experiment engine:

- RecBole stays a clean upstream-like training/evaluation base.
- RecClaw only adds configs, wrappers, result collection, and comparison logic.
- You can update or replace RecBole without rewriting the whole RecClaw layout.

RecClaw never copies RecBole into itself and does not require RecBole source
changes for v0.

## Layout

```text
RecClaw/
├── README.md
├── .gitignore
├── recclaw_program.md
├── configs/
│   ├── task_ml1m.yaml
│   ├── bpr.yaml
│   ├── lightgcn.yaml
│   └── bpr_lr_candidate.yaml
├── scripts/
│   ├── run_baseline.sh
│   ├── run_candidate.sh
│   ├── collect_result.py
│   ├── compare_runs.py
│   └── sync_to_server.sh
├── results/
│   ├── baseline/
│   ├── candidates/
│   └── results.csv
└── notes/
    └── experiment_log.md
```

## RecBole Discovery

RecClaw looks for the adjacent RecBole repo in this order:

1. `RECBOLE_ROOT`
2. `RECBole_ROOT`
3. `../RecBole` relative to the current RecClaw project root

Example:

```bash
export RECBOLE_ROOT=~/projects/RecBole
```

The scripts expect this entrypoint to exist:

```text
$RECBOLE_ROOT/run_recbole.py
```

## Local Compatibility Notes

This repository was adapted to the local RecBole version detected in
`~/projects/RecBole`:

- The entrypoint is `run_recbole.py`.
- The config keys `eval_args`, `metrics`, `topk`, and `valid_metric` are valid.
- `mode: full` is accepted in YAML and the local RecBole version internally
  normalizes it to `{'valid': 'full', 'test': 'full'}` at runtime.

No RecBole source file was modified for this RecClaw v0 scaffold.

## Dataset Note

The current local RecBole checkout already contains `dataset/ml-100k`, but
`ml-1m` was not found during inspection. Since RecClaw v0 fixes the task to
`ml-1m`, the first run may need RecBole to download or prepare that dataset.

If your environment does not auto-download datasets, prepare `ml-1m` inside
the dataset location expected by RecBole before running baselines.

## Fixed Task Setup

`configs/task_ml1m.yaml` fixes the v0 experiment environment to:

- dataset: `ml-1m`
- task: general recommendation
- primary metric: `NDCG@10`
- secondary metric: `Recall@10`
- evaluation: full ranking
- fixed `split`, `group_by`, `order`, and `mode`

Model-specific config files stay separate:

- `configs/bpr.yaml`
- `configs/lightgcn.yaml`

## Run Baselines

Run BPR:

```bash
bash scripts/run_baseline.sh bpr
```

Run LightGCN:

```bash
bash scripts/run_baseline.sh lightgcn
```

Run both:

```bash
bash scripts/run_baseline.sh all
```

Each baseline run will:

- call the adjacent RecBole repo
- write a copy of the console log to `results/baseline/`
- parse metrics with `scripts/collect_result.py`
- append one row to `results/results.csv`

## Run A Candidate

An example candidate override is included at:

```text
configs/bpr_lr_candidate.yaml
```

Run it like this:

```bash
bash scripts/run_candidate.sh bpr configs/bpr_lr_candidate.yaml
```

You can also pass your own extra config file:

```bash
bash scripts/run_candidate.sh lightgcn /path/to/my_candidate.yaml
```

The candidate script will:

- combine `task_ml1m.yaml`
- combine the model baseline config
- add your extra config override(s)
- save the run log under `results/candidates/`
- append the parsed result to `results/results.csv`

## Collect Results Manually

Parse a RecBole log and print JSON:

```bash
python scripts/collect_result.py results/baseline/baseline_bpr_YYYYMMDD_HHMMSS.log
```

Parse and append to CSV:

```bash
python scripts/collect_result.py \
  results/candidates/candidate_bpr_YYYYMMDD_HHMMSS.log \
  --append-csv results/results.csv \
  --run-id candidate_bpr_demo
```

## Compare Runs

Compare two log files directly:

```bash
python scripts/compare_runs.py \
  results/baseline/baseline_bpr_YYYYMMDD_HHMMSS.log \
  results/candidates/candidate_bpr_YYYYMMDD_HHMMSS.log
```

Compare two rows from `results/results.csv` by run id:

```bash
python scripts/compare_runs.py \
  --csv results/results.csv \
  --baseline-id baseline_bpr_YYYYMMDD_HHMMSS \
  --candidate-id candidate_bpr_YYYYMMDD_HHMMSS
```

The default comparison rule is:

- `candidate ndcg@10 > baseline ndcg@10` -> `keep`
- otherwise -> `discard`
- if the candidate run failed or has no valid metric value -> `crash`

## Future Extensions

This v0 scaffold is intentionally config-first. A sensible next path is:

1. config-only search
2. sampler-level changes
3. loss-level changes
4. module-level changes
5. richer experiment memory and automatic retry logic

The current project stops at the smallest verifiable loop.