# RecClaw

RecClaw is a recommender-system research agent built around RecBole. It
proposes mechanism-level candidates, materializes them in a controlled
executable space, runs isolated comparisons, and learns from outcomes without
silently changing the experiment protocol.

This branch is the compact functional Research Line release. Historical Pilot
records, generated result databases, version-by-version launch scripts, and
nightly process documents are intentionally excluded from the release tree.
See [docs/research_line/README.md](docs/research_line/README.md) for the exact
scope and evidence boundary.

## Main components

- `src/recclaw_core/experiments/helix_abc_v1/`: three-arm Research Line runtime,
  canonical identities, state stores, orchestration, training contracts, Meta
  control, and audit projections.
- `src/recclaw_core/helix/` and `src/recclaw_evidence_guard/`: EvidencePort and
  Evidence Guard integration.
- `recclaw_ext/`: executable recommender-model, loss, sampler, and composition
  extensions.
- `configs/`: action-space, candidate, dataset, and execution configuration.
- `scripts/research_line.py`: side-effect-free Producer, Router, Search Memory,
  and Meta-Research utilities.
- `scripts/campaign_train_worker.py`: common training worker.
- `scripts/run_candidate.py`: isolated candidate execution.
- `scripts/plan_research_line_comparison.py`: matched comparison-plan generator.
- `tests/`: retained functional and regression coverage.

## Quick checks

```bash
python3 scripts/analysis/lint_recclaw_space.py
python3 -m pytest -q tests/experiments/helix_abc_v1
```

## Local pilot entry

Use a fresh output directory for every development run:

```bash
python3 scripts/run_reflection_pilot.py \
  --rounds 50 \
  --proposal-source llm \
  --search-intensity algorithm_first \
  --gpu-id 0
```

Generate a matched command plan without modifying the original checkout:

```bash
python3 scripts/plan_research_line_comparison.py \
  --old-root ../RecClaw-original \
  --rounds 50 \
  --proposal-source heuristic
```

Keep `RecClaw_LabLog` and generated experiment outputs out of runtime inputs.
They may be used only for human-facing analysis and reporting.
