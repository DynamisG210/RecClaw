# RecClaw

RecClaw is a recommender-system research agent workspace built around RecBole.
It is used to propose, validate, implement, run, and reflect on candidate
recommendation algorithms under a fixed evaluation protocol.

The current runtime focuses on controlled BPR/LightGCN action-space exploration
with local extension code under `recclaw_ext/`.

## Main Pieces

- `configs/action_space.yaml`: runtime boundary for what the agent may change.
- `configs/candidate_registry.yaml`: runnable candidate catalog.
- `scripts/agent.py`: Observe -> Plan -> Propose -> Validate -> Run -> Reflect loop.
- `scripts/research_line.py`: explicit research-ability line with Candidate
  Producers, Research Router, Search Memory, and Meta-Research advisory state.
- `scripts/run_candidate.py`: isolated candidate execution.
- `scripts/build_experience_summary.py`: reflection memory and search steering.
- `scripts/plan_research_line_comparison.py`: paired command-plan generator for
  comparing this research-line version against `RecClaw-原版`.
- `recclaw_ext/`: local model/loss/sampler extensions.
- `recclaw_program.md`: operating manual for the agent and experiments.

## Quick Checks

```bash
python3 scripts/analysis/lint_recclaw_space.py
python3 -m unittest discover -s tests
```

## Pilot Entry

Use an isolated output directory for each run:

```bash
python3 scripts/run_reflection_pilot.py \
  --rounds 50 \
  --proposal-source llm \
  --search-intensity algorithm_first \
  --gpu-id 0
```

## Comparison Plan

Generate a matched command plan against the original RecClaw without modifying
the original folder:

```bash
python3 scripts/plan_research_line_comparison.py \
  --old-root ../RecClaw-原版 \
  --rounds 50 \
  --proposal-source heuristic
```

Keep `RecClaw_LabLog` out of runtime inputs. Use it only for human-facing
analysis, plots, and reports.
