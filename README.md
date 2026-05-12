# RecClaw

RecClaw is a lightweight workspace for recommender-system method exploration.

The current stage focuses on local candidate design around fixed BPR and
LightGCN baselines.

## Server Experiment Flow

Start from a clean `results/` state, then establish same-protocol baselines
before running the agent loop:

```bash
bash scripts/run_baseline.sh
python scripts/agent.py --loop-mode mixed --rounds 5
```

Useful agent modes:

- `tuning`: conservative parameter/config proposals only.
- `mixed`: default balance of runnable tuning and implementation review queue.
- `explore`: allows auto-implementation of `code_required` proposals through the LLM allowlist.
- `auto`: asks the LLM planner to choose the next loop action each round.

For a no-LLM smoke check, run:

```bash
python scripts/agent.py --rounds 1 --dry-run --disable-candidate-proposals
```

`run_time` and `latency_ms` are intentionally separate. `latency_ms` should
come from an explicit inference-timing benchmark or model output; the result
collector leaves it empty when only wall-clock run time is available.
