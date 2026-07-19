# Reproducibility

## Source verification

From the frozen pre-Canary commit/tag recorded in the release handoff:

```bash
git status --porcelain
sha256sum -c SOURCE_SHA256SUMS
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m recclaw_phase1.ab002_s0 --check
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/phase1/scan_secrets.py --root . --include-untracked
```

`git status --porcelain` must remain empty after verification.

## Dry-run verification

```bash
git ls-files -z | sort -z | xargs -0 sha256sum > /tmp/source.before
.venv/bin/recclaw-ab002-launch-control \
  --runtime-root /tmp/ab002-plan-control \
  --search-seed 42 --gpu-id 0 --pair-id AB002-SEED-42 \
  --broker-url http://127.0.0.1:18080 \
  --baseline-dir /external/fixed-baseline --dry-run
git ls-files -z | sort -z | xargs -0 sha256sum > /tmp/source.after
cmp /tmp/source.before /tmp/source.after
test ! -e /tmp/ab002-plan-control
```

Treatment uses the dedicated Treatment entry point with GPU1 for seed 42. Canary uses seed 9001, Control GPU0, Treatment GPU1, and pair ID `AB002-SEED-9001`.

## Execution-time preflight

Record without secret values:

- release commit/tree/tag and source checksum-file digest;
- S0 ID, source-tree digest, leakage-audit digest, and exact Control/Treatment materialization manifests;
- Python, dependency freeze, RecBole, PyTorch, CUDA, GPUs, and host identity;
- exact dataset and two baseline-log digests;
- dataset and RecBole before-state digests;
- external capacity plus absent run/output/temp/cache roots;
- broker health, pair-bound cache path, and non-empty secret variables;
- Control zero-Guard scan and Treatment frozen Guard/overlay digests.

`RECCLAW_AB002_START_AUTHORIZED=YES` is a mechanical accident guard, not scientific or permission authority.

The frozen gpu5 Canary path is one credential-in-memory orchestration command:

```bash
recclaw-ab002-run-canary \
  --runtime-root /NAS2020/Workspaces/DMGroup/tingrangan/RecClaw_evidence_guard_ab_002/runtime_<release> \
  --baseline-dir /home/tingrangan/projects/RecClaw/results/baseline \
  --expected-tag <pre-canary-tag>
```

It requires non-empty `RECCLAW_LAB_LLM_API_KEY` and `RECCLAW_LAB_LLM_BASE_URL`, starts the paired broker, writes the exact preflight, runs only seed-9001 Control/Treatment, stops the broker, runs the fixed controlled fail-open/quarantine probes, and writes `canary_report.json`. Credentials and the ephemeral broker client token are never written. The resulting report keeps `gate_status=NOT_STARTED` and `full_ab_authorized=false`; an independent read-only Canary review is still required.

## Frozen execution layout

```text
broker/paired_llm.sqlite3
runs/{control,treatment}/search_seed_{9001,42,43,44,45,46,47}/...
pair_inputs/search_seed_{42,43,44,45,46,47}.json
three_seed_baseline_report.json
simple_rule_comparison.json
analysis/paired_analysis.json
analysis/curve_data.json
analysis/running_best_curves.svg
analysis/neutral_outcome_audit.json
```

All runtime outputs, temporary candidate source, caches, logs, checkpoints, and model files remain outside the Git source tree. Raw artifacts are preserved even when a run is rejected or inconclusive.

## Post-run integrity and analysis

After every run, capture dataset and RecBole after-state digests. Build arm-blinded raw-run envelopes outside both online loops and apply the frozen neutral policy. Only neutral-eligible three-seed confirmations enter the primary endpoint.

The fixed post-GO sequence is: run the three-seed LightGCN comparator, create randomly assigned blind packets with `recclaw-ab002-blind-run`, and run `recclaw-ab002-confirm-blind-run` on each packet. Blind packets copy candidate execution artifacts and the minimal confirmation source only; they exclude `agent.py`, Guard feedback, and the arm mapping. Keep the mapping outside neutral scoring.

Run final analysis only from the six pair inputs and comparator report:

```bash
.venv/bin/python -m recclaw_phase1.ab002_final_analysis \
  --runtime-root /external/recclaw-ab002 \
  --output-dir /external/recclaw-ab002/analysis
```

Do not modify endpoints, thresholds, auditor policy, S0, model, provider, prompts, or resource envelopes after Canary or Full A/B outcomes are observed.
