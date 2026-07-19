# RecClaw Phase 1: Evidence Guard A/B

This branch contains the clean-start matched AB-002 development experiment:

- Control: Original RecClaw from a reconstructed, leak-audited pre-search S0;
- Treatment: the same S0 plus the frozen development Evidence Guard integration;
- six paired search seeds, a separate three-round Canary, a common paired `gpt-5.4` broker, neutral raw-run auditing, three-seed confirmation, and paired final analysis.

AB-001, failed research-loop code, historical search outputs, ADMMSLIM, Stage 1B runtime, OPE, and new profile families are outside this delivery.

## Status and authority

```text
authority = NONE
evidence_class = DEVELOPMENT_ONLY
formal_acceptance = false
Canary = NOT_STARTED
Full A/B = NOT_STARTED
```

Local tests, successful training, and neutral-audit eligibility do not approve a gate or establish a research claim.

## Verify the source

Python `>=3.11,<3.14` is required.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m recclaw_phase1.ab002_s0 --check
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/phase1/scan_secrets.py --root . --include-untracked
```

## Safe dry-run

```bash
.venv/bin/recclaw-ab002-launch-control \
  --runtime-root /external/recclaw-ab002 \
  --search-seed 42 --gpu-id 0 --pair-id AB002-SEED-42 \
  --broker-url http://127.0.0.1:18080 \
  --baseline-dir /external/fixed-baseline \
  --dry-run
```

Dry-run creates no runtime directory, starts no training, and contacts no LLM. The launcher rejects wrong seed/pair bindings, wrong GPU rotation, an in-tree runtime root, and baseline-byte drift at execution.

## Repository map

- `phase1/s0/ab002/`: closed reconstructed S0 and known-leakage audit;
- `phase1/overlays/`: Treatment-only Guard hook overlay;
- `src/recclaw_core/exploration/`: frozen development Evidence Guard;
- `src/recclaw_phase1/`: arm launcher, paired broker, neutral auditor, and analysis;
- `configs/phase1/ab002/`: experiment, arm, neutral-audit, and decision contracts;
- `records/phase1/`: milestone-level development records only.

Read [the experiment definition](docs/phase1_ab002_experiment.md) and [reproducibility guide](docs/reproducibility.md) before execution.
