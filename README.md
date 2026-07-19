# RecClaw Phase 1: Evidence Guard A/B

This branch is the minimal pre-Canary delivery for a contemporaneous development comparison:

- Control: the final effective action-space self-reflection RecClaw;
- Treatment: the same RecClaw with the frozen development Evidence Guard;
- baseline source identity: commit `0b44db72f2e44bfbf8139b43c9624e1e89f52b35`, tree `3f9049509e5e09ae59a0d6aba79a5c2094dd3c2c`.

The failed permission-expanded/research-loop implementation, historical search outputs, ADMMSLIM follow-up environment, stopped work packages, and broad v2 infrastructure are not part of this delivery tree. They remain in the external provenance archive.

## Status and authority

The code and preparation records are development artifacts:

```text
authority = NONE
evidence_class = DEVELOPMENT_ONLY
formal_acceptance = false
Canary = NOT_STARTED
Full A/B = NOT_STARTED
```

The Evidence Guard does not authorize execution, admit authoritative evidence, update accepted claim state, promote artifacts, or approve a gate.

## Fresh setup

Python `>=3.11,<3.14` is required. The inspected legacy gpu5 Python 3.8 environment is not eligible for the frozen Guard integration.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Run the secret scan:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/phase1/scan_secrets.py --root . --include-untracked
```

## Safe dry-run

Dry-run only builds a plan. It does not create the runtime root, contact the LLM provider, reserve a GPU, or start training.

```bash
.venv/bin/recclaw-ab002-launch-control \
  --runtime-root /tmp/recclaw-ab002-dry-run \
  --search-seed 42 --gpu-id 0 --pair-id PAIR-42 \
  --broker-url http://127.0.0.1:18080 \
  --baseline-dir /external/fixed-baseline \
  --dry-run
```

Use the dedicated Treatment entry point for the other arm. Both launchers refuse a runtime root inside, equal to, or containing the source checkout.

## Repository map

- `src/recclaw_core/exploration/`: frozen Evidence Guard and Original RecClaw adapters;
- `src/recclaw_phase1/`: paired broker, arm materializer/launcher, analysis, and neutral outcome auditor;
- `configs/phase1/ab002/`: fixed Claim/Protocol, fresh S0, arm and outcome contracts;
- `phase1/overlays/`: frozen Treatment-only agent overlay;
- `records/phase1/`: only the five milestone-level machine records;
- `docs/`: current architecture, trustworthy-evidence line, experiment, reproducibility, and the active RC2 specification.

See [the experiment contract](docs/phase1_ab002_experiment.md) and [reproducibility guide](docs/reproducibility.md) before any execution. Do not start the Canary without a verified Python/RecBole environment and explicit user authorization.
