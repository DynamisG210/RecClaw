# Reproducibility

## Source verification

From a fresh clone or worktree at tag `phase1-ab002-pre-canary`:

```bash
git status --porcelain
sha256sum -c SOURCE_SHA256SUMS
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python scripts/phase1/scan_secrets.py --root . --include-untracked
```

`git status --porcelain` must be empty after verification. Bytecode is disabled so test imports do not dirty the tree.

## Dry-run verification

Hash the tracked tree before and after each launcher dry-run. A dry-run must print a plan but create no runtime directory and change no source byte.

```bash
git ls-files -z | sort -z | xargs -0 sha256sum > /tmp/source.before
.venv/bin/recclaw-ab002-launch-control \
  --runtime-root /tmp/ab002-plan-control \
  --search-seed 42 --gpu-id 0 --pair-id PAIR-42 \
  --broker-url http://127.0.0.1:18080 \
  --baseline-dir /external/fixed-baseline --dry-run
git ls-files -z | sort -z | xargs -0 sha256sum > /tmp/source.after
cmp /tmp/source.before /tmp/source.after
test ! -e /tmp/ab002-plan-control
```

Treatment uses `recclaw-ab002-launch-treatment` with the same pair inputs. The materialization-only mode is allowed to create an external repetition source tree but never writes into the checkout.

## Execution-time preflight

Before Canary, record without secret values:

- exact delivery commit, tree, tag, and `SOURCE_SHA256SUMS` digest;
- Python >=3.11 environment identity and dependency freeze;
- RecBole, PyTorch, CUDA, GPU, and host identity;
- exact dataset and baseline digests from the fixed contract;
- external NFS capacity and repetition-root absence;
- broker database path, endpoint health, and secret-variable presence;
- Control source scan proving zero Guard files/imports;
- Treatment integration hash replay and overlay result;
- paired GPU assignment and start authorization.

Treatment materialization does not depend on an ambient Git executable. The
launcher applies the single frozen UTF-8 overlay with strict hunk/context
matching and verifies the resulting `scripts/agent.py` digest before writing
the runtime manifest. A missing Git binary therefore cannot change or weaken
the Treatment source projection.

The launcher also projects the exact `RECBOLE_ROOT` from the frozen AB-002
contract into both arms. Candidate execution therefore does not depend on an
ambient shell variable or on an adjacent source checkout.

The shared ML environment contains dependencies only, not the RecClaw package.
At child-process launch the Control arm clears ambient Python and Evidence Guard
paths and disables the user site; the Treatment arm then receives only its
materialized, digest-bound integration path.

The launcher additionally requires `RECCLAW_AB002_START_AUTHORIZED=YES` for execution. This environment variable is a mechanical accident guard, not a permission or scientific gate.

## Outputs

Every output belongs below the external runtime root:

```text
broker/paired_llm.sqlite3
runs/control/search_seed_{42,43,44}/...
runs/treatment/search_seed_{42,43,44}/...
analysis/ab002_analysis.json
analysis/ab002_running_best.svg
analysis/neutral_outcome_audit.json
```

Do not copy raw outputs, secrets, model checkpoints, datasets, temporary arm source trees, or execution logs into this Git branch. Only milestone-level machine records may be added in a later explicitly scoped delivery update.
