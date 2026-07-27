# M6G G2 Independent Adversarial Audit

## Scope and method

This audit reviewed the G2 result as a fresh-process, black-box contract
exercise rather than relying on implementation-local assertions. It covered
the C-private evidence snapshot, exact identity confinement, validation-task
lifecycle, ordinary-round and budget semantics, and the three V13 Frontier
projections. It did not call a Broker, Provider or training runtime.

## Adversarial cases

- Three exact observations for one candidate, protocol and comparator
  accumulate across seeds 2026, 2027 and 2028 and reach the Guard as a
  multi-seed bundle.
- An observation for another candidate is present in the same private ledger
  but absent from the target candidate's Guard snapshot.
- A validation task executes the exact frozen program and seed in a future
  ordinary SearchRound.
- Validation rounds debit one ordinary execution each while making zero
  proposal and zero LLM calls.
- A Guard-ineligible metric is visible in Observed Frontier but cannot enter
  SearchEligible Frontier.
- A preliminary one-seed result can enter only the preliminary
  SearchEligible projection and cannot enter Confirmed Frontier.
- Confirmed Frontier remains empty without both an explicit confirmed
  post-selection row and the matching frozen evaluator digest.
- Pilot descriptive analysis computes no B-A, C-B or C-A treatment effect.

## Verification evidence

```text
PYTHONPATH=src .../python -m pytest -q \
  tests/experiments/helix_abc_v1/test_m3_helix_composition.py \
  tests/experiments/helix_abc_v1/test_m4_precanary.py \
  tests/experiments/helix_abc_v1/test_v13_frontier_analysis.py \
  -k "not research_and_common_runtime_bytes_remain_frozen"

31 passed, 1 deselected, 3 subtests passed
```

The deselected assertion binds the superseded V12/M2 frozen source bytes. V13
must not rewrite that historical identity.

The affected legacy Pilot/Campaign check additionally produced 19 passes. Its
two failures are outside G2: one legacy test still invokes the already-versioned
Research feedback API without its required mechanism identity, and the other
correctly detects that the permanently superseded V12 source digest no longer
matches the V13 worktree. Neither failure was hidden or used as G2 evidence.

`compileall` and `git diff --check` both passed for the G2 scope.

## Verdict

`PASS — P0=0 / P1=0 / P2=0`

G2 establishes development search evidence and scheduling semantics only. It
does not establish final confirmation, treatment effect, Main readiness or
external acceptance.
