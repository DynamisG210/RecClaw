# RecClaw M6R Resume Autonomous Stop Report

## Stop verdict

```yaml
program: HELIX-ABC-001
milestone: M6
pilot_version: V4
pilot_seed: 9204
verdict: AUTONOMOUS_HARD_STOP
pilot_verdict: NOT_READY
authority: NONE
evidence_class: DEVELOPMENT_ONLY
formal_acceptance: false
M7_started: false
M8_started: false
```

M6R itself remains passed. The one conditionally authorized fresh Pilot V4
did not reach `GO`. The recovery command explicitly gives a fresh Pilot no
ad-hoc runtime patch budget and requires a hard stop on any non-GO result.
No V5, Main freeze, M7, M8, push, promotion or formal claim was started.

## Exact blockers

### 1. Frozen post-run audit API mismatch

After all nine ordinary executions closed, the frozen audit path called
`TrainingSingleWriterExperimentStoreV1.integrity_check()`. That type has no
such method. The authoritative runner terminated with `AttributeError` and
wrote `PILOT_FAILURE.json` with `verdict=NOT_READY`.

### 2. Pilot readiness is independently non-GO

A read-only application of the frozen readiness function returned
`NOT_READY`: six successes and three runtime failures give failure rate
`0.3333333333333333`, above the frozen `0.25` ceiling, and one opaque instance
has zero successful support.

All three failed runs selected LightGCN and failed with:

```text
AttributeError: 'dok_matrix' object has no attribute '_update'
```

The content-bound environment freezes SciPy 1.15.3 while the frozen RecBole
LightGCN code calls that unavailable private method.

### 3. Actual shared-root write

All nine workers created a default RecBole log file under the project-root
`log/` directory rather than an Arm-private/result root. The files are empty,
but the shared write is a real confinement violation.

## Affected invariants

- A Pilot must complete its frozen audit and return `GO`.
- Pilot failure rate must be at most 0.25 and every opaque instance must have
  at least two successful executions.
- Package-owned runtime preflight must cover the mechanisms the Pilot may
  select, including the common LightGCN anchor.
- Training writes must remain inside the exact Arm-private/result capability.
- M7/M8 may start only after a GO Pilot and exact Main freeze.

## Preserved evidence

- Pre-outcome commit:
  `cbfbcbfd9793261a7f29364fd66b09582c68cc26`.
- Contract content digest:
  `4739c76f39f983bb2e8c50e0f6e222c725041f8c8180bc2d880bc2b62dc3cde0`.
- Result root:
  `results/research_line/m6_pilot_9204_v4`.
- Result-tree digest:
  `162b452ec63600d8c734940bd22fbf0f590a8847823cfae7a82473aba83e485f`.
- Broker: 23/23 successful calls, 0 retries, 317,111 actual tokens.
- Execution: 9 rounds, 9 starts, 9 `FINISHED` claims, 9 debits.
- Resources: 1,884,066 ms GPU device time and 523,352 GPU cost microunits;
  no frozen per-Arm/round ceiling violation.
- Runtime identity: one release/ABI/purpose/metric/resource identity across
  A/B/C; nine exact per-run bindings.
- Guard: six calls in one C-private evidence ledger.
- Store: SQLite integrity `ok`, zero foreign-key violations.
- V1/V2/V3 remain sealed and were not reused.
- Treatment mapping was not written or inspected. No effect comparison or
  NDCG-based policy decision was performed.

Mechanical closure does not convert the Pilot to GO.

## Attempted repairs

None after V4 started. This is deliberate: the recovery authorization forbids
an ad-hoc patch or another Pilot after the fresh Pilot exposes a mismatch or
returns non-GO. V4 has not been modified, rerun or reinterpreted.

## Safest next options

1. **Recommended if continuation is desired:** separately authorize a new
   bounded M6E recovery milestone. It should make only three concrete changes:
   close the training-store audit interface, content-bind a tested
   LightGCN/SciPy-compatible runtime, and redirect/verify RecBole logging under
   the exact instance-private root. Before any new Pilot, run fixed training
   canaries for LightGCN and the non-anchor mechanism path plus the full
   post-run audit. A later Pilot must use a new version, seed and roots.
2. Keep M6 stopped and retain M0-M6R plus V4 as development-only engineering
   evidence.
3. Do not waive the failure-rate/support gate, ignore the shared write, patch
   V4 in place, or reinterpret the manually reconstructed diagnostics as an
   authoritative GO.

## Current trustworthy state

- M0-M5 and M6R remain locally checkpointed development work.
- Pilot V4 is closed `NOT_READY`.
- M6 has no exact Main freeze.
- M7/M8 are not started.
- No three-arm scientific effect, Evidence Guard increment, accepted evidence
  or formal claim has been established.
- No remote branch, PR, release, promotion or authority ledger was modified.
