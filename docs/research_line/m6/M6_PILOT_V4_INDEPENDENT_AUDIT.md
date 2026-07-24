# M6 Pilot V4 Independent Audit

Verdict: `FAIL`

```yaml
P0: 0
P1: 3
P2: 0
pilot_verdict: NOT_READY
authority: NONE
evidence_class: DEVELOPMENT_ONLY
formal_acceptance: false
M7_started: false
M8_started: false
```

## What did close mechanically

- The exact frozen V4 contract and source projection still verify.
- All 23 broker calls succeeded with no retry or reused response. Actual broker
  cost was 308,034 input tokens, 9,077 output tokens and 317,111 total tokens.
- All nine SearchRounds closed exactly once. All nine execution claims are
  `FINISHED` and debited once; all three triplet barriers are `7`.
- SQLite integrity is `ok`, foreign-key violations are zero, all nine raw
  result envelopes are indexed, and the common resource audit reports no
  per-Arm/round ceiling violation.
- A/B/C execution claims bind one runtime release, runner ABI, purpose, metric
  contract and resource contract. There are nine exact per-run runtime
  bindings.
- The C-private evidence ledger is intact and contains six Guard calls.
- The V1/V2/V3 Pilot roots and records were not reused or modified.

These facts establish useful mechanical closure only. They do not override the
failed Pilot gate.

## P1-001 — Authoritative audit path cannot complete

After all nine rounds, `RealPilotOrchestratorV1.pilot_audit()` called
`self.store.integrity_check()`. The active store is
`TrainingSingleWriterExperimentStoreV1`, which does not expose that method.
The frozen runner exited with:

```text
AttributeError:
'TrainingSingleWriterExperimentStoreV1' object has no attribute
'integrity_check'
```

The authoritative script therefore wrote `PILOT_FAILURE.json` with
`verdict=NOT_READY` and did not produce the frozen neutral audit, ITT rows,
readiness packet, runtime-identity packet or Pilot execution result.

## P1-002 — Frozen Pilot readiness is independently NOT_READY

A read-only invocation of the already-frozen analysis function over the nine
sealed raw envelopes did not use effect size and returned:

```yaml
row_count: 9
success_count: 6
failure_count: 3
failure_rate: 0.3333333333333333
failure_rate_ceiling: 0.25
support_by_opaque_instance: [0, 3, 3]
verdict: NOT_READY
```

All three failures belong to one still-opaque instance. Each selected
LightGCN, and each worker failed with the same runtime error:

```text
AttributeError: 'dok_matrix' object has no attribute '_update'
```

The frozen RecBole LightGCN implementation calls this removed/private SciPy
method. No treatment mapping was written or inspected, and no NDCG or Arm
effect comparison was performed.

## P1-003 — Observed write-confinement violation

Each of the nine RecBole executions created one empty default logger file
under the shared project path `log/`. This path is outside every Arm-private
runtime root and outside the V4 result root. The nine files have tree digest
`0eea1c54d8a258343164bd162dc1469bce26a1602ec5651e836a05ea23ef0bea`.

Although the files are empty and no source byte changed, this is an actual
shared-root write and fails the frozen confinement invariant.

## Accounting interpretation

The broker database is the authority for actual physical calls and actual
provider tokens: 23 calls and 317,111 tokens. The neutral resource ledger
contains 27 `PHYSICAL_LLM_CALL` and 371,086 `BILLED_TOKEN_DEBIT` units because
the first-round B/C shared replay is charged to each Arm's budget projection.
Per-Arm/round ceilings pass, but these projections must not be reported as the
actual physical provider totals.

## Conclusion

V4 is sealed `NOT_READY`. The recovery command gives a fresh Pilot no
ad-hoc patch budget and requires a hard stop when it is not GO. The audit API,
LightGCN runtime compatibility and shared logger root may not be repaired in
place, and V4 may not be rerun. M7 and M8 remain forbidden.
