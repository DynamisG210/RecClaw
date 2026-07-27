# V14 Result Audit

## Verdict

`NON-GO — P0=0, P1=2`

V14 is permanently sealed and cannot be patched, rerun or reused. The first
training execution produced a valid ML-1M development result, but exceeded
two frozen resource ceilings. A second implementation defect then propagated
the expected CommonExecutionGuard rejection as an exception instead of
closing the SearchRound with a typed common-execution-failure result.

## Observed result

- candidate: `bl1_1bfdb8d3f321304e770e`
- model: `LightGCNComposableV2`
- training exit: `SUCCESS`
- development NDCG@10: `0.2054`
- wall time: `1,477,269 ms`
- GPU device time: `1,477,269 ms`
- normalized GPU cost: `410,352 microunits`
- filesystem confinement: `PASS`
- shared-root side-effect audit: `PASS`

The frozen ceilings were:

- GPU device time: `900,000 ms`
- GPU cost: `250,000 microunits`
- wall time: `1,500,000 ms`

The result was therefore correctly ineligible under the frozen common
resource contract even though training itself succeeded.

## Runtime closure defect

`CommonTrainingExecutionGuardV1.close_result()` returned a rejected
`CommonResultClosureV2` and no admissible `RawResultEnvelopeV2`.
`PilotTrainingLauncherV1` raised immediately before it:

1. persisted the rejected common closure;
2. debited actual GPU/wall resources into the round ledger;
3. emitted a typed no-metric common-execution-failure envelope;
4. completed normal feedback and round closure.

The sealed database consequently contains one `OPEN` round, one finished
execution claim, 12 registered artifacts and no round feedback/closure.

## Attribution impact

- only one opaque Arm instance started;
- no Evidence Guard POST decision occurred;
- no Search Memory, Meta observation or frontier update occurred;
- no B-A or C-B effect is estimable;
- the development metric is runtime diagnostic evidence only;
- no held-out data was accessed.

## Root-cause classes

1. `LOCAL_BACKEND_RESOURCE_INCOMPATIBILITY`: the frozen Main-grade training
   recipe is about 64% above the GPU-time ceiling and 64% above the normalized
   GPU-cost ceiling on the local RTX 4070 Laptop GPU for this candidate.
2. `REJECTED_RESULT_CLOSURE_GAP`: an expected common rejection aborts the
   orchestrator instead of becoming a typed closed failure row.

These are distinct from the closed V13 Provider-schema failure class.

## Recovery boundary

The successor must:

- persist and account rejected common results exactly once;
- close their SearchRounds without admitting their metrics to search;
- keep the frozen resource ceilings unchanged;
- qualify a faster contract-equivalent backend or an execution-equivalent
  performance implementation without changing training semantics;
- use a new source checkpoint, contract, Pilot version, seed, state and root.

Changing the ceilings after seeing V14 is not authorized and is not proposed.
