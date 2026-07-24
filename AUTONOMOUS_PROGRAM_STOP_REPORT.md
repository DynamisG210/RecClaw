# RecClaw Research Line Autonomous Program Stop Report

## Stop verdict

```yaml
program: HELIX-ABC-001
milestone: M6
verdict: AUTONOMOUS_HARD_STOP
pilot_verdict: NOT_READY
authority: NONE
evidence_class: DEVELOPMENT_ONLY
formal_acceptance: false
M7_started: false
M8_started: false
```

The autonomous M1–M8 program stopped at M6. Pilot V3 did not reach `GO`
after the two bounded M6 repair rounds permitted by the Master Goal. No
further implementation, Pilot version, Main Campaign, push, promotion, or
formal claim is authorized.

## Exact blocker

The M6 Pilot training launcher and the frozen M1 common runtime release use
different runner contracts:

- `PilotTrainingLauncherV1` emits
  `recclaw.package-owned-pilot-training-runner.v1`;
- `SingleWriterExperimentStoreV1.mark_execution_started()` accepts only
  `recclaw.fake-non-training-runner.v1`;
- the current materialization, binding, permit, CommonExecutionGuard, and
  close-result path are also rooted in the fake non-training runner release.

V3 therefore failed before the training subprocess was spawned:

```text
InvariantViolation: execution start receipt does not bind the claim
```

This is not safely repairable by changing another Pilot-only string. A
training-capable, content-bound CommonExecution runtime release must be
versioned and reviewed end to end.

## Affected invariant

The package-owned launcher may start an ordinary execution only when the
receipt, claim, binding, permit, runner ABI, runtime release, result closure,
and budget debit all bind the same exact execution contract.

Weakening the store check, relabeling the training runner as the fake runner,
or bypassing CommonExecutionGuard would violate the execution identity and
three-arm common-runtime invariants.

## Evidence

### Frozen V3 identity

- pre-outcome commit:
  `6bded5498da90309a8d11cb5079f6ec820afadfb`
- pre-outcome tree:
  `48ff97200af3734088a6423581865fdd6562d5ac`
- Pilot contract content digest:
  `61a1b35b84dcd5fdb12528348de1a488957e8a53eac775b80b70611d7892d5ea`
- Pilot contract SHA-256:
  `6469497764772df84345251fe9e5510a9bea700520aa04cfa4543713a1f210af`
- Pilot seed: `9203`
- result root: `results/research_line/m6_pilot_9203_v3`
- failure packet SHA-256:
  `588a36bc22a675ed50d80f83e01c8fa4e4d68b7032f77dbc46e23803a5181fe9`

### Mechanical state at failure

- broker calls: `1 / 23`, all successful, `0` retries;
- billed broker tokens: `14170`;
- execution claim: present and still `CLAIMED`;
- execution-start receipt: content-addressed and indexed;
- ordinary execution debit: `0`;
- training backend starts: `0`;
- NDCG or other training outcome observed: `false`;
- A/B/C result comparison performed: `false`.

The indexed receipt binds the committed claim, permit, round, run, and binding
digests, but carries the Pilot training ABI. The store rejects it because
`state_store.py` hard-codes the fake non-training ABI. The current
close-result path contains the same fake-runner and non-outcome-bearing smoke
assumptions, so changing only the start-receipt comparison would remain
incomplete.

## Attempted repairs

1. Pilot V1, seed `9201`, failed before training because the Pilot adapter
   compared `CommonDecision.PASS.value` to the shorthand `"PASS"`.
2. Repair round 1 created Pilot V2, seed `9202`, and fixed that exact enum
   comparison. V2 then exposed the paired gate shorthand mismatch:
   `GateStatus.ALLOW.value` is `ALLOW_DEVELOPMENT_FAKE_RUN`, not `"ALLOW"`.
3. Repair round 2 created Pilot V3, seed `9203`, replaced both shorthands with
   their frozen enum values, added exact regression coverage, reran targeted,
   activated, Evidence Guard, BL-ICF, legacy, identity, and environment gates,
   and passed preflight.
4. V3 advanced past both common decisions and created the execution claim and
   receipt, then exposed the substantive training-runtime release mismatch
   above.

V1, V2, and V3 are closed. Their broker responses, seeds, state, and artifacts
must not be reused in a future Pilot or Main initial state.

## Independent severity assessment

```yaml
P0: 0
P1: 1
P2: 0
```

The P1 is an M6-blocking execution-contract incompatibility. It caused no
scientific contamination and no unaccounted training/GPU execution because
the system failed closed before spawning the backend.

## Safest next options

1. **Recommended:** open a separately authorized recovery milestone for one
   versioned, training-capable CommonExecution runtime release. Preserve the
   existing fake non-training release as a closed M1/M4 smoke profile; bind a
   new exact runner ABI, evaluation purpose, start receipt, raw result,
   binding, permit, store verification, and close-result policy for Pilot
   training. Re-run the affected M1/M4 contract and adversarial suites before
   freezing a fresh Pilot with a new seed.
2. Keep M6 stopped and use M0–M5 only as development evidence. This preserves
   all current guarantees but cannot support a three-arm training experiment.
3. Redesign the Pilot around the fake non-training runner. This is not
   recommended because it cannot establish Main training feasibility.

Do not fix this by allowing an arbitrary runner ABI, accepting the ABI from
the receipt itself, renaming the training runner to the fake runner, or
skipping the store/CommonExecutionGuard checks.

## Recommended recovery scope

The smallest coherent recovery is:

1. define a closed training runtime policy/release with one exact runner ABI
   and evaluation purpose;
2. make receipt/raw-result store verification consume that pre-bound policy,
   retaining the existing fake-runner default behavior;
3. bind materialization, `CandidateExecutionBindingV2`, permit, launcher,
   receipt, raw result, and close-result to the same release digest;
4. add positive training-receipt and negative substitution/replay tests;
5. rerun the affected CommonExecution and pre-Canary suites;
6. only after an independent P0/P1-free review, create a fresh Pilot version,
   fresh seed, and fresh output root.

This requires new user authorization because the autonomous repair budget is
exhausted and the Master Goal hard-stop condition has fired.

## Current trustworthy state

- M0–M5 remain locally checkpointed development work.
- M6 implementation and three sealed Pilot attempts are preserved locally.
- M6 is `NOT_READY`; no exact Main freeze exists.
- M7 and M8 were not started.
- No training outcome, scientific arm contrast, Evidence Guard effect, or
  formal claim has been established.
- No remote branch, PR, release, promotion, or authority ledger was modified.
