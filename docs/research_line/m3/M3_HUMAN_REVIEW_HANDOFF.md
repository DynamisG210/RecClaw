# M3 Human-Review Handoff

> - Milestone: `M3 — Evidence Guard Adapter + Helix Composition`
> - Local verdict: `PASS_WITH_NONBLOCKING_P2`
> - Authority: `NONE`
> - Evidence class: `DEVELOPMENT_ONLY`
> - Formal acceptance: `false`
> - Next eligible milestone: `M4`

## Delivered composition

```text
A: Candidate/Result → NullEvidencePortV1 ┐
B: Candidate/Result → NullEvidencePortV1 ├→ same DeterministicHelixFusionV1
C: Candidate/Result → guard_adapter.py   ┘
                         ↓
              exact Guard core + C-private audit ledger
```

The selected Guard evaluator is an exact-byte copy from
`phase1/evidence-guard-ab002@0c868c2`:

```text
d47df73feec97a01f2528cbf110b62c473d16414fcfc94ffefaaad3ff0a7c1af
```

Only `src/recclaw_core/helix/guard_adapter.py` imports that package. The adapter
returns shared typed adjudication, not the full event. Full PRE/POST events
remain in C's separate `DEVELOPMENT_ONLY/EVIDENCE_AUDIT` ledger.

## Deterministic boundary behavior

- Null returns `NOT_ADJUDICATED`, never `ALLOW` or `ADMISSIBLE`.
- PRE block advances to the next item in the same frozen slate.
- No Producer refresh, LLM call, proposal or execution budget is added.
- POST produces the exact eight-field Compact Feedback.
- Fusion Bridge maps confirmation, diagnostic, exclusion and protocol-branch
  outcomes deterministically.
- Search Memory receives only a narrow instruction and Compact Feedback digest,
  never the full audit event.
- wrong-arm Candidate/Result substitution fails closed.

## Verification

- exact Guard-core tests: 13 passed;
- M3 composition/E2E/adversarial: 13 passed;
- M0-M2/BL/legacy targeted regressions: 117 passed;
- exact core identity and unique-import scan passed;
- reference dispositions matched all four arm-blind cases;
- M1 common and M2 Research frozen bytes remain unchanged.

## Explicitly not established

M3 does not establish real broker/process isolation, real training, actual
comparator delta, Canary, Pilot, Main, scientific benefit, evidence authority,
formal acceptance or permission release.

## M4 entry conditions

M4 may start from the M3 checkpoint if:

1. M1 common, M2 Research and M3 Guard/Fusion identities remain frozen;
2. the three-arm broker uses opaque assignment and arm-private roots;
3. only the broker owns state-store and Guard-ledger writers;
4. fake E2E proves one round, at most one ordinary execution and one feedback;
5. crash, duplicate, contention, stop and filesystem-isolation negatives pass;
6. the output is a sealed `READY_FOR_CANARY` review packet, not a Canary run.
