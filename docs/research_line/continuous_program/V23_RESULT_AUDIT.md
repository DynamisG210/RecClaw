# V23 Result Audit

## Verdict

V23 is permanently sealed as `NON_GO`, with `P0=0` and `P1=1`.
It is development failure evidence only and provides no treatment-effect or
formal-acceptance evidence.

## Natural terminal state

- The supervisor exited naturally with exit code `1`.
- No V23 Pilot, worker, or supervisor process remains.
- The temporary LLM configuration and supervisor script are absent.
- The frozen output root remains intact and must not be reopened.
- Eight rounds are closed, B round 3 is open, and six later slots were never
  opened.
- All 8 execution claims reached `FINISHED`.
- All 13 Broker calls reached `SUCCESS` with retry count zero.

## Failure

The B round-3 `lineage_refiner` call returned:

```text
parent_candidate_id=cand-1b8e552d7879c18ef420544c
```

The frozen response schema required `^cand-[a-f0-9]{24}$`, while the
production `CandidateInstanceIdV1` constructor emits `cand-` followed by the
full 64-hex SHA-256 digest. The schema therefore made an exact canonical
parent echo impossible. Production correctly rejected the returned identifier
as absent from the Arm-private exact lineage, but only after all four B round-3
Producer calls had completed.

This is a Provider-output-schema versus canonical-identity contract defect,
not a mechanism failure, GPU failure, VPN failure, or evidence of an Arm
treatment effect.

## Required recovery

The repair must be made in the canonical Provider-to-lineage binding path:

1. The model must not author or echo an opaque exact-parent identity when
   `parent_policy=REQUIRE_EXACT_PRIOR_PARENT`.
2. The runtime must bind the already-known Arm-private exact parent
   deterministically.
3. Candidate-instance registration must occur only after canonical parent
   resolution.
4. The Provider schema and Broker release must be versioned and requalified.
5. V23 must become a permanent regression fixture.
6. A fresh V24 may be frozen only after the affected M6I and Provider gates
   pass.

No V23 response, state, candidate, seed, database, or output root may be reused.
