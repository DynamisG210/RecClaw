# M2 Independent Adversarial Audit

## Verdict

`PASS_WITH_NONBLOCKING_P2`

```text
P0 = 0
P1 = 0
P2 = 2
authority = NONE
evidence_class = DEVELOPMENT_ONLY
formal_acceptance = false
```

This fresh-context pass reviewed the M2 implementation, quality-gate logic,
fixture lineage, generated artifacts and the M0/M1 regression boundary. It is
an independent development review inside this autonomous run, not external
acceptance.

## What was challenged

- repeated or mislabeled physical calls masquerading as independent Producers;
- unequal model, BL projection, schema, Proposal or Token contracts across the
  three agentization modes;
- post-hoc role assignment, missing falsification, incomplete lineage and
  discovery credit given to control/repair;
- incomplete candidate-pool or Router reason traces;
- Candidate-ID leakage, one-Producer, one-family and tuning-only Meta collapse;
- a versioned Meta digest that never changes next-round behavior;
- mutable proposals or mutable Search Memory predecessor chains;
- Evidence Authority fields or Guard/Fusion imports entering Research code;
- B/C Research identity drift and Arm A calling Research components;
- M1 common runtime release drift.

## P1 findings resolved before verdict

1. The first authority scan recursively inspected the opaque shared BL Program,
   whose frozen common schema itself contains `claim_ceiling`. The corrected
   boundary audits only Research-owned lineage, utility and memory fields;
   Research code neither reads nor interprets that shared field.
2. The quality gate initially did not prove unique physical-call IDs, role
   context/memory/RNG scopes or controlled model/BL/schema equality. Those
   conditions now fail the gate.
3. Meta initially emitted a new policy but the Controller did not consume it.
   The next round now applies versioned Producer token allocation, mechanism
   axis targeting, memory retrieval policy and Router priors. The gate and E2E
   require an activated policy effect.

## Gate evidence

- Agentization: `PASS_INDEPENDENT_MULTI_AGENT`.
- Selected mode: `BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1`.
- Meta: `PASS_VERSIONED_META`.
- Standalone readiness: `PASS`.
- Producer lineage complete: `1.0`; post-hoc relabel: `0`.
- Falsification slot, control/repair credit separation and complete route trace:
  present.
- Research package has no Guard/Fusion import and the Guard package is absent.
- M1 common release projection and source-manifest digests remain unchanged.
- 18 M2 targeted/E2E/adversarial tests and 99 prior targeted/regression tests
  passed.

## Nonblocking P2

`M2-P2-001`: the agentization fixture validates the independent call contract,
lineage, mechanism diversity and outcome-masked evaluator deterministically; it
does not establish that a real LLM will preserve those rates. M5 must validate
the selected physical broker plan and actual token/latency records.

`M2-P2-002`: Search Memory is an immutable typed in-process chain in M2. Its
integration with the durable single-writer store, crash recovery and
cross-process exclusivity remains an M4 pre-Canary gate.

Neither risk is a reason to add speculative runtime infrastructure to M2.

## Conclusion

All mandatory M2 automatic-continuation conditions are satisfied with no
unresolved P0/P1. The Research Capability Line is standalone and Search
Utility-only, and M3 may add the Guard only through its separate adapter.
