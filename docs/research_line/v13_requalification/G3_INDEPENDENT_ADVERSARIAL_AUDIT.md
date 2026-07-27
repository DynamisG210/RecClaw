# M6G G3 Independent Adversarial Audit

## Scope and method

The audit treated the Main Git object database, not the current working file,
as the reference subject. It verified path-to-blob binding, raw content,
dependency loading, direct `RecClawAgent` behavior, the V13 factory and the
actual common-route call site.

## Findings

No P0, P1 or P2 finding remains.

Adversarial checks established:

- modifying the Main working copy cannot change the loaded Original subject;
- a missing or substituted commit path/blob fails source materialization;
- the V13 factory constructs only `PinnedOriginalMainAdapterV1`;
- V13 Original routing remains active when Meta is absent;
- missing `original_priority` is rejected on the V13 path;
- `status=implemented` is added only after Common eligibility;
- Research, Meta and Guard-private fields are rejected by the projection;
- priority, proposal bonus, pending implementation, algorithm-first,
  plateau, duplicate and validated-focus behavior are produced by pinned
  `plan()`, not a parallel score;
- keep/revise/crash and memory transitions are produced by pinned
  `reflect()` and `remember()`;
- the direct pinned-source reference trace equals the V13 A trace for the
  fixed golden context.

## Verification

```text
PYTHONPATH=src .../python -m pytest -q \
  tests/experiments/helix_abc_v1/test_v13_original_main.py

10 passed
```

Affected Original/controller/Canary regressions:

```text
PYTHONPATH=src .../python -m pytest -q \
  tests/experiments/helix_abc_v1/test_v13_original_main.py \
  tests/experiments/helix_abc_v1/test_m0_contracts.py \
  tests/experiments/helix_abc_v1/test_m2_research_capability.py \
  tests/experiments/helix_abc_v1/test_m5_real_canary.py \
  -k "not three_round_fake_upstream_canary_closes_records_and_active_meta"

43 passed, 1 deselected
```

The deselected legacy Canary reaches a third metric-free validation result
after exhausting the frozen validation seeds. V13 correctly refuses to invent
another seed or relabel it confirmed. It is not the V13 route or G3
differential harness.

`compileall` and `git diff --check` passed.

## Verdict

`PASS — P0=0 / P1=0 / P2=0`

The comparator-value dependency is assigned to G4 and remains a hard
pre-Pilot requirement.
