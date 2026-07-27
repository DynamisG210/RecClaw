# M6G G1 Independent Adversarial Audit

## Scope

The review covered only the active V13 path from EvidencePort PRE/POST through
deterministic admission, Research controller, Meta, Search Memory and Producer
prompt projection. It did not assess G2 or later stages and did not call a
Broker, training runtime or Provider.

## Initial independent verdict

`FAIL — P0=0 / P1=1 / P2=0`

The independent reviewer found that a `PROTOCOL_BRANCH_TASK` correctly removed
the `SearchUtilityEventV2` from `FusedSearchFeedbackV2`, but the orchestrator
still constructed a mechanism belief from the pre-admission event. That belief
could therefore enter Search Memory through a path not authorized by the fused
feedback.

The same review confirmed:

- all-PRE-blocked slate exhaustion preserved controller, Meta and Search Memory;
- no CompactFeedback, Guard reason, claim ceiling, Guard digest or raw artifact
  locator reached Research, Meta or Producer prompts;
- the focused pre-fix suite passed 27 tests plus 3 subtests.

## Remediation

The orchestrator now creates a mechanism belief only when the admitted
`FusedSearchFeedbackV2` itself contains `search_utility_event`.

`ResearchLineControllerV1.close_round_v13()` independently rejects any
task-only feedback carrying mechanism beliefs. A dedicated adversarial test
exercises the exact protocol-branch smuggling case.

## Requalification evidence

```text
PYTHONPATH=src .../python -m pytest -q \
  tests/experiments/helix_abc_v1/test_v13_scientific_attribution.py \
  tests/experiments/helix_abc_v1/test_m4_precanary.py

28 passed, 3 subtests passed
```

Affected Campaign and Meta regressions:

```text
PYTHONPATH=src .../python -m pytest -q \
  tests/experiments/helix_abc_v1/test_v13_scientific_attribution.py \
  tests/experiments/helix_abc_v1/test_m4_precanary.py \
  tests/experiments/helix_abc_v1/test_meta_v17_campaign.py \
  tests/experiments/helix_abc_v1/test_campaign_pilot_readiness.py \
  -k "not frozen_campaign_pilot_contract_is_pre_outcome_and_exact"

49 passed, 1 deselected, 3 subtests passed
```

The deselected test verifies the superseded V12 frozen source digest. It must
remain unchanged and is not a V13 regression.

## Final audit-cycle verdict

`PASS — P0=0 / P1=0 / P2=0`

The independent finding's stated minimal acceptance condition is now enforced
both at the caller and controller contract boundaries, and the exact negative
case is covered.
