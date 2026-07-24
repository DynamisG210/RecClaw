# M2 Human-Review Handoff

> - Milestone: `M2 — Research Capability + Quality Gate`
> - Local verdict: `PASS_WITH_NONBLOCKING_P2`
> - Agentization: `PASS_INDEPENDENT_MULTI_AGENT`
> - Meta: `PASS_VERSIONED_META`
> - Standalone readiness: `PASS`
> - Authority: `NONE`
> - Evidence class: `DEVELOPMENT_ONLY`
> - Formal acceptance: `false`
> - Next eligible milestone: `M3`

## Delivered capability

The standalone Research Capability Line now contains:

- three closed Producer modes under one equal total-resource envelope;
- four preassigned independent roles with distinct call, prompt, context,
  memory and RNG identities;
- deterministic control/ablation and repair services with no discovery credit;
- a Search-Utility-only Static Router with closed hard gates and full pool
  trace;
- immutable eight-field Developmental Mechanism Belief and Search Memory
  snapshots;
- a round-boundary versioned Meta policy that changes next-round Producer
  allocation, axis targets, memory policy and Router priors;
- two-round B/Null standalone execution with exactly one ordinary opportunity
  and one feedback per opened round;
- unchanged Original Arm A behavior and no Research invocation from A.

The selected Research treatment is:

```text
Producer mode = BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
Router = StrongStaticRouterV1 + VERSIONED_POLICY_UPDATE
Agentization verdict = PASS_INDEPENDENT_MULTI_AGENT
Meta verdict = PASS_VERSIONED_META
```

## Gate artifacts

- `RESEARCH_CAPABILITY_QUALITY_GATE_V1.json`
  - content digest:
    `9485496e20aaa3ce2fc11b3739b5b4af81ad97c6782c6864a425635c5d4de450`
- `RESEARCH_LINE_STANDALONE_READINESS_V1.json`
  - content digest:
    `c4e7b80e43bf7004b0ef43844a559dbe1e10d6e9026f60dc3d64c805ec9d28a5`

The independent fixture improves semantic uniqueness from `0.25` batched and
`0.75` neutral to `1.0`, mechanism-axis coverage from `0.25` and `0.75` to
`1.0`, and duplicate rate from `0.75` and `0.25` to `0.0`, at the same 400
fixture token cost. These are outcome-masked development-fixture measurements,
not a real-LLM or scientific-effect claim.

## Verification

- M2 targeted/E2E/adversarial: 18 passed;
- M1 runtime: 25 passed;
- corrected M0: 26 passed;
- BL-ICF: 27 passed;
- legacy Research Line: 4 passed;
- legacy Candidate Proposal: 17 passed.

The M1 common release projection remains
`97c4247af6f0a2fb2fbccd64663e2a6ed25cc1bc98aab2f5630e6d41d6f67347`.

## Explicitly not established

M2 does not establish:

- real LLM Producer quality or actual broker cost;
- durable three-arm Search Memory/crash recovery;
- Evidence Guard, Evidence Authority or scientific admissibility;
- real training, Canary, Pilot, Main or a Research effect;
- formal acceptance or permission release.

## M3 entry conditions

M3 may start if:

1. the M2 checkpoint contains only the recorded M2 files;
2. the M1 common runtime and M2 Research source bytes remain frozen;
3. the only Guard import adjacency is the new `helix/guard_adapter.py`;
4. A/B continue using `NullEvidencePortV1`, C alone uses the Guard port;
5. Fusion remains common and deterministic;
6. no Guard output becomes Router score, Producer credit or Meta input.
