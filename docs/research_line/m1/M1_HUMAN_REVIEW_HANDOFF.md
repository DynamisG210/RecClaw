# M1 Human-Review Handoff

> - Milestone: `M1 — Common BL Executable Vertical Slice`
> - Local verdict: `PASS_WITH_NONBLOCKING_P2`
> - Authority: `NONE`
> - Evidence class: `DEVELOPMENT_ONLY`
> - Formal acceptance: `false`
> - Next eligible milestone: `M2`

## Delivered slice

M1 establishes one package-owned common execution path shared by A/B/C:

```text
MechanismProgram
→ CompileReport
→ CommonExecutionGuard.plan_check
→ deterministic Materializer
→ ExecutionTrustClassification
→ CandidateExecutionBindingV2
→ CommonExecutionGuard.pre_execute
→ committed single-use execution claim
→ FakeNonTrainingRunnerV1
→ CommonExecutionGuard.close_result
→ RawResultEnvelopeV1
```

The executable profile closes BPR-MF and LightGCN anchors plus non-default
negative sampling, ranking objectives, graph propagation, regularization or
geometry, SSL/contrastive mechanisms, and package-owned architecture/operator
templates. Non-anchor NGCF, SGL, DirectAU and UltraGCN recipes run through the
same deterministic non-training path.

The runtime deliberately rejects valid BL programs that require
candidate-controlled Python. M1 does not create a general arbitrary-code
execution surface.

## Exact common identity

```text
common release projection =
97c4247af6f0a2fb2fbccd64663e2a6ed25cc1bc98aab2f5630e6d41d6f67347

executable profile =
2344a77980678093e7c768e4782a5cee6eafb82b2ffe770e247dab3d44a1ddf9

runtime source manifest =
038ebf3186ff0c67d7ebe7f5d799ea2d0f4aa6ee141f6a42d9767a05fb01a5fd
```

The release projection is byte-identical for A, B and C.

## Verification

- M1 targeted/E2E/adversarial: 25 passed;
- corrected M0 regression: 26 passed;
- BL-ICF mechanism space: 27 passed;
- legacy Research Line: 4 passed;
- legacy Candidate Proposal: 17 passed;
- JSON, canonical static/dynamic identity, import-boundary, outside-CWD,
  functional-EOL and diff checks passed.

The execution record is `M1_EXECUTION_RECORD.json`; the fresh adversarial pass
is `M1_INDEPENDENT_AUDIT.md`; open pre-Canary risks are in
`M1_RISK_REGISTER.json`.

## Explicitly not established

M1 does not establish:

- real LLM operation or ordinary model training;
- a production arbitrary-code sandbox or Stage 1B runtime permission;
- Research Producers, Router, Meta-Learning, Search Memory or their quality
  gates;
- Evidence Guard, evidence admissibility, Claim Ceiling or Protocol Branch;
- three-arm broker/orchestration, Canary, Pilot or Main validity;
- scientific effect, formal acceptance or permission release.

## M2 entry conditions

M2 may start from the M1 local checkpoint if:

1. the M1 commit contains only the recorded M1 files;
2. the three known EOL-only paths and untracked master instruction remain
   excluded;
3. the common release projection and its source/resource bytes are frozen;
4. M2 is limited to WP4/WP4.5 Research Capability and Quality Gate;
5. M2 does not implement Evidence Authority or import Evidence Guard.

`M1-P2-001` is nonblocking for synthetic M2/M3 and becomes blocking before
M5. No speculative sandbox work is required in M2.
