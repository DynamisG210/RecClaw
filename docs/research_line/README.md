# Research Line functional release

This directory describes the compact Research Line release surface: the
four-role producer path, outcome-blind portfolio/resource controls, the
round/campaign runtime, and the legacy compatibility projections consumed by
the existing Helix stack.

## Included

- the integrated Research Line runtime and its canonical three-arm state model;
- Provider physical-call, consumer logical-call, and candidate-instance
  identity separation;
- Arm-private mutable ownership and cross-Arm isolation;
- common materialization, training, result, observation, and round-closure
  paths;
- Research Capability producers, Router, Search Memory, Meta control, and
  frontier admission;
- the EvidencePort/Helix boundary and typed evidence projections;
- executable BL-ICF/model resources, generic workers, and self-contained
  functional regression tests.

## Excluded from the release tree

- milestone and nightly process documents;
- V13-V25 contracts, audit packets, recovery notes, and launch wrappers;
- generated Pilot result databases, logs, Broker outputs, and materialized
  artifacts;
- machine-specific qualification and remote-supervision helpers;
- historical and external-backend integration tests whose prerequisites are
  outside this compact source release.

Historical process documents, generated artifacts, and machine-specific
evidence remain outside this public release surface.

## Evidence boundary

This is a functional code release, not a research-result release. No run
results, formal-effect claim, Main authorization, held-out result, or formal
acceptance is included here. Ongoing experiments must remain in their
environment-owned run roots and must not be back-written into this branch.

## Evidence Guard -> Helix handoff

The Research Line side is ready for a narrow Evidence Guard handoff. The
existing `src/recclaw_core/helix/` package is the only allowed adjacency to
`recclaw_evidence_guard`; keep that trust boundary intact.

### Implemented contracts

- `ResearchContext`, `ProducerOutcome`, `BehaviorProjection`, and
  `ResearchTaskQueueV2` live in `src/recclaw_core/research_line/interfaces.py`.
- `project_common_execution_feedback()` emits the Helix
  `SearchUtilityEventV2`, a frozen comparison identity, a typed episode/closure
  when applicable, and a durable task-queue transition.
- `ResearchTaskQueueV2.to_legacy_task()` preserves the existing
  `ResearchTaskV1` projection; the no-portfolio route remains accepted by the
  router and campaign runtime.
- `src/recclaw_core/helix/contracts.py` defines the guard-facing immutable
  `CandidateEnvelope`, `RawResultEnvelope`, `PortAdjudication`, and
  `CompactFeedback` objects. `EvidenceGuardPortV1` performs PRE/POST calls;
  `HelixFusionBridgeV1` maps compact feedback to Search Memory destinations.

### Boundary for the teammate

1. At the campaign runner boundary, after a candidate binding is resolved and
   before `run_with_physical_context()`/the worker launch, construct a
   `CandidateEnvelope` with the candidate semantic/program/plan digests, opaque
   arm instance id, `COMMON_PASS`, planned protocol, model/comparator, seed
   ids, action family, and purpose. Call `EvidenceGuardPortV1.pre_run()`.
2. Do not launch when PRE is `BLOCK`. A PRE `ALLOW` is the only authorization
   to execute the candidate; `NullEvidencePortV1` deliberately remains
   `NOT_ADJUDICATED`.
3. After the common result closure is durably written, construct a
   `RawResultEnvelope` with the raw-result and closure digests, observed
   protocol, model/comparator, seed-run records, run status, artifact identity
   status, and normalized metrics. Call `EvidenceGuardPortV1.post_run()`.
4. Feed the returned compact feedback through `HelixFusionBridgeV1`; only the
   Research Line `SearchUtilityEventV2`, typed episode/closure, and task-queue
   transition may update Search Memory/frontier state. Guard ledger rows stay
   development-only and private to the run root.

### Focused verification and experiment entry

The minimum offline check is:

```text
python -m pytest -q tests/research_line/test_interfaces.py tests/research_line/test_bootstrap.py tests/research_line/test_campaign.py tests/research_line/test_execution.py tests/research_line/test_interpreter.py tests/research_line/test_portfolio_profile_builder.py tests/research_line/test_router_portfolio_v2.py tests/research_line/test_runtime.py tests/research_line/test_single_round.py
```

Use `scripts/run_research_line_single_round.py --preflight` for a no-call
round composition check. The bounded Helix experiment entry is
`scripts/run_research_line_standalone.py`; supply teammate-owned private
`--api-config` and `--run-root` values, then use `--resume` for continuation.
This release does not claim a formal run result and does not perform a
Provider/GPU/remote run as part of publication.

### Not yet complete

The Research Line production runner is not yet wired to invoke
`EvidenceGuardPortV1` around each physical attempt. The Evidence Guard
teammate owns that adapter wiring, the Helix feedback projection, and the
independent Helix experiment run.

## Stable entry points

- `scripts/run_research_line_single_round.py`
- `scripts/run_research_line_standalone.py`
- `src/recclaw_core/research_line/`
- `src/recclaw_core/experiments/helix_abc_v1/`
