# M0 Human-Review Handoff

> - Milestone: `M0 — Contract Kernel + Single-Writer Store`
> - Local verdict: `PASS`
> - Authority: `NONE`
> - Evidence class: `DEVELOPMENT_ONLY`
> - Formal acceptance: `false`
> - M1 status: `NOT_STARTED`

## Delivered slice

M0 implements only the machine-readable A/B/C experiment contract, the closed
Producer and Meta enums, one-session/equal-resource contracts, fixture-only
Controller seams, `NullEvidencePortV1`, deterministic M0 Fusion, and
`SingleWriterExperimentStoreV1`.

The store uses SQLite WAL, `synchronous=FULL`, foreign keys, and
`BEGIN IMMEDIATE`. Its schema contains exactly:

```text
scheduled_slots
rounds
round_events
arm_state
resource_ledger
execution_claims
triplet_barrier
artifact_index
```

It enforces one slot open, one round per Arm/seed/index, one execution claim,
one feedback, the three-Arm barrier, stop-before-open, payload-bound
idempotency, conservative recovery, and atomic artifact write/indexing.

## Original controller resolution

The exact pre-Research-Line source is:

```text
commit = 2d8c881354e1b536a6c66d7dfbb977e0c5090e50
path = scripts/agent.py
blob_sha1 = c40334b72dbb9557eced2bd081915b5333156fdf
```

`a96db67d98ccddde58e4fa641d8ccea92f10c33a` is the first commit adding
`scripts/research_line.py`, and its parent is the source commit above. The
golden fixture binds proposal schedule, selection order, one feedback
consumption, configured stop, and M0 resource debits.

## Verification

The machine execution record is
`docs/research_line/m0/M0_EXECUTION_RECORD_V1.json`.

Results:

- M0 targeted: 24 passed;
- existing Research Line: 4 passed;
- existing Candidate Proposal: 17 passed;
- existing BL-ICF mechanism space: 27 passed;
- SQLite integrity and foreign-key checks: clean;
- exact M0 table count: 8;
- activated public dataclass models: 6, below the roadmap limit of 18;
- direct import and forbidden-field boundaries: passed.

The BL regression suite required the two versions already locked in
`requirements-search-space.txt`; they were installed in `/tmp` only.

## Explicitly not established

This handoff does not establish:

- CommonExecutionGuard implementation or runtime conformance;
- BL materialization or Runner execution;
- Producer, Router, Meta, or Search Memory runtime capability;
- Evidence Guard integration or evidence admissibility;
- Canary, Pilot, Main, scientific benefit, permission release, or formal
  acceptance.

No real LLM, materialization, RecBole training, Runner, Canary, Pilot, Main,
commit, or push occurred.

## M1 entry conditions

M1 may start only after:

1. the user explicitly authorizes M1 in a new task;
2. this M0 diff, execution record, migration digest, and golden source tuple
   receive human review with no unresolved P0/P1;
3. the three pre-existing EOL-only files remain untouched;
4. M1 stays limited to the common BL executable vertical slice and does not
   introduce a real LLM or ordinary training;
5. the M1 contract freezes one exact CommonExecutionGuard implementation shared
   byte-identically by A/B/C.

