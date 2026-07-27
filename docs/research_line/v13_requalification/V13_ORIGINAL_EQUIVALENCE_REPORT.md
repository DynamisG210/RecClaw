# V13 Arm A Original Equivalence Report

## Decision

Arm A V13 no longer uses the handwritten `OriginalRuntimeAdapterV1` as its
decision subject. `RealCanaryProposalBrokerV1.create_v13()` constructs
`PinnedOriginalMainAdapterV1`, which executes the exact pinned Main
`RecClawAgent`.

The historical fixture and V12 adapters remain only for superseded contracts
and old unit fixtures. The V13 factory and V13 routing test cannot select them.

## Exact subject identity

- Main commit: `2d8c881354e1b536a6c66d7dfbb977e0c5090e50`
- `scripts/agent.py` blob:
  `c40334b72dbb9557eced2bd081915b5333156fdf`
- content-bound source release:
  `e5651684e852a25dfa24af00c41ab8a3b848dc0cfe62c3ea7aa0c1fb3c522027`

The loader resolves each declared path at the pinned commit, reads the exact
Git blob, verifies both Git blob SHA-1 and raw-byte SHA-256, materializes it in
an isolated temporary root, and imports it. It does not import the dirty
working copy of `scripts/agent.py`.

The release contains the exact pinned bytes for:

- `scripts/agent.py`
- `scripts/action_space.py`
- `scripts/collect_result.py`
- `scripts/compare_runs.py`
- `configs/action_space.yaml`
- `configs/candidate_proposal_schema.yaml`

## Thin allowed projection

The adapter performs only these experiment-required translations:

1. a CommonExecutionGuard-passing BL executable program becomes a wired
   Original registry entry;
2. the program's real model and entrypoint are retained;
3. `status=implemented` is derived from Common execution eligibility, not
   injected before the common gate;
4. `priority` is consumed unchanged from the Original proposal response's
   `original_priority` field;
5. the exact Original decision is projected back to the common candidate ID
   and round transition.

No Research Router score, Search Memory, Meta state or Evidence Guard field is
visible to the Original subject.

## Directly executed Original behavior

The following methods are called from the pinned source:

- `refresh_candidate_proposals()` for proposal-refresh scheduling;
- `plan()` for registry/proposal merge, scoring, pending implementation,
  algorithm-first and plateau behavior, duplicate handling and validated
  focus;
- `reflect()` for keep/revise/discard/crash decisions;
- `remember()` for feedback consumption.

The V13 A route is no longer conditional on Meta being present. Meta and the
Original controller are independent control planes.

## Differential evidence

The golden suite covers:

- proposal refresh and force refresh;
- proposal bonus;
- registry plus accepted-proposal merge;
- pending implemented priority;
- algorithm-first filtering;
- plateau penalties and forced-family bonus;
- execution-signature duplicate handling;
- validated-focus behavior;
- keep, revise and crash feedback;
- one feedback consumption per round;
- selection order;
- common one-execution/one-proposal-call budget projection;
- frozen final-round stop projection.

For fixed source, state, candidates, comparator inputs and seed, the direct
reference trace equals the V13 Arm A non-projection trace.

## Matched-comparator closure

G4 closed the earlier comparator dependency. Exact matched-comparator results
are used when available; absence remains the typed value `NOT_AVAILABLE` and
does not borrow the temporally adjacent same-Arm result. The pinned Original
therefore receives either an exact comparison or the same explicit missing
comparison state as the Research path.

No Broker, Provider, training runtime or Pilot was run in G3.
