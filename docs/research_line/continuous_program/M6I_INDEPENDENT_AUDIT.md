# M6I Final Current-Byte Independent Audit

## Verdict

`PASS — P0=0 / P1=0 / P2=1`

This audit mechanically recomputed the current-byte invariants from the raw
reports, per-seed exact-scheduler summaries, source-file hashes, permanent
failure fixtures, and fresh Provider probe receipts. It did not accept any
generator's top-level `PASS` as sufficient. Authority remains `NONE`, evidence
remains `DEVELOPMENT_ONLY`, and formal acceptance remains false.

## Canonical closure

- Mutable-state inventory: 648 findings classified, zero unowned
  treatment-dependent objects, and zero forbidden global mutable objects.
- Identity: `ProviderPhysicalCallId`, `ConsumerLogicalCallId`, and
  `CandidateInstanceId` are separate. The active policy is `ARM_PRIVATE`.
- Parent binding: the Provider cannot author an exact lineage parent. The
  runtime binds either the explicit root or an exact prior same-Arm parent
  before candidate registration.
- Pre-execution rejection: Provider semantic rejection and Provider process
  failure close the round as typed no-execution terminals; actual usage is
  preserved while the allocation ledger is capped at the frozen ceiling.
- State machine: every frozen proposal, task, result, no-result, and Meta
  branch reaches a typed terminal and advances the Meta boundary exactly once
  where applicable.
- Regression fixtures: V19, V20, V21, and the V23 truncated-parent failure are
  permanent replay evidence and pass on the repaired current bytes.

## Recomputed evidence

- All six A/B/C orders pass. Five hundred deterministic schedules cover 4,000
  multi-round order instances.
- Lightweight synthetic qualification passes 100 seeds × 50 rounds/Arm:
  15,000/15,000 terminal, 5,000 closed triplet barriers, 10,000 B/C Meta
  boundaries, and zero open rounds, duplicate claims, duplicate feedback,
  unsafe cache hits, foreign parents, cross-Arm reads/writes, cross-Arm
  physical calls, or Confirmed-frontier entries.
- Exact production-scheduler qualification passes 100 seeds × 50 rounds/Arm:
  15,000/15,000 terminal, 5,000 closed triplet barriers, 10,000 B/C Meta
  boundaries, and zero open rounds, unfinished claims, duplicate claims,
  duplicate feedback, or cross-Arm physical identities.
- Every exact-scheduler seed exercised all six order prefixes, a typed Provider
  failure, and the four result-fault classes: common execution failure,
  resource-ceiling rejection, training failure, and quarantine/inconclusive.
- The fresh real-Provider/no-training probe made 12 successful `gpt-5.4`
  calls across all six orders. It found no cross-Arm call or lineage reuse,
  changed no SearchMemory/Meta/frontier state, executed no training, and left
  no credential residue.
- The active regression boundary passes 457 tests plus 400 subtests. Thirteen
  sealed historical-identity or local-environment assertions are separately
  deselected; they do not execute the active M6I path and were not edited to
  manufacture a green result.

## Exact source identity

- Qualified implementation commit:
  `591878c8bfea6bf4a0104a57ec2a8651b9f46f0b`
- Exact source archive SHA-256:
  `d5d4cd347bb48005ef9f6e93e913b6fbcef38d8b852a7b363ef8c6daa5046e67`
- Current source projection digest:
  `c81641b1f4f72efd6cf86ee99e21252e15525a6c65e25a40472aa6dc4b0808b5`
- Exact scheduler report SHA-256:
  `1e7f0208449d179ff42427228133c89b5c7ac7f74b600e9ff5c443a2b28be2aa`
- Provider isolation report SHA-256:
  `bed1690bd5283e025c26a3edff8b47650fc29e63327c216081557af27978b013`

The audit independently hashed every path in the exact report's source
projection and reproduced the projection digest. The before/after projection
inside the exact run is unchanged.

## Evidence rejection and repair

The first current-byte stress attempt at commit `3a65e5d` failed with a
`NameError` while handling a Provider-authored lineage parent. That run was
rejected as evidence. Commit `591878c` added the missing error import and
updated the exact fake Provider to the runtime-owned parent contract. All
affected targeted gates and the complete exact 100×50 qualification were then
rerun and passed.

## P2 boundary and next action

P2 remains 1 only because the broad historical suite still contains 13
sealed-identity or environment-specific assertions outside the active M6I
path. This does not authorize a scientific claim.

M6I now satisfies the current-byte pre-Pilot gate. The next legal action is a
local checkpoint commit, followed by live inspection and qualification of one
common backend/runtime for a fresh V24 five-round-per-Arm full-training Pilot.
No previous Pilot outcome was used to change the treatment, 66-semantics
profile, dataset, metric, or analysis semantics.
