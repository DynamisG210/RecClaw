# M6I Final Independent Audit

## Verdict

`PASS — P0=0 / P1=0 / P2=1`

This is a mechanical cross-evidence audit of the current source bytes. It
recomputed invariants from the source projection, raw report contents,
per-seed exact-scheduler summaries, permanent failure fixtures and Provider
probe receipts. It did not accept a generator's top-level `PASS` as sufficient.
Authority remains `NONE`; evidence remains `DEVELOPMENT_ONLY`; formal
acceptance remains false.

## Canonical observation semantics

The transition matrix and production classifier now use the same function in
`integrated_state_core.py`.

| Class | Search Memory | Meta admitted | Meta boundary | Belief | Observed | SearchEligible | Confirmed | Task | Round | Resource ledger |
|---|---|---|---|---|---|---|---|---|---|---|
| admitted success | update | yes | exactly once | allowed | yes | yes | no | normal lifecycle | terminal | actual |
| preliminary / requires confirmation | update | yes | exactly once | allowed | yes | preliminary only | no | validation enqueued | terminal | actual |
| protocol branch | task projection | no | exactly once | no | yes | no | no | protocol task lifecycle | terminal | actual |
| diagnostic / engineering / Guard quarantine | no result update | no | exactly once | no | yes | no | no | none or typed diagnostic | terminal | actual |
| not admissible / no-search-update | no | no | exactly once | no | yes | no | no | none | terminal | actual |
| common execution / resource / training failure | failure fact only | no | exactly once | no | yes | no | no | active task cancelled | terminal | actual |
| Provider failure before proposal | no | no | exactly once | no | no | no | no | none | typed broker-failure terminal | actual broker debit |
| all PRE blocked | no | no | exactly once | no | no | no | no | active task cancelled when applicable | no-execution terminal | actual broker debit |

Ordinary search rounds never write Confirmed frontier state. Preliminary
signals remain explicitly preliminary.

## Recomputed closure evidence

- Ownership: 639 mutable findings classified; zero unowned
  treatment-dependent objects; zero forbidden global mutable objects.
- Identity: ProviderPhysicalCallId, ConsumerLogicalCallId and
  CandidateInstanceId are separate; active policy is `ARM_PRIVATE`.
- State machine: every frozen proposal, task, result and Meta branch reaches a
  typed terminal; V19, V20 and V21 fixture bytes match their recorded hashes.
- Order invariance: all six A/B/C orders pass; 500 deterministic schedules
  cover 4,000 multi-round order instances.
- Synthetic qualification: 100 seeds × 50 rounds/Arm, 15,000/15,000 terminal,
  zero open rounds, duplicate claims/feedback, unsafe cache hits, foreign
  parents, cross-Arm reads/writes or Confirmed frontier entries.
- Exact production scheduler qualification: 100 seeds × 50 rounds/Arm,
  15,000/15,000 terminal, 5,000 closed triplet barriers, 10,000 B/C Meta
  boundaries, zero open rounds, unfinished claims, duplicates or cross-Arm
  physical identities. Each seed exercised Provider failure, four typed result
  faults and all frozen task fixtures.
- Provider isolation: 12 fresh `gpt-5.4` calls across all six orders, zero
  training, distinct Arm-private physical/logical/candidate identities,
  unchanged SearchMemory/Meta/frontier state, and no credential residue.
- Source projection: every current qualified source hash equals the exact
  100×50 report and the before/after projection is unchanged.
- Sealed Meta: V19 checkpoint SHA-256 remains
  `27e3118e178e4ebd45f54dd9b0e7239af9a1997d021fc648a800166bc52e2173`.

## Regression boundary

The active suite passes `443 tests + 390 subtests`. Ten tests are separately
deselected because they assert sealed historical or obsolete local-environment
conditions: a missing untracked V12 contract, local SciPy ABI drift, obsolete
V18/V13 identity constants, and the intentionally existing sealed V13 root.
They do not execute the active M6I path and were not altered to manufacture a
green result. This is the sole P2.

## Final decision

M6I satisfies the current-byte pre-Pilot gate. The next legal action is a local
checkpoint commit followed by selection and qualification of one common
backend/runtime for a fresh five-round-per-Arm full-training Pilot. No Pilot
outcome has been used in this audit.
