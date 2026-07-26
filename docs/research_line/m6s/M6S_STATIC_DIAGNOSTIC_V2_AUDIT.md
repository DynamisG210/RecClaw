# M6S Static Diagnostic Pilot V2 local audit

## Verdict

```yaml
diagnostic_verdict: PASS_WITH_NONBLOCKING_P2
chain_result: CHAIN_PASS
P0: 0
P1: 0
P2: 1
authority: NONE
evidence_class: DEVELOPMENT_ONLY
formal_acceptance: false
main_eligibility: false
audit_independence: LOCAL_FRESH_CONTEXT_NOT_EXTERNAL_OR_INDEPENDENT
```

The one-round static-policy diagnostic achieved its defined purpose: verify
the non-Meta A/B/C path with real Broker calls and real bounded training. It
does not satisfy the formal Pilot, Meta promotion, Main-freeze or scientific
effect gates.

## Verified execution chain

- Contract V2 used unused search seed `9207`, a new experiment identity, new
  SQLite stores and new roots.
- A used Original plus Null Evidence; B used Research Line plus Null Evidence;
  C used the same Research controller identity plus Evidence Guard.
- B/C used `RESEARCH_STATIC_V2`; Meta versions remained exactly `B=1, C=1`.
- Five provider process calls completed successfully. The common treatment
  ledger charged the intended `A=1, B=4, C=4` physical-call costs.
- Three SearchRounds, three execution claims, three GPU training starts and
  three typed feedback closures completed.
- C produced exactly one PRE and one POST Guard event.
- All three immutable audit snapshots verified; SQLite integrity was `ok`,
  foreign-key violations were empty, and the triplet barrier closed.
- A/B/C used one content-bound Training Runtime Release V2 identity. All three
  filesystem mount audits passed.
- Budget accounting closed without a violation. Source bytes were unchanged
  by execution.
- The affected M6/M6E/M6F/M6S targeted suite passed `51/51` both before and
  after execution.

## Descriptive result

```text
A NDCG@10 = 0.0337
B NDCG@10 = 0.0337
C NDCG@10 = 0.0337

B-A = 0.0000
C-B = 0.0000
C-A = 0.0000
```

This is not evidence that Research Line or Evidence Guard has zero effect.
There is one observation per Arm and no Meta. More importantly, A selected
candidate `bl1_9644...` while B/C selected `bl1_ecdf...`, but all three
compiled to the same mechanism-semantics digest and the same SGL training
projection. The candidate IDs differ because research/hypothesis payload text
differs; the executable mechanism does not.

## P2 — effect non-discrimination

The diagnostic proves chain executability, isolation, accounting and the
Static Router/Guard composition. It does not prove the chain can discriminate
Research capability effects in one round because this seed produced a
cross-Arm semantic collision.

Before the formal Meta-bearing Pilot, the pre-outcome design should require
support for multiple executable mechanism-semantics outcomes across its
scheduled pools and should report semantic collision rate. This must be
enforced without selecting seeds or candidates from observed NDCG.

## V1 repair closure

V1 is permanently sealed with zero training executions. Its failure was the
missing content-bound task-authorization input in the isolated worktree, not a
Broker or training failure. V2 added an exact-byte pre-Broker contract check
for that input and passed it. V1 was not reopened, patched, rerun or reused.

## Boundary

Meta V2 remains independently `INCONCLUSIVE`; this diagnostic neither changes
that verdict nor writes into Meta calibration, validation or held-out roots.
Formal experiments and the formal Pilot remain required to use an activated,
qualified Meta checkpoint.
