# M5 Independent Audit

## Scope and status

- Milestone: M5 real development Canary
- Authority: `NONE`
- Evidence class: `DEVELOPMENT_ONLY`
- Formal acceptance: `false`
- Auditor verdict: `PASS_WITH_NONBLOCKING_P2`
- P0: 0
- P1: 0
- P2: 3

This audit independently inspected the frozen contract, the broker ledger, the
neutral experiment store, the C-only Guard ledger, the sealed mapping, materialized
BL-ICF artifacts, source identities, and the activated regression suites. It did
not infer readiness from the executor's verdict alone.

## Independently recomputed gates

| Gate | Recomputed result |
|---|---:|
| Frozen contract content digest | `bb9a15128a0d23d24566b5beddee162fe98a8c1b2c7859423cb41d5cedf22e4e` |
| Broker calls | 15 success, 0 failure, 0 retry |
| Maximum tokens in one call | 14,186 of 20,000 |
| Broker total tokens | 200,735 |
| Broker aggregate latency | 1,894,932 ms |
| Search rounds | 9 total, 9 closed |
| Ordinary execution claims | A=3, B=3, C=3 |
| Round feedback events | 9 |
| Triplet barriers | rounds 1–3 each bitmap 7, next index authorized |
| Common execution starts | 9 |
| Training backend starts | 0 |
| Guard calls | 6 in the C-only ledger |
| A/B EvidencePort state | `NOT_ADJUDICATED` in all rounds |
| C EvidencePort state | `ADJUDICATED` in all rounds |
| B/C selected candidate equality | true in all rounds |
| B/C actual token-debit equality | true in all rounds |
| Cross-arm mutation | 0 |
| Source identity mismatch | 0 |
| Neutral SQLite integrity / FK | `ok` / 0 violations |
| Broker SQLite integrity | `ok` |
| Guard SQLite integrity | `ok` |
| Result tree | 115 files, 707,797 logical bytes |
| Result tree manifest digest | `353917d045c95701219d2b44ad5e1a2e0fe454cd4d0232e4935e251dcb8e3b97` |

The three B/C candidate pairs are identical before the EvidencePort boundary.
Only C has a Guard ledger and adjudicated feedback. No ordinary model training was
performed, so the Canary evaluates real broker, materialization, interface smoke,
identity, isolation, accounting, and record closure—not recommendation quality.

## Regression evidence

- M4/M5 focused suite: 14 tests passed.
- M0–M5, Evidence Guard, and BL-ICF activated suite: 136 tests passed.
- Legacy candidate, reflection, Research Line, lint, and agent-loop suite:
  120 tests passed.
- The broad one-environment discovery command was not used as gate evidence
  because the audit-only environment intentionally lacks PyYAML and the RecBole
  environment intentionally lacks `rfc8785`. The dependency-appropriate suites
  above cover the same modules and passed.

## Non-blocking P2 observations

1. The Canary intentionally used package-owned non-training interface smoke.
   Bounded real candidate training, GPU metering, and recommendation endpoints
   remain M6 Pilot obligations.
2. The external RecBole checkout is dirty but was exact-byte bound for M5. M6
   must freeze an isolated reproducible runtime/package identity before Pilot and
   Main; the dirty checkout cannot silently become the Main trust root.
3. Real broker latency was about 31.6 minutes for 15 calls. Pilot must establish
   the full campaign's runtime/cost feasibility without changing the frozen
   A-versus-B/C call plan based on outcome quality.

## Conclusion

M5 satisfies the master-goal continuation condition:
`READY_FOR_PILOT_REVIEW`, P0=0, P1=0, cross-arm mutation=0, identity mismatch=0,
and closed budget accounting. This is local development evidence only. It does
not approve Pilot, establish permission release, evaluate NDCG, support a
scientific claim, or authorize Main outside the active master instruction.
