# M3 Independent Adversarial Audit

## Verdict

`PASS_WITH_NONBLOCKING_P2`

```text
P0 = 0
P1 = 0
P2 = 3
authority = NONE
evidence_class = DEVELOPMENT_ONLY
formal_acceptance = false
```

## Scope

The fresh pass inspected the exact Guard-core transfer, unique import
adjacency, shared Envelopes, C-private ledger, PRE/POST create-once behavior,
restart recovery, Null semantics, common Fusion, Compact Feedback, deterministic
memory mapping, same-slate selection and arm-blind reference fixture.

It also re-ran the frozen M0-M2 and BL/legacy regressions and recomputed M1/M2
source identities.

## P1 findings resolved

1. Null PRE was selected correctly but its trace was indistinguishable from
   `ALLOW`. Fusion now records `SELECT_CURRENT_NOT_ADJUDICATED`.
2. The initial unique-import test inspected imported symbols rather than the
   `ImportFrom.module`, so it could miss a forbidden adjacency. It now proves
   `guard_adapter.py` is the only shared direct import.
3. Candidate and Result Envelopes initially lacked explicit `COMMON_PASS`,
   SHA-256 and opaque Arm binding. The adapter now rejects wrong-arm
   substitution and malformed mechanical identities without re-running common
   validation.
4. The reference fixture was initially only structurally checked. It now
   independently predicts four PRE/POST cases and is crossed against actual
   Guard output.

## Evidence

- exact Guard core SHA-256:
  `d47df73feec97a01f2528cbf110b62c473d16414fcfc94ffefaaad3ff0a7c1af`;
- migrated Guard core tests: 13 passed;
- M3 composition/adversarial tests: 13 passed;
- prior targeted regressions: 117 passed;
- A/B use Null and create no Guard root; C alone creates its private ledger;
- PRE block traverses only the original slate and produces no refresh or extra
  proposal;
- POST after restart reconstructs the exact committed PRE request;
- Compact Feedback has exactly eight fields and contains no full audit event;
- frozen M1 common and M2 Research identities are unchanged.

## Nonblocking P2

`M3-P2-001`: the M3 ledger supports one writer instance and deterministic
create-once/restart behavior. Cross-process writer ownership and kill/restart
orchestration remain M4 gates.

`M3-P2-002`: comparator delta is present in the eight-field contract but remains
`null` in the non-training synthetic result. M5 real-result projection must
supply its frozen comparator delta.

`M3-P2-003`: the exact Guard core was copied immediately before the task
manifest was written. The manifest subsequently bound the same source
commit/path/hash before any adapter semantics or test result existed. This is a
recorded process-order deviation, not an identity or treatment ambiguity.

## Conclusion

M3 has no unresolved P0/P1. Research source remains frozen; the Guard cannot
rank, reward, write Search Memory or alter Candidate/Result bytes. M4 may
compose the three-arm fake orchestration using these exact identities.
