# M0 Independent Audit

> - Milestone: `M0 — Contract Kernel + Single-Writer Store`
> - Audit verdict: `PASS_WITH_NONBLOCKING_P2`
> - P0: `0`
> - P1 after bounded repair: `0`
> - Authority: `NONE`
> - Evidence class: `DEVELOPMENT_ONLY`
> - Formal acceptance: `false`

## Identity audit

The audit independently recomputed the historical M0 identities before making
any repair:

- base commit: `7aeca9278bbe7aac4aaa5de38d67d507c7e172b7`;
- base tree: `ca29f0c1b1883f723a66ebd72b4dc42ba6e112db`;
- upstream divergence: `0 ahead / 0 behind`;
- historical changeset:
  `f034fd0e6402974c1671149eb7631d0eb71dcbf1e8fc96a0c8693251f613a082`;
- execution record:
  `571d80dcc0bbbbbb1ac5f6430b05b75a3ed0726c9a50bd63472ad2e9533f2fca`;
- task manifest:
  `5c6ecec5df4cbd8001ef71d40ca9946d6d0b0f5f6277708e01766285e320951c`;
- experiment contract file:
  `b0b54a18893190e0e1bee976a2bf93322903431bfa15b982c07e404a3200e39c`;
- experiment contract semantic identity:
  `9b8f5a08941939991dd78a4fc9b671c861a4ab6445b9501b31f1af7908d795a7`;
- migration:
  `aabac96bdcdbc61d85c6302fdb56647eac009d85a7025e589f64c10c86a56c49`;
- Original source commit/blob:
  `2d8c881354e1b536a6c66d7dfbb977e0c5090e50` /
  `c40334b72dbb9557eced2bd081915b5333156fdf`;
- golden fixture:
  `a6dc9b346b1ab3f4ea53b9eeade9411c747cd1139fbb0aeb45700235c62c3c7b`.

All historical implementation-file hashes matched the immutable execution
record. The three known EOL-only paths still have zero functional diff under
`git diff --ignore-space-at-eol` and are excluded from the checkpoint.

## Adversarial findings and bounded repair

Fresh single-fault probes found four related P1 false-allows:

1. `ExperimentContractV1.from_dict()` ignored unknown fields and accepted a
   type-coerced representation under the same loaded identity.
2. B and C received different initial Search Memory digests because the Arm
   code entered the empty-memory preimage.
3. `open_round()` accepted a caller-supplied before-state digest that differed
   from the committed `arm_state`.
4. Research-owned fixture payloads accepted common spelling variants of
   explicitly forbidden Evidence Authority and Guard-control fields.

The first permitted repair pass changed only the two affected implementation
files and their focused tests. It did not modify the historical execution
record or handoff, add future runtime components, change A/B/C, or expand
budget or authority.

The corrected implementation-files digest is:

```text
658d7b7d92a570f9b973ee556e786d5384e6b62093542a2cf8b2fb71c0549086
```

Exact historical-to-current hashes are recorded in
`M0_AUDIT_CORRECTION_RECORD_V1.json`.

## Verification

Executed after the repair:

```text
M0 targeted                         26 passed
existing Research Line               4 passed
existing Candidate Proposal         17 passed
BL-ICF mechanism space              27 passed
```

The BL suite ran in `/tmp/recclaw_m0_audit_venv` with the exact
`requirements-search-space.txt` versions. SQLite still reports WAL,
`synchronous=FULL`, foreign keys enabled, integrity `ok`, and exactly the
eight M0 tables. Targeted diff checks found only the already identity-bound
final blank lines in seven historical M0 files; no other whitespace error is
present in the M0 scope.

The original failure probes now reject unknown contract fields, produce equal
B/C initial memory digests, reject a wrong round predecessor, and reject the
tested forbidden-field aliases.

## Nonblocking P2

- Process-wide deployment exclusivity for the single state-store writer is not
  yet proven across two independent service instances. SQLite transaction and
  uniqueness behavior is proven in-process; cross-instance contention remains
  an M4 pre-Canary gate.
- Cross-Arm budget-snapshot equality is frozen and testable at the session
  contract layer, but the M0 store does not itself compare a matched triplet's
  three budget snapshots. The neutral broker/orchestrator must enforce and
  adversarially test this in M4.

Neither P2 is used to claim M1 runtime isolation or campaign readiness.

## Verdict boundary

The corrected M0 subject has no known P0/P1 within its activated scope and may
enter M1. This is development evidence only. It does not establish external
review acceptance, execution permission, CommonExecutionGuard conformance,
Research Capability, Evidence Guard behavior, Canary/Pilot/Main readiness, or
any scientific claim.
