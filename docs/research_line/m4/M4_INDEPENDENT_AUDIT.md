# M4 implementation-independent adversarial audit

## Scope and status

- Milestone: `M4`
- Lane: exploration
- Authority: `NONE`
- Evidence class: `DEVELOPMENT_ONLY`
- Formal acceptance: `false`
- Audit verdict: `PASS_WITH_NONBLOCKING_P2`
- P0: 0
- P1: 0
- P2: 3

This is a separate adversarial pass over the finished M4 changes and activated
tests. It is not external scientific acceptance, runtime permission
certification, or approval to reuse Canary output as Pilot/Main evidence.

## Evidence inspected

- M0 single-writer schema, transaction boundaries, triplet barrier and recovery.
- M1 CommonExecutionGuard release identity and non-training runner.
- M2 Producer/Router/Meta controller and Search Memory boundary.
- M3 Null/Guard ports, same-slate selector, deterministic Fusion and C-private
  Guard ledger.
- M4 opaque assignment, runtime-root capability, UID isolation probe, fake
  broker, three-arm scheduler, neutral projection, four-axis reconstruction and
  sealed review packet.
- 154 activated targeted/regression tests.

## Adversarial findings closed

1. Arm-private materialization files initially shared unqualified artifact
   relative paths. M4 now indexes them under the opaque instance namespace while
   preserving exact source bytes for Common result closure.
2. Full Compact Feedback initially reached the M2 Search Memory validator. M4
   now sends only the deterministic Helix Fusion instruction and raw search
   digest to the Research controller; the full event remains C-private.
3. A review packet could initially be emitted from an arbitrary result
   sequence. Packet creation now requires the exact opaque triplet, completed
   one-execution rounds, no training start, closed budgets, SQLite integrity,
   a `7` triplet bitmap, next-index authorization and exactly one C PRE/POST
   audit pair.
4. A forged filesystem capability could initially substitute a sibling root.
   `ArmPrivateRootsV1` now rejects incomplete namespace sets, non-child paths,
   symlinked roots and inode aliases; traversal, symlink, hardlink, `/proc`,
   sibling read and sibling write negatives pass.

## Remaining P2 items

1. M4 exercises arm-scoped capabilities plus Linux numeric-UID DAC. The exact
   real Canary container/mount projection remains an M5 preflight input and must
   pass before any real broker or candidate execution.
2. M4 uses fixed synthetic metrics only to exercise result and Guard closure.
   They have no effect-evidence status and must not tune M5 policy or thresholds.
3. M4 activates one fake scheduled round per Arm. The exact three-to-five-round
   Canary contract and multi-round environment recovery are deliberately bound
   at M5 entry before any Canary outcome is visible.

## Boundary conclusion

Development evidence shows that the M4 fake path preserves one round, at most
one ordinary execution, one feedback, equal total resource ceilings, opaque
assignment, B/C controller equality, A/B Null versus C Guard composition,
C-private full audit, deterministic Fusion, neutral blinding and four-axis
record reconstruction. It does not establish real-environment isolation,
real-LLM availability, candidate-training validity, Pilot readiness, Main
permission, or any scientific effect.
