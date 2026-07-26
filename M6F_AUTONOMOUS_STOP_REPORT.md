# RecClaw M6F Autonomous Stop Report

## Stop verdict

```yaml
program: HELIX-ABC-001
milestone: M6F_BROKER_OBSERVABILITY_AND_FAILURE_CLOSURE_V1
verdict: AUTONOMOUS_HARD_STOP
M6F_verdict: FAIL
P0: 0
P1: 1
P2: 0
fresh_pilot_authorized: false
authority: NONE
evidence_class: DEVELOPMENT_ONLY
formal_acceptance: false
M7_started: false
M8_started: false
```

M6F did not clear its mandatory `P0=0/P1=0` independent-audit gate. No fresh
Pilot, Main freeze, M7, M8, push, promotion, or formal claim was started.
Pilot V5 remains permanently sealed and non-reusable.

## Exact blocker

The independent audit reproduced one P1 at the required crash boundary
`after exit receipt, before Broker row`.

`BrokerProcessRunnerV2` can durably persist a response-bearing process receipt,
outcome, and response file before `CodexCliCanaryBrokerV1` commits the Broker
SQLite row. After a simulated crash at that point, replay finds no Broker row,
but `call_with_session()` rejects the pre-existing response output before
delegating to the process runner's durable recovery path.

Observed independent reproduction:

```json
{
  "second_call_error": "uncommitted broker output path already exists",
  "typed_outcome": false,
  "typed_receipt": false,
  "spawn_count": 1,
  "output_exists": true
}
```

No duplicate spawn or retry occurred. However, the replay error carries no
typed receipt/outcome, so orchestration cannot invoke
`BrokerFailureClosureV1`; an opened SearchRound can remain mechanically
incomplete. This violates an explicit M6F crash/restart acceptance criterion.

## What passed before the stop

- BrokerProcessReleaseV2 resolved with release digest
  `631cc487f73b39e384ebdbb1c50a020ce08268459858176ce802ce4be908e155`.
- Separate bounded stdout/stderr capture, typed process records, evidence-based
  classification, deterministic redaction, nominal failure closure, immutable
  SQLite snapshots, and the association-free neutral audit passed their
  exercised focused paths.
- The local fake-process suite passed 13/13 tests.
- The Original-shape real conformance probe passed once with receipt
  `e59ae7e0aad51a9aa9132cae40b2f1ab7b117531df89b99c8c1be348888c2a12`
  and 14,057 tokens.
- The Producer-shape real conformance probe passed once with receipt
  `2cf692dac5c21805213c08741efb51056588e70fd33fbd5879a6a7fe81517a55`
  and 13,593 tokens.
- The conformance packet digest is
  `981874bda5cd9adf16cc91e4946fc42989a67abceaa5f55bd6ee775f448ee009`.
  The diagnostic root contains 24 files and 38,324 bytes with tree digest
  `cbc4948c6185ffbc60305f948a8053721d19708d4fd005bc2f07a6a0da3d151f`.
- The complete Helix suite passed 155/155 under the frozen Training Runtime
  Release V2 environment:

  ```text
  PYTHONPATH=src \
  /root/projects/RecClaw_m6_training_runtime_v2/bin/python \
    -m unittest discover \
    -s tests/experiments/helix_abc_v1 -p 'test_*.py'
  ```

These results establish the exercised observability and conformance paths.
They do not override the independently reproduced crash-closure defect.

## Preserved identities and boundaries

- Entry commit: `fe1ee557876a08937ea27518c6aaac1b54558567`.
- M6F implementation head before this stop record:
  `aef24db10b346ba0d97e4a33c38bbdf043d1d226`.
- Corresponding tree: `e6be5b46992087140d0b657b34b093d7660ba774`.
- Pilot V5 remains 12 files and 251,040 bytes with digest
  `affa504ec72ff85461beb4cc2f6a15f55e7211da39209f1aac03f26bef7e7b59`.
- Pilot V5 provider-level root cause remains `UNKNOWN`.
- Training Runtime Release V2 remains
  `c7ab04e5c8425f7a43cd84a0c63f1871bb02effe1c31b6ebcfe78e704b83ba23`.
- No V1-V5 artifact, state, request, seed, or output was reused or modified.
- No treatment mapping was unblinded and no NDCG/effect result guided M6F.
- Research, Producer, Router, Meta, Evidence Guard, BL-ICF, treatment,
  readiness-threshold, budget, and training semantics were not changed.
- Raw Broker stdout/stderr remained private and did not enter prompts, Search
  Memory, Router, Meta, Evidence Guard, or scientific feedback.
- The user's parallel Meta worktree and unrelated dirty files were not staged
  or modified by this stop action.

The independent audit is
`docs/research_line/m6f/M6F_INDEPENDENT_AUDIT.md`, SHA-256
`de136176fa3ca5364a546e08f20e7c46aadb113bd48f9ff0909ca3894da303c8`.

## Attempted repair

None after the independent P1 finding. The active M6F command requires an
immediate hard stop when the independent audit has any P0 or P1. Applying a
post-audit patch and self-declaring the same audit gate passed would violate
that rule.

## Safest next option

If continuation is desired, authorize one narrowly scoped M6F repair pass:

1. route pre-existing response/receipt/outcome artifacts through the durable
   runner reconciliation path before rejecting output;
2. reconstruct and commit the exact Broker row without spawning again;
3. return the same typed outcome/receipt so the round closes exactly once;
4. add a Broker-level crash test for the response-bearing
   receipt-to-row boundary and its conflicting replay;
5. rerun focused, full, identity, isolation, and fresh independent P0/P1
   audits.

That repair must not add retry, refund, fallback, replacement proposals,
training, Guard calls, Search Memory updates, or any Meta/treatment change.
Until separately authorized and independently passed, no Pilot V6 is allowed.
