# M6F Independent Audit

## Verdict

```yaml
milestone: M6F_BROKER_OBSERVABILITY_AND_FAILURE_CLOSURE_V1
verdict: FAIL
P0: 0
P1: 1
P2: 0
fresh_pilot_authorized: false
authority: NONE
evidence_class: DEVELOPMENT_ONLY
formal_acceptance: false
```

M6F does not clear its required `P0=0/P1=0` gate. I made no implementation
change, provider call, or Pilot run, and did not open a sealed V5 SQLite
database.

## Audited subject

```yaml
branch: feat/research-line-abc
base: fe1ee557876a08937ea27518c6aaac1b54558567
commits:
  - df931c5900b4ee56c82342b1ccea59bbd1369c12
  - 00e20c8a4337d4f75215c70287f1d85ae2494140
  - 36c2ac74c0e1aecfa968ab1030611537d74bae51
  - aef24db10b346ba0d97e4a33c38bbdf043d1d226
head: aef24db10b346ba0d97e4a33c38bbdf043d1d226
tree: e6be5b46992087140d0b657b34b093d7660ba774
result_root: results/research_line/m6f_conformance_v2
result_tree: 24 files, 38324 bytes
result_tree_digest: cbc4948c6185ffbc60305f948a8053721d19708d4fd005bc2f07a6a0da3d151f
```

## P1-001 — receipt-to-Broker-row crash cannot recover

Affected invariant:

```text
after exit receipt, before Broker row
→ replay the same durable process evidence
→ persist/replay the Broker row
→ close the opened round exactly once
→ no retry, refund, fallback, Guard, training, memory, or frontier update
```

`BrokerProcessRunnerV2` durably writes its response-bearing receipt/outcome and
response file before `CodexCliCanaryBrokerV1` commits the Broker SQLite row.
After a crash at that boundary, `_stored()` finds no row, but
`call_with_session()` rejects the existing response output before delegating
to `BrokerProcessRunnerV2` recovery. The error has no typed receipt/outcome.
`ThreeArmPreCanaryOrchestratorV1` therefore cannot call
`close_broker_failure()`, so the opened SearchRound may remain incomplete.

Independent local fake-process reproduction:

```text
1. Complete one fake Broker call in a temporary root.
2. Preserve response plus request/start/receipt/outcome artifacts.
3. Remove only the temporary Broker row, simulating crash before row commit.
4. Reconstruct the Broker and replay the exact logical call.
```

Observed:

```json
{"second_call_error":"uncommitted broker output path already exists",
 "typed_outcome":false,"typed_receipt":false,
 "spawn_count":1,"output_exists":true}
```

No retry occurred, but exactly-once failure closure also did not occur. This is
P1 because it violates an explicitly required crash boundary and can leave
Pilot state mechanically incomplete. It is not P0 because no treatment
exposure, cross-arm mutation, provider duplication, V5 mutation, or scientific
result corruption was observed. The absent boundary test and concrete defect
are one finding; no separate P2 is counted.

## Gate summary

| Gate | Result | Evidence |
|---|---|---|
| V5 seal/non-reuse | PASS | 12 files, 251,040 bytes, digest `affa504ec72ff85461beb4cc2f6a15f55e7211da39209f1aac03f26bef7e7b59`; V5 cause remains `UNKNOWN`. |
| Broker release | PASS | Release `631cc487f73b39e384ebdbb1c50a020ce08268459858176ce802ce4be908e155`; executable `cbacbb9726262ef558b4af0438a1b2a5bba9076132401d947b5b4d2bf92ab0e4`; schema `e043b49e10b6e3bd67a1da40a18bfcab0cfb987c37d1b27222b1ecc207c167b4`. |
| Capture/classifier/redaction | PASS for exercised paths | Separate bounded streams, hashes, sizes, truncation flags, receipt linkage, closed classifier, deterministic public redaction; current focused suite 13/13 PASS. |
| Original real probe | PASS | 14,057 tokens; receipt `e59ae7e0aad51a9aa9132cae40b2f1ab7b117531df89b99c8c1be348888c2a12`; report `6eca9a9b373e97fc93932a513f81d57e518300e8b17a61e9c96699d0b16f97ed`. |
| Producer real probe | PASS | 13,593 tokens; receipt `2cf692dac5c21805213c08741efb51056588e70fd33fbd5879a6a7fe81517a55`; report `0c547955e005f9a33e252ea00bbfeed2def29ddbbcfde325d5bbd32b4ad7b956`. |
| Probe isolation/no retry | PASS | Packet digest `981874bda5cd9adf16cc91e4946fc42989a67abceaa5f55bd6ee775f448ee009`; zero SearchRounds, no treatment state, one start per probe. |
| Nominal failure closure/accounting | PASS | One session debit, `BROKER_PROCESS_FAILURE/COMMON_NO_EXECUTION`, zero execution/Guard, stopped remainder, identical replay, conflicting replay rejected. |
| Required crash matrix | **FAIL** | P1-001 fails `after exit receipt, before Broker row`; runner-local tests do not cover the Broker-row boundary. |
| Immutable snapshots/neutral audit | PASS for tested path | Backup plus `mode=ro&immutable=1`, integrity/hash checks, zero sidecars; aggregate projection exposes no assignment association. |
| Frozen M1/M6E identities | PASS | M1 release `0a616e...`, common projection `97c424...`, source manifest `038ebf...`; Training Runtime Release V2 remains `c7ab04e5...`. |
| Meta/Producer/Router/Guard/BL/training scope | PASS by diff | No policy, role/prompt, Router, Guard/Fusion, BL compiler/profile, threshold, treatment, or training behavior semantics changed. |

Focused independent command:

```text
PYTHONPATH=src /root/projects/RecClaw_m6_training_runtime_v2/bin/python \
  -m unittest tests.experiments.helix_abc_v1.test_m6f_broker_observability -v
13/13 PASS
```

Broader suites were not rerun after the confirmed P1 because the governing
command requires immediate hard-stop on any P0/P1.

## Disposition

M6F is `FAIL` with `P0=0/P1=1/P2=0`. Passing fake/real probes establish
current executability and observability, not crash-complete closure. No fresh
Pilot, Main freeze, M7, or M8 is authorized. V5 remains sealed and
non-reusable. A minimal receipt/outcome-to-row replay repair and focused crash
test require separate authorization and a fresh independent M6F audit.
