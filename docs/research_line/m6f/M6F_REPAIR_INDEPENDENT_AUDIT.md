# M6F Narrow Repair Independent Audit

## Verdict

```yaml
milestone: M6F_BROKER_OBSERVABILITY_AND_FAILURE_CLOSURE_V1
subject: receipt_outcome_response_to_broker_row_repair
verdict: FAIL
P0: 0
P1: 2
P2: 0
fresh_pilot_authorized: false
authority: NONE
evidence_class: DEVELOPMENT_ONLY
formal_acceptance: false
```

The repair restores the Broker row for an exact replay after the row is lost,
without respawning, for successful, process-failed, and response-bearing
semantic-failed calls. It does not yet make conflicting replay fail closed
across those paths. The two independently reproduced failures below violate
the explicit zero-retry and conflicting-replay requirements, so the repair
cannot clear the M6F `P0=0/P1=0` gate.

I made no provider call, started no Pilot, and changed no implementation. The
only repository write made by this audit is this report.

## Audited subject

```yaml
branch: feat/research-line-abc
commit: e409d102221aeafa6de24317f489980ca180d22a
tree: 79d3fe33991890e51d825eea629b47dfbe81886e
parent_stop_commit: 66da51c511e355fd634eba753cb55a4bbe8dc3bb
changed_files:
  - src/recclaw_core/experiments/helix_abc_v1/broker_process.py
  - src/recclaw_core/experiments/helix_abc_v1/canary_broker.py
  - tests/experiments/helix_abc_v1/test_m6f_broker_observability.py
```

The implementation change moves the orphan-response rejection from
`CodexCliCanaryBrokerV1` to `BrokerProcessRunnerV2`, after the runner first
checks for an exact durable call root. That is sufficient for exact replay,
but the runner's call root is keyed by the request-envelope digest rather than
by a durable logical-call identity binding.

## P1-001 — conflicting process-failure replay respawns

Independent fake-process reproduction:

1. Execute a process-failed Broker call and persist its receipt/outcome.
2. Delete only its Broker SQLite row, simulating the audited
   receipt-to-row crash boundary.
3. Replay the same `logical_call_id` and proposal session with a different
   prompt.

Observed:

```json
{
  "spawn_count": 2,
  "durable_call_roots": 2,
  "first_receipt": "eea6509c3748a42d3dfc348d4f5f8befdc58f9ace9834b5703a6bf09f70ad9e2",
  "second_receipt": "987d625ff503f8f59be0ee4f6752b84a051c306fd9e6661a747e514cfa27908f",
  "receipts_differ": true,
  "final_row_status": "FAILED",
  "final_error_type": "PROCESS_EXIT_FAILURE"
}
```

A process-failed call normally has no response output. After its Broker row is
lost, the conflicting prompt produces a different request-envelope digest and
therefore a new call root; the new orphan-output guard has no file to detect.
The fake subprocess is launched a second time. This is a duplicate physical
attempt for one logical call and violates zero retry, budget integrity, and
conflicting-replay fail-closed behavior.

## P1-002 — conflicting Broker request reinterprets durable response

`expected_proposal_count` is included in the Broker SQLite
`request_digest`, but it is not bound into `BrokerRequestEnvelopeV2`. With the
Broker row lost, a replay can keep the same logical ID, proposal session, and
prompt while changing only this Broker-level field. The runner treats it as
the same durable process call, but the Broker reinterprets the old response
under the new expectation.

Independent observations:

```json
{
  "success_then_conflicting_expected_count": {
    "spawn_count": 1,
    "request_digests_differ": true,
    "same_receipt": true,
    "first_status": "SUCCESS",
    "recovered_row_status": "FAILED",
    "recovered_error_type": "SCHEMA_VALIDATION_FAILURE"
  },
  "semantic_failure_then_conflicting_expected_count": {
    "spawn_count": 1,
    "request_digests_differ": true,
    "same_receipt": true,
    "first_status": "SCHEMA_VALIDATION_FAILURE",
    "recovered_row_status": "SUCCESS",
    "second_call_returned_success": true
  }
}
```

The request digests were:

```text
initial expected count 0:
d55b594192b95d160d10bd309c1439fdc309ef4e8ac1c173761ff9581e9899c2

conflicting expected count 1:
1473fe78f3000e4514ef04decf0d34e629da5751746ea42c5b29814e44bfa808
```

This path does not respawn, refund, or fall back, but it does not fail closed:
it can reverse the terminal meaning of the exact same response and even turn a
previous semantic failure into a successful Broker row.

## Gate results

| Gate | Result | Evidence |
|---|---|---|
| Exact successful replay after row loss | PASS | Same response and receipt, one spawn, one reconstructed row in the focused test. |
| Exact process-failure replay after row loss | PASS | Same typed outcome/receipt, one spawn, one reconstructed failure row in the focused test. |
| Exact response-bearing semantic-failure replay | PASS | Same typed failure/receipt, one spawn, one reconstructed failure row in the focused test. |
| Conflicting replay fails closed | **FAIL** | P1-001 and P1-002. |
| No respawn/retry at the crash boundary | **FAIL** | P1-001 launches the process twice. |
| No refund/fallback/replacement/Guard/training/memory update | PASS for exercised exact-replay paths | No such behavior was added by the repair diff; existing closure tests pass. This does not neutralize the duplicate spawn. |
| Meta/Producer/Router/Guard/BL/training isolation | PASS | The repair commit changes only Broker process/Broker persistence code and its focused test. |
| V5 seal and non-reuse | PASS | Reverified 12 files, 251,040 bytes, tree digest `affa504ec72ff85461beb4cc2f6a15f55e7211da39209f1aac03f26bef7e7b59`. No V5 database was opened. |

## Verification

Focused suite:

```text
PYTHONPATH=src \
/root/projects/RecClaw_m6_training_runtime_v2/bin/python \
  -m unittest \
  tests.experiments.helix_abc_v1.test_m6f_broker_observability -v

14/14 PASS
```

Full activated Helix suite:

```text
PYTHONPATH=src \
/root/projects/RecClaw_m6_training_runtime_v2/bin/python \
  -m unittest discover \
  -s tests/experiments/helix_abc_v1 -p 'test_*.py'

156/156 PASS
```

These green suites establish the covered exact-replay paths but do not cover
the two conflicting-replay cases reproduced independently above.

## Existing real conformance evidence

The prior real Original-shape and Producer-shape probe results remain reusable
for process invocation conformance and were not rerun. Relative to the
previous M6F implementation, the repair does not change the package-owned
release resource, executable, argv construction, environment projection,
fresh-call `Popen` invocation, stream capture, timeout, parser, classifier, or
redaction semantics. It changes only the ordering of the pre-existing-output
guard on recovery.

```yaml
broker_release_digest: 631cc487f73b39e384ebdbb1c50a020ce08268459858176ce802ce4be908e155
broker_release_resource_sha256: f314fb4c991e279f3617cd808621a65d872e94191bc0e229cf4fda13942e6d63
original_probe: PASS
original_receipt: e59ae7e0aad51a9aa9132cae40b2f1ab7b117531df89b99c8c1be348888c2a12
producer_probe: PASS
producer_receipt: 2cf692dac5c21805213c08741efb51056588e70fd33fbd5879a6a7fe81517a55
conformance_packet_digest: 981874bda5cd9adf16cc91e4946fc42989a67abceaa5f55bd6ee775f448ee009
conformance_result_tree:
  file_count: 24
  total_bytes: 38324
  digest: cbc4948c6185ffbc60305f948a8053721d19708d4fd005bc2f07a6a0da3d151f
```

The real probes establish the unchanged fresh process invocation. They do not
establish crash-replay conflict safety.

## Disposition

The narrow repair is `FAIL` with `P0=0/P1=2/P2=0`. No fresh Pilot, Main
freeze, M7, or M8 is authorized. V5 remains sealed and non-reusable.

The remaining repair must durably bind one complete Broker request identity to
each `logical_call_id` before process execution. Exact replay may reconstruct
the missing row from the same evidence; any differing Broker request identity
must fail before spawn and before response reinterpretation.
