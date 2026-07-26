# M6F Final Independent Audit

## Verdict

```yaml
milestone: M6F_BROKER_OBSERVABILITY_AND_FAILURE_CLOSURE_V1
audited_commit: b7892acbf1473516ca765aa0b1142e193c5415a8
audited_tree: add18a06a11d5f77de1663d597f8755fc61344f7
verdict: FAIL
P0: 0
P1: 1
P2: 0
fresh_pilot_authorized: false
authority: NONE
evidence_class: DEVELOPMENT_ONLY
formal_acceptance: false
```

The second repair correctly binds the complete Broker request identity into
the process-runner envelope and blocks conflicting recovery after the Broker
row is missing. It does not enforce the same identity when the Broker row is
still present. A stored row can therefore be silently replayed under a
different proposal-generation session or a different content-bound Broker
release. This violates the required stable logical-call binding and prevents
M6F from clearing its `P0=0/P1=0` gate.

I made no Provider call, started no Pilot, inspected no treatment effect, and
changed no implementation. The only repository write made by this audit is
this report.

## Audited subject

```yaml
branch: feat/research-line-abc
commit: b7892acbf1473516ca765aa0b1142e193c5415a8
tree: add18a06a11d5f77de1663d597f8755fc61344f7
historical_audit_commit: 2ca2b29
changed_files_since_historical_audit:
  - src/recclaw_core/experiments/helix_abc_v1/broker_process.py
  - src/recclaw_core/experiments/helix_abc_v1/canary_broker.py
  - src/recclaw_core/experiments/helix_abc_v1/resources/broker_process_release_v2.json
  - tests/experiments/helix_abc_v1/test_m6f_broker_observability.py
```

## P1-001 — stored Broker rows do not enforce the full logical-call identity

`CodexCliCanaryBrokerV1.call_with_session()` calls `_stored()` before entering
`BrokerProcessRunnerV2`. `_stored()` compares only `request_digest`. It does
not compare the requested `proposal_generation_session_id` with the stored
session and does not compare the current `broker_release_digest` with the
stored release.

The Broker-level request digest binds prompt, expected proposal count, model,
reasoning effort, schema, and service tier. It intentionally does not contain
the proposal-generation session, while a release-only change such as the
content-bound timeout policy can leave that request digest unchanged.
Consequently, both conflicts bypass the new private logical-call binding
because the stored-row fast path returns first.

Independent local fake-process reproduction:

```json
{
  "prompt_conflict": "REJECTED",
  "expected_proposal_count_conflict": "REJECTED",
  "different_session_accepted": true,
  "different_release_accepted": true,
  "release_digest_changed": true,
  "stored_session": "session-a",
  "stored_release_remained_old": true,
  "spawn_count": 1
}
```

The absence of a second spawn is necessary but insufficient. The required
behavior is to fail every conflicting logical-call replay before spawn, not to
return evidence created under a different session or release. Silent
cross-session replay can misattribute proposal-session consumption; silent
cross-release replay defeats the content-bound runtime identity used to
authorize the call. These are two manifestations of one incomplete stored-row
identity check and are counted as one P1.

## Gate results

| Gate | Result | Evidence |
|---|---|---|
| Exact success recovery after receipt/outcome/response but missing Broker row | PASS | Focused recovery test reconstructs the identical row and typed evidence with one spawn. |
| Exact process-failure recovery after missing Broker row | PASS | Focused recovery test returns the identical typed receipt/outcome and reconstructs one failure row without respawn. |
| Exact response-bearing semantic-failure recovery | PASS | Focused recovery test reconstructs the same semantic failure and row without response reinterpretation. |
| Different prompt conflicts before spawn | PASS | Independent reproduction rejected the request; spawn count remained one. |
| Different expected proposal count conflicts before spawn | PASS | Independent reproduction rejected the request; spawn count remained one. |
| Different proposal session conflicts before spawn | **FAIL** | Stored-row fast path returned the prior success under `session-b` while the row remained bound to `session-a`. |
| Different relevant Broker release conflicts before spawn | **FAIL** | A Broker with a different timeout-bound release digest returned the row produced under the old release. |
| Stable logical-call binding | **FAIL** | Missing-row recovery is bound; stored-row replay is not fully bound. |
| No retry/refund/fallback/replacement/training/Guard/memory update | PASS for exercised repair paths | No duplicate spawn or added treatment behavior was observed; existing closure/orchestration tests passed. This does not cure the identity violation. |
| Resource debit semantics | PASS for exercised exact-recovery paths | Prior successful Producer calls plus a later failed call retain cumulative physical-call, token, and wall-time debits; exact row recovery preserves the recorded debit. |
| Immutable snapshots and neutral audit | PASS for exercised paths | Snapshot/association-free focused test passed; immutable SQLite inspection produced `integrity_check=ok`, no WAL/SHM sidecars, and only aggregate conformance counts. |
| V5 seal/non-reuse | PASS | Recomputed 12 files, 251,040 bytes, digest `affa504ec72ff85461beb4cc2f6a15f55e7211da39209f1aac03f26bef7e7b59`; V5 cause remains `UNKNOWN`. |
| Meta/Producer/Router/Guard/BL/training isolation | PASS by diff | The repair changes only Broker process identity/persistence code, its release resource, and focused tests; no treatment, policy, Guard, BL-ICF, budget, readiness, or Training Runtime V2 semantics changed. |

## Broker release and conformance evidence

The package-owned release recomputed exactly:

```yaml
broker_release_digest: 2d88e3df3486f2369c4487c66c333dbf046796c8d38e449e506c184c7f9fadd1
release_digest_valid: true
```

The fresh treatment-free M6F v3 evidence was internally content-consistent:

```yaml
result_root: results/research_line/m6f_conformance_v3
result_tree:
  file_count: 26
  total_bytes: 38328
  digest: 9fff29e8f209029a3b474c0400d559f900b9e62fe8a65494fc40e6d3c2e5fa3c
conformance_packet_digest: d2a9c2d20e675a74696f9b5361dde5a8cd4fb71a8cda72feb76fdb1db2da366e
original_shape:
  verdict: PASS
  receipt: cdcb3306601c41bde96bc3c09a2b8da673e6c3a76adbff4694d6c7ea12bd818a
  tokens: 14167
  report_digest: db9426abac7ce8c4fb8f2c8e53d102303713bb189a10594c7c17db3810521212
producer_shape:
  verdict: PASS
  receipt: 3e029b4cc243c72a5b48b74a9bc84a2f0dc83570a93e407284ee16528b867e24
  tokens: 13635
  report_digest: 68839a4cd902cda350655d2b76f666c1a341e36c63e6d58bf3849d0aa3fd2624
aggregate_broker_rows:
  status: SUCCESS
  count: 2
  total_tokens: 27802
search_rounds_opened: 0
treatment_state_used: false
```

For both real probes, the request/start/receipt/outcome digests, report links,
logical-call binding files, and stdout/stderr content hashes recomputed
correctly. Both receipts record exit code zero and
`provider_request_confirmation=RESPONSE_RECEIVED`. These results establish the
new release's fresh-call conformance, but do not override P1-001.

## Verification

Focused suite under frozen Training Runtime Release V2:

```text
PYTHONPATH=src \
/root/projects/RecClaw_m6_training_runtime_v2/bin/python \
  -m unittest \
  tests.experiments.helix_abc_v1.test_m6f_broker_observability -v

14/14 PASS
```

Full activated Helix suite under the same frozen runtime:

```text
PYTHONPATH=src \
/root/projects/RecClaw_m6_training_runtime_v2/bin/python \
  -m unittest discover \
  -s tests/experiments/helix_abc_v1 -p 'test_*.py'

156/156 PASS
```

The green suites establish the covered recovery paths. Neither suite exercises
the stored-row session/release conflicts independently reproduced above.

## Disposition

M6F remains `FAIL` with `P0=0/P1=1/P2=0`. A fresh Pilot, Main freeze, M7, and
M8 are not authorized. Pilot V5 remains sealed and non-reusable.

The minimal remaining repair is to make `_stored()` compare the complete
logical-call identity needed by the process envelope—at minimum the proposal
session and current Broker release in addition to the Broker request digest—
and fail closed before returning any stored success or failure. It must retain
the current zero-retry, no-refund, no-fallback, no-training, no-Guard, and
no-memory behavior, followed by a fresh independent audit.
