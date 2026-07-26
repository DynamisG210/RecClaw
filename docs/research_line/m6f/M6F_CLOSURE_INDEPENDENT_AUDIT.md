# M6F Closure Independent Audit

## Verdict

```yaml
milestone: M6F_BROKER_OBSERVABILITY_AND_FAILURE_CLOSURE_V1
audited_commit: 5dab3765ea7b8b96e6894a735d84e6204b7fab04
branch: feat/research-line-abc
verdict: PASS
P0: 0
P1: 0
P2: 0
fresh_pilot_authorized: true
authorization_scope: ONE_FRESH_PILOT_ONLY
pilot_v5_reuse_authorized: false
main_freeze_authorized: false
M7_authorized: false
M8_authorized: false
authority: NONE
evidence_class: DEVELOPMENT_ONLY
formal_acceptance: false
```

Commit `5dab376` clears the stored-row identity defect reported by the preceding
independent audit. Fresh adversarial testing found no remaining P0, P1, or P2.
The M6F `P0=0/P1=0` gate is therefore clear and one genuinely fresh Pilot is
authorized. This is not authority to reopen or reuse V5, freeze Main, begin M7
or M8, push, promote, or make a formal scientific claim.

I made no Provider call, started no Pilot, opened no V5 SQLite database,
inspected no treatment effect, and made no implementation change. The only
repository write made by this audit is this report.

## Governing scope and audited subject

The audit applied the active M6F execution manifest, the governing closure
command, and all three preceding M6F independent reports:

- `M6F_INDEPENDENT_AUDIT.md`
- `M6F_REPAIR_INDEPENDENT_AUDIT.md`
- `M6F_FINAL_INDEPENDENT_AUDIT.md`

The latest repair changes only:

```text
src/recclaw_core/experiments/helix_abc_v1/canary_broker.py
tests/experiments/helix_abc_v1/test_m6f_broker_observability.py
```

`CodexCliCanaryBrokerV1._stored()` now validates the Broker request digest,
`proposal_generation_session_id`, and current content-bound Broker release
before returning either a stored success or a stored typed failure. Missing-row
recovery remains protected by the durable logical-call binding to the complete
process request-envelope digest.

Known unrelated dirty paths and pre-existing result roots were preserved and
were not treated as M6F implementation evidence.

## Independent replay and conflict matrix

A fresh temporary-root fake-process harness ran under the frozen Training
Runtime Release V2 interpreter. It exercised all combinations of:

```text
terminal state:
  SUCCESS
  PROCESS_EXIT_FAILURE
  response-bearing SCHEMA_VALIDATION_FAILURE

Broker row state:
  STORED_ROW
  MISSING_ROW_AFTER_DURABLE_RECEIPT_OUTCOME_RESPONSE

replay:
  EXACT
  conflicting prompt
  conflicting expected_proposal_count
  conflicting proposal_generation_session_id
  conflicting Broker release
```

Results:

| Path | Exact replay | Four identity conflicts | Spawn count | Response reads during conflicts |
|---|---:|---:|---:|---:|
| Stored success | PASS | 4/4 rejected | 1 | 0 |
| Missing-row success | PASS | 4/4 rejected | 1 | 0 |
| Stored process failure | PASS | 4/4 rejected | 1 | not applicable |
| Missing-row process failure | PASS | 4/4 rejected | 1 | not applicable |
| Stored semantic failure | PASS | 4/4 rejected | 1 | 0 |
| Missing-row semantic failure | PASS | 4/4 rejected | 1 | 0 |

Aggregate result:

```yaml
exact_replays: 6/6_PASS
conflicting_replays: 24/24_REJECTED
duplicate_spawns: 0
maximum_spawn_count_per_logical_call: 1
conflict_response_output_reads: 0
conflict_created_or_replaced_rows: 0
```

Each exact replay preserved the same call result or typed error, receipt, and
outcome. Each missing-row exact replay reconstructed exactly one Broker row.
For response-bearing paths, `Path.read_bytes()` was guarded at the durable
response output; every conflict was rejected before that output could be read
or reinterpreted. Thus changing `expected_proposal_count` cannot reverse a
stored response's terminal meaning, and changing prompt, session, or release
cannot create another physical attempt.

## Required identity and evidence gates

| Gate | Result | Evidence |
|---|---|---|
| Complete stored-row identity | PASS | Request bytes, proposal session, and current release are checked before stored success or failure replay. |
| Complete missing-row identity | PASS | Prompt/count/session/release conflicts all hit the durable full-envelope logical-call binding before recovery or spawn. |
| Exactly-once success/process/semantic replay | PASS | Six fresh exact replays preserved terminal evidence; missing-row paths reconstructed one row each. |
| No response reinterpretation | PASS | Twenty-four conflicting calls caused zero reads of response outputs. |
| No retry or second process | PASS | Every scenario ended with one physical spawn. |
| No refund/fallback/replacement proposal | PASS | No such branch was entered or added; identity conflicts fail closed. |
| No training/Guard/Search Memory/frontier update | PASS | The repair touches only stored Broker identity validation and its focused test; focused orchestration checks retained zero Guard and execution use on Broker failure. |
| Resource accounting | PASS | Focused tests preserve prior physical-call/token/wall-time debits, debit one proposal-generation session on failure closure, and do not refund replayed work. |
| Failure closure | PASS | Exact failure closure remains idempotent, conflicting closure is rejected, all remaining slots stop, and no execution claim is created. |
| Immutable/neutral audit | PASS | Fresh focused tests verified immutable snapshots with `integrity_check=ok`, matching hashes, zero SQLite sidecars, and an association-free projection without assignment, candidate, treatment, or metric fields. |
| Budget and role isolation | PASS | No threshold, Pilot budget, treatment, Meta, Producer, Router, Evidence Guard, BL-ICF, or Training Runtime semantics changed. |

## Broker release, real conformance, and final fake evidence

The package-owned Broker release remains content-bound as:

```yaml
broker_release_digest: 2d88e3df3486f2369c4487c66c333dbf046796c8d38e449e506c184c7f9fadd1
release_resource_sha256: d10e4e2ec1bf105469ff30e2faa3c95ff7fabbbd5d49e4cfa06d65c6a94cd44a
broker_executable_sha256: cbacbb9726262ef558b4af0438a1b2a5bba9076132401d947b5b4d2bf92ab0e4
response_schema_digest: e043b49e10b6e3bd67a1da40a18bfcab0cfb987c37d1b27222b1ecc207c167b4
release_policy_test: PASS
```

The treatment-free real conformance v3 packet was inspected without a new
Provider call. It remains internally consistent and reusable because
`5dab376` changes only the stored-row fast path and does not change the
release, executable, argv, environment, timeout, capture, parser, classifier,
or fresh-call path:

```yaml
result_root: results/research_line/m6f_conformance_v3
files: 26
total_bytes: 38328
recorded_tree_digest: 9fff29e8f209029a3b474c0400d559f900b9e62fe8a65494fc40e6d3c2e5fa3c
packet_digest: d2a9c2d20e675a74696f9b5361dde5a8cd4fb71a8cda72feb76fdb1db2da366e
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
search_rounds_opened: 0
treatment_state_used: false
```

The post-repair fake-only v7 packet is also clean:

```yaml
result_root: results/research_line/m6f_conformance_fake_v7
fake_suite_verdict: PASS
tests: 14/14
stderr_sha256: 4e4c939d9557a2c75687a0af0da27ce046f42ac37aa755804eaadb608db33699
stdout_sha256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
real_probes: NOT_RUN
```

`real_probes: NOT_RUN` is correct for v7: it is final local repair evidence,
while the unchanged real invocation contract is covered by v3.

## V5 seal

Pilot V5 remains permanently sealed and non-reusable:

```yaml
path: results/research_line/m6_pilot_9205_v5
files: 12
total_bytes: 251040
tree_digest: affa504ec72ff85461beb4cc2f6a15f55e7211da39209f1aac03f26bef7e7b59
provider_level_root_cause: UNKNOWN
opened_by_this_audit: false
reuse_authorized: false
```

The governing manifest and all preceding audits agree on this identity. The
latest repair has no V5 path, and this audit neither opened a V5 database nor
read, copied, patched, reran, or reused any V5 request, state, seed, or output.

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

## Disposition

M6F is `PASS` with `P0=0/P1=0/P2=0`. One fresh Pilot is now authorized under
the already-frozen Pilot contract, budget, endpoints, treatment boundaries,
and Training Runtime Release V2. It must use a new result root and must not
reuse V5.

This authorization ends at the fresh Pilot. Its result still requires the
governing post-Pilot checks and independent decision before any Main freeze,
M7, M8, promotion, or scientific claim.
