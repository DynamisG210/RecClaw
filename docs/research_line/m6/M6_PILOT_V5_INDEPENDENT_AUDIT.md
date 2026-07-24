# M6 Pilot V5 Independent Audit

Verdict: `FAIL`

```yaml
P0: 0
P1: 1
P2: 2
pilot_verdict: NOT_READY
authority: NONE
evidence_class: DEVELOPMENT_ONLY
formal_acceptance: false
authorized_attempts: 1
attempts_consumed: 1
remaining_attempts: 0
sealed: true
reusable: false
M7_started: false
M8_started: false
M7_authorized: false
M8_authorized: false
```

## Independent conclusion

The one authorized frozen Pilot V5 attempt was made and failed during its first
Broker call. The persisted Broker row is `FAILED/PROCESS_FAILURE`;
`PILOT_FAILURE.json` is `NOT_READY`. No response, returned model, token count,
proposal, training start, execution claim, result artifact, metric, treatment
effect, or Pilot outcome exists.

This is a failed and consumed attempt, not a zero-cost preflight failure and not
a completed Pilot. The V5 root is sealed and non-reusable. The frozen
single-attempt policy forbids a retry or in-place repair. This audit grants no
authority to create V6, resume M7/M8, or promote any claim.

## Governing authorization and frozen identity

The M6E resume authorization is bound by SHA-256
`3ce6bf8199c903927d159f85aa5cbe68dc6e28466001abed33bf26d3120542e8`:
exactly one fresh Pilot V5 was permitted after M6E PASS, and M7/M8 could resume
only after Pilot `GO` plus an independent `P0=0/P1=0` audit. Those conditions
are not met.

The executed frozen command was:

```bash
PYTHONPATH=src /root/projects/RecClaw_m6_training_runtime_v2/bin/python scripts/run_m6_pilot_v5.py --contract docs/research_line/m6/DEVELOPMENT_PILOT_CONTRACT_V5.json --output-root results/research_line/m6_pilot_9205_v5
```

Exact frozen identities:

| Identity | Value |
|---|---|
| Experiment | `HELIX-ABC-DEVELOPMENT-PILOT-9205-V5` |
| Search seed | `9205` |
| Assignment commitment | `eb724d56680b620bf1a9e9d58f6301079607c244ae882f01d9bb9e50ca1e0a73` |
| Store contract identity | `b9be2c4d31c3e7ef3e4eceb1f75db70992e2d182676d0abd8779eea52f533daf` |
| Contract SHA-256 | `02709a3d8a1217201f32a2f66ec4599cec0f44e3ea3917fbe6bf213230863f7e` |
| Contract content digest | `b953f056305014bba05858bcafcc88cdd7483d4b187b3aa90f55cda151017f84` |
| Manifest SHA-256 | `e106762c0cd6c22939c91d904920937f8abb0cebb6f3d6c22844668df48d69df` |
| Manifest content digest | `294cdf799e6c503ea21417650e1e2598e21deb320d92d47b67ffea0f7b2c4b42` |
| Source base commit | `cd93c6db015450e351aa587ba46df8a854c1a954` |
| Source base tree | `457caab324919046195eeef54e42681f99468433` |
| Source projection digest | `3cc983d91b65c5a988b47949018faecb6b8304812a7f68e111da9aad663fba39` |
| Runtime release digest | `c7ab04e5c8425f7a43cd84a0c63f1871bb02effe1c31b6ebcfe78e704b83ba23` |
| Runner ABI | `recclaw.package-owned-search-training-runner.v1` |
| Execution purpose | `DEVELOPMENT_PILOT_OFFLINE_TOPN` |

All 37 files in the frozen source projection matched their declared SHA-256
values; the attempt introduced zero projected-source changes. The runtime
identity remained Python 3.10.20, NumPy 1.26.4, SciPy 1.12.0, Torch
2.10.0+cu128 with CUDA available, RecBole 1.2.1 at clean commit
`7b02be5ec80a88310f2d04a27a82adfcbb5dc211` and tree
`ca6386c4121ce2aae478ced7e136894ac1d7c218`.

## Environment and persisted failure

`ENVIRONMENT_PREFLIGHT.json` has verdict `PASS`. It records the frozen
`gpt-5.4` catalog entry, `codex-cli 0.144.1`, `CHATGPT` login state, the
declared runtime versions, clean RecBole identity, exact ML-1M file hashes,
mount namespace `PASS`, and numeric-UID sibling read/write denial.

The Broker database contains exactly one row:

```yaml
logical_call_id: pilot-original-9205-1
request_digest: 40a11f4d8c7f6a3a44441a402f15d316f27e8b30e117abc17a85d8ef81878ebd
status: FAILED
error_type: PROCESS_FAILURE
response_present: false
response_digest: null
returned_model: null
input_tokens: null
cached_input_tokens: null
output_tokens: null
total_tokens: null
latency_ms: null
retries: 0
```

`PILOT_FAILURE.json` records:

```yaml
error_type: CanaryBrokerError
reason: Codex broker call failed with process status 1
verdict: NOT_READY
```

The known cause boundary stops there. Codex process status `1` is persisted,
but stderr was not persisted. Therefore the underlying process failure cause
is unknown; this audit does not infer authentication, connectivity, model,
CLI, schema, timeout, or provider behavior from absent evidence. The passing
preflight establishes only that its declared checks passed before the call.

## Mechanical database and execution state

All three SQLite databases returned `integrity_check=ok` and zero
foreign-key violations:

- `broker_private/broker.sqlite3`
- `runtime/neutral/experiment.sqlite3`
- `runtime/evidence_audit/inst-8cb6b3ca29c8258d64212d05/evidence_guard.sqlite3`

The neutral store contains exactly:

```yaml
arm_states: 3
arm_state_summary: ACTIVE at next_round_index 1
scheduled_slots: 9
scheduled_slot_statuses:
  OPENED: 1
  PLANNED: 8
rounds: 1
round_status: OPEN
round_events:
  ROUND_OPENED: 1
resource_ledger:
  PROPOSAL_GENERATION_SESSION: 1
execution_claims: 0
artifact_index_rows: 0
triplet_barriers: 3
barrier_closed_bitmaps: [0, 0, 0]
guard_calls: 0
training_backend_starts: 0
```

There is no proposal response or output, no execution debit or launch
confirmation, no raw result envelope, no metric, no readiness packet, and no
treatment-effect or outcome record. The open round and active arm states are
incomplete failure residue; they do not establish execution or scientific
evidence.

## Result-tree identity and audit-side-effect provenance

Immediately after the Pilot process exited and before SQLite inspection, the
root `results/research_line/m6_pilot_9205_v5` had the authoritative
post-execution identity:

```yaml
projection: SHA256(RFC8785(sorted [{path, sha256, size_bytes}]))
file_count: 8
total_bytes: 185504
tree_digest: 4997f503d04bf25c4fa626a63c2ec1327721aa2c748223eb73b41b88f6ad2b75
```

The three typed root records have exact SHA-256 values:

- `ENVIRONMENT_PREFLIGHT.json`:
  `8f0a0496ced7eff97da832ef2b22e0afdc55424bceeb857ec9c2ef2f5831be3f`
- `FROZEN_CONTRACT_IDENTITY.json`:
  `568099de51452ecb7852c7e510a4390a2838a9471924571b7c057112f4936cd6`
- `PILOT_FAILURE.json`:
  `a86abc51e77de70948f0b3c26cb1b2e190882bb185ead5af4b23d4d58674350e`

Subsequent audit queries opened the neutral state and evidence-guard databases
with SQLite URI `mode=ro`. Despite that requested mode, SQLite created and
left one 32,768-byte `-shm` sidecar and one zero-byte `-wal` sidecar beside
each of those two databases. The inspection therefore added four persistent
files and changed the final root identity to:

```yaml
projection: SHA256(RFC8785(sorted [{path, sha256, size_bytes}]))
file_count: 12
total_bytes: 251040
tree_digest: affa504ec72ff85461beb4cc2f6a15f55e7211da39209f1aac03f26bef7e7b59
added_by_audit:
  files: 4
  bytes: 65536
  state_sidecars:
    shm_bytes: 32768
    wal_bytes: 0
  evidence_guard_sidecars:
    shm_bytes: 32768
    wal_bytes: 0
```

The core SQLite database hashes and every pre-existing file hash remained
unchanged. The sidecars are audit inspection artifacts, not Pilot execution
outputs. They are retained in place to preserve provenance; this audit did not
delete or normalize them. The 8-file identity is the post-Pilot/pre-inspection
record, while the 12-file identity is the final sealed on-disk record.

A post-termination process inspection found no running
`run_m6_pilot_v5.py`, `run_m6_pilot.py`, `pilot_train_worker.py`, or
worktree-bound Codex worker.

## Findings

### P1-001 — Frozen Pilot gate failed

The first and only Broker call ended `FAILED/PROCESS_FAILURE`, and the Pilot
persisted `NOT_READY` before any response or training start. The authorized
attempt is consumed. This blocks Pilot `GO`, seals V5, and keeps M7/M8
unauthorized.

### P2-001 — Post-termination inspection exposed assignment associations

The primary post-termination mechanical database inspection displayed
`arm_code`-to-opaque-instance associations. This was unnecessary for the
aggregate failure audit and is a procedural information-boundary finding.
This document intentionally does not reproduce, interpret, or use those
associations.

The exposure did not reveal or become linked to proposals, responses, results,
metrics, effects, or outcomes because none exist. It therefore does not create
a treatment-effect conclusion or change the P1 failure classification, but
future sealed-Pilot inspection should use association-free aggregate queries.

### P2-002 — Read-only SQLite inspection changed the evidence tree

The audit's URI `mode=ro` SQLite inspection created persistent SHM/WAL
sidecars for the neutral state and evidence-guard databases. No core database
or prior artifact byte changed, and the sidecars contain no new Pilot outcome,
but the inspection itself changed the result tree from the 8-file
post-execution identity to the 12-file final identity. This is an audit
provenance and evidence-preservation defect. It is classified P2 because it
did not alter core records, failure classification, scientific content, or
the already-failed gate.

## Final disposition

Pilot V5 is `FAIL/NOT_READY`, sealed, exhausted, and non-reusable. No retry,
rerun, in-place patch, Broker/provider call, V6 construction, treatment
unblinding, M7 work, or M8 work is authorized by this record.
