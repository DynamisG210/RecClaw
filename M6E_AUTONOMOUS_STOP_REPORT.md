# RecClaw M6E Resume Autonomous Stop Report

## Stop verdict

```yaml
program: HELIX-ABC-001
milestone: M6
pilot_version: V5
pilot_seed: 9205
verdict: AUTONOMOUS_HARD_STOP
pilot_verdict: NOT_READY
P0: 0
P1: 1
P2: 2
authority: NONE
evidence_class: DEVELOPMENT_ONLY
formal_acceptance: false
M7_started: false
M8_started: false
```

M6E remains passed. Its audited Training Runtime Release V2, typed store-audit
port, training canaries, filesystem capability, and P0/P1 closure were not
invalidated. The one conditionally authorized fresh Pilot V5 attempt did not
reach `GO`, so seed 9205 and the complete V5 root are permanently sealed and
non-reusable. No V6, Main freeze, M7, M8, push, promotion, or formal claim was
started.

## Exact failure

The frozen environment preflight passed. The first Broker process was then
invoked exactly once for `pilot-original-9205-1` and exited with process status
1. Its ledger row is `FAILED/PROCESS_FAILURE`; it contains no response, model
identity, latency, or token usage. `PILOT_FAILURE.json` records
`CanaryBrokerError` and `verdict=NOT_READY`.

The runner did not persist the subprocess stderr. The supported cause boundary
is therefore only "Codex subprocess exited 1"; authentication, connectivity,
model availability, schema, CLI, timeout, and provider behavior remain
unresolved rather than inferred.

## Work completed before failure

```yaml
broker_process_invocations: 1
broker_successes: 0
broker_retries: 0
provider_request_confirmation: unknown
opened_rounds: 1
closed_rounds: 0
proposal_generation_sessions: 1
proposal_responses: 0
ordinary_training_executions: 0
training_backend_starts: 0
guard_calls: 0
gpu_device_time_ms: 0
gpu_cost_microunits: 0
token_usage: unavailable
```

Nine slots were prospectively scheduled, but only one round was opened. There
are no execution claims, artifacts, raw results, metrics, readiness rows,
frontiers, or treatment effects. No scientific comparison can be made from
this Pilot.

## Preserved identities

- Pre-outcome commit:
  `8d56f04362eaa269833b28a7cc73fde4097daa5b`.
- Contract content digest:
  `b953f056305014bba05858bcafcc88cdd7483d4b187b3aa90f55cda151017f84`.
- Source projection digest:
  `3cc983d91b65c5a988b47949018faecb6b8304812a7f68e111da9aad663fba39`.
- Training runtime release:
  `c7ab04e5c8425f7a43cd84a0c63f1871bb02effe1c31b6ebcfe78e704b83ba23`.
- Final result root:
  `results/research_line/m6_pilot_9205_v5`.
- Final result-tree digest:
  `affa504ec72ff85461beb4cc2f6a15f55e7211da39209f1aac03f26bef7e7b59`
  over 12 files and 251,040 bytes.
- Independent audit:
  `docs/research_line/m6/M6_PILOT_V5_INDEPENDENT_AUDIT.md`, SHA-256
  `0396ee1446ce6737d047541a6fdfedb860a5f7eb526e387a48b9344bdd4af3c9`.

The frozen 37-file source projection still matches exactly, all three SQLite
stores report integrity `ok` with zero foreign-key violations, and no Pilot,
Broker, or training worker remains alive.

## Procedural findings

The independent audit records two P2 findings. A post-termination mechanical
query exposed arm-code to opaque-instance associations even though no outcome
existed. Later SQLite `mode=ro` queries created state/guard WAL-SHM sidecars,
changing the package projection from the initial 8-file post-exit tree to the
final 12-file sealed tree without changing core databases or typed artifacts.
Future failure audits should use association-free aggregate queries and
immutable SQLite snapshots.

## Final disposition

Pilot V5 is `FAIL/NOT_READY`; the single-attempt budget is exhausted. Under the
M6E recovery command, no retry, rerun, in-place patch, V6 construction, Main
freeze, M7, or M8 is authorized. Further progress requires a new explicit user
authorization that defines how to diagnose the unpersisted Broker process
failure without reopening V5.
