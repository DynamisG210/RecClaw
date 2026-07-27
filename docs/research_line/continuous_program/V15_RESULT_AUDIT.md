# V15 Result Audit

## Verdict

`NON-GO` — P0=0, P1=1. V15 is permanently sealed and may not be patched, resumed, rerun or reused.

## What succeeded

- The laboratory `gpt-5.4` broker completed four physical calls with zero retry: 14,917 total tokens and 48,920 ms aggregate latency.
- The first selected `BPRComposableV2` candidate trained successfully and produced development NDCG@10 `0.2061`.
- Runtime, dataset, execution recipe, start receipt, filesystem, artifact and resource identities all closed.
- The actual training result was correctly converted to `COMMON_EXECUTION_FAILURE` with metric source `NOT_ADMITTED_RESOURCE_CEILING`.
- `CommonExecutionGuard` persisted `TRAINING_RESULT_REJECTED`, reason `TRAINING_RESOURCE_CEILING_EXCEEDED`; the execution claim reached `FINISHED`.
- No rejected metric was admitted to Search Memory, Meta or frontier.

## Failure

The physical run consumed 1,368,131 ms GPU/wall time and 380,036 normalized GPU-cost units. The frozen limits were 900,000 ms GPU time, 1,500,000 ms wall time and 250,000 cost units.

After the typed result and claim had already closed, `_run_arm()` passed the actual over-ceiling resource quantities to `close_round()`. The single-writer resource ledger correctly rejects a debit above the frozen round allocation, raising:

`InvariantViolation: GPU_DEVICE_TIME_MS debit exceeds the frozen round ceiling`

The process therefore ended with one open round even though the common result and execution claim were terminal.

## Root cause

The recovery separated evidence admission from resource rejection, but still used one value for two different meanings:

1. actual observed physical consumption, which must remain immutable for audit and cost reporting;
2. budget allocation debit, which cannot exceed the frozen per-round ceiling.

The smallest correct successor change is to preserve actual consumption in the immutable raw result/resource-accounting artifact while debiting the enforced round budget at the frozen allocation for a typed resource rejection. It must also report the actual overage on the neutral cost projection; no rejected metric or search update is allowed.

## Recovery boundary

- Do not modify the V15 root or databases except for the permanent seal marker.
- Create a new source checkpoint, Pilot version, unused seed and fresh root.
- Add targeted success, over-ceiling and adversarial transaction tests proving result, claim, resource allocation and round all close exactly once.
- Re-run the fixed backend gates. The three V8 canaries remain valid because the training runtime bytes and release are unchanged unless the successor fix changes the release-bound source projection.
