# M6 Pilot V5 Pre-Outcome Audit V1 Failure

## Verdict

```yaml
verdict: FAIL
P0: 0
P1: 1
P2: 0
pilot_authorized: false
broker_or_provider_calls: 0
v5_output_root_absent: true
authority: NONE
evidence_class: DEVELOPMENT_ONLY
formal_acceptance: false
```

Pilot V5 is not authorized. The frozen mutable global model-cache identity no
longer matches the live file, and the provider-free environment preflight fails
closed before Broker use.

## P1-1 — Frozen global model-cache identity drift

Frozen by `DEVELOPMENT_PILOT_CONTRACT_V5.json`:

```text
path:
  /mnt/c/Users/gtrho/.codex/models_cache.json
expected SHA-256:
  c8a701e26610dec14fbd1d97dd0666d44898bbee9302fc2831ec462bb7955e3e
expected ETag:
  W/"28f8708945f9f68f16b9ce4aad1641f2"
```

Observed independently:

```text
actual SHA-256:
  cc0e71b5d666401b789460648398ee2b0f2c0d2897de9b90493bc8c9528d1d78
actual ETag:
  W/"f29c60174f2df8c7104fcc7d0bddd254"
```

The exact provider-free probe returned:

```text
contract_verifier=PASS
environment_preflight=FAIL:PreCanaryInvariantError:Codex model catalog ETag changed
```

This P1 blocks the single Pilot call. No attempt was consumed.

## Other audited categories

All other required pre-outcome categories passed:

- frozen HEAD `ebc98494c393b52a9ce31fe5b9b47fcabd25e058`,
  tree `68d73fb21cf0991a86abc7b72a19adcf348aad4c`;
- true M6E checkpoint
  `57c9d520cd3788ab355acdef4f8a5ced81e8585d`, tree
  `85cfcbea43400b7406be4cbdf2fd84dd1a3252b3`;
- V5 source base `c6431eba7f2003d3a630e40511902039431bbfd7`,
  tree `acda815d3c098d7ea9d25066bfa14542aa441bde`;
- finalized M6E gate `PASS`, `P0=0`, `P1=0`, with runtime release
  `c7ab04e5c8425f7a43cd84a0c63f1871bb02effe1c31b6ebcfe78e704b83ba23`;
- V5 contract canonical digest
  `3070b44e66f28586e2f40857eb456c009c9c5da5fc8bed04ed6cc4a3de3ea515`
  and file SHA-256
  `eadb160fc35f884307c133f02c5699fcc105144b942e95e18dc707d10d148326`;
- V5 manifest canonical digest
  `954e3396e2adbabbcae04324ff5c86f757092cbf797682bf12afb30b92c5ca99`;
- source projection digest
  `7bb5abb9f27164f6496ccf2d42bfc99064c3b1b664e60677d4c5c71e000efe6c`,
  with all 36 projected source hashes matching;
- response schema, BL fixture, training profile, Codex executable, ML-1M
  dataset files, and clean RecBole commit/tree identities matching;
- sealed V1–V4 roots and shared historical log evidence matching their frozen
  file counts, byte counts, and tree digests;
- sealed seeds exactly `[9201, 9202, 9203, 9204]`, making `9205` the smallest
  unused seed;
- fresh V5 experiment, assignment, store, state, result, audit, Broker, memory,
  filesystem-capability, and prospective runtime-binding identities distinct
  from V1–V4;
- V4/V5 A/B/C definitions, common runtime, budgets, failure/readiness rules,
  one-attempt/no-retry policy, blinding, Research/Router/Producer/Meta,
  Guard/Fusion, and analysis semantics unchanged;
- full canonical store-audit capability present and provider-free focused tests
  passing;
- no treatment outcome inspected.

The V5 output root remained absent:

```text
/root/projects/RecClaw_research_line_abc/results/research_line/m6_pilot_9205_v5
```

No Broker/provider call occurred.

## Authorization status

```yaml
M6E_gate: PASS
fresh_pre_outcome_audit: FAIL
Pilot_V5_single_call: NOT_AUTHORIZED
attempt_consumed: false
output_root_created: false
```

Do not run the frozen Pilot command until this P1 is prospectively repaired and
the resulting frozen pre-outcome state passes a new independent audit with
`P0=0` and `P1=0`.
