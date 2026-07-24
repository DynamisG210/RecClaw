# M6 Pilot V4 Pre-Outcome Audit

Verdict: `READY_TO_EXECUTE_ONE_FRESH_PILOT`

```yaml
P0: 0
P1: 0
P2: 0
authority: NONE
evidence_class: DEVELOPMENT_ONLY
formal_acceptance: false
```

## Frozen subject

- Base M6R checkpoint:
  `0eea407f19a26e9ce893c40a450cf814be89c43e`.
- Pilot contract:
  `DEVELOPMENT_PILOT_CONTRACT_V4.json`.
- Contract content digest:
  `4739c76f39f983bb2e8c50e0f6e222c725041f8c8180bc2d880bc2b62dc3cde0`.
- Contract SHA-256:
  `796b9c3d83c67afb5e58606ac4080438df2c8ccb5732a8b231fefe25b571d4cd`.
- Fresh seed/root:
  `9204` / `results/research_line/m6_pilot_9204_v4`.

## Entry-gate evidence

- The M6R audit is `PASS_WITH_NONBLOCKING_P2`, `P0=0/P1=0`. Its sole P2
  required a new Pilot contract, seed, root, store and broker state; V4
  supplies all five.
- `PilotStoreContractV2` and `FreshPilotOrchestratorV2` are additive. The
  historical `RealPilotOrchestratorV1`, seed 9203, experiment identity and
  nonce remain available only as the sealed V3 integration target.
- The V4 run script rejects V1 contract schema, seeds 9201/9202/9203, a
  mismatched store identity, assignment, budget, A/B/C composition, BL common
  projection, runner ABI, runtime release or execution purpose before
  constructing the broker.
- Local environment preflight passed with exact dataset bytes, clean RecBole
  commit/tree, Python/package versions, Codex CLI/login/model catalog, mount
  namespace and numeric-UID isolation.
- The training compatibility preflight returned
  `TRAINING_RUNTIME_COMPATIBLE` under runtime release
  `9401d9c5f5f096057d183cf0e4dc4f2f76bbc85030099d17646ecdad5e8ad375`
  without a broker call.
- Targeted tests passed `18/18`; the activated M0-M6R experiment suite passed
  `114/114`.
- V1/V2/V3 contracts, failure records and result roots have zero diff from
  the M6 hard-stop commit. The V4 result root did not exist during this audit.
- The existing unrelated EOL/user changes are excluded from the V4
  checkpoint.

## Frozen execution rule

Exactly one execution of V4 is authorized. Once the output root is created,
there is no ad-hoc runtime or contract patch budget. Any contract/runtime
mismatch, P0/P1, cross-arm contamination, unresolved identity ambiguity or
non-GO result seals V4 and stops M6. M7/M8 remain forbidden unless V4 is GO
and the exact Main freeze separately passes.
