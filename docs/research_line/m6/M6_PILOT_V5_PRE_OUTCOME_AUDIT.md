# M6 Pilot V5 Repaired Pre-Outcome Independent Audit

## Verdict

```yaml
audit_verdict: PASS
P0: 0
P1: 0
P2: 0
pilot_v5_single_call_authorized: true
remaining_authorized_attempts_before_execution: 1
attempt_consumed: false
broker_calls: 0
provider_calls: 0
v5_output_root_absent: true
authority: NONE
evidence_class: DEVELOPMENT_ONLY
formal_acceptance: false
```

The repaired Pilot V5 pre-outcome state clears the required
`P0=0/P1=0` gate. Exactly one execution of the frozen Pilot V5 command is
authorized. This audit did not execute that command, instantiate a real Broker,
call a provider, create the V5 output root, inspect treatment outcomes, or
authorize M7/M8.

## Findings

### P0

None.

### P1

None.

### P2

None.

## Scope and frozen identity

The audit was performed provider-free and read-only against:

```text
worktree:
  /root/projects/RecClaw_research_line_abc
branch:
  feat/research-line-abc
frozen repaired docs commit:
  5a07dccff2eca667fa9961f5acb2f73bda872b74
frozen repaired docs tree:
  69ff1f4db28603ec6c348aee5d857624d8b51d62
source base commit:
  cd93c6db015450e351aa587ba46df8a854c1a954
source base tree:
  457caab324919046195eeef54e42681f99468433
M6E checkpoint commit:
  57c9d520cd3788ab355acdef4f8a5ced81e8585d
M6E checkpoint tree:
  85cfcbea43400b7406be4cbdf2fd84dd1a3252b3
upstream relation:
  ahead 22, behind 0
```

The source-base and M6E checkpoint trees independently reproduced their
claimed identities. Pre-existing unrelated dirty paths were preserved and
excluded from this audit.

## Closure of the prior V1 P1

The mutable global Codex model cache is no longer a runtime identity
dependency.

The repaired contract binds:

```yaml
mode: FROZEN_SELECTED_MODEL_CATALOG_PROJECTION_V1
path: /root/projects/RecClaw_research_line_abc/src/recclaw_core/experiments/helix_abc_v1/resources/pilot_v5_model_catalog_snapshot.json
sha256: c00da1e1bc082c2fccb3c9eb832079ab3bbffae2976db74c467f76f2843b1c1c
record_schema: recclaw.frozen-selected-model-catalog-projection.v1
selected_model_slugs:
  - gpt-5.4
etag: 'W/"f29c60174f2df8c7104fcc7d0bddd254"'
```

Independent source inspection established:

- the package-owned projection contains exactly the selected `gpt-5.4`
  catalog row;
- `verify_contract_v5()` hashes the projection path declared by the contract
  and rejects a mode other than
  `FROZEN_SELECTED_MODEL_CATALOG_PROJECTION_V1`;
- `pilot_environment_preflight()` reads the contract-declared projection,
  checks its ETag, and confirms `gpt-5.4` is present;
- the relevant runtime code
  (`run_m6_pilot_v5.py`, `run_m6_pilot.py`, and `real_canary.py`) contains no
  hard-coded reference to
  `/mnt/c/Users/gtrho/.codex/models_cache.json`; and
- the global path retained inside the projection's `source_observation`
  metadata records provenance only. It is neither opened nor identity-checked
  by the Pilot runtime.

The repair commit is bounded to the freezer, V5 verifier, package-owned
projection, and focused Pilot test. It does not change Arm, research, routing,
guard, training, readiness, or analysis semantics.

## Exact content, source, and external identities

Canonical digests were recomputed as
`SHA256(RFC8785(object without content_digest))`:

| Artifact | Claimed content digest | Recomputed | File SHA-256 |
|---|---|---|---|
| V5 contract | `b953f056305014bba05858bcafcc88cdd7483d4b187b3aa90f55cda151017f84` | exact | `02709a3d8a1217201f32a2f66ec4599cec0f44e3ea3917fbe6bf213230863f7e` |
| V5 task manifest | `294cdf799e6c503ea21417650e1e2598e21deb320d92d47b67ffea0f7b2c4b42` | exact | `e106762c0cd6c22939c91d904920937f8abb0cebb6f3d6c22844668df48d69df` |
| M6E conformance packet | `cb8376e5189bc60a9a60e776892b1bdc39004c480c47ab9c8fc07e406c3601ca` | exact | `69a274938b2c73a29d1ee4ed3b893128a05d5d7ffc5d49fc5894de605647f144` |

The V5 source projection reproduced as:

```yaml
algorithm: SHA256(RFC8785(source.files mapping))
claimed: 3cc983d91b65c5a988b47949018faecb6b8304812a7f68e111da9aad663fba39
recomputed: 3cc983d91b65c5a988b47949018faecb6b8304812a7f68e111da9aad663fba39
projected_files: 37
hash_mismatches: 0
```

An initial diagnostic encoded the projection as a list rather than the
repository's frozen mapping and therefore did not reproduce the digest. The
probe was corrected to the freezer's actual
`sha256_digest(source_files)` algorithm above; all individual file hashes and
the canonical mapping digest then matched.

All eight declared live external identities matched:

- response schema;
- package-owned selected-model catalog projection;
- Codex executable;
- BL-ICF anchor fixture;
- Pilot training profile; and
- the three ML-1M dataset files.

All lineage hashes also matched exactly:

```yaml
M6E_independent_audit_sha256: 0fb306da466b45c9f453419ed8531af832edd4a5659d7707bc7ffdfd0ac9b798
M6E_packet_sha256: 69a274938b2c73a29d1ee4ed3b893128a05d5d7ffc5d49fc5894de605647f144
M6E_execution_record_sha256: 2e63264d006c315e272387eb67049f3c656749847ff971bc6e50b2aa16004b64
V4_contract_sha256: 796b9c3d83c67afb5e58606ac4080438df2c8ccb5732a8b231fefe25b571d4cd
V4_failure_record_sha256: 3b202dee1308e584d4d04c53f652689a14cc5d4d0fba5d1a89541f4b9b76c003
V1_pre_outcome_failure_sha256: f214573ab5c2074ff5c570a43200c3efde3d434d437ac4865da121dacca621fc
```

`verify_contract_v5()` returned `PASS`.

## M6E gate, runtime release, and audit capability

The finalized M6E packet independently returned:

```yaml
verdict: PASS
P0: 0
P1: 0
runtime_release_digest: c7ab04e5c8425f7a43cd84a0c63f1871bb02effe1c31b6ebcfe78e704b83ba23
contract_runtime_release_digest: c7ab04e5c8425f7a43cd84a0c63f1871bb02effe1c31b6ebcfe78e704b83ba23
store_audit_contract_digest: 9f2924d274e039e87d5758f9c31a3892732589e15d3fad2d170dad61aee5c713
full_audit_report_digest: 5f308c0831090ccfcb9cf579cd59e1f49ebb198130df12874cf472687cc4c0fa
filesystem_capability_policy_digest: f70210b22c78f4c10f89b602a02e908631ed2223f068dbc0ec76d6b6e78ce95c
```

`require_m6e_conformance_packet()` returned `PASS` and reproduced the exact
runtime release binding. The canonical typed
`ExperimentStoreAuditPortV1` and `StoreIntegrityReportV1` remain present;
base and training stores share the same audit implementation, and the full
authoritative audit rehearsal remains bound by the packet.

Focused provider-free tests covering the Pilot and M6E environment/audit path
passed:

```text
34/34 PASS
```

## Provider-free environment preflight

The exact V5 environment preflight completed without constructing or invoking
a Broker:

```yaml
verdict: PASS
model_available: gpt-5.4
codex_cli_version: codex-cli 0.144.1
broker_login_state_probe: CHATGPT
runtime:
  python: 3.10.20
  numpy: 1.26.4
  scipy: 1.12.0
  torch: 2.10.0+cu128
  cuda: true
  recbole: 1.2.1
recbole_clean_commit: 7b02be5ec80a88310f2d04a27a82adfcbb5dc211
recbole_clean_tree: ca6386c4121ce2aae478ced7e136894ac1d7c218
recbole_import_root: /root/projects/RecBole_m6_runtime
dataset_hashes: PASS
mount_namespace: PASS
numeric_uid_isolation:
  own_read: true
  sibling_read_denied: true
  sibling_write_denied: true
```

The Codex checks were limited to `--version` and `login status`; no
`codex exec`, Broker dispatch, provider request, or proposal generation
occurred.

## Sealed V1-V4 evidence

Each root was recomputed as
`SHA256(RFC8785([{path, sha256, size_bytes}, ... sorted by path]))`:

| Sealed root | Files | Bytes | Recomputed tree digest | Result |
|---|---:|---:|---|---|
| V1 seed 9201 | 14 | 172657 | `f2c76e7f7c3efea542928d954d6ec28b4bc3f8dd8513dc8963e6798d5171f7e8` | PASS |
| V2 seed 9202 | 17 | 181962 | `cfd8a3abc71a7d9db4f4ea937139d49c44f1410056482b6f16332240636036ec` | PASS |
| V3 seed 9203 | 15 | 181108 | `5daf01110948b23987d8ca898ce566bb3733d14a3d574f99d25e89e7f82916fe` | PASS |
| V4 seed 9204 | 215 | 1148945 | `162b452ec63600d8c734940bd22fbf0f590a8847823cfae7a82473aba83e485f` | PASS |
| historical shared `log/` | 9 | 0 | `0eea1c54d8a258343164bd162dc1469bce26a1602ec5651e836a05ea23ef0bea` | PASS |

The canonical V1-V5 contract content digests are pairwise distinct:

```text
V1 8e5c513092659ddb267a78e035e55ce25c2a2479b6503c576319e8dbeb0771cd
V2 41212930dca6b10161abac935eb470be33c07191e9def3237e2f1055a061c8f8
V3 61a1b35b84dcd5fdb12528348de1a488957e8a53eac775b80b70611d7892d5ea
V4 4739c76f39f983bb2e8c50e0f6e222c725041f8c8180bc2d880bc2b62dc3cde0
V5 b953f056305014bba05858bcafcc88cdd7483d4b187b3aa90f55cda151017f84
```

No V1-V4 result, response, state, seed, assignment, or memory was reused.

## Fresh V5 identity

The sealed registry is exactly `[9201, 9202, 9203, 9204]`; `9205` is
therefore the smallest unused seed greater than 9204.

```yaml
search_seed: 9205
experiment_id: HELIX-ABC-DEVELOPMENT-PILOT-9205-V5
assignment_commitment: eb724d56680b620bf1a9e9d58f6301079607c244ae882f01d9bb9e50ca1e0a73
store_contract_identity_digest: b9be2c4d31c3e7ef3e4eceb1f75db70992e2d182676d0abd8779eea52f533daf
result_and_audit_root: results/research_line/m6_pilot_9205_v5
broker_records_root: results/research_line/m6_pilot_9205_v5/broker_private
runtime_root: results/research_line/m6_pilot_9205_v5/runtime
state_store: results/research_line/m6_pilot_9205_v5/runtime/neutral/experiment.sqlite3
memory_namespace: DEVELOPMENT_ONLY/SEARCH_MEMORY
memory_initial_head: null
prior_memory_imported: false
```

Across V1-V5, experiment IDs, contract content digests, assignment
commitments, and store identities are each pairwise distinct. The B/C memory
writers are namespaced by the new V5 experiment ID. Prospective runtime
bindings inherit the new experiment/store/assignment identity and the common
M6E runtime release; no binding or state artifact has yet been materialized.

## Unchanged Pilot semantics

Direct V4/V5 comparisons were exact for:

- `analysis`;
- `arm_composition`;
- `budget_per_arm_round`;
- `guard_and_fusion`;
- `lineage_policy`;
- `main_fixed_fields`;
- `research`; and
- the BL-ICF common release projection.

The frozen invariants remain:

```yaml
same_A_B_C_definitions: true
same_runtime_release_for_all_arms: true
same_budget_contract: true
assignment_opaque: true
failure_rate_ceiling: 0.25
minimum_successful_support_per_opaque_instance: 2
readiness_must_not_use_effect_size: true
no_ndcg_tuning: true
broker_retries: 0
authorized_attempts: 1
rerun: false
in_place_patch: false
no_patch_or_retry_after_start: true
if_non_go: SEAL_AND_HARD_STOP
```

No treatment mapping, NDCG value, relative Arm result, or other Pilot outcome
was opened or used.

## Commands and observed results

All commands were provider-free.

```text
git -C /root/projects/RecClaw_research_line_abc status --short --branch
git -C /root/projects/RecClaw_research_line_abc rev-parse HEAD
git -C /root/projects/RecClaw_research_line_abc log --oneline --decorate -10
git -C /root/projects/RecClaw_research_line_abc show --stat --oneline cd93c6d
git -C /root/projects/RecClaw_research_line_abc show cd93c6d -- \
  scripts/freeze_m6_pilot_v5_contract.py \
  scripts/run_m6_pilot_v5.py \
  src/recclaw_core/experiments/helix_abc_v1/resources/pilot_v5_model_catalog_snapshot.json \
  tests/experiments/helix_abc_v1/test_m6_pilot.py
```

Result: exact commits/trees and bounded repair scope above; unrelated dirt was
preserved.

```text
wsl.exe -d Ubuntu --exec /usr/bin/env \
  PYTHONDONTWRITEBYTECODE=1 \
  /root/projects/RecClaw_m6_training_runtime_v2/bin/python -c \
  <read-only canonical/hash/preflight/seal probe>
```

The probe:

- recomputed contract, manifest, packet, source, external, lineage, and sealed
  root identities;
- called `verify_contract_v5()`;
- called `require_m6e_conformance_packet()`;
- called `pilot_environment_preflight()` without constructing a Broker;
- compared V4/V5 frozen semantic fields and V1-V5 identities; and
- checked output-root and Broker-database absence after all operations.

Result: all gates and identities above passed.

```text
wsl.exe -d Ubuntu --exec /usr/bin/env \
  -C /root/projects/RecClaw_research_line_abc \
  PYTHONPATH=src \
  /root/projects/RecClaw_m6_training_runtime_v2/bin/python \
  -m unittest \
  tests.experiments.helix_abc_v1.test_m6_pilot \
  tests.experiments.helix_abc_v1.test_m6e_environment_closure
```

Result: `34/34 PASS`.

## Side-effect and authorization state

After all checks:

```yaml
output_root:
  /root/projects/RecClaw_research_line_abc/results/research_line/m6_pilot_9205_v5
output_root_absent: true
broker_database_absent: true
broker_calls: 0
provider_calls: 0
attempt_consumed: false
```

Exactly one Pilot V5 call is now authorized:

```bash
PYTHONPATH=src /root/projects/RecClaw_m6_training_runtime_v2/bin/python scripts/run_m6_pilot_v5.py --contract docs/research_line/m6/DEVELOPMENT_PILOT_CONTRACT_V5.json --output-root results/research_line/m6_pilot_9205_v5
```

This authorization is limited to the one frozen attempt. A non-`GO` result,
any P0/P1, or any frozen-identity failure requires sealing and hard-stop; it
does not authorize an in-place patch or rerun. M7/M8 remain unauthorized
unless the completed Pilot and its independent outcome audit satisfy their
existing gates.
