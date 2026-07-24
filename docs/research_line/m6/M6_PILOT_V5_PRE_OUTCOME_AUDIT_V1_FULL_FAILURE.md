# M6 Pilot V5 Fresh-Context Pre-Outcome Audit V1 Full Failure

## Verdict

```yaml
audit_verdict: FAIL
P0: 0
P1: 2
P2: 0
pilot_v5_single_call_authorized: false
remaining_authorized_attempts_before_execution: 1
broker_or_provider_called: false
v5_output_root_created: false
authority: NONE
evidence_class: DEVELOPMENT_ONLY
formal_acceptance: false
```

Pilot V5 is blocked before its first Broker/provider call. The frozen external
model-catalog identity has drifted, and the V5 exact-contract verifier does not
enforce the catalog SHA-256 that the contract declares. The later environment
preflight does fail closed on the changed ETag, so no Broker use is possible
through the frozen command, but `P1=0` is not established.

No source was modified, no treatment outcome was inspected, no Pilot state was
opened, and no V5 output root was created during this audit.

## P1 findings

### P1-1 — Frozen model-catalog external identity is no longer present

The frozen V5 contract binds:

```text
models_cache_path:
  /mnt/c/Users/gtrho/.codex/models_cache.json
expected SHA-256:
  c8a701e26610dec14fbd1d97dd0666d44898bbee9302fc2831ec462bb7955e3e
expected ETag:
  W/"28f8708945f9f68f16b9ce4aad1641f2"
```

The independent live read found:

```text
actual SHA-256:
  cc0e71b5d666401b789460648398ee2b0f2c0d2897de9b90493bc8c9528d1d78
actual ETag:
  W/"f29c60174f2df8c7104fcc7d0bddd254"
frozen model gpt-5.4 present in current catalog: true
```

Model presence does not repair the frozen-byte/ETag mismatch. The exact
provider-free execution preflight reproduced:

```text
contract_verifier=PASS
environment_preflight=FAIL:PreCanaryInvariantError:Codex model catalog ETag changed
```

This is a frozen external-environment identity failure and blocks Pilot V5
before Broker use.

### P1-2 — The declared model-catalog SHA-256 is not enforced

`scripts/run_m6_pilot_v5.py::verify_contract_v5()` verifies the source
projection, response schema, BL fixture, training profile, and Codex executable,
but omits `broker.models_cache_path/models_cache_sha256` from its exact-file
checks. The shared `pilot_environment_preflight()` validates the catalog ETag
and requested model presence, but not the declared catalog SHA-256.

The independent probe is discriminative evidence: `verify_contract_v5()` passed
while the catalog bytes differed from the contract. The current ETag mismatch is
still caught by the later preflight, but a byte change retaining the frozen ETag
would not be rejected by either exact-identity layer. A declared controlled
external digest therefore lacks complete enforcement.

This is independent of P1-1: merely refreshing the frozen ETag/hash would not
close the verifier gap.

## Cleared audit categories

### Git and freeze identities

```text
branch:
  feat/research-line-abc
M6E checkpoint:
  commit 57c9d520cd3788ab355acdef4f8a5ced81e8585d
  tree   85cfcbea43400b7406be4cbdf2fd84dd1a3252b3
V5 source base:
  commit c6431eba7f2003d3a630e40511902039431bbfd7
  tree   acda815d3c098d7ea9d25066bfa14542aa441bde
frozen pre-outcome HEAD:
  commit ebc98494c393b52a9ce31fe5b9b47fcabd25e058
  tree   68d73fb21cf0991a86abc7b72a19adcf348aad4c
upstream relation at audit entry:
  ahead 19, behind 0
```

`57c9d52` is the true M6E PASS checkpoint. `c6431eb` is the later V5 source
projection base, and `ebc9849` adds only the frozen V5 contract and task
manifest. Both ancestry edges were verified.

The pre-existing unrelated dirty paths were preserved and excluded:

```text
notes/method_change_space.md
recclaw_ext/models/lightgcn_lw.py
scripts/agent.py
RecClaw_Codex_Autonomous_M1_M8_Master_Goal.md
RecClaw_Codex_Autonomous_M1_M8_Master_Goal.md:Zone.Identifier
results/agent_state_summary.json
results/research_line/m6e_canaries/
results/research_line/m6e_superseded_cpu_fallback_canaries/
results/research_line/m6e_superseded_pre_recursive_ro_canaries/
results/research_line/m6e_superseded_proc_root_canaries/
results/research_routes.jsonl
```

### M6E gate and audit capability

The finalized M6E packet reproduced:

```yaml
verdict: PASS
P0: 0
P1: 0
P2: 1
packet_content_digest: cb8376e5189bc60a9a60e776892b1bdc39004c480c47ab9c8fc07e406c3601ca
packet_sha256: 69a274938b2c73a29d1ee4ed3b893128a05d5d7ffc5d49fc5894de605647f144
independent_audit_sha256: 0fb306da466b45c9f453419ed8531af832edd4a5659d7707bc7ffdfd0ac9b798
training_runtime_release_digest: c7ab04e5c8425f7a43cd84a0c63f1871bb02effe1c31b6ebcfe78e704b83ba23
store_audit_contract_digest: 9f2924d274e039e87d5758f9c31a3892732589e15d3fad2d170dad61aee5c713
full_audit_report_digest: 5f308c0831090ccfcb9cf579cd59e1f49ebb198130df12874cf472687cc4c0fa
filesystem_capability_policy_digest: f70210b22c78f4c10f89b602a02e908631ed2223f068dbc0ec76d6b6e78ce95c
allowed_handler_families: [BPR, LightGCN, NGCF, SGL]
```

`require_m6e_conformance_packet()` reproduced PASS and checks packet canonical
content, `P0/P1`, active release identity, all four handler canaries, forced
runtime-failure classification, and the full authoritative audit rehearsal.

`ExperimentStoreAuditPortV1` is shared by base and training stores. The
authoritative `pilot_audit()` uses that port, including SQLite integrity,
foreign keys, required tables, round/claim/feedback uniqueness, triplet
barriers, and exact artifact-index checks. Focused tests for this path passed.

### Contract, manifest, source, and remaining external identities

```yaml
contract_content_digest:
  3070b44e66f28586e2f40857eb456c009c9c5da5fc8bed04ed6cc4a3de3ea515
contract_sha256:
  eadb160fc35f884307c133f02c5699fcc105144b942e95e18dc707d10d148326
manifest_content_digest:
  954e3396e2adbabbcae04324ff5c86f757092cbf797682bf12afb30b92c5ca99
manifest_sha256:
  eb5d5e04cfe7ebce329f3f30775b9c1208ae437a03a62298390b908a50cbaf76
source_projection_digest:
  7bb5abb9f27164f6496ccf2d42bfc99064c3b1b664e60677d4c5c71e000efe6c
source_files_checked: 36
source_hash_mismatches: 0
```

The response schema, BL fixture, training profile, Codex executable, and all
three ML-1M dataset files reproduced their frozen SHA-256 values. The sole
external mismatch was the model catalog in P1-1.

The RecBole runtime remained clean at:

```text
commit 7b02be5ec80a88310f2d04a27a82adfcbb5dc211
tree   ca6386c4121ce2aae478ced7e136894ac1d7c218
```

### Sealed V1–V4 lineage and smallest unused seed

All five contract canonical digests reproduced. The sealed Pilot registry was
exactly `[9201, 9202, 9203, 9204]`; therefore `9205` is the smallest unused
Pilot seed greater than 9204.

The sealed roots reproduced without reading treatment semantics:

| Root | Files | Bytes | Tree digest |
|---|---:|---:|---|
| `m6_pilot_9201_v1` | 14 | 172657 | `f2c76e7f7c3efea542928d954d6ec28b4bc3f8dd8513dc8963e6798d5171f7e8` |
| `m6_pilot_9202_v2` | 17 | 181962 | `cfd8a3abc71a7d9db4f4ea937139d49c44f1410056482b6f16332240636036ec` |
| `m6_pilot_9203_v3` | 15 | 181108 | `5daf01110948b23987d8ca898ce566bb3733d14a3d574f99d25e89e7f82916fe` |
| `m6_pilot_9204_v4` | 215 | 1148945 | `162b452ec63600d8c734940bd22fbf0f590a8847823cfae7a82473aba83e485f` |
| shared historical `log/` | 9 | 0 | `0eea1c54d8a258343164bd162dc1469bce26a1602ec5651e836a05ea23ef0bea` |

No V1–V4 seed, response, state, result, or memory reuse was found.

### Fresh V5 identities and absent output root

```yaml
search_seed: 9205
experiment_id: HELIX-ABC-DEVELOPMENT-PILOT-9205-V5
assignment_commitment: eb724d56680b620bf1a9e9d58f6301079607c244ae882f01d9bb9e50ca1e0a73
store_contract_identity_digest: b9be2c4d31c3e7ef3e4eceb1f75db70992e2d182676d0abd8779eea52f533daf
result_root: results/research_line/m6_pilot_9205_v5
audit_root: results/research_line/m6_pilot_9205_v5
audit_storage: ROOT_LEVEL_TYPED_AUDIT_FILES_V2
broker_records_root: results/research_line/m6_pilot_9205_v5/broker_private
runtime_root: results/research_line/m6_pilot_9205_v5/runtime
state_store: results/research_line/m6_pilot_9205_v5/runtime/neutral/experiment.sqlite3
memory_namespace: DEVELOPMENT_ONLY/SEARCH_MEMORY
prior_memory_imported: false
output_root_absent: true
```

V1–V5 experiment IDs, contract digests, assignment commitments, and store
identity digests were pairwise distinct.

A provider-free prospective binding derivation, using identical sentinel
candidate/round/run inputs for V4 and V5, proved the binding sets disjoint.
The V5 prospective binding digests were:

```text
A 5677080112ce02bd48c4dff854917dab359121db3f4e71d2a26da7d9241cbb0e
B 971348f2be82cb98e708cb41df124738bf973f446e54c724b5f42ebdfe0863f0
C 48efea9f8d78c1e5380616a4eb79acec4ebb3c1549ff1694711e719678e8e376
```

These are prospective identity proofs, not treatment outcomes or executed-run
records. Actual bindings remain uncreated because Pilot V5 did not start.

### Unchanged treatment, runtime, budget, blinding, and analysis semantics

V4 and V5 were exactly equal for:

```text
analysis
arm_composition
BL/common release projection
per-Arm per-round budget
Guard and Fusion identities
lineage policy
Main fixed fields
Research/Producer/Router/Meta identities and gate labels
```

The source delta from M6E checkpoint `57c9d52` to V5 source base `c6431eb`
contained only:

```text
scripts/freeze_m6_pilot_v5_contract.py
scripts/run_m6_pilot.py
scripts/run_m6_pilot_v5.py
src/recclaw_core/experiments/helix_abc_v1/real_pilot.py
tests/experiments/helix_abc_v1/test_m6_pilot.py
```

No Research controller, Producer, Router, Meta, Evidence Guard, Fusion,
BL-ICF compiler/profile, or Pilot analysis source changed. The `real_pilot.py`
delta adds the V5 seed/store/guard/assignment wrapper and retains the common
orchestration semantics; `run_m6_pilot.py` only makes the verifier and
orchestrator type injectable for the V5 wrapper.

The frozen rules remain:

```yaml
authorized_attempts: 1
rerun: false
in_place_patch: false
broker_retries: 0
no_patch_or_retry_after_start: true
assignment_opaque: true
readiness_uses_effect_size: false
failure_rate_ceiling: 0.25
minimum_success_per_opaque_instance: 2
same_A_B_C_common_runtime: true
same_budget_contract: true
```

## Commands and results

All commands were provider-free. No command invoked the V5 run entrypoint.

```text
git rev-parse HEAD HEAD^{tree}
git log --oneline --decorate -12
git status --short --branch
git show -s --format='%H %T %s' 57c9d52
git show -s --format='%H %T %s' c6431eb
git show -s --format='%H %T %s' ebc9849
git rev-list --left-right --count HEAD...origin/feat/research_line
git merge-base --is-ancestor 57c9d52 c6431eb
git merge-base --is-ancestor c6431eb ebc9849
```

Result: identities and ancestry above; ahead 19, behind 0.

```text
PYTHONPATH=src /root/projects/RecClaw_m6_training_runtime_v2/bin/python -
```

Read-only canonical audit operations:

- recomputed V1–V5 contract and V5 manifest content digests;
- checked 36 source hashes and all declared external/dataset hashes;
- invoked `verify_contract_v5()` and `require_m6e_conformance_packet()`;
- recomputed the sealed-root projection
  `SHA256(RFC8785([{path,sha256,size_bytes}, ... sorted by path]))`;
- compared V4/V5 frozen semantic fields;
- derived disjoint prospective V4/V5 runtime bindings without materializing
  either root.

Result: all cleared categories above passed; model-catalog hash mismatch found.

```text
PYTHONPATH=src /root/projects/RecClaw_m6_training_runtime_v2/bin/python -
```

Provider-free exact preflight probe:

```text
contract_verifier=PASS
environment_preflight=FAIL:PreCanaryInvariantError:Codex model catalog ETag changed
```

```text
PYTHONPATH=src /root/projects/RecClaw_m6_training_runtime_v2/bin/python \
  -m unittest \
  tests.experiments.helix_abc_v1.test_m6_pilot \
  tests.experiments.helix_abc_v1.test_m6e_environment_closure
```

Result: `33/33 PASS`.

## Frozen run command and authorization status

The exact frozen command remains:

```bash
PYTHONPATH=src /root/projects/RecClaw_m6_training_runtime_v2/bin/python scripts/run_m6_pilot_v5.py --contract docs/research_line/m6/DEVELOPMENT_PILOT_CONTRACT_V5.json --output-root results/research_line/m6_pilot_9205_v5
```

Authorization status:

```yaml
M6E_prerequisite: PASS
fresh_context_pre_outcome_gate: FAIL
Pilot_V5_single_call: NOT_AUTHORIZED
reason:
  - P1-1 frozen model-catalog identity drift
  - P1-2 declared catalog SHA-256 not enforced
safe_state:
  output_root_absent: true
  attempt_consumed: false
  Broker_calls: 0
```

Do not run the command. Any prospective repair must preserve the sealed
V1–V4 lineage, create a new frozen pre-outcome source/contract identity as
required by the governing policy, enforce every declared external digest, and
undergo another fresh-context pre-outcome audit before Broker use.
