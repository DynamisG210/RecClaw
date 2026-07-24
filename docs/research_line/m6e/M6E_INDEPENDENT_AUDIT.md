# M6E Independent Audit

## Verdict

```yaml
M6E_verdict: PASS
audit_verdict: PASS_WITH_NONBLOCKING_P2
P0: 0
P1: 0
P2: 1
authority: NONE
evidence_class: DEVELOPMENT_ONLY
formal_acceptance: false
```

M6E clears the required independent `P0=0/P1=0` gate. The active
training release, store-audit port, closed filesystem projection, four
package-owned CUDA canaries, controlled runtime failure, authoritative
post-run audit, regression suites, and predecessor sealing all passed
independent checks.

This audit does not authorize or execute a fresh Pilot, a Broker/provider
call, M7, or M8.

## Findings

### P0

None.

### P1

None.

### P2-1 — fresh Pilot identity remains a prospective post-M6E action

`FreshPilotOrchestratorV2` is correctly blocked by the current
`PENDING_INDEPENDENT_AUDIT` packet before any Broker attribute is accessed.
Its presently wired fallback still uses the sealed V4
`PilotStoreContractV2` / seed-9204 identity and
`M6-PILOT-9204-OPAQUE-V4` assignment nonce.

This is nonblocking for M6E because no fresh Pilot was launched and section
10 of the governing command schedules fresh-Pilot identity selection only
after M6E PASS. Before the one authorized fresh Pilot is invoked, the
controller must:

1. determine the smallest unused Pilot seed greater than 9204 from the
   sealed registry;
2. freeze a new Pilot version, experiment ID, contract, state, roots, Broker
   records, runtime bindings, and source projection;
3. keep V1-V4 state and responses unused; and
4. run the finalized M6E conformance preflight before the first Broker call.

The existing entrypoint must not be invoked merely because the M6E packet is
later finalized to PASS.

## Audited state

- Worktree: `/root/projects/RecClaw_research_line_abc`
- Branch: `feat/research-line-abc`
- Entry commit: `fe81dc111f13e79052eb8c68aff70ada46cdecb1`
- Entry tree: `d79954f898fc05bfa94abe6d4208433398a7a3ec`
- Pending packet content digest:
  `5419b9a7ac4a590fba3d3e26dc09240ce56e1bf04ee007d31e1196bcdded5e5c`
- Pending packet file SHA-256:
  `9208b473fc54bdec451a3c2a838e619ea6ce4bfd849662952e9505415494c4f3`
- Source projection digest:
  `6a131217a64406bfd5cc021d91b2148637696d36a53bd3985fb55d1058a2aec3`

The manifest, compatibility matrix, pending packet, and final-canary
projection all reproduced their claimed canonical content digests. Every
file in the packet source projection reproduced its claimed SHA-256.

The unrelated dirty paths `notes/method_change_space.md`,
`recclaw_ext/models/lightgcn_lw.py`, and `scripts/agent.py` had no semantic
diff under `git diff --ignore-cr-at-eol`; their visible changes are line
ending noise and were excluded from M6E.

## Runtime Release V2

The independently resolved active release is:

`c7ab04e5c8425f7a43cd84a0c63f1871bb02effe1c31b6ebcfe78e704b83ba23`

Independent validation returned no release failures. The frozen environment
resolved as:

- Python `3.10.20`
- NumPy `1.26.4`
- SciPy `1.12.0`
- Torch metadata `2.10.0`
- Torch runtime `2.10.0+cu128`
- CUDA build `12.8`
- RecBole `1.2.1`
- RFC 8785 `0.1.4`
- CUDA available with one
  `NVIDIA GeForce RTX 4070 Laptop GPU`

The bounded compatibility matrix was independently reproduced:

| SciPy | `dok_matrix._update` | Result |
|---|---:|---|
| 1.15.3 | absent | incompatible |
| 1.14.1 | absent | incompatible |
| 1.13.1 | absent | incompatible |
| 1.12.0 | present; probe value `1.0` | compatible |

The selected environment therefore closes the V4 LightGCN failure by a
dependency-only change. No assignment or `setattr` monkeypatch for
`dok_matrix._update` was found. The RecBole runtime remained clean at commit
`7b02be5ec80a88310f2d04a27a82adfcbb5dc211`, tree
`ca6386c4121ce2aae478ced7e136894ac1d7c218`, with no core edit.

The release binds:

- filesystem policy digest
  `f70210b22c78f4c10f89b602a02e908631ed2223f068dbc0ec76d6b6e78ce95c`;
- store-audit contract digest
  `9f2924d274e039e87d5758f9c31a3892732589e15d3fad2d170dad61aee5c713`;
- exact launcher, worker, schemas, lock, profile, handler registry, RecBole
  source, dataset, metric, resource, and close-chain identities.

Code inspection confirmed all three Arms use the same active release and
filesystem policy; only their exact private roots and resulting capability
identities differ.

## Canonical store-audit port

`ExperimentStoreAuditPortV1` is the canonical typed port for both store
variants. `RealPilotOrchestratorV1` obtains the adapter and executes a
preflight audit before Broker use; `pilot_audit()` delegates to the same port
rather than duplicating SQL.

The BPR full-triplet store was reopened with SQLite URI
`mode=ro&immutable=1` and passed the canonical audit:

- report digest:
  `5f308c0831090ccfcb9cf579cd59e1f49ebb198130df12874cf472687cc4c0fa`;
- `PRAGMA integrity_check`: `ok`;
- foreign-key violations: `0`;
- exact tables:
  `arm_state`, `artifact_index`, `execution_claims`, `resource_ledger`,
  `round_events`, `rounds`, `scheduled_slots`, `triplet_barrier`;
- table counts respectively: `3`, `14`, `1`, `8`, `24`, `3`, `3`, `1`;
- one claim in `FINISHED`, execution-debited, start-confirmed state;
- round, claim, feedback, barrier, and artifact-index checks all true.

The independently produced report was byte-for-field identical to the
canary's typed report and matched the packet digest.

## Filesystem confinement

The authoritative capability recursively makes the complete mount tree
read-only, then reopens only:

- the exact run-private root;
- private `/tmp`, `/var/tmp`, and `/dev/shm` bind mounts;
- `/dev/dxg` as the WSL GPU device capability; and
- the worker's resolved `/proc/<pid>` self-control bind mount.

The stable capability binds `/proc/self`; the worker resolves that identity
to its own PID. `/proc` itself remains read-only. Project, RecBole, dataset,
Windows drives, WSL mounts, source, configs, shared results, and shared log
roots remain read-only.

A live namespace check of the final BPR worker independently observed only
the exact run root, `/dev/dxg`, `/dev/shm`, `/proc/241933`, `/tmp`, and
`/var/tmp` as writable. All 39 mount rows were subsequently bound into the
worker result and side-effect audit. Each authoritative canary reproduced:

- no missing writable mount;
- no unexpected writable mount;
- valid mount-audit digest;
- unchanged protected-root before/after digest;
- private cwd, log, home, cache, checkpoint, and temp roots;
- RecBole's default log under the run-private `work/log/<model>` path; and
- CUDA available with one bound GPU.

Earlier M6E evidence generations were correctly excluded from authority:
the pre-recursive generation left WSL submounts writable, the next
generation caused CUDA error 304 by making `/proc` read-only, and the next
temporarily reopened all of `/proc`. Only the narrowed 9440-9444 generation
is projected as final evidence.

## Fixed canaries and controlled failure

The published final projection is:

- path:
  `results/research_line/m6e_final_canaries/M6E_FINAL_CANARY_PROJECTION.json`;
- canonical content digest:
  `2d7311e37dbc128b0894853c9ca2423bb8d45c9a8ebee832278d065fdcec520c`;
- file SHA-256:
  `326238bf500c93a37a44cb33b4587475d13c63af246be638eaaf02f567eaae0a`;
- authoritative execution tree:
  `results/research_line/m6e_narrow_final_canaries`;
- tree: `129` files, `1,007,573` logical bytes;
- tree digest:
  `fad609d9f4680a0256a337a9b0f8255ef562e09bcd7f28d5cfd88fb42a59b374`.

| Seed | Family | Terminal result | CUDA | Confinement | Typed raw digest |
|---:|---|---|---|---|---|
| 9440 | BPR | success | PASS | PASS | `f5aa50b903576c34413bd1b487cfab1ec0a2472fd576b26fe1b2c75c6ffdbca9` |
| 9442 | LightGCN | success | PASS | PASS | `90c9157653e7a65fcfe82963181b57858c634ce7fe462642aa09cca43f23f34d` |
| 9443 | NGCF | success | PASS | PASS | `e3ee7e267bdd7f57a39dddb0c321f1fbf6613c56740d3c65dfd3e939c0598eab` |
| 9444 | SGL | success | PASS | PASS | `8576ab493311c1c028914663233e6c03f01765b6646f7673aa0c4f46288af619` |
| 9441 | BPR forced failure | `RUNTIME_FAILURE` | PASS | PASS | `d079793d55e0fd30855db48be10b15f3c1d851cf4768abe90edf22c2b9f51830` |

For every row, the independently reconstructed typed chain matched:

`confirmation -> receipt -> raw output -> resource accounting -> common close -> raw-result envelope`.

The forced failure closed the claim, result, resource debit, round, and
authoritative audit mechanically. It emitted no normalized metrics and its
readiness input contained `success_count=0`, `failure_count=1`; it was not
counted as a successful training outcome.

The BPR rehearsal then completed the exact full authoritative
`pilot_audit()` before readiness-input construction: three rounds, three
feedback records, one execution, one closed triplet barrier, and the store
report above. Fixed canaries made zero Broker/LLM/provider calls and their
metrics did not enter Search Memory, Frontier, Meta, treatment comparison,
or readiness effect analysis.

## Broker gate

The pending packet has null audit counts and verdict
`PENDING_INDEPENDENT_AUDIT`. An independent constructor probe of
`FreshPilotOrchestratorV2` raised
`fresh Pilot is blocked before Broker use until M6E PASS` with zero Broker
attribute accesses. The packet's content digest, active release digest,
handler-family coverage, forced-failure classification, and authoritative
audit fields are checked by the preflight gate.

This audit file is the independent input needed to finalize that packet; the
current pending packet was not modified by the auditor.

## Regression and adversarial verification

Independent commands and results:

```text
python -m unittest \
  tests.experiments.helix_abc_v1.test_m6e_environment_closure \
  tests.experiments.helix_abc_v1.test_m6r_training_runtime
38/38 PASS

python -m unittest discover \
  -s tests/experiments/helix_abc_v1 -p test_*.py
140/140 PASS

python -m unittest \
  tests.evidence_guard.test_core_v1 \
  tests.test_bl_icf_mechanism_space
40/40 PASS

python -m unittest discover -s tests -p test_*.py
147/147 PASS
```

The focused suite covers both store adapters, missing/altered audit
capability, dependency substitution, old-release immutability, shared-root
and symlink escape, writable WSL submount rejection, cross-root capability
binding, typed runtime mismatch, forced-failure classification, handler
coverage, and the pre-PASS Broker gate.

## Sealed predecessors and semantic isolation

The sealed tree projection from the M6E task manifest reproduced exactly:

| Root | Files | Bytes | Tree digest |
|---|---:|---:|---|
| V1 seed 9201 | 14 | 172657 | `f2c76e7f7c3efea542928d954d6ec28b4bc3f8dd8513dc8963e6798d5171f7e8` |
| V2 seed 9202 | 17 | 181962 | `cfd8a3abc71a7d9db4f4ea937139d49c44f1410056482b6f16332240636036ec` |
| V3 seed 9203 | 15 | 181108 | `5daf01110948b23987d8ca898ce566bb3733d14a3d574f99d25e89e7f82916fe` |
| V4 seed 9204 | 215 | 1148945 | `162b452ec63600d8c734940bd22fbf0f590a8847823cfae7a82473aba83e485f` |
| shared historical `log/` | 9 | 0 | `0eea1c54d8a258343164bd162dc1469bce26a1602ec5651e836a05ea23ef0bea` |

Historical identities also remained exact:

- fake release:
  `0a616ee205f494a161e7b10424181f37247347de8c5d714328dfa2f3f1ffb41d`;
- fake source manifest:
  `038ebf3186ff0c67d7ebe7f5d799ea2d0f4aa6ee141f6a42d9767a05fb01a5fd`;
- fake common projection:
  `97c4247af6f0a2fb2fbccd64663e2a6ed25cc1bc98aab2f5630e6d41d6f67347`;
- V4 training release:
  `9401d9c5f5f096057d183cf0e4dc4f2f76bbc85030099d17646ecdad5e8ad375`.

The sensitive diff was empty for Pilot analysis, Research controller and
capability, Real Canary/Producer path, Router/Meta Helix path, Evidence
Guard, BL-ICF mechanism definitions/profile/anchor fixtures, and the frozen
Pilot training profile. The `real_pilot.py` changes are limited to the
canonical audit port, Runtime Binding V2/filesystem capability integration,
and the pre-PASS conformance gate; thresholds, budgets, treatment
definitions, and readiness semantics are unchanged.

## Required next action

Finalize the M6E packet with this audit's SHA-256 and `P0=0`, `P1=0`,
`P2=1`; create the dedicated local M6E checkpoint without pushing. Then
resolve P2-1 prospectively before the single authorized fresh Pilot. No
fresh Pilot may reuse V1-V4 identity or state.
