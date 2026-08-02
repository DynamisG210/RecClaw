# Q4 final independent audit

Date: 2026-08-03 (Asia/Shanghai)
Verdict: **PASS_WITH_FINDINGS**
Authority: **DEVELOPMENT_ONLY**; no external acceptance, milestone promotion, outcome-aware superiority, or formal scientific conclusion is granted.

## Audit identity and scope

The audited source is commit `3615a8c9b2cc6d473df6fb1348fd09c47a883426`, parent `63e44341bd1da982f00a9db7de2be7e779c7a63d`, tree `4492bc796707f24080a9ae799023b9d73b15b3c4`. The source worktree `/root/projects/RecClaw_prospective_policy_comparison` was clean, its branch had no upstream, and no remote ref contained the Q4 commit. The independent branch `feat/research-line-final-independent-audit` and worktree `/root/projects/RecClaw_final_independent_audit` were created from that exact commit with WSL-native Git. The unrelated dirty saved checkout at `/root/projects/RecClaw` was not used as the source and was not modified.

This audit read Q4 code, sealed receipts, Provider broker databases, materialized candidates, qualification evidence, resource and matched-training outputs, negative evidence, and the first-parent history. It adds only this report, its canonical receipt, and an artifact-level audit test. It does not alter Q4 implementation, results, frozen inputs, selections, outcomes, or seals.

## Lineage and seal verification

The WSL-native first-parent traversal from P0/pre-freeze-ready through Q4 contains 24 commits, including the preserved R1, Q0R, and Q1 blocking/correction commits. The required milestone identities are:

| Milestone | Commit | Parent | Tree | Canonical receipt SHA-256 |
|---|---|---|---|---|
| P0 / pre-freeze-ready | `ce9104ac67993ea5184ef4337403954ee50b543e` | `1af05904230485f82b503adc051ae3b1b6833b47` | `58d88bd96fd39ea0b5d907ebfec30cd8b76ab92d` | `aede1d9fbbded5b0145e4ff48ff6d7d19d2bdbb40959dfe73c61fa5bb8cf8de9` |
| R1 | `11dae330dbbbbf3a8108b19e7f9b9020326f0a20` | `e43bb78acbfbe1e616cc320cf9ff6b55b2144286` | `3d9bf25a9dbab869ce16e9f6df213f01f139925b` | `46c4ac25d068e84221913f5f3cd1fa7f5f7f3452e6ea44ac01f77bdbdd0a41de` |
| R2 | `3505885c69737064ba5bd59a5aa8c96e5309d15d` | `11dae330dbbbbf3a8108b19e7f9b9020326f0a20` | `dd2ff040814af2fc21d01655abbd4d35ca2e31e9` | `5ed2ebddc5f0724fd63849cdac9b3155f039192f2002b895d73d0c7cee665393` |
| F1 | `0041d1cd4dafb3e1a1e2aced97c6a4988db72fc8` | `3505885c69737064ba5bd59a5aa8c96e5309d15d` | `a9395474954743ab88e3604156cdc2a812f99258` | `59d576b0bbc48fcef17d601acdd53c2205df8f775d5f58a01f5d7314ac8d27fe` |
| Q0 | `ec8c419aa678bca1ab7468c1ae96b822256b53f0` | `0041d1cd4dafb3e1a1e2aced97c6a4988db72fc8` | `b63410ff78b877bde0ce3619feb1d5039500d244` | `4bd58902c18c1d55da78c5855d23d6c20dedcc84ecf465af56e96fac748f88a5` |
| Q0R2 | `278391ab47c78508af211978045ad5573d6fc135` | `5372da07829c92d73b530855fded95de4f57059b` | `4ef40aa98ddb3ddd1020c2a8ffc7b0dee0752edf` | `3db80f559e474687f70cdc7aa0eeb0487c4b12fb7053444cfcc35a5d1252f4be` |
| Q1 | `d10083908a64a0d05a0cbbb1a6eddc2055b2f710` | `0fcde9458074cd1d32591c0b36f3e1289275565e` | `4f44fa0165292a442ec9e5b3ddafb2335e3ae36c` | `db890b3d5d845773e4f3aff76b93ed2fe3e239fdf8c376e5b750fbf6e3e44ecd` |
| Q2 | `49f9225517d76582500bcf80bf7fc6a5d112f137` | `d10083908a64a0d05a0cbbb1a6eddc2055b2f710` | `71ae16d237211f3852782765b34de8c9fadefc5d` | `5b8fd3a35a2e80196ecda195a400cf90b4c6e7aa70a2a09f976c09dad840c131` |
| Q3 | `fe1678d286fc710202265d145ed58940b1a373f7` | `49f9225517d76582500bcf80bf7fc6a5d112f137` | `11658cd121bcc074f4f1436efe25e32ab3eb239e` | `bf614d1b3b579c3b8e689f5003bd2437df9478c7eb202ab013c1453e9ef75672` |
| Soak | `63e44341bd1da982f00a9db7de2be7e779c7a63d` | `fe1678d286fc710202265d145ed58940b1a373f7` | `af8803eabaa89f56c5d53afbf60bc201187d5556` | `8c86853f6676fa03da317d0211e27f747299f9df851530f68b0c02ae5034cad3` |
| Q4 | `3615a8c9b2cc6d473df6fb1348fd09c47a883426` | `63e44341bd1da982f00a9db7de2be7e779c7a63d` | `4492bc796707f24080a9ae799023b9d73b15b3c4` | `304b0a0b1649983fefda9bbc3d50528eb75c40644c5e4d98d61661c851a30281` |

For every row, the receipt blob at its milestone commit is byte-identical to the same path in the Q4 tree. The docs and result copies of the Q4 canonical receipt are also byte-identical.

## Q4 evidence and DEVELOPMENT_ONLY gates

1. **Real runnable path — pass.** The four shared-pool proposals are four successful `gpt-5.4` Provider calls with zero retry. Each of the three selected arms has one independent successful implementation call with zero retry. All seven broker rows have `outcome_json IS NULL`. STATIC and CURRENT_F1 materialize recommendation-specific models, pass actual construction/API/behavior qualification, pass the legal resource probe, and complete matched 100-epoch BPR-parent and candidate training. The one-epoch qualifier is only an executability gate and is not used as effect evidence.
2. **End-to-end validity — pass with one disclosed missing arm.** STATIC and CURRENT_F1 each complete a full development Episode. OUTCOME_AWARE independently materializes but fails construction with `KeyError`/`INTERFACE`; resource, mechanism, matched training, and Episode are therefore legally deferred or missing. That negative evidence is retained and missingness is not converted to zero.
3. **Serves open recommender research — pass for pilot closure.** The accepted comparators select from one frozen four-spec open-research pool. STATIC uses canonical opaque-spec order, CURRENT_F1 uses the accepted F1 direction ordering, and OUTCOME_AWARE uses the accepted Q3 acquisition policy with fixed 15% exploration and seed `56331`. Selection is pre-outcome. The two executable candidates exhibit non-parent behavioral deltas and recommendation-model mechanisms; they are not static candidates, hyperparameter-only substitutions, or inert wrappers.
4. **No prohibited substitute — pass for pilot closure.** The audited execution uses no 66-item preimplementation inventory, candidate-specific repair, manual patch, fallback-to-parent result, mock result, hidden missingness, success picking, retry, cross-policy outcome reuse, or held-out access. Qualifier smoke execution is not promoted to experiment evidence. Full effect values come only from the matched 100-epoch development runs.

The frozen pool has 4/4 valid specs. STATIC and CURRENT_F1 select `38a26af380e16a8ef7e2ada045f75be761e0ff38eda747f1c2e83a830db12b64`; OUTCOME_AWARE selects `146af84a1c273a5df82f1f5d3141f14874272367121c14344b5a9ec1d560609a`.

The old seed-`54302` STATIC and CURRENT_F1 resource-prefix launches remain present as two `INVALID_PROTOCOL_BINDING` launches, with zero training batches, zero effect authority, and zero mechanism authority. Their original manifests, resource receipts, and start confirmations remain sealed. Fairness v2 changes only the resource-prefix seed to the accepted Q0R2 seed `54102` while byte-binding the original selection, resolver, implementer, and qualifier. Because no valid training batch or authority existed under the invalid binding and the candidate was neither regenerated nor retried, `candidate_retry_count = 0` is correct.

The physical ledger closes at 10 GPU launches: two qualifiers, two invalid protocol-bound resource starts, two legal resource probes, and four matched training runs. All recorded matched runs are explicitly serial, with maximum concurrency reported as one; held-out reads and stage/candidate retries are zero.

## Development results and interpretation

| Policy | Selected / parent NDCG@10 | Best parent-relative development effect | Mechanism | Cost per informative Episode | Full Episode completion |
|---|---:|---:|---|---:|---:|
| STATIC | `0.2004 / 0.2068` | `-0.0064` | `NON_IDENTIFIABLE` | `622143 ms` | `1.0` |
| CURRENT_F1 | `0.1996 / 0.2068` | `-0.0072` | `NON_IDENTIFIABLE` | `693288 ms` | `1.0` |
| OUTCOME_AWARE | missing | missing, not zero | `NOT_ASSESSED` | `INF_UNDEFINED_NO_SMOOTHING` | `0.0` |

These are single-seed, parent-relative development observations. Neither executable Episode has a parent-equivalent mechanism-off execution, so both mechanism states are correctly `NON_IDENTIFIABLE`. OUTCOME_AWARE has no effect or mechanism authority.

## Findings

- **F-01 — Medium, claim-limiting, does not invalidate Q4 pilot closure.** STATIC and CURRENT_F1 select the same frozen research spec but intentionally use independent Provider implementations, producing different candidate source trees. Their `-0.0064` versus `-0.0072` difference therefore mixes implementation-realization variance with policy execution and cannot be interpreted as a causal selection-policy difference. Q4 makes no such superiority claim; any later comparative claim needs a frozen common implementation or an implementation-randomness design.
- **F-02 — Low, auditability.** The ten-launch total is compositionally supported by two qualification receipts, two preserved invalid start confirmations, two legal resource worker results, and four matched worker results. Serial fields support the reported maximum concurrency of one, but not every launch class has timestamped start/end intervals; concurrency one cannot be independently reconstructed from timestamps alone.
- **F-03 — Low, portability.** Several sealed receipts retain historical absolute source/NAS paths. All referenced bytes are present in the relocated package and the relative `SHA256SUMS` verifies, but consumers should resolve package-relative paths rather than assume historical absolute paths remain live.
- **F-04 — Coverage limitation, not an implementation defect.** The sealed pre-finalization audit records a broader remote run with 11 passes and 6 permission failures against an inaccessible historical R1/R2 path. This audit did not install dependencies, probe a GPU, or repeat full training. Fresh local contract tests and byte verification passed; remote environment reachability and training replay remain unverified.

No finding justifies altering sealed Q4 evidence. No hidden negative evidence, outcome leakage, candidate repair, retry, fallback/mock result, or held-out use was found.

## Fresh verification

- `.../python -m pytest -q test_q4_final_independent_audit.py` — **3 passed**; recomputed all 182 package entries and checked Provider, seed-correction, launch-accounting, missingness, and authority facts.
- `.../python -m pytest -q test_prospective_policy_comparison.py test_open_meta_vnext.py test_open_meta_f1.py test_open_meta_q3_projection.py` — **26 passed**.
- First adjacent invocation without `PYTHONPATH=src` — **collection failed** for two modules with `ModuleNotFoundError: recclaw_core`; retained as an environment invocation failure.
- Corrected `env PYTHONPATH=src .../python -m pytest -q test_resource_scheduling.py test_idea_quality.py test_mechanism_characterization.py test_multiround_soak_contract.py` — **38 passed, 6 dependency deprecation warnings**.
- First `sha256sum --quiet -c` invocation from the repository root — **failed to open all 182 relative entries**; retained as a wrong-working-directory invocation failure.
- Corrected invocation from `results/research_line/q4_prospective_policy_comparison_20260803_01` — **182/182 passed**.
- Seven broker databases — **7 rows, 7 SUCCESS, 7 null outcomes, model set `{gpt-5.4}`**.
- A parallel WSL-native receipt-blob query returned four results and timed out on seven concurrent WSL calls; one serial batched native-Git query then completed the remaining comparisons successfully.

Key Q4 byte identities: canonical receipt `304b0a0b1649983fefda9bbc3d50528eb75c40644c5e4d98d61661c851a30281`; physical receipt `5b0e6914d17960ffec025a65df6aaeb445f53ab7e3878616a60f0080891d6100`; result package `c7d246a88e3857fedbade67887fcdde431497c4dc960913fe2c686f240492f8b`; fairness-v2 contract `76e8ee755b457cc8b0d93af33e9b2d67fe2793100b22393378f665dd26dfce29`; verification audit `3909b7df70c99536b914900eda551b59e208272ca6d5f27063bb71a7edeb31b3`; `SHA256SUMS` file `1d1f053b5561f2e9eab9f7997530d2e85f303a75d05129217205becbdce686bf`.

## Permitted claim

Q4 may claim only that a real, outcome-blind, shared-pool, three-policy **DEVELOPMENT_ONLY prospective pilot** executed under the sealed budget and information boundary; two policies completed negative parent-relative development Episodes, while OUTCOME_AWARE produced a preserved interface failure and a missing Episode. It may not claim OUTCOME_AWARE superiority, policy ranking from the observed effects, mechanism identification, held-out generalization, or a formal scientific result.
