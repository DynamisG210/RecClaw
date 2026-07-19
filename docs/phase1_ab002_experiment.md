# Phase-1 AB-002 experiment

## Scientific question

Starting from the same pre-search Original RecClaw state, under the same ML-1M protocol, model, candidate space, runner, and total research budget, does the frozen development Evidence Guard improve the quality and performance of the full iterative search trajectory?

This is a paired development A/B. It is not formal claim acceptance, permission release, or production promotion.

## Frozen design

- S0: `AB002-RECONSTRUCTED-CLEAN-START-S0-V1`.
- Control: the reconstructed S0 with no Guard import, contract, state, feedback, or output.
- Treatment: the same S0 plus the digest-bound frozen Guard files and one treatment overlay.
- Environment: ML-1M, BPR/LightGCN, RecBole 1.2.1.
- Protocol: random per-user 0.8/0.1/0.1 split, uniform one-negative training sampling, full-sort validation/test, NDCG@10 primary.
- Agent LLM: `gpt-5.4`, temperature 0.2, timeout 300 seconds, retries 3; no reasoning-effort or service-tier field is added.
- Canary: paired seed 9001, three rounds per arm, at most six candidate executions per arm with one implementation attempt per round, excluded from final outcomes.
- Full A/B: paired search seeds 42–47, 20 scheduling rounds and at most 20 candidate-training attempts per arm and repetition.
- Training: ordinary candidates use seed 2026; selected candidates and the fixed LightGCN comparator use 2026, 2027, and 2028.

The exact machine contract is [experiment_contract.json](../configs/phase1/ab002/experiment_contract.json); [experiment_contract.md](../configs/phase1/ab002/experiment_contract.md) is its concise human projection. Endpoints and decision thresholds are frozen before Canary.

## S0 boundary

No single historical commit contains both the final valid common runner and a clean pre-first-iteration library. S0 is therefore explicitly reconstructed:

- common runner logic comes from the clean Phase-1 source baseline;
- action space, registry, candidate configs, and local implementations come from pre-search commit `b3dc5a7e33fd1cbec60b5dfb9ae7097c67d0b5e8`;
- post-start candidate family names and historical performance thresholds are removed from the three agent-readable common files;
- candidate execution-budget instrumentation is identical in both arms;
- dynamic memory, proposals, result rows, search trees, and reflection artifacts are absent.

Exact bytes and provenance are in [initial_state_manifest.json](../phase1/s0/ab002/initial_state_manifest.json). The bounded known-leakage result is in [historical_leakage_audit.json](../phase1/s0/ab002/historical_leakage_audit.json). Historical best values remain reporting-only and are never copied into either arm or the runtime Guard contract.

## Common paired LLM broker

Both arms use `/v1/pairs/AB002-SEED-{seed}/{control|treatment}/chat/completions`. Pair IDs are seed-bound, so cache cannot cross search seeds. Identical canonical requests at the same per-arm sequence reuse one exact response; divergent requests receive independent upstream calls.

The external SQLite ledger stores the canonical request, response bytes, provider request ID, returned model, usage, latency, status, and bounded error diagnostics. It stores neither bearer token nor upstream API key. Each arm has the same maximum request and requested-output-token envelope. Guard feedback does not schedule extra LLM calls.

## Neutral outcome boundary

The online Treatment Guard affects the search loop but is not the final judge. The frozen neutral raw-run auditor:

- rejects arm-bearing input envelopes;
- reads no Guard classification;
- checks dataset binding, split, sampling, full-sort candidate universe, seed, metric, RecBole configuration, log/result identity, and candidate artifacts;
- returns only development performance-analysis eligibility.

The primary endpoint is the selected new protocol-valid candidate's three-seed mean NDCG@10 minus the three-seed LightGCN comparator mean, paired by search seed. If an arm discovers no eligible new candidate, its frontier is the fixed comparator. Missing or malformed confirmation evidence remains inconclusive rather than being silently repaired.

Final analysis reports all six pairs, paired mean and median, deterministic paired bootstrap interval, exact sign-flip result, individual and aggregate curves, intention-to-treat outcomes, Guard-applied secondary analysis, and all failures/fail-open events.

## Gate boundary

Canary execution may start only after the preflight record proves exact source/S0/Guard/environment/dataset/shared-state identities and non-empty secrets without recording secret values. On a host without Git, the source proof additionally requires an external release-identity sidecar that binds commit/tree/tag, the retained release archive, `SOURCE_SHA256SUMS`, and the complete clean-tree manifest. The implementation agent may evaluate the mechanical Canary criteria and assemble a review packet, but it cannot approve Canary GO for the six-pair Full A/B.
