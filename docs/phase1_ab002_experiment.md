# Phase-1 AB-002 experiment

## Question

Under one fixed ML-1M protocol, does the frozen Evidence Guard improve the quality of experiment feedback and the admissible search curve of Original RecClaw without excessive false blocking or cost?

This is a development comparison, not a formal scientific or permission gate.

## Frozen design

- Control source: commit `0b44db72f2e44bfbf8139b43c9624e1e89f52b35` plus the common required search-seed harness.
- Treatment source: the same Control source plus `phase1/overlays/treatment_agent.patch` and the five frozen Guard files.
- Environment: ML-1M, BPR/LightGCN candidate families, RecBole 1.2.1 semantics recorded in the fixed contract.
- Protocol: random per-user 0.8/0.1/0.1 split, one uniform training negative, full-sort evaluation, NDCG@10 primary.
- Search repetitions: seeds 42, 43, and 44.
- Training seed: 2026; validation seeds 2026, 2027, and 2028.
- Budget: scheduling rounds 1 through 20 for every repetition.
- S0: fresh and empty. No historical memory, proposals, result rows, search tree, reflection records, or history-best value may be seeded.
- Execution: matched pairs run contemporaneously with rotating GPU assignment after Canary authorization.

Historical v4m metrics are reference lines only. They are not the contemporaneous Control and do not establish a causal effect.

## Common paired broker

Both arms use a local OpenAI-compatible broker path:

```text
/v1/pairs/{PAIR_ID}/{control|treatment}/chat/completions
```

The broker stores requests/responses in an external SQLite ledger. It stores neither the client bearer token nor the upstream API key. Identical canonical requests at the same pair sequence share one exact upstream response. Different requests receive separate upstream calls and an explicit divergence classification.

## Outcomes

Primary outcomes are final best admissible NDCG@10, the admissible running-best curve, discrete curve area, budget/time to reference thresholds, conversion to valid/informative results, and across-seed stability.

Mechanism outcomes include revision/branch behavior, quarantined feedback, protocol mismatch, blocker classification, independently adjudicated valid-action false blocks, Guard latency, and broker request divergence.

The Neutral Outcome Auditor applies the frozen policy and can return `POSITIVE`, `NEUTRAL_USEFUL`, `NEGATIVE`, or `INCONCLUSIVE`. It must return `INCONCLUSIVE` when independent false-block adjudication or all three matched pairs are missing.

## Stop boundary

This delivery stops before Canary. The inspected gpu5 Python 3.8.10 environment is incompatible with the Python >=3.11 frozen integration. Before Canary, an external owner must materialize and verify an eligible environment, recheck dataset and baseline bytes, GPU/storage state, package digests, broker connectivity, and non-empty secrets without printing values, then explicitly authorize the start.
