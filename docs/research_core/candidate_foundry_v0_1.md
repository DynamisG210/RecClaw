# Candidate Foundry v0.1

Candidate Foundry v0.1 turns a controlled mechanism-negative result into
reusable typed memory and shows that memory can steer a future candidate queue.

Golden replay steps:

1. Read Phase 9.5-style structured fixture artifacts.
2. Ingest `mem_bpr_rankcut_activation_metric_mismatch`.
3. Retrieve that memory for a future BPR rank-surrogate query.
4. Validate candidate cards.
5. Rank candidates with deterministic policy.
6. Downgrade repeat batch-local rank-cut candidates.
7. Write selected candidate queue and claim-boundary report.

The replay intentionally avoids LLM calls, RecBole, metric smoke, formal
evaluation, registry promotion, and `results/results.csv`.
