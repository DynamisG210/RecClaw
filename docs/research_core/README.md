# RecClaw Research Core v0.1

Research Core v0.1 is a slim RecClaw-native Candidate Foundry foundation.

It demonstrates typed memory ingestion, deterministic memory retrieval, candidate-card
validation, policy ranking, golden fixture replay, and claim-boundary verification.
It does not claim recommender metric improvement, search-quality improvement,
formal success, autonomous discovery, M5 readiness, or runtime-kernel readiness.

Run the golden fixture without GPU, LLM calls, RecBole, or formal result writes:

```bash
python -m recclaw_core.foundry \
  --fixture artifacts/research_core/golden_foundry_fixture \
  --output tmp/research_core_demo
```

The replay ingests the BPR rank-cut diagnostic negative memory, retrieves it for
a future BPR rank-surrogate query, deterministically downgrades a repeat
batch-local rank-cut candidate, and writes a selected candidate queue plus a
claim-boundary report.
