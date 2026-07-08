# Research Core v0.1

Research Core v0.1 adds a functional RecClaw-native Candidate Foundry skeleton
with typed memory ingestion, retrieval, deterministic policy ranking, and
fixture replay.

Run the fixture replay with:

```bash
python -m recclaw_core.foundry \
  --fixture tests/fixtures/research_core/golden_foundry_fixture \
  --output /tmp/recclaw_research_core_demo
```
