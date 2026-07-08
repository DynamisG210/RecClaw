# Mainline Extraction Plan

Keep in mainline:

- `recclaw_core/` slim Candidate Foundry runtime
- `configs/research_core/` schemas and policy weights
- `tests/test_research_core_*.py`
- `docs/research_core/`
- `artifacts/research_core/golden_foundry_fixture/`
- `artifacts/research_core/bpr_rankcut_diagnostic_memory.yaml`

Keep BPR rank-cut only as a diagnostic extension:

- local loss/model code may remain
- tests may remain
- do not promote to `candidate_registry`
- do not add to default `action_space`

Archive or keep out of runtime:

- one-off Stage/P/M builder scripts
- raw model predictions
- blind review packets
- prompt packs
- large architecture reports
- Codex execution logs

Next search-quality, formal, M5, runtime-kernel, or large-loop experiments must
be opened as separate projects.
