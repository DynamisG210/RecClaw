# Research Core v3 Evidence Archive Index

This archive index is intentionally small. It does not move the full
`RecClaw_Architecture/` evidence zoo into runtime and does not make old stage
builders part of the RecClaw-native Candidate Foundry v0.1 surface.

Archive here or in a separate evidence PR:

- one-off P0/P1/P2/M1.5/M1.5-bis builders
- Stage 2b/2c/2d/2e/2f calibration scripts and raw predictions
- blind review packets and prompt packs
- paired old-loop proxy artifacts
- raw model outputs
- repeated architecture verifiers used only for evidence adjudication

Keep in mainline:

- `recclaw_core/`
- `configs/research_core/`
- `docs/research_core/`
- `artifacts/research_core/golden_foundry_fixture/`
- `artifacts/research_core/bpr_rankcut_diagnostic_memory.yaml`
- focused tests for the slim Candidate Foundry v0.1 replay

The supported scope remains `bounded_candidate_foundry_milestone`.
Search-quality, metric-improvement, formal-success, M5, and runtime-kernel
claims remain unsupported.
