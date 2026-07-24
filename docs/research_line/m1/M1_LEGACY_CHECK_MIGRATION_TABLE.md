# M1 Legacy Check Migration Table

M1 treats the legacy files below as migration evidence only. The experimental
path has one package-owned truth: `CommonExecutionGuardV1` plus the frozen
BL-ICF executable profile.

| Legacy source | Useful mechanical behavior | M1 owner | Deliberately excluded |
|---|---|---|---|
| `configs/candidate_proposal_schema.yaml` | required fields, runner/base-model closure | BL program schema and `plan_check` | old BPR/LightGCN-only action-space truth |
| `scripts/validate_candidate_proposal.py::path_is_allowed` | normalized relative-path containment | materializer path closure | string protocol heuristics |
| `scripts/implement_candidate_proposal.py::ALLOWED_WRITE_ROOTS` | write-root allowlisting and overwrite checks | Arm-private deterministic materializer | shared `recclaw_ext/models`, registry mutation |
| `scripts/implement_candidate_proposal.py` compile/import checks | syntax/import/entrypoint availability | package-handler source identity and import attestation | caller-supplied module/callable/import path |
| `scripts/implement_candidate_proposal.py` smoke | launch failure visibility | bounded non-training interface smoke | 3-epoch training, metric stagnation |
| `scripts/agent.py` candidate health and family budget | runner/budget completeness | common budget and runner closure | utility history, family downweight/freeze |
| `scripts/agent.py` seed validation | records that extra runs consume real budget | excluded from Main SearchRound | seed-validation and multi-seed training |
| `scripts/run_candidate.py` registry/config/result checks | runner identity and result closure | V2 binding, committed claim, fake runner, close-result | bare candidate IDs, registry lookup, host dynamic import |
| BL kernel/provider | strict bytes, compile, capability closure | package-owned compiler re-run | caller-injected provider/report |

This migration does not change the legacy compatibility path. It prevents that
path from acting as a second experimental validator or runner.
