# Non-Meta Research Line Readiness Audit V1

Date: 2026-07-26 (Asia/Shanghai)

Scope: Research Capability Line excluding Meta learning. This is a local
engineering and experiment-readiness audit. It is `DEVELOPMENT_ONLY`, has
`authority=NONE`, and is not an independent acceptance or a scientific result.

## Verdict

The non-Meta execution chain is functional after the fixes in this successor
release, but the current four-recipe Pilot search profile is not sufficient for
the planned formal Main campaign.

```yaml
engineering_chain: PASS_AFTER_FIXES
research_evidence_boundary: PASS
three_arm_budget_and_substrate_alignment: PASS
fresh_pilot_preparation: CONDITIONAL
formal_main_readiness: BLOCKED
expected_effect_direction: PLAUSIBLY_POSITIVE_BUT_NOT_ESTABLISHED
```

## Runtime path reviewed

```text
role-scoped Producers
  -> typed CandidateProposalV2
  -> BL-ICF compile and executable-semantic deduplication
  -> Search-Utility hard gates and deterministic ranking
  -> CommonExecutionGuardV1 on the complete frozen pool
  -> ranked same-slate Helix selection
  -> NullEvidencePortV1 (A/B) or EvidenceGuardPortV1 (C)
  -> package-owned materialization and training
  -> raw result closure and four-axis resource accounting
  -> compact Search-only feedback and mechanism belief
```

## Necessary fixes completed

1. **Router decision is now executable.** The Router previously recorded its
   highest-scoring candidate but passed the slate to Helix in Producer order.
   `NullEvidencePortV1` therefore executed the first allowed Producer item
   instead of the Router winner. The route now exposes a closed
   `ranked_candidate_ids` sequence; Helix receives that exact order, and a C-arm
   PRE block advances through the same ranked slate.
2. **Semantic duplicates no longer preserve Producer-order bias.** Hard-gate
   eligible proposals are grouped by BL-ICF executable semantics, the
   highest-scoring representative survives, and the remaining duplicates are
   recorded as `SEMANTIC_DUPLICATE`.
3. **The package-level Research controller export is no longer the M0 fixture
   seam.** It now resolves to the runtime `research_controller.py`
   implementation.
4. **Search Memory now carries mechanism information.** The prompt projection
   includes the selected proposal's executable mechanism axis, intent,
   competing explanations, confounds, next discriminative test, compact
   outcome feedback, and the immutable snapshot digest. It remains
   Search-Utility-only and receives no full Guard event.
5. **Pilot claim/execution alignment is closed.** The Pilot response schema now
   admits only four exact executable field combinations. Unsupported free-text
   claims such as residual propagation, degree tempering, a new sampler, or a
   new loss cannot alter the scientific projection of the compiled program.
   The raw response remains in the private Broker record for audit, while the
   executable program states only the mechanism actually run.

## Evidence from the sealed Static Diagnostic V2

The V2 outcome remains sealed and is not reinterpreted as a new experiment.
Its already-recorded Producer responses were replayed locally without Broker or
training calls to inspect the old routing path.

- Four Research Producers generated only two unique executable semantics:
  LightGCN and SGL (`semantic uniqueness = 0.50`).
- Under the Router policy, the intended winner was the
  `falsification_designer` LightGCN proposal with score `0.6712`.
- The old orchestration executed the first allowed SGL proposal instead.
- Therefore the observed A/B/C equality in that one-round diagnostic was
  partly caused by the now-fixed Router-to-Helix ordering defect. It still does
  not establish that LightGCN would outperform SGL or that B would outperform
  A.

## Alignment with the intended Research Capability Line

| Requirement | Status | Finding |
|---|---|---|
| Search Utility only | aligned | Runnable probability, useful signal, frontier potential, information gain, cost, and blocker risk remain the Router inputs. No Claim Ceiling, Evidence Admission, Protocol Branch, or accepted-evidence writer was added. |
| Research/Evidence separation | aligned | Research source has no direct Evidence Guard or Fusion dependency. Full Guard events remain outside Search Memory; only deterministic compact feedback crosses the bridge. |
| Same BL-ICF and CommonExecutionGuard in A/B/C | aligned | The three Arms use the same executable profile, common mechanical gate, training release, protocol, and resource ceilings. |
| One SearchRound, at most one training execution | aligned | Producer and Router activity remains inside the round. Same-slate PRE fallback creates no extra proposal session or execution opportunity. |
| Mechanism-oriented Producers | partially aligned | Role prompts are distinct and mechanism-oriented, but the Pilot runtime can execute only four anchor-equivalent recipes. Role differentiation therefore has little room to become executable differentiation. |
| Falsification and controls | not yet runtime-complete | A falsification slot exists, but live proposals do not yet carry a closed same-parent contrast. `ControlAblationBuilderV1` and `RepairEngineerV1` currently create audit metadata only; they do not generate runnable control or repair candidates. |
| Search adaptation independent of Meta | partially aligned | The static Router and mechanism-aware compact memory now work independently. Utility estimates remain Producer self-assessments, and the current memory head is process-local rather than reconstructable after restart. |
| Research Capability Quality Gate | insufficient for Main | V1 passed on a handcrafted outcome-masked fixture. The real Static V2 pool achieved only `0.50` semantic uniqueness, below the fixture threshold of `0.75`; V1 therefore does not establish live runtime quality. |

## Remaining blockers before a formal campaign

### P1 — Executable mechanism capacity

The old Pilot mapper exposed six nominal backbone/objective combinations. The
additive claim-aligned V2 schema now admits only the four combinations that the
current runtime can execute exactly: BPR, LightGCN, NGCF, and SGL. The formal
design schedules 50 rounds per Arm-search-seed with a fixed ordinary training
seed. Repeating these four programs cannot support a meaningful mechanism-search
frontier or the intended RecClaw research highlight.

Before Main, a new content-bound executable release must add genuine
mechanism-level interventions whose program semantics change the actual
training computation. Merely changing labels, hypotheses, mechanism-axis text,
or ordinary hyperparameters is not sufficient.

### P1 — Live falsification/control closure

At least one Producer path must emit an executable same-parent control or
falsification contrast with typed parent lineage. Repair candidates must remain
non-discovery and should appear only for an observed runnable blocker. The live
quality gate must evaluate these candidates, not metadata synthesized after
generation.

### P1 — Search Memory recovery

The single-process adaptive path is functional, but its substantive compact
memory head is not reconstructed from the transactional experiment store after
a process restart. A multi-round Main run must either bind a restart-safe
Search Memory snapshot or explicitly hard-stop without behavioral resume.

## Non-blocking limitations

- Router utility is currently based on Producer self-assessment. Meta may
  calibrate it, but Meta activation must not be used to hide poor live semantic
  diversity.
- The one-round static diagnostic is an executability observation, not a
  Research Line effect estimate.
- The local review and tests below are not an independent P0/P1 audit.

## Gate required when Meta is ready

Do not start the fresh formal Pilot merely because a Meta checkpoint becomes
available. First require one outcome-blind runtime Research Capability gate
over the exact frozen Broker schema and executable release:

1. Router winner equals the first executable same-slate candidate.
2. Every scientific mechanism statement is derived from the bound executable
   program.
3. Live proposal pools meet the predeclared semantic-uniqueness threshold and
   report collision rate.
4. Runnable falsification/control lineage is present.
5. Search Memory restart behavior is frozen.
6. B/C retain identical Producer, Router, Meta checkpoint, BL-ICF, budget,
   initial state, and backend identities before Guard-derived trajectory
   differences.

Only after that gate and the Meta promotion gate both pass should a new,
unused-seed Pilot contract be frozen.

## Verification performed

- Frozen training runtime:
  `python -m unittest discover -s tests/experiments/helix_abc_v1 -p test_*.py -q`
  — 164 tests passed.
- Evidence Guard and BL-ICF boundary suites:
  `python -m unittest -q tests.evidence_guard.test_core_v1
  tests.test_bl_icf_mechanism_space` — 40 tests passed.
- Targeted Pilot/Broker regression:
  `python -m unittest -q
  tests.experiments.helix_abc_v1.test_m6f_broker_observability
  tests.experiments.helix_abc_v1.test_m6_pilot` — 25 tests passed.
- Twelve reproducible `__pycache__` directories under `src`, `tests`, and
  `scripts` were removed. Sealed Pilot outputs, SQLite/WAL audit state, frozen
  contracts, and the Master Goal identity input were preserved.

No Broker call or training execution was performed. These checks establish
local engineering consistency only; they do not estimate B-A or C-B.
