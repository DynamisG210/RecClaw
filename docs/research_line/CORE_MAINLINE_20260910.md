# Core mainline source snapshot — 2026-09-10

This branch publishes the current native Research Line source for independent
review and continued validation. It is an engineering delivery, **not** a claim
that final research quality, statistical superiority, or cross-system dominance
has been established. It is not merged into `main`.

## Source and scope

### Follow-up: prepared proposal identity

The first correction at `dee56e9b` was **not end-to-end complete**: actual
replica2 R2 training succeeded, but execution-to-lineage feedback still failed.
The follow-up uses the paired qualified candidate's bound parent and execution
contract at both new handoff and saved-prepared selection, while preserving all
other research intent. The checkpoint reader reuses the writer's existing
same-slate monotonic progression check. `execution.py` lineage checks remain
unchanged and still reject actual hypothesis drift.

Latest complete 491-file source-map SHA256:
`b36f37053f259932b276191d59abd302253c1217de15941a99b5a3be2db29d60`.
Delta from `dee56e9b` SHA256:
`83756321a95b2d68fd5652e79b2247cf12adf498bcc280a1afa6220a22b2e60d`.
Full correction from the initial snapshot SHA256:
`fcb234e148b002606014a10cfa981caab7033def660e1f37b84cb7ed31bcd343`.

The core owner's offline native-entry replay of the actual stopped R2 reused
the saved full metric `0.0613`, train seed `54201` and `71518 ms`, preserved
physical/prepared bytes, persisted feedback and R2 trace, advanced state to R3,
and reached the next native research boundary with the metric in context.
Network, Provider and training calls were disabled. Real hypothesis drift was
rejected and already-bound outcomes were unchanged; the original eight
regression checks also passed. This closes the reproduced offline path, not
yet an assertion of successful live continuation or improved search quality.
The full result remains negative relative to the root; neither failure nor its
cost is erased. The earlier stage and its narrower tests are retained below.

The initial publication at `28331b10` matches the snapshot described below.
A subsequent two-file correction in `runtime.py` and `campaign.py` separates
the frozen research proposal identity from the adapter-bound implementation
identity. It restores candidate rerouting, attempt accounting and same-round
checkpoint progression after parent-label binding; it does not change the
initial Implementer request, models, prompts or training configuration.

Corrected 491-file experimental source-map SHA256:
`480574291027a6e4cf8648d4db728575010031589f7a485785ce1ff396713128`.
Core-owner patch SHA256:
`9b51336cdb993d42a9cb602ec583eb221a6d024adb0f5567414c42445fc2eef6`.
The owner's offline replay of the saved replica2 R2 and native inputs changed
from 3 failed / 5 passed to 8 passed, including remaining-candidate selection,
preserved prior attempts, idempotent checkpoint persistence and rejection of
real drift. The first Implementer request was identical; the corrected replay
visited four distinct proposals instead of repeating one. Private checkpoint
fixtures are not published. Lightweight artifact-free accounting tests are
included separately: all three passed with the project's Python 3.10 Linux
runtime (`python -m pytest -q tests/test_prepared_proposal_identity.py`).
Windows import still lacks `fcntl`; the server's system Python 3.8 is too old
for the project's `TypeAlias` import and lacks pytest. Neither was used to
claim runtime validation.

These are regression checks, not improved recommendation metrics. The original
fixed-source stability failure and real implementation failures remain evidence.
Actual replica2 recovery belongs to its experiment owner and must retain the
source transition and all prior costs; healthy runs are not hot-modified by this
Git publication. Whole-project integration and final quality remain unverified.

### Initial snapshot

The publication starts from Git commit `abc91650` (the existing
`feat/research_line_helix_stability` branch). That commit is the publication base,
not a claim about the exact Git ancestry of the experiment snapshot.

The source of truth is the delivered `RecClaw_p1_final_b_20260910` snapshot:

- Archive SHA256: `11c907f7125226bece8f38ad1b1a7d78ad91eede5e2aeb8f559ca5d2f77c6849`.
- Complete 491-file experimental source-map SHA256:
  `5db7a06560d4f1e98091d7004b52f087dc7f9b4f72e9af23c708e3a72cda6bfa`.
- All 317 delivered `src/` files are included, together with the native entry,
  worker and candidate entry points, associated model code, dependency list,
  adapters and reusable input templates. The 491-file digest does **not** describe
  the whole Git tree: machine launchers, experiment receipts, run records and
  private input bundles are not newly published here.
- The exact `src/` mapping SHA256 is
  `9c6dd09b4692a7fbed2ec3887cf8bbadbec9f026661fc2c0835361f7e696b576`
  (SHA256 of the UTF-8 JSON path-to-SHA256 map, sorted keys and compact separators).
- Unrelated pre-existing `scripts/agent.py` and reflection-pilot hardening on the
  Git branch are preserved; this publication does not replace those with older
  copies from the experiment bundle.

The core retains four-role research, shared opportunity selection, executable
parent context, faithful implementation and measured outcome/cost feedback.
It includes the already-delivered hook-parent-context, custom-marker, research
window, operational-definition, text-normalization and candidate-provenance
changes. BL primary-objective ownership is an associated adapter responsibility,
not a universal research-policy improvement.

Search-space modules and E1 model/adapter code are dependencies supplied with the
snapshot, not evidence that all separately managed P4–P7 overlays are synchronized
to this branch. The newer E1 `director_sequential`/on-demand experimental strategy
is not imported into this native four-role release. Historical source resources
and legacy machine-specific defaults remain byte-preserved; they are not a
portable launch configuration or authorization to access those paths.

## Running and validation

The native entry is `scripts/run_research_line_standalone.py`. Use Linux and the
project's compatible Python/RecBole environment; the native runtime imports
`fcntl` and is not a Windows-native training entry. Inspect `--help` in that
environment. Configure `RECCLAW_PROJECTS_ROOT`, `RECCLAW_SEARCH_DATA_ROOT`,
`RECCLAW_RECBOLE_ROOT` and `RECCLAW_PYTHON_EXECUTABLE` for your own machine before
formal execution, and supply the intended root/prior, fixed data/evaluator,
provider configuration and isolated writable output paths. Credentials and
experimental root/prior bundles are intentionally not included.

The current validation protocol uses two independent fresh native B campaigns
of 15 rounds, the original full training recipe, train seed 54201, search seed
54202, and the established Terra/Luna medium role routing. Search seed does not
control remote LLM sampling. Native per-attempt limits remain; qualification,
probe, full training, revisions and real failures all count toward reported
cost. No heldout feedback or cross-replica candidate/cache transfer is allowed.
This describes the study boundary, not an instruction to duplicate running jobs.

The delivered runtime passed 10 targeted CPU parent/objective/value-gradient
and adjacent-hook checks in its original Linux environment. Publication review
also checks staged source identity and syntax. A Windows run of the existing
package tests initially yielded 3 passed / 1 failed because creating a symlink
requires a privilege absent on the host; native CLI import also requires Linux
`fcntl`. Neither environmental failure is relabeled as a successful test.
Four delivered source files retain extra blank lines at EOF; these are known
whitespace-only findings preserved for source identity, not behavioral fixes.

Publication checks subsequently passed the native CLI `--help` import and all
four existing package tests on Linux. The Linux archive environment lacks Git
and its default NFS pytest temporary directory has an ownership mismatch; a new
isolated `--basetemp` resolved the package-test setup error. The original-source
Git test is instead run in the complete local Git worktree. The repository's
old M0 contract fixture was aligned with the delivered experiment-contract JSON
and its current BL-ICF resource identity; no runtime or experimental source was
changed to satisfy the test.

Final acceptance must use actual multi-round useful candidates, faithful
execution, feedback-driven subsequent choices, stability and complete costs.
C's unproven incremental benefit is not a prerequisite for accepting a strong B.
Do not infer final acceptance or a safe whole-project `main` merge from this
snapshot or targeted checks alone.
