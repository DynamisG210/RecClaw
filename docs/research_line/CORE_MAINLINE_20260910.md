# Core mainline source snapshot — 2026-09-10

This branch publishes the current native Research Line source for independent
review and continued validation. It is an engineering delivery, **not** a claim
that final research quality, statistical superiority, or cross-system dominance
has been established. It is not merged into `main`.

## Source and scope

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
