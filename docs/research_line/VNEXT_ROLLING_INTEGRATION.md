# Research Line vNext Rolling Integration

## Frozen identity

- Frozen v1: `d2817397d0c71d5122f55e81a2c60c72b42eaf4b`
- RC0: `4306d9ba13157d3c9ed6b877d33558ba9a40c449`
- Integration branch: `feat/research-line-vnext-integration`
- Integration worktree: `/root/projects/RecClaw_research_line_vnext_integration`

The integration branch was created from RC0 with WSL-native Git. Existing
worktrees, including Strategic Reset, are not integration inputs.

## Accepted intake

The following BC commits are an exact single-parent chain rooted at RC0 and
were integrated without rewriting their identities:

1. `a581b33884c73837dcf307e7d6b7ab3789bfee6a`
2. `45a97f5082c1aebb0df0d74b9c8d812d27377157`
3. `8580a41b357a96ff1d34ac44cd57a8e77c99ad3d`
4. `824643c4155b85d7e12dffaf77827a0eda50aadf`

No later BC commit and no A0 or D0 commit is accepted in this integration
state.

## RC0 to RC1 state

| Boundary | State | Evidence or blocker |
|---|---|---|
| RC0 nine-contract schema | Integrated | RC0 contract tests and public exports |
| Shared origin-blind package boundary | Integrated | Accepted BC package tests |
| Mechanical RecBole qualification | Integrated | Positive one-epoch and negative interface fixtures |
| Capability admission and registry | Integrated | Accepted deterministic admission tests |
| G local orchestration | Integrated | Direct response to package, qualification, admission, and registry; no Provider or outcome |
| A0 OpenSpec and Resolver runtime | Waiting | No accepted A0 implementation commit |
| BC Next Fresh Profile | Waiting | No accepted Next Profile commit in the authorized intake set |
| D0 TypedResearchEpisode runtime | Waiting | No accepted D0 implementation commit |
| RC1 field freeze | Blocked | Requires the accepted vertical slice through deterministic Next Fresh Profile |
| R1/R2 launchers | Blocked | Wave gates and owner runtimes above are incomplete |

The local orchestration boundary intentionally ends at
`VersionedCapabilityRegistry`. It does not build a profile, mutate the current
campaign, call a Provider, consume an outcome, or create a scientific episode.

## Commit intake checklist

1. Record the supervisor-accepted commit SHA and owner lane.
2. Verify the commit exists and is a descendant of RC0.
3. Verify the supplied range is a single-parent chain with the declared base.
4. Review the exact file surface against owner boundaries.
5. Reject Strategic Reset, old V wrappers, attempt families, ledger/database/
   authority/state additions, compatibility branches, silent catalog fallback,
   Provider/GPU/held-out calls, outcomes, and sealed-output changes.
6. Run the owner-targeted tests and structural lint at the exact accepted tip
   in a clean worktree.
7. Integrate without rebasing or squashing. Use fast-forward only when the
   accepted chain directly extends G; otherwise merge the accepted lineage so
   the owner commit identities remain reachable unchanged.
8. Resolve only G-owned conflicts locally. Return owner-internal conflicts as
   a minimal correction request.
9. Run focused integration tests, then the relevant Research Line regression
   set and structural lint.
10. Review the final diff and history, create a clean local G commit, and
    record accepted commits, checks, blockers, and the next intake condition.

## Conflict ownership

| Surface | Owner |
|---|---|
| OpenSpec production, eligibility, Resolver rules and reason codes | A0 |
| Implementer, package, RecBole adapter, Qualifier, admission, registry, Next Profile internals | BC |
| Episode closure, failure/evidence classification and memory authority | D0 |
| Public exports, shared schema migration, old-entry migration, orchestration, launchers and end-to-end wiring | G |

If runtime evidence requires an RC0 semantic or representation change, the
originating owner supplies the smallest reproducer and proposed correction.
G applies the shared migration once, records it as RC1, and updates all
integrated callers. G does not copy owner logic to bypass a conflict.
