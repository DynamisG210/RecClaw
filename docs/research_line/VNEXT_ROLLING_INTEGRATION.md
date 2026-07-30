# Research Line vNext Rolling Integration

## Frozen identity

- Frozen v1: `d2817397d0c71d5122f55e81a2c60c72b42eaf4b`
- RC0: `4306d9ba13157d3c9ed6b877d33558ba9a40c449`
- Integration branch: `feat/research-line-vnext-integration`
- Integration worktree: `/root/projects/RecClaw_research_line_vnext_integration`
- BC Next Profile merge: `83eb440ab561ad787d5513cc6afc16507b99c3ab`
- A0 OpenSpec merge: `f06e0c868f9b5088855f01cae8bd4f33c92746a0`
- D0 Episode closure merge: `b7fc787688c528084e3f100d4000b71d00e057a2`

The integration branch was created from RC0 with WSL-native Git. Existing
worktrees, including Strategic Reset, are not integration inputs.

## Accepted intake

The following BC commits are an exact single-parent chain rooted at RC0 and
were integrated without rewriting their identities:

1. `a581b33884c73837dcf307e7d6b7ab3789bfee6a`
2. `45a97f5082c1aebb0df0d74b9c8d812d27377157`
3. `8580a41b357a96ff1d34ac44cd57a8e77c99ad3d`
4. `824643c4155b85d7e12dffaf77827a0eda50aadf`
5. `095a812c66367cf19544e8275cb0d57a5a067009`
6. `5d986d1fe53645dcef31256c8db5f0de160c486e`

The last two commits were integrated through an explicit merge because G and
BC had diverged at `824643c4155b85d7e12dffaf77827a0eda50aadf`.
Their original identities and parent chain remain reachable unchanged. No A0
or D0 logic was copied into G.

The accepted A0 lineage is:

1. `d0551ea402a41842fd6d52e71e4affd0649c3494`
2. `35cdcb55fc1bb69ff3770fdb8338462b0dce99a0`

The accepted D0 lineage is:

1. `5c6b064fa90f6d495e226831f0ec86fc9e89d391`

Both owner lineages were integrated through explicit merges from their RC0
base. Their business files remain byte-identical to the accepted tips.

## RC0 to RC1 state

| Boundary | State | Evidence or blocker |
|---|---|---|
| RC0 nine-contract schema | Integrated | RC0 contract tests and public exports |
| Shared origin-blind package boundary | Integrated | Accepted BC package tests |
| Mechanical RecBole qualification | Integrated | Positive one-epoch and negative interface fixtures |
| Capability admission and registry | Integrated | Accepted deterministic admission tests |
| BC Next Fresh Profile | Integrated | Deterministic profile and receipt tests plus accepted local vertical |
| A0 OpenSpec and Resolver runtime | Integrated | Four-role projection and five deterministic results; no catalog fallback |
| D0 Episode closure | Integrated | Scientific Episode and engineering diagnostic permissions remain separate |
| G local orchestration | Integrated | Open draft through Next Fresh Profile plus negative D0 diagnostic |
| RC1 field representation | Locally exercised | Full Wave 1 local vertical passes; formal designation remains external |
| R1/R2 launchers | Not started | Provider, outcome and later Wave authority are outside this integration |

The local orchestration boundary intentionally ends at
`ProfileBuildReceiptV1`. It does not mutate the current campaign, call a
Provider, consume an outcome, or create a scientific episode. The positive
path performs one one-epoch development smoke; the negative path stops at
`API_CONTRACT` with `INTERFACE`, leaves unit/smoke `NOT_RUN`, and records zero
smoke executions. That failure creates only an engineering diagnostic with
`mechanism_memory_allowed = false`. A passing qualification without an outcome
creates neither a `TypedResearchEpisodeV1` nor a scientific closure.

## Canonical Wave 1 local identities

- OpenSpec: `75af40196bac5424147d66c3a699292e8804b1acc9818f8f471e50820ed9d9dc`
- Innovation resolution: `3a11bfdad39a98026fd65ad89ffb90c1a5e228a6db7bbdf62db6c18dcf012cf5`
- Positive qualification: `68e88491cf960b8a4b89fff38339936b4499d52892559c7afa090554dd201cd0`
- Negative qualification: `e6a73ae6d33b51827747ef0dd4df797be67f04af786b085c16a682ed5ae8bc70`
- Engineering diagnostic: `c297bc75843e24aaa0b785d9eb8e1d4f69c944ed48b77ad9b9c4de8121d216b9`
- Qualified capability: `c34f7ed240a038bc7a8c6ccd30acfe6a96987396dfd52da7911a2d352b155967`
- Registry: `3d1e3a06d8200fcf683e052447fd9c18581c4da753d302631a343a8f71b3dec5`
- Build manifest: `39bdcdd1fcfe943e7948f7ba1d211e2a0d714352213ee97eda99cfaf8d7be631`
- Next Profile: `815c4fe0d3e81ac3a2723e66aeace84e8fb2f4bb9bda7791427228b4c8aa8e20`
- Profile receipt: `8d0b7cedfde60ef678733951b0f1d217c2ce8574a4e9097b4acd87b096003740`

## Wave 1 integrated evidence

1. The high-change interaction-gate draft resolves to
   `INNOVATION_REQUIRED` against all 66 frozen capabilities, with no exact,
   nearest, static, or catalog fallback.
2. Mechanically equivalent A/B source projections produce the same blind
   Implementer request; producer, context, origin and outcome are absent.
3. Candidate roots are fresh and local; the allowlist, no-symlink rule,
   source-tree/package identities and implementation-receipt bytes remain
   unchanged through qualification.
4. The positive interaction-gated model passes real RecBole construction,
   loss, predict, full-sort and one one-epoch development smoke.
5. Admission produces one `QualifiedCapabilityV1`, a deterministic 67-entry
   registry-derived Next Profile, and a next-fresh-only profile receipt.
6. The negative candidate stops at `API_CONTRACT/INTERFACE`; unit and smoke
   are `NOT_RUN`, smoke count is zero, and D0 grants engineering diagnostics
   but no mechanism memory.
7. Current profile and slate bytes remain unchanged; the new capability is
   ineligible until `NEXT_FRESH_CAMPAIGN`.

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
9. Capture qualification, capability, registry, profile and profile-build
   receipt identities for the integrated local vertical.
10. Run focused integration tests, then the relevant Research Line regression
    set and structural lint.
11. Review the final diff and history, create a clean local G commit, and
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
