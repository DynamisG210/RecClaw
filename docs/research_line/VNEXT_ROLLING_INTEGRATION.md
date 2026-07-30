# Research Line vNext Rolling Integration

## Frozen identity

- Frozen v1: `d2817397d0c71d5122f55e81a2c60c72b42eaf4b`
- RC0: `4306d9ba13157d3c9ed6b877d33558ba9a40c449`
- Integration branch: `feat/research-line-vnext-integration`
- Integration worktree: `/root/projects/RecClaw_research_line_vnext_integration`
- BC Next Profile merge: `83eb440ab561ad787d5513cc6afc16507b99c3ab`
- A0 OpenSpec merge: `f06e0c868f9b5088855f01cae8bd4f33c92746a0`
- D0 Episode closure merge: `b7fc787688c528084e3f100d4000b71d00e057a2`
- Local Wave 1 accepted / RC1 candidate base:
  `4d493939bb1e118ae9c99c89e8b73077078c3ab8`
- Candidate tree: `3fd503618a664025e8f7a4c751ffdb646787a3d7`
- D1 accepted merge: `3fe57ba8a1c50334907eb2878d70b8664e402617`
- F0 accepted merge: `5489f73d196b29d1c6602d7f7ea14f9f417774e8`
- E0 accepted merge: `e6ac840a76fa8a50880df935bc93c0fb473b262d`

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

Wave 2 accepted owner commits:

- D1 `9d4937f48a3f37bdc031888bb5fb19222b25de96`, parent
  `5c6b064fa90f6d495e226831f0ec86fc9e89d391`
- F0 `ea348c3d301672d8fe090fe76c14e1a7241862da`, parent
  `4d493939bb1e118ae9c99c89e8b73077078c3ab8`
- E0 `a07bd2ccf1effb11702a738668db0af7d83c47d2`, parent
  `4d493939bb1e118ae9c99c89e8b73077078c3ab8`

D1, F0 and then E0 were integrated through explicit merge commits. Their owner
files are unchanged.

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
| RC1 field representation | Local candidate accepted | Independent Wave 1 gate passed at exact commit `4d493939`; no tag, push or release |
| D1 Scientific Episode adapter | Integrated | Only D0-approved `MECHANISM_MEMORY` closures project to the existing belief contract |
| F0 Open Meta interface | Integrated | Static, explicit-support Idea/Experiment policies plus canonical replay and future activation |
| E0 Search Adapter | Integrated | Exact current 66, fail-closed Idea acquisition and next-fresh 67-entry Search projection |
| Wave 2 intake harness | Complete locally | E0/D1/F0 accepted identities and receipts are attached; formal gate remains external |
| R1/R2 launcher | Dry-run boundary only | Complete canonical prefreeze identity is required; Provider and experiment execution are impossible through this boundary |

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

The exact local candidate record is
`docs/research_line/vnext/WAVE1_RC1_CANDIDATE.json`. It binds the accepted
commit and tree, frozen authority hashes, merge identities, owner tips and
the accepted E0/D1/F0 intake slots. It is not a tag or release manifest.

## Wave 2 owner intake checklist

Each E0, D1 or F0 intake must arrive from the supervisor with all of the
following fields before G attaches it:

1. The exact accepted commit SHA and its declared parent SHA.
2. A clean owner worktree at that exact tip.
3. The exact owner file surface and a SHA-256 manifest binding those bytes.
4. A SHA-256 receipt for the owner-targeted tests at the exact accepted tip.
5. A SHA-256 receipt for the owner structural lint at that exact tip.
6. One canonical public `module:attribute` entrypoint.
7. Evidence that the accepted commit descends from RC0 or the declared
   already-accepted owner parent.
8. A diff review showing no G-owned launcher, orchestration or shared schema
   migration and no forbidden Provider/outcome/held-out or state surface.

G records those values in `Wave2OwnerIntakeV1`, verifies the owner byte
surface before and after integration, and preserves the owner commit and
parent identities through an explicit merge whenever G and the owner have
diverged. `Wave2IntegrationHarnessV1` only records the three future ports:
E0 Search Adapter, D1 Scientific Episode adapter and F0 Open Meta interface.
It neither imports nor executes their implementations. An entrypoint mismatch
produces only the lane, field, expected value and observed value for the owner
to correct.

E0, D1 and F0 are recorded with exact owner byte manifests, test receipts,
structure receipts, public entrypoints and merge commits. Uncommitted,
unaccepted or cross-owner bytes are not intake.
The machine-readable frozen checklist is
`docs/research_line/vnext/WAVE2_INTAKE_CHECKLIST.json`.
The receipt preimages are
`docs/research_line/vnext/D1_F0_INTAKE_RECEIPTS.json`.
E0 has a separate exact receipt at
`docs/research_line/vnext/E0_INTAKE_RECEIPT.json`.

## Canonical Wave 2 local identities

- D1 success belief:
  `9e8c48344e992f64ebc0b3bdb4ab9b769d5600de1ee743d702b0311a3fd3f250`
- D1 mechanism-negative belief:
  `d5a28c46d77abce37ae9c6292d5f3fa3f174a3af52fbac5da8dc1a8d3ed77d41`
- F0 Idea decision:
  `4f04690e315d8f8a0830487b7727578e29f59b4242f074be498bdc8c29a0f451`
- F0 Experiment decision:
  `b0284474665ddefab9128cdbc3d6fe1378d9a5ed56314df61ea6ec5c898f2d98`
- F0 explicit out-of-support Experiment decision:
  `745c5f1a84465935ce5ea9cf68d471352c6c1ff3b689f69b44998b771156d99d`
- F0 Idea activation:
  `83a41c6bc2aae0d94729d784c87c96542f5b4b58dfa994953bc4ef2d0f8aac9d`
- F0 Experiment activation:
  `00df12882874ec1d70fd509055b530a50f2784bdd728fc1fde4ec6cdf92e4bcd`
- F0 replay dataset:
  `d36199367d0de52660858be2c4d2f2e209cfaf439741a916be9f83aaedb044e7`

D1 refuses identity drift, engineering/interface/package/runtime/resource,
Provider, protocol, missing-outcome and inconclusive closures before any
Search Memory write. Its output is the existing qualitative
`DevelopmentalMechanismBeliefV1`; it does not create a
`SearchUtilityEventV2` or numeric frontier observation.

F0 keeps Idea and Experiment inputs, budgets, decisions and activation
boundaries separate. Unsupported low-change/configuration-like ideas and
capabilities outside the frozen slate return explicit `OUT_OF_SUPPORT`,
`DEFER` and zero allocations. Both policies remain
`RESEARCH_STATIC_VNEXT`; activation is only `NEXT_FRESH_CAMPAIGN` for Idea
and `NEXT_ROUND` for Experiment, with `promotion_authorized = false` and no
current-policy replacement.

## Canonical complete Wave 2 evidence

The canonical local engineering receipt is
`docs/research_line/vnext/WAVE2_INTEGRATED_GATE_RECEIPT.json`, with byte
SHA-256
`fcb431c0ae8738f243ba10021770a2c507bee2b6d8454df622c2dc05700332eb`.
It records evidence for independent supervisor evaluation; it explicitly sets
`scientific_gate_authorized = false` and does not promote Wave 2.

- Current Search profile: 66 entries, profile digest
  `952ba027cb8dbfd8831fca6d33ce39db1aabd4dbc41534b089250e0dc1253b15`
- Current adapted-profile canonical bytes:
  `53b5bfb6a6ccc090b2a306519e47326060bd729cf17d5e10abba6d65e6c93f52`
- Frozen source profile canonical bytes:
  `a8f44e5ebeb8aa9dc3a96c26b849e65bd8f458c83dbe818b97774ca8f55e4e38`
- Current frozen slate:
  `2ea4433e6843ce46ec0492a2649014c20cd4363610e6d1b3ef1b4ba9cc1e0baa`
- Current Router trace:
  `c7ea9d5b6f85724ea0c22a54e64167fc3606047bbf2db5718a060186a40a9e6b`
- Next-fresh Search profile: 67 entries, digest
  `ee60ecf3835150c4af7ace0bfc9f459a5c247e094db8233c5075b78c1c39a228`
- Qualified next binding:
  `347b6c627a0c1e6e51ef892184dee1cd96a7ad113d524745c5463a2655db6afe`
- Next-fresh Router trace:
  `d28bd5ffd5dbf2c0c8e3d2a42948c6f4154382e0e558c22387a27282e188c56d`
- Integrated F0 replay dataset:
  `5243364c5d87836cb23541fe13411bd8cab7e17d5919fccf2d79680d2c0d1f1e`
- Integrated D1 rejection matrix:
  `1ed17ed89329703ec4d3cdc29c7adbfe0f5560f8c064b92aa4cf466f0b36845a`

E0 sends only `SEARCH_READY` current-profile bindings to the unchanged
`StrongStaticRouterV1`. `INNOVATION_REQUIRED`,
`DEFERRED_PROTOCOL_CHANGE`, `UNSUPPORTED`, `INVALID_SPEC`, unqualified open
specs and exact-66 fallback attempts do not enter the current frozen slate.
A qualified outside-profile capability remains unavailable in the current
campaign and becomes selectable only through the distinct 67-entry
`NEXT_FRESH_CAMPAIGN` profile, using its already-qualified entrypoint.

F0 composition remains identity-only: Idea consumes spec acquisition
identities, while Experiment consumes identities from an E0-frozen active
profile/slate. Its decision identity is passed as an audit projection to the
existing Router. No F0 schema is merged into E0 and no E0 schema is
redefined by F0.

## R1/R2 prefreeze and dry-run boundary

`docs/research_line/vnext/R1_R2_PREFREEZE_TEMPLATE.json` is intentionally
incomplete. Unknown source, runtime, schema, prompt, tool, endpoint, model,
credential-identity, seed, namespace, root and DB identities remain `null`;
G does not invent them. Before any launch authority can be considered, one
copy must be completed and serialized as canonical, digest-stable JSON.

The G loader fails closed unless the caller supplies that one file and its
exact SHA-256 digest and every required identity is present. It freezes:

- one shared A/B model identity and proposal-call contract, including
  granularity, token budget, call count, failure rules and no retry;
- proposal budget `8` per side;
- fresh A/B identity plus isolated R1/R2 lineage, seed, outcome namespace,
  memory namespace, root and DB references;
- held-out absence, shared Implementer and Qualifier identities, manual
  candidate patch prohibition and `DEVELOPMENT_ONLY` qualification;
- missingness and analysis-plan identities before outcomes.

The current launcher operation is deterministic dry-run validation only. Its
receipt always records zero Provider calls, zero experiment runs, zero
outcomes consumed and `launch_authorized = false`.

The R1 gate reserves empty receipt slots for at least four fresh specs from at
least two Producer roles, at least two qualified capabilities, and at least
one real `STRUCTURAL`, `INTERACTION` or `PROPAGATION` change. Prefreeze files
must keep those slots empty. Smoke, mock or development fixtures cannot
populate them; only later real R1 receipts may be evaluated by an accepted
owner boundary.

## General commit intake checklist

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
| Search Adapter implementation | E0 |
| Scientific Episode adapter implementation | D1 |
| Open Meta interface implementation | F0 |

If runtime evidence requires an RC0 semantic or representation change, the
originating owner supplies the smallest reproducer and proposed correction.
G applies the shared migration once, records it as RC1, and updates all
integrated callers. G does not copy owner logic to bypass a conflict.

The G-owned Wave 2 wiring adds no Producer, Search, Router, Runner, Evaluator,
Interpreter, Meta learner or Guard implementation. Accepted F0 supplies only
its static pre-learning owner interface. G adds no database, ledger, authority
service, state machine, Broker, retry or cooldown behavior.
