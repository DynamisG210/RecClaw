# E0 Search Adapter Intake

Baseline: `4d493939bb1e118ae9c99c89e8b73077078c3ab8`

Contract set: accepted RC0 `OpenResearchSpecV1`,
`CapabilityResolutionV1`, `QualifiedCapabilityV1`,
`ExecutableProfileVNext`, and `AcquisitionDecisionV1`. E0 does not change
their representation or semantics.

## Consumer surface

Implementation:

`src/recclaw_core/experiments/helix_abc_v1/search_adapter.py`

The G-owned wiring path is:

1. call `adapt_current_search_profile()` to project the exact current 66;
2. use `acquire_candidate_idea()` or `acquire_open_idea()` at the Idea
   boundary;
3. call `bind_search_candidate()` only for a capability in the active profile;
4. call `freeze_experiment_slate()` before routing;
5. call `route_frozen_experiment_slate()` to invoke the unchanged
   `StrongStaticRouterV1`;
6. after accepted registry/profile construction, call
   `activate_next_fresh_search_profile()` with a distinct campaign ID;
7. consume the selected binding's accepted `executable_entrypoint`.

`predecessor_executable_entries()` supplies the exact 66-entry input expected
by the accepted BC profile builder.

E0 deliberately does not modify package `__init__` exports, old orchestration
entrypoints, Runner materialization, or end-to-end launchers. Those remain G
owned.

## Canonical fixture

Fixture ID: `e0-search-adapter-canonical-v1`

Path:

`tests/experiments/helix_abc_v1/fixtures/e0_search_adapter_cases_v1.json`

SHA256:
`1bebe835a9e079d33f0e5025912788ca2694128a1fb74b7278ccd0e532ced445`

The fixture freezes:

- the exact current profile ID, digest, canonical-byte digest and length;
- 66 total entries split as 24 BPR and 42 LightGCN;
- current and next-fresh campaign identities;
- one experiment budget snapshot;
- current-search and qualified-next positive cases;
- unqualified, unsupported, current-campaign, and silent-fallback negatives.

## What the tests establish

- all 66 current candidates remain exact and produce the same
  `StrongStaticRouterV1` route trace with or without E0;
- only `SEARCH_READY` carries a current Search proposal;
- `INNOVATION_REQUIRED`, `DEFERRED_PROTOCOL_CHANGE`, `UNSUPPORTED`, and
  `INVALID_SPEC` cannot enter a current frozen slate;
- a qualified capability remains unavailable in its predecessor campaign;
- the accepted next profile activates only under a distinct fresh campaign;
- its qualified capability can be selected through the unchanged Router;
- selected execution uses the already-qualified registry entrypoint and does
  not invoke Implementer again;
- a qualified proposal that resolves to an exact fixed-66 recipe is rejected,
  so there is no nearest or silent catalog fallback;
- Idea and Experiment decisions use different feature, budget, and eligibility
  schemas and bind the profile/slate/campaign identities;
- predecessor profile and slate canonical bytes remain unchanged.

## Structured correction record

| Stage | Observed failure | Root cause | Minimal correction | Contract impact |
|---|---|---|---|---|
| First E0 targeted run | Innovation fixture resolved `UNSUPPORTED` | The test environment omitted the draft's frozen `recbole-runtime` dependency and `implementation_units` budget | Supply the exact declared dependency and budget in the fixture environment | None; the Resolver correctly failed closed |
| First E0 targeted run | Reconstructing a proposal with `dataclasses.replace` failed RFC 8785 canonicalization | `CandidateProposalV4` intentionally stores a deep-frozen `mappingproxy`; `replace` fed that private representation back through the constructor | Rebuild the negative proposal through the normal constructor helper | None; production code unchanged |
| Expanded negative-state run | Identity-boundary test raised `NameError: replace` | Cleanup removed the import while another immutable profile test still used it | Restore the test-only import | None; production code unchanged |

No candidate, threshold, protocol, seed, evidence class, or comparison rule was
changed to make a check pass.
