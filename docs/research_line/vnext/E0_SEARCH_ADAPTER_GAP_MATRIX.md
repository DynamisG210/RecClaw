# E0 Search Adapter Gap Matrix

Baseline: `4d493939bb1e118ae9c99c89e8b73077078c3ab8`

Owner boundary: E0 adapts accepted Wave 1 contracts to the existing Search
consumer. It does not change the fixed catalog, Producers, Resolver,
Implementer, Qualifier, Registry, Profile Builder, Router, Runner, Evaluator,
public exports, or orchestration entrypoints.

## Actual baseline gaps

| Required boundary | Accepted baseline evidence | Actual gap | E0 action |
|---|---|---|---|
| Fixed 66 as the current executable profile | `campaign_runtime.py` exposes exactly 66 package-owned mechanisms and `open_spec.py` projects their resolver identities | No first-class, campaign-bound Search consumer projection exists | Mechanically project the exact 66 identities and entrypoints without changing source profile bytes |
| Idea acquisition | A0 projects `OpenResearchSpecV1` and returns the frozen five-state resolution | No `AcquisitionDecisionV1` consumer routes the resolution to Search, Innovation, defer, or reject | Add a deterministic Idea acquisition boundary with distinct Idea schemas and budget snapshot |
| Experiment acquisition | `StrongStaticRouterV1` routes `CandidateProposalV4` | The Router does not know capability qualification, profile identity, slate identity, or activation boundary | Freeze a profile-bound experiment slate before invoking the unchanged Router |
| No unqualified execution | A0 prevents non-`SEARCH_READY` resolutions from claiming a current match | A broad BL-ICF-valid open program can still be passed directly to the Router | Require every routed proposal to bind an exact capability entry in the active profile |
| Next-profile activation | BC builds `ExecutableProfileVNext` from registry entries and records `NEXT_FRESH_CAMPAIGN` | No consumer activates the result only for a distinct fresh campaign | Validate registry/profile identity and create a fresh campaign Search projection without mutating the predecessor |
| Qualified capability consumption | `QualifiedCapabilityV1` binds the qualified package entrypoint | No bridge proves it can enter the existing Router path without another Implementer call | Bind a Search proposal to the active qualified entry and pass that proposal through `StrongStaticRouterV1` unchanged |
| No silent fixed-66 fallback | A0 resolution records `no_silent_fallback=true` | Search consumption has no independent check that a qualified program was not mapped to an exact catalog recipe | Require qualified Search projections to compile while remaining outside the exact 66 execution recipes |
| Current campaign byte isolation | BC profile receipt states `current_profile_unchanged=true` | Availability and slate bytes are not checked at the Search consumer boundary | Keep immutable profile/slate objects and test their bytes before and after next-profile construction and activation |
| Identity and budget auditability | RC0 defines `AcquisitionDecisionV1` | There is no actual Idea/Experiment decision using separate schemas | Emit canonical decisions with distinct schema digests, frozen budget snapshots, active profile/slate identities, and reason codes |

## Frozen observations

- Current profile ID: `BL_ICF_EXECUTABLE_PROFILE_V2`
- Current profile digest:
  `952ba027cb8dbfd8831fca6d33ce39db1aabd4dbc41534b089250e0dc1253b15`
- Current profile canonical-bytes SHA256:
  `a8f44e5ebeb8aa9dc3a96c26b849e65bd8f458c83dbe818b97774ca8f55e4e38`
- Current profile canonical byte length: `31995`
- Exact executable count: `66`
- Adjacent accepted baseline tests:
  `29 passed` across A0, admission, next-profile, and G local orchestration.

## Intended consumer path

```text
existing CandidateProposalV4
  -> A0 projection and resolution
  -> E0 Idea acquisition
  -> SEARCH_READY only
  -> active-profile capability binding
  -> frozen experiment slate
  -> unchanged StrongStaticRouterV1

qualified next capability
  -> accepted VersionedCapabilityRegistry / ExecutableProfileVNext
  -> distinct fresh-campaign activation
  -> qualified capability binding (existing entrypoint; no Implementer)
  -> frozen experiment slate
  -> unchanged StrongStaticRouterV1
```

The adapter ends at a selected, profile-bound executable entry. G owns wiring
that selection into old orchestration and Runner entrypoints.
