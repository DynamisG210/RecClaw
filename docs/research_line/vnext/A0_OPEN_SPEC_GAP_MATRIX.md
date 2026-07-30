# A0 Open Spec / Capability Resolver Gap Matrix

Baseline: `4306d9ba13157d3c9ed6b877d33558ba9a40c449`

| A0 requirement | RC0 / frozen Search v1 state | Minimal A0 change | Explicit boundary |
|---|---|---|---|
| `OpenResearchSpecV1` | RC0 freezes the contract and canonical identity; no producer adapter exists | Add one deterministic adapter that preserves each `CandidateProposalV4` scientific field and supports an open high-change draft without requiring a catalog program | Do not rewrite Producer roles, context assembly, or Provider transport |
| Four Producer projections | Four frozen roles emit `CandidateProposalV4`; prompts currently require one of the 66 exact executable programs | Project all four roles through the same adapter, retaining role, hypothesis, competing explanation, control, falsifier, evidence, and context/protocol/profile bindings | Do not add a fifth Producer or role-specific implementation path |
| High-change eligibility | Search v1 rejects programs outside the exact executable catalog | Accept structured capability changes to core representation, objective, relation, propagation, training, structure, or another non-parameter mechanism; reject parameter/config-only claims as high-change | Do not approximate an open change with a nearby catalog item |
| Five-state resolution | RC0 freezes `CapabilityResolutionV1` and its five enum values; no resolver exists | Add one precedence-ordered deterministic resolver over spec validity, protocol compatibility, exact profile match, dependency support, and budget support | Do not call an LLM or add routing/qualification state |
| Exact Search match | Frozen profile exposes 66 mechanism identities and semantic digests | Match only a declared exact current-profile semantic identity; emit `SEARCH_READY` with the exact capability ref/digest | Do not use nearest-neighbor, static-candidate, or config fallback |
| Outside-profile route | Contract can represent `NOT_EXPRESSIBLE` but has no executable classification path | Route a valid, protocol-compatible, supported, budget-compatible capability diff to `INNOVATION_REQUIRED` | Innovation is eligible only for the next fresh campaign |
| Protocol change | Frozen protocol is referenced by ref/digest | Route incompatible protocol ref/digest or required protocol features outside the frozen manifest to `DEFERRED_PROTOCOL_CHANGE` | Do not change the protocol or start an experiment |
| Unsupported feasibility | No A0 dependency/budget decision implementation exists | Route unavailable dependencies or an exceeded frozen budget to `UNSUPPORTED` with deterministic reason codes | Do not add retry, transport, environment, or GPU behavior |
| Invalid spec | RC0 dataclass validates construction but not A0 semantic consistency | Reject contradictory expressibility/high-change declarations and malformed adapter drafts as `INVALID_SPEC` | Do not repair or reinterpret invalid input |
| Campaign immutability | RC0 freezes next-campaign activation semantics; no A0 mutation path exists | Keep resolver and adapters pure and return only immutable contracts | Do not modify the current profile or slate |
| Fixtures and regression | RC0 has contract fixtures only | Add canonical fixtures for all five states, all four roles, ordering determinism, protocol change, high-change, invalid input, and no silent fallback | Do not invoke Provider, held-out data, RecBole training, or GPU |

Wave 1 integration surface:

- producer output or an open high-change draft in;
- `OpenResearchSpecV1` out;
- frozen profile/protocol/capability-support projections plus a spec in;
- `CapabilityResolutionV1` out;
- no current-campaign mutation or execution side effect.
