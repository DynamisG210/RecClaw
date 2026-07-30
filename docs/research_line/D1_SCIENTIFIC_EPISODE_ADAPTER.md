# D1 Scientific Episode Adapter

## Scope

D1 connects a D0-approved `TypedResearchEpisodeV1` to the existing
`DevelopmentalMechanismBeliefV1` input accepted by
`SearchMemoryWriterV1`. It does not call the writer, mutate Search Memory,
construct frontier state, invoke an interpreter, or depend on Evidence Guard.

## Actual gap matrix

| Existing boundary | Episode availability | D1 decision |
|---|---|---|
| Mechanism hypothesis and competing explanation | Present | Project directly |
| Matched comparator and protocol identity | Present through Episode plus frozen comparison identity | Revalidate before projection |
| Positive versus negative mechanism evidence | Typed by `failure_class` | Project the outcome digest to exactly one evidence side |
| Next discriminative test | Present | Project directly |
| Search Memory belief input | Existing V1/V2 consumer | Return the existing V1 contract |
| Numeric comparator delta, metric contract, seed and resource measurements | Only content references are present | Do not synthesize `SearchUtilityEventV2` |
| Origin or source-Arm metadata | Not part of the adapter input | Reject source/origin-shaped mechanism axes |

`DevelopmentalMechanismBeliefV1` is used because it can preserve qualitative
mechanism-positive and mechanism-negative evidence without inventing a numeric
effect. The existing V2/Search Utility contracts require numeric facts that are
not fields of `TypedResearchEpisodeV1`.

## Admission

Projection is allowed only when:

- the closure lane is `MECHANISM_MEMORY`;
- `mechanism_memory_allowed` is true;
- closure, Episode and frozen comparison identities match;
- protocol, comparator and outcome identities match;
- every D0 closure stage is `PASS`;
- the Episode is an executed `NONE` or `MECHANISM` scientific terminal;
- evidence is `DEVELOPMENT_EXPERIMENT` or `FORMAL_EXPERIMENT`;
- qualification evidence was not used as scientific evidence.

All other paths fail with a stable `ScientificEpisodeAdapterReasonV1` code.
The adapter returns a belief only; G owns public exports and runtime wiring.
