# F0 Open Meta Interface Gap Matrix

Baseline: `4d493939bb1e118ae9c99c89e8b73077078c3ab8`

F0 is a pre-learning interface and static-policy delivery. It does not
implement F1 learning, shadow evaluation, promotion, or automatic policy
updates.

| F0 requirement | Accepted baseline state | Minimal F0 delivery | Preserved boundary |
|---|---|---|---|
| Separate Idea and Experiment policies | RC0 has one `AcquisitionDecisionV1` envelope with an explicit stage and distinct schema identities, but no executable open-policy input or decision | Add distinct `IdeaPolicyInputV1` / `ExperimentPolicyInputV1`, distinct budget types, and stage-checked allocations | No shared feature or budget schema; no use of fixed-space Meta features |
| Bounded policy inputs | Wave 1 provides context, OpenSpec, capability/profile, and Typed Episode identities | Accept current context/protocol/profile identities, opaque OpenSpec or capability identities, and identity-only historical Episode summaries | No producer/origin/source label, raw outcome, interpretation, cost, held-out/split, or current unknown outcome feature |
| Explicit support result | No accepted F0 support-domain contract exists | Return `IN_SUPPORT` or `OUT_OF_SUPPORT` with reason codes, scope, and an explicit `RESEARCH_STATIC_VNEXT` mode | Out-of-support returns `DEFER` and zero allocation; it never maps to the 66-item catalog or parameter/configuration tuning |
| Static pre-learning baseline | Existing learned Meta is tied to the fixed executable profile | Add deterministic canonical-identity ordering within open high-change Idea scope and the frozen Experiment slate | No learner, weights, policy update, cross-domain learning, or capability injection |
| Replay data boundary | Existing replay paths belong to fixed-space Meta or runtime state | Add one canonical JSON dataset writer/reader containing policy version, input digest, allocations, acquisition decisions, and activation identity/boundary | No DB, ledger, service, attempt, authority, retry, or mechanism-belief write |
| Activation | Next Profile already binds `NEXT_FRESH_CAMPAIGN`; fixed-space Meta owns its existing boundaries | Bind Experiment decisions to `NEXT_ROUND` and Idea decisions to `NEXT_FRESH_CAMPAIGN` | Static F0 scheduling does not replace current policy; promotion is not implemented |
| Deterministic evidence | No F0 fixtures/tests exist | Add in-support, explicit out-of-support, budget isolation, information blindness, activation, replay round-trip, and stable-digest tests | No Provider, GPU, runner, evaluator, held-out, or outcome-bearing call |
| G integration ownership | Public exports and end-to-end wiring are G-owned | Deliver an internal module, owner tests, canonical identities, and this intake note | F0 does not edit public exports or old entry points |

## Static support domains

`IdeaPolicy` supports only open, high-change, currently non-expressible
ResearchSpec identities. It allocates ideation direction slots and bounded
implementation/qualification slots. Parameter-only, configuration-only, and
currently expressible subjects are explicit out-of-support results.

`ExperimentPolicy` supports only executable capability identities already
present in the frozen current slate. A subject outside that slate is an
explicit out-of-support result and receives no experiment budget.

Both policies use canonical opaque-identity ordering solely as an explainable
pre-learning fixture. Historical Episode summary identities remain audit
inputs but are not ranking features in the static decisions.

## G intake

G may expose the new F0 contracts and functions through the package public API
and wire them to E0 after reviewing the owner commit. G should retain:

- separate Idea and Experiment input/budget schemas;
- `RESEARCH_STATIC_VNEXT`, its reason, and its support scope;
- Experiment `NEXT_ROUND` versus Idea `NEXT_FRESH_CAMPAIGN`;
- no current-policy replacement without a future, separately authorized F1
  promotion result;
- the canonical replay file as data only, with no state or belief authority.
