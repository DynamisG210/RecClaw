# R1/R2 Prefreeze Status

## Status

`BLOCKED_PREFREEZE`

The fresh R1/R2 scientific and call contract is locally closed, but the one
authorized endpoint/auth/schema qualification probe did not pass. No
`R1_PREFREEZE_READY_RECEIPT.json` exists, and no R1 worker may start.

## Accepted source identity

- accepted G commit: `fbcb410e9bdf5052ed48a7e54152eca3917c8dd9`
- accepted G parent: `56d3156f53d17d4d850368ce7a093fa368957d81`
- accepted G tree: `53e8389cd04dc162c48e6e47b7f1f1920a90febd`
- accepted Wave 2 commit: `56d3156f53d17d4d850368ce7a093fa368957d81`
- accepted Wave 2 tree: `30b239b7ac85cd8de176d4b719b327ba02b96e0b`

The workflow, audit, accepted G source, parent, tree, predecessor manifest, and
predecessor blocked receipt were verified before this worktree was created.

## Frozen fresh contract

- model: exact `gpt-5.4`
- real transport:
  `LabApiCanaryBrokerV1.call_with_session`
- granularity: one preassigned Producer role and one proposal per call
- call count: 8 per side
- proposal budget: 8 per side
- token ceiling: 6000 per call
- role schedule: each of the four existing Producer roles appears twice
- A/B: symmetric independent replication lanes with identical call-contract
  digest and isolated roots, databases, candidates, packages, outcomes,
  memories, lineages, and seed plans
- response: strict fresh OpenSpec JSON schema; no fixed-66 candidate fields
- tools: none
- retry: zero
- `CONTENT_NOT_JSON` and every other response-contract failure: terminal,
  consumed slot, no replacement
- successful-response selection: forbidden
- schema relaxation: forbidden
- valid proposals must mechanically resolve to `INNOVATION_REQUIRED`
- shared accepted Implementer, origin-blind projection, and Qualifier
- manual candidate patch: forbidden
- qualification evidence: `DEVELOPMENT_ONLY`
- held-out: absent

The scientific policy freezes denominators, missingness, thresholds,
qualification gate, and analysis before any outcome. R1 requires, per side,
at least four fresh specs across at least two Producer roles and at least two
qualified capabilities; at least one real structural, interaction, or
propagation change is required overall. The shared negative fixture is
mandatory.

## Single authorized probe

Exactly one physical Provider call was made with the production fresh OpenSpec
schema and a pre-authored non-research sentinel. The endpoint returned HTTP 400
before a model response because one exact schema keyword was unsupported.

Non-sensitive classification:

- status: `BLOCKED`
- classification: `HTTP_400_EXACT_SCHEMA_KEYWORD_UNSUPPORTED`
- physical Provider calls: 1
- retry count: 0
- returned model: absent
- research candidates generated: 0
- OpenSpecs projected: 0
- Resolver calls: 0
- candidate roots: 0
- admission: 0
- training: 0
- outcomes: 0
- held-out reads: 0
- secrets or sensitive headers persisted: false

The credential-config, credential, endpoint, request, Provider error, release,
schema, and unsupported-keyword identities are represented only by irreversible
digests in the checked-in receipts. No key value, endpoint URL, or sensitive
header is present.

## Canonical artifacts

- `R1_R2_PREFREEZE_MANIFEST.json`:
  `baad034df0bf423bf2ffc90847848d461be5fdbef1d8b4ff8d68386d6e508ab4`
- `PREFREEZE_BLOCKED_RECEIPT.json`:
  `c491f2f1776f912b4ea1cab5d7db66ddc2d83d6ea2a138c9dc9e09d6c0e7fef3`
- `FRESH_OPEN_SPEC_ENDPOINT_PROBE_RECEIPT_V1.json`:
  `35c38fff6802a11d41f06a45258dce0df9fb712854030add7246b7111815e370`
- shared A/B call-contract digest:
  `71b64a34767f362a31cbab08611b298475059c96aee522b729138d9a2b11b05e`

The manifest is canonical JSON. Its only null required identity is
`provider_identity.returned_model`; the endpoint produced no model response.
The fail-closed validator rejects the manifest on that exact field and also
requires verified authentication, exact schema support, and an exact returned
`gpt-5.4` before READY.

Adjacent verification inadvertently exercised three existing synthetic
one-epoch qualification fixtures. These were not fresh R1 candidates and
created no persistent candidate root or admission, but they were training
fixtures and are recorded explicitly in the blocked receipt. Fresh R1 training
remained zero.

## Exact blocker and stop boundary

The current fresh response contract cannot be verified on the configured
endpoint, and exact returned-model/authentication identity remains unclosed
because schema rejection occurred before completion. The frozen no-retry and
no-schema-relaxation rules prohibit another probe or an in-place schema change
in this prefreeze.

R1 has no legal start condition in this branch. A future attempt requires a new,
explicitly authorized pre-outcome contract version and a separately authorized
single probe; it may not reuse this attempt identity, silently remove the
unsupported rule, change model/endpoint/protocol, or treat this infrastructure
failure as candidate-quality evidence.
