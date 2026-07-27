# V13 Scientific Attribution Architecture Decision

## Decision

V13 freezes one comparison substrate and three controller treatments:

```text
A = exact pinned Main RecClaw decision core + common BL-ICF V2
B = Research Capability V13 + Meta V18 + common BL-ICF V2
C = exact B + EvidenceGuardPortV1
```

The primary estimands remain `B-A` and `C-B`; `C-A` is secondary. Pilot and
Main use the same executable profile, runtime, dataset split, materializer,
runner, budgets and analysis code.

## Runtime path

```text
Lab API gpt-5.4
  -> A: PinnedOriginalMainAdapterV1
  -> B/C: four discovery Producers -> common eligibility -> Router -> Meta V18
  -> CandidateEnvelope
  -> B: NullEvidencePortV1 / C: EvidenceGuardPortV1
  -> CommonExecutionGuard -> package-owned Training Runtime Release V5
  -> RawResultEnvelopeV2
  -> typed EvidencePort POST
  -> DeterministicHelixAdmissionV13
  -> FusedSearchFeedbackV2
  -> SearchUtilityEventV2, typed task, or no update
  -> Controller / Meta / Search Memory
  -> Observed and SearchEligible analysis projections
```

`ConfirmedFrontier` is not computed by the Pilot. It requires the later frozen
post-selection evaluator.

## Authority and isolation

- Research-visible state contains only common Search Utility and closed task
  references.
- Guard-private claim, protocol, reason, evidence-use and audit fields do not
  enter Producer, Router or Meta inputs.
- B and C share the same Research and slow-Meta policy. Their fast states and
  stores remain Arm-private.
- Guard-blocked candidates receive no extra proposal, execution, retry or
  refund.
- Arm A imports exact Git blobs from Main commit
  `2d8c881354e1b536a6c66d7dfbb977e0c5090e50`; it does not import the dirty
  working-copy `scripts/agent.py`.

## Frozen implementation choices

- executable profile: `BL_ICF_EXECUTABLE_PROFILE_V2`, 66 unique semantics;
- Meta: `META_VNEXT_V18_SUPPORT_AWARE`;
- Broker: single JSON-schema laboratory API request, model `gpt-5.4`, zero
  retry;
- training: `TRAINING_RUNTIME_RELEASE_V5`;
- one SearchRound: one ordinary candidate execution opportunity plus one
  result feedback;
- five Pilot rounds per Arm, seed `9215`;
- Pilot output root:
  `/root/projects/RecClaw_campaign_pilot_9215_v13`.

The output root is absent and the Pilot has not started.
