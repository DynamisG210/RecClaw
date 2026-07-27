# G6 independent adversarial audit

Verdict: **PASS — P0=0, P1=0, P2=3**

Scope: Meta checkpoint identity, frozen calibration evidence, CandidateProposalV4
feature compatibility, the 66-entry executable profile, the 64-entry Research
discovery subset, B/C identity, Arm-private fast state, Producer control,
Guard-field exclusion, and non-collapse behavior. No Pilot, Provider call, new
training outcome, or Main execution was used.

## Material finding and correction

The previous campaign adapter disabled Fast for ML-1M but still evaluated the
slow policy with `task_scale=1.0`. The frozen development corpus only supported
the slow/fast task-scale interactions through `0.417513333333333`. That was an
unlabelled slow-policy extrapolation.

V18 fixes the actual failure mode rather than relabelling V17:

- learned rank features are projected independently into their frozen
  development ranges before scoring;
- the original CandidateProposalV4 pool and identities remain unchanged;
- the actual task context is retained in the audit and Fast gate;
- Fast is disabled unless the actual task context is supported and every
  candidate axis in the slate is calibrated;
- architecture, geometry and sampling receive no invented axis coefficient;
  they retain common static-utility and generic structural scoring;
- the two roots remain executable anchors/matched controls, while the 64
  non-root compositions form the Research discovery universe.

This is a new content-bound V18 policy bundle. It inherits the exact promoted
V17 slow fit because the frozen corpus contains no outcome-bearing examples for
the three new axes; refitting the same corpus cannot create that missing
information. The change is therefore a support projection and activation
policy, not a fabricated learned improvement.

## Adversarial checks

- Modified checkpoint bytes, parent bytes, feature support, profile digest,
  feature schema, router digest, activation boundary, or Pilot-outcome flag
  fail V18 loading.
- All 64 discovery compositions produce finite support-projected scores. The
  profile still contains all 66 executable entries.
- Seven controlled utility scenarios select all seven mechanism axes; selected
  Producer counts are `2/2/2/1`, so neither axis nor Producer identity is a
  hard-coded winner.
- B and C share the exact slow policy and control policy. Their fast state
  digests differ by opaque Arm identity while their initial semantic
  projections are equal.
- Three successful out-of-support rounds advance the fast boundary without
  writing a fast observation.
- V18 falsification directives are typed `FALSIFICATION`, remain discovery
  Producers, and do not masquerade as the old control slot.
- Rank/support/checkpoint fields contain no Claim Ceiling, Evidence Admission,
  Protocol Branch, Guard event, or evidence-use input.
- The 29 V17/control/architecture regressions pass unchanged.

## P2 limits

1. Fast remains unavailable for this ML-1M Pilot. It keeps its positive frozen
   evidence but contributes no runtime treatment here.
2. Three new axes have generic-transfer scoring only. Their eventual effects
   must come from ordinary development observations, not from a retroactive
   Meta claim.
3. The inherited slow ranker was axis-concentrated in old replay. V18 adds no
   unvalidated reward shaping; uncovered-axis Producer targeting and real
   utility sensitivity are verified, and concentration remains a Pilot
   diagnostic.

These are transparent development limits, not P0/P1 attribution or execution
failures.
