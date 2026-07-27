# V13 Meta Requalification Report

## Promoted policy

- checkpoint: `META_VNEXT_V18_SUPPORT_AWARE`
- checkpoint SHA-256:
  `0ef232905cb07abada89322f631908743b31b74527f36c59efc4c00f7b56fd9c`
- policy-bundle digest:
  `2d42ac7b9ced6057d3e20e19f1f470338ed00b0fcbc70e790bdce7fa44706024`
- promotion-decision digest:
  `ce5783e7f39f29135422dff3d3fb4a377d3a11a1283677a0a29ba4517064223e`
- activation boundary: `NEXT_CAMPAIGN`
- Pilot outcomes used: no

V18 retains the V17 outcome-trained slow coefficients after projecting the
changed V13 feature contract into the frozen calibration support. It binds
CandidateProposalV4, the 66-semantics executable profile, the new Research
feature schema and the closed common-only feedback admission path.

## Calibration evidence

- 84 development episodes / 336 candidate results;
- 48 fresh held-out episodes / 192 candidate results;
- slow versus static mean net value: `+0.0407538912`, 80% interval
  `[+0.0283492554, +0.0544886913]`;
- slow+fast versus slow mean net value: `+0.0059484685`, 80% interval
  `[+0.0029822312, +0.0090377809]`.

## Declared limitations

The actual ML-1M task context is `task_scale=1.0` and
`task_density=0.2843119865`. `task_scale` lies outside the frozen Fast support
ceiling (`0.4175133333`), so Fast is disabled rather than extrapolated. This
does not disable slow Meta control. Architecture, geometry and sampling use
generic transfer coefficients, not unsupported axis-specific outcome claims.

B and C load the same slow checkpoint. Fast state, when supported in a later
campaign, remains Arm × SearchSeed private.
