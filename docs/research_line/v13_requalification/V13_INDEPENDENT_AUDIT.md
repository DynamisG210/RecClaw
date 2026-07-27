# V13 G8 Independent Adversarial Audit

## Verdict

`PASS — P0=0, P1=0, P2=3 declared Meta limitations`

This audit is independent of Pilot outcomes: no Provider call, ordinary
training execution or Pilot root was created.

## High-value checks

1. Arm A uses `PinnedOriginalMainAdapterV1` and exact Main Git blobs. The
   handwritten predecessor adapter is unreachable from `create_v13()`.
2. A/B/C bind the same 66-semantics executable profile and common training
   release. B/C non-Guard policies are equal.
3. The Lab API release binds the current CandidateProposalV4 JSON Schema and
   `gpt-5.4`; the earlier V12 schema release is not reused.
4. Training Runtime Release V5 binds the current campaign runtime profile,
   source bytes, environment, dataset partition and real execution recipe.
5. Constructing V13 creates no round, calls no Provider and starts no training.
6. The run entrypoint imports without creating the Pilot output root.
7. Pilot analysis reads `RAW_RESULT_ENVELOPE_V2` and `ROUND_FEEDBACK`. The
   earlier draft incorrectly queried V1/`ROUND_CLOSED`; this was corrected
   before contract freeze.
8. Automated Pilot completion cannot self-declare GO. It reports
   `PENDING_INDEPENDENT_AUDIT`, requires one completed exact validation-task
   lifecycle, and never computes a treatment effect or Confirmed Frontier.
9. The frozen source manifest points to committed source HEAD `a0956c6`; the
   external laboratory secret is neither copied into the contract nor stored
   in the repository.

## Verification

- focused G8 freeze/runtime/analysis and G7 attribution checks: 11 passed;
- all V13, Lab API, M6R training and M6E environment tests: 99 passed;
- executable-family loss/backward checks: 9 passed within that suite;
- Python compilation: pass;
- frozen contract build and independent verify: pass;
- staged diff check: pass.

The old M3 test that requires current evolved source bytes to equal the frozen
superseded V12 contract remains intentionally outside the V13 suite. V12 was
not modified or run.

## P2 declarations

The three G6 limitations remain:

1. Fast residual is disabled for the current out-of-support ML-1M scale.
2. Architecture, geometry and sampling lack axis-specific outcome-trained
   Meta coefficients.
3. Inherited slow-policy concentration must be observed in the development
   Pilot.

None changes the frozen protocol or invalidates the G8 start gate.
