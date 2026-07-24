# M1 Independent Adversarial Audit

## Verdict

`PASS_WITH_NONBLOCKING_P2`

```text
P0 = 0
P1 = 0
P2 = 1
authority = NONE
evidence_class = DEVELOPMENT_ONLY
formal_acceptance = false
```

This was a fresh-context adversarial pass over the completed M1 diff. It is an
independent implementation review within the autonomous development run; it
does not substitute for external human acceptance or runtime permission.

## Scope inspected

- the complete Proposal → compile → plan → materialize → trust → binding →
  pre-execute → committed claim → fake Runner → close-result → RawResult path;
- A/B/C release-projection equality;
- BL executable-profile coverage and unsupported-program rejection;
- candidate-controlled program, materialization, import, path, artifact,
  receipt, raw-result and claim forgery attempts;
- ordinary-execution debit and single-use claim behavior;
- package import outside the repository working directory;
- Research/Evidence authority import and field boundaries;
- real-LLM, ordinary-training, RecBole-core and external-side-effect absence.

## Findings resolved before verdict

The review found and closed concrete P1 false-allows:

1. caller compile projections could differ only by tuple/list representation;
2. a BL-valid custom-model program could reach a package template despite the
   executable profile excluding candidate-controlled code;
3. a caller could present a forged or stale materialization report instead of
   a fresh derivation from exact bytes;
4. unmanifested files could exist below the candidate root;
5. start receipts and raw artifacts were not initially re-bound to the
   committed claim;
6. close-result artifact closure did not initially bind both receipt and raw
   output;
7. the state-store execution lifecycle source was not initially part of the
   common release identity.

Regression tests now exercise each case.

## Gate evidence

- 25 M1 targeted/E2E/adversarial tests passed.
- 26 corrected M0 tests passed.
- 27 BL-ICF compiler/space tests passed.
- 4 legacy Research Line and 17 legacy Candidate Proposal tests passed.
- BPR-MF, LightGCN and non-anchor NGCF/SGL/DIRECTAU/UltraGCN recipes complete
  the deterministic fake vertical slice.
- A/B/C produce the same common release projection digest:
  `97c4247af6f0a2fb2fbccd64663e2a6ed25cc1bc98aab2f5630e6d41d6f67347`.
- unsupported SIMPLEX and valid-but-candidate-controlled custom-model
  programs fail closed.
- no real LLM call or ordinary training occurred.

## Nonblocking P2

`M1-P2-001`: M1 proves a closed package-owned fake-handler interface and
filesystem closure, not OS-level confinement for future real training. This
does not block synthetic M2/M3 work. M4 must establish the pre-Canary
confinement and cross-arm negatives before M5 may run.

Adding a general sandbox during M1 would exceed the vertical slice and violate
the instruction to avoid speculative defensive infrastructure.

## Conclusion

All M1 automatic-continuation conditions are satisfied with no unresolved
P0/P1. M2 may begin from the M1 checkpoint while treating the common runtime
release as frozen.
