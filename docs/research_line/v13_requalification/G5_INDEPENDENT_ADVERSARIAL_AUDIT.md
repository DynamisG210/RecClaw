# M6G G5 Independent Adversarial Audit

## Scope and method

The audit treated the typed operator catalog, generated BL-ICF programs,
package-owned handlers, CommonExecutionGuard and deterministic materializer as
one executable profile. It did not infer executability from prompt text.

## Findings

No P0, P1 or P2 finding remains in G5 scope.

- The profile contains two frozen bases and 16 typed operators across seven
  scientific axes.
- The compatibility rules generate 66 unique executable semantics: 2 base
  controls, 16 single-operator candidates and 48 compatible two-operator
  candidates.
- The 66 programs are generated from operators; they are not 66 hand-written
  recipe entries.
- Same-axis and same-component mutations are excluded before program
  generation.
- All 66 programs compile, pass CommonExecutionGuard, materialize
  deterministically and resolve to one exact package-owned recipe.
- Program, semantics and execution-recipe identities are all 66/66 unique.
- Provider output is a typed base/primary/secondary composition. A legacy
  recipe-only response fails the frozen schema.
- Four BPR and five LightGCN family canaries collectively cover every
  operator and complete real `calculate_loss()` plus backward propagation.
- Successful profile use remains identical for A/B/C; the profile contains
  no Arm or Evidence-Guard condition.
- The executable-profile digest is separated from the broader campaign
  runtime digest, so G6 Meta requalification cannot silently redefine the
  search space.

## Verification

```text
6 targeted profile tests passed
93 affected V13 and historical regressions passed
2 documented historical tests deselected
3 subtests passed
29 Meta control-plane regressions passed
compileall PASS
git diff --check PASS
```

The RecBole warnings are upstream sparse-tensor and pandas deprecation
warnings; every loss/backward canary completed with finite loss and gradients.

## Verdict

`PASS - P0=0 / P1=0 / P2=0`

G5 does not authorize a Pilot. The frozen profile changes Producer semantics,
feature support and candidate-pool geometry, so the existing Meta checkpoint
must be requalified in G6 before use.
