# BL-ICF Mechanism Space v1

Status: development-only implementation contract

Authority: `NONE`

Evidence class: `DEVELOPMENT_ONLY`
Formal acceptance: `false`

## Scope

`BL_ICF_MECHANISM_SPACE_V1` is the search space for static, offline,
implicit-feedback, ID-only collaborative top-N ranking. It freezes the accepted
recommendation protocol while opening model and training mechanisms.

This package is for the future Research-only versus Full Helix comparison. It
does not replace, reinterpret, or rerun the frozen Original versus Evidence-only
narrow-space experiment. It also does not add sequential, session, metadata,
multimodal, knowledge-graph, cold-start, online/OPE, LLM-recommender, or
evaluation-protocol search.

## Contract layers

The implementation is deliberately not a finite hyperparameter table.

1. `Known Primitive Registry` closes 16 typed mechanism axes and currently
   contains more than 200 package-owned primitives.
2. `Compositional Mechanism Grammar` defines structural operators such as
   add, remove, replace, split, fuse, gate, route, precompute, fixed-point
   approximation, closed-form derivation, propagation-to-constraint
   compilation, and full model synthesis.
3. `Custom Mechanism Escape Hatch` permits a from-scratch component or model
   when it supplies a mathematical and algorithmic definition, typed ports,
   train-only data roles, family-boundary justification, minimal implementation
   plan, matched control, ablation, failure modes, cost, and claim ceiling.

The generic kernel dispatches a `MechanismProgramV1` to a package-owned
provider. A caller can submit a program, but cannot inject its own schema,
registry, compiler, or capability policy into the public compile path.

## Compilation and identity

The compiler performs strict JSON parsing, JSON Schema validation, exact search
space identity checks, primitive and operator resolution, typed port and data
role checks, DAG validation, parameter validation, family-boundary checks,
protocol-impact routing, change-budget checks, and capability derivation.

It emits separate identities for distinct purposes:

- `search_space_digest`: exact-byte closure of the provider and controlled
  resources;
- `mechanism_program_digest`: exact submitted program identity;
- `mechanism_semantics_digest`: structure-derived identity that is invariant to
  opaque component renaming and narrative-only edits;
- `candidate_id`: short deterministic handle derived from the exact program;
- `profile_ref`: exact scientific-profile identity carried through the program,
  resolved IR, and execution binding.

Compile status is fail-closed:

- `VALID_WIRED`: only an exact package-recognized built-in BPR or LightGCN
  runtime signature;
- `VALID_NEEDS_IMPLEMENTATION`: a valid mechanism that still requires a
  candidate-local implementation;
- `PROTOCOL_BRANCH_REQUIRED`: a scientifically meaningful proposal that changes
  a frozen protocol field and cannot enter the current comparison;
- `CAPABILITY_DENIED`, `INVALID`, `UNSUPPORTED`, or `INTERNAL_ERROR`: not
  runnable in the selected space.

A development execution binding must repeat every identity, grant exactly the
compiler-derived capability set, bind that envelope by content digest, and use
exactly these write roots:

```text
recclaw_ext/generated/<candidate_id>/
artifacts/<run_id>/
```

The binding does not grant Stage 1B execution permission, admit evidence, or
promote a candidate. It also cannot turn an unimplemented candidate into
`VALID_WIRED`.

## Anchor expressibility

Test-only fixtures cover BPR-MF, LightGCN, NGCF, SGL, SimGCL/XSimGCL, NCL,
LightGCL, DirectAU, SimpleX, GF-CF, and UltraGCN-style mechanisms. They establish
grammar expressibility rather than research equivalence or metric performance.

The UltraGCN-style fixture contains no message-passing component. It represents
fixed-point structural constraints with distinct user-item and top-K item-item
relations and an explicit propagation-to-constraint rewrite.

Fixtures and anchor names are outside the runtime prompt projection. The
Proposal LLM receives primitive and grammar contracts, not answer recipes.

## Extension to additional recommender spaces

The generic kernel is intentionally family-neutral. A future space should be
added as a new provider package with its own:

- versioned family contract and supported profile kinds;
- primitive registry and compositional grammar;
- program schema and capability policy;
- exact-byte resource closure;
- compiler and safe prompt projection;
- positive expressibility fixtures and single-fault negative fixtures.

It is then added to the static provider catalog. Cross-family composition stays
denied until a versioned bridge explicitly defines type compatibility,
scientific-profile compatibility, identity semantics, and capability effects.
This permits eventual expansion across recommender-system families without
turning one family schema into an unreviewable universal union.

Public kernel and report versions remain stable across providers. Family-specific
IR belongs to the provider, while protocol, budget, implementation, run, result,
and memory records should carry the same space, program, semantics, candidate,
and profile identities end to end.

## Development checks

From the repository root, with the search-space dependencies installed:

```text
python tools/v2/build_bl_icf_space_resources.py --check
python tools/v2/build_bl_icf_anchor_fixtures.py --check
PYTHONPATH=src python -m unittest tests.test_bl_icf_mechanism_space
PYTHONPATH=src python scripts/mechanism_space.py list
```

Resource builders are deterministic. Editing the provider or a controlled
resource requires regenerating the closure and fixtures, then rerunning all
checks.
