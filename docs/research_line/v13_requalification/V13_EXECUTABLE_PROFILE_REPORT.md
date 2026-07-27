# V13 Executable Profile Report

## Frozen profile

- profile: `BL_ICF_EXECUTABLE_PROFILE_V2`
- executable-profile digest:
  `d483faa471c3f26d321daa89c6946ab67a11a95d459d2d5c095c9da419ee5da1`
- full runtime-profile digest:
  `745b67509bb920a2345e99b8ef4c7ba3daffb8a3d62188ecc1e1978a3a002b87`
- total unique executable semantics: 66
- roots: 2
- single-operator programs: 16
- compatible two-operator programs: 48
- non-root discovery candidates: 64

The bounded space covers sampling, ranking objective, graph propagation,
regularization/geometry, self-supervision and package-owned architecture
operators. Programs have depth at most two, no duplicated axis, deterministic
materialization and content-addressed identity. Arbitrary Python is not part of
the search space.

## Qualification

- compile: 66/66;
- CommonExecutionGuard: 66/66;
- materialization and execution-recipe identity: 66/66;
- semantic uniqueness: 66/66;
- recipe uniqueness: 66/66;
- BPR family loss/backward canaries: 4/4;
- LightGCN family loss/backward canaries: 5/5.

A, B and C bind this exact digest. The frozen Pilot and intended Main runtime
profiles are identical; the Pilot is not a four-anchor canary and is not a
21-recipe lower-claim substitute.
