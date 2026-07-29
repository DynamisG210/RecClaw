# Research Line functional release

This directory describes the compact functional release frozen from source
commit `b621a5c78e05c5c52ffc55d4090c0707af80cc27`.

## Included

- the integrated Research Line runtime and its canonical three-arm state model;
- Provider physical-call, consumer logical-call, and candidate-instance
  identity separation;
- Arm-private mutable ownership and cross-Arm isolation;
- common materialization, training, result, observation, and round-closure
  paths;
- Research Capability producers, Router, Search Memory, Meta control, and
  frontier admission;
- EvidencePort and Evidence Guard integration;
- executable BL-ICF/model resources, generic workers, and self-contained
  functional regression tests.

## Excluded from the release tree

- milestone and nightly process documents;
- V13-V25 contracts, audit packets, recovery notes, and launch wrappers;
- generated Pilot result databases, logs, Broker outputs, and materialized
  artifacts;
- machine-specific qualification and remote-supervision helpers;
- historical and external-backend integration tests whose prerequisites are
  outside this compact source release.

Those files remain recoverable in the local archive:

`/root/projects/RecClaw_research_line_archive_2026-07-29_b621a5c`

The archive contains 834 files (8,128,507 bytes). Its
`SHA256_MANIFEST.txt` digest is:

`dd148c2bc4da3d1e2a9c36177a7ff213e656fe7212ae131e8268d063b2a146af`

## Evidence boundary

This is a functional code release, not a formal research-result release.

- M6I current-byte engineering isolation reached `P0=0` and `P1=0` in the
  archived development audit.
- A later five-round development Pilot closed its full engineering chain, but
  its small-sample observations do not establish a formal treatment effect.
- No 50-round Effect Pilot or single-seed 100-round formal trial is claimed by
  this release.
- No Main authorization, held-out result, or formal acceptance is implied.

Ongoing experiments continue in a separate worktree and branch. Their outputs
must not be back-written into this frozen functional release.

## Stable entry points

- `scripts/research_line.py`
- `scripts/campaign_train_worker.py`
- `scripts/run_candidate.py`
- `scripts/plan_research_line_comparison.py`
- `src/recclaw_core/experiments/helix_abc_v1/`

The intended release tag is `research-line-functional-v1.0.0`.
