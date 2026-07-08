# Architecture Lab v0

## Purpose

Architecture Lab v0 is the gated replacement for directly merging
`feat/architecture-exploration-integration`.

It must be built on top of `main` after Research Core v0.1 and must use
Research Core gating. It must not open broad architecture search by default.

## Pilot Shape

Run an 8-card diagnostic pilot:

- 4 architecture cards
- 4 local-extension control cards

The 4 architecture cards are exactly:

- graph-attention propagation
- clean residual propagation
- low-rank or router interaction head
- transformer-style interaction block under the current general-rec protocol

## Hard Boundaries

Do not include:

- SASRec or BERT4Rec
- sequence dataloaders
- temporal split changes
- multimodal feature dependencies
- formal runs
- registry promotion
- RecBole core edits
- default `Custom` base-model access in main action space
- `architecture_first.enabled=true`

## Gates

Each card must pass:

- schema validation
- parent or conceptual-anchor audit
- protocol-drift audit
- allowed implementation-root audit
- runner-contract design audit

At most one architecture card may proceed to import or interface smoke in v0.
No architecture card proceeds to formal evaluation.

## Success Rule

Architecture Lab v0 succeeds only if:

- at least 2 of 4 architecture cards are schema valid
- at least 1 of 4 architecture cards passes interface or import smoke
- there is 0 protocol drift
- there are 0 RecBole core edits
- cost is not materially higher than local-extension controls

Metric improvement is not required and must not be claimed.

## Future Reuse From Old Branch

Allowed ideas to reintroduce later:

- mechanism composition
- novelty claim
- expected failure mode
- ablation parent
- implementation complexity
- base architecture label
- runner-contract checks
- allowed implementation roots

These ideas must be reintroduced through Research Core gates, not by merging
the old branch.
