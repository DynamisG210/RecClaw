# Research Core Next Stage Plan

This directory records human planning notes for the stage after Research Core
v0.1. These notes are not runtime policy, search memory, candidate scoring
input, or evidence of recommender improvement.

Research Core v0.1 supports only this claim:

```text
Research Core v0.1 adds a functional RecClaw-native Candidate Foundry skeleton
with typed memory ingestion, retrieval, deterministic policy ranking, and
fixture replay.
```

Do not claim search-quality improvement until a matched pilot supports it. Do
not claim algorithm discovery until formal multi-seed evidence and rival audit
exist.

## Stage Order

1. Keep `feat/architecture-exploration-integration` out of main and treat it as
   an experimental architecture-lab prototype.
2. Run the Foundry vs old-loop 20/20 matched pilot design before any larger
   loop.
3. In parallel with that design, prepare a LightGCN follow-through micro-loop
   because LightGCN is the strongest historical positive frontier.
4. Rebuild architecture exploration as gated Architecture Lab v0 on top of
   Research Core. Do not merge the old architecture branch.
5. Expand only by evidence: 20/20, then 50/50, then 100/100, and only then
   consider 200/200.

## Files

- `architecture_branch_retrospective.md`
- `foundry_vs_old_loop_20_20_pilot.md`
- `lightgcn_followthrough.md`
- `architecture_lab_v0.md`
