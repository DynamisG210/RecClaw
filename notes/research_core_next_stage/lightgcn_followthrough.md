# LightGCN Follow-Through Micro-Loop

## Purpose

Run a small design and smoke path around the historically strongest positive
frontier: LightGCN residual, norm, and depth-variance mechanisms.

This micro-loop can be prepared alongside the 20/20 pilot, but it must not be
treated as formal evidence or as an algorithm-discovery claim.

## Candidate Scope

Prioritize:

- LightGCN residual, norm, and depth variance
- LightGCN edge-dropout residual-norm repair
- clean residual propagation

Do not include:

- Transformer
- sequence recommendation
- multimodal recommendation
- default Custom architecture

## Procedure

1. Generate 3 LightGCN candidate cards.
2. Run deterministic policy ranking.
3. Select the top 1 candidate.
4. Build an implementation-readiness packet.
5. Run activation smoke.
6. If activation is clean and cheap, run isolated single-seed metric smoke.

## Metrics

Record:

- `NDCG@10`
- `Recall@10`
- `MRR@10`
- `Hit@10`
- `Precision@10`
- `ItemCoverage@10`
- activation diagnostics
- runtime status

## Result Classes

Use exactly one result class:

- `positive_smoke_signal`
- `informative_negative`
- `activation_metric_mismatch`
- `implementation_blocker`

Single-seed smoke can guide the next step, but it is not formal validation.
