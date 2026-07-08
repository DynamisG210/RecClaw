# Foundry vs Old Loop 20/20 Matched Pilot

## Purpose

Test whether Research Core Candidate Foundry produces more useful research
signal than the old RecClaw loop under the same candidate budget.

This is not a formal recommender evaluation. It must not write
`results/results.csv`, promote registry candidates, run M5, or claim metric
improvement.

## Arms

Arm A: old RecClaw loop.

- candidate source: existing `scripts/agent.py` and action-space proposal path
- memory: old agent memory only, or structured Research Core memory disabled
- policy: original loop logic
- budget: 20 candidates

Arm B: Research Core Candidate Foundry.

- candidate source: memory and mechanism-grounded candidate cards
- memory: typed Research Core memory enabled
- policy: deterministic policy ranking
- budget: 20 candidates

## Candidate Bounds

Allowed first-pilot families:

- BPR
- LightGCN
- ADMMSLIM diagnostic only if already scaffolded

Disallowed:

- `Custom` architecture
- Transformer or sequence recommendation
- multimodal recommendation
- formal runs
- multi-seed formal validation
- registry promotion
- M5
- runtime-kernel extraction

Recommended Foundry allocation:

- 8 LightGCN cards
- 6 BPR cards
- 4 ADMMSLIM diagnostic cards
- 2 policy-selected free slots

The old-loop arm should use its own normal strategy for 20 candidates.

## Candidate Gates

Each candidate can advance only through these gates:

1. Proposal or candidate-card validation.
2. Parent-anchor, duplicate, and memory-collision audit.
3. Implementation-readiness packet.
4. Activation smoke for selected top candidates only.
5. Isolated metric smoke only when activation is clean and cheap.

Stop a candidate at the first failed gate and preserve the failure as diagnostic
evidence, not formal evidence.

## Metrics

Report these metrics for both arms:

- `schema_valid_rate`
- `parent_anchor_rate`
- `memory_citation_rate`
- `duplicate_collision_rate`
- `readiness_pass_rate`
- `activation_pass_rate`
- `metric_smoke_attempt_rate`
- `metric_smoke_signal_rate`
- `negative_information_value`
- `implementation_blocker_rate`
- `cost_per_useful_signal`
- `best_metric_smoke_delta`

## Useful Research Signal

Count a candidate as useful if at least one is true:

- it creates a clear parent-matched metric-smoke plan
- activation smoke proves a mechanism active or inactive
- metric smoke produces positive, informative negative, or activation-metric
  mismatch evidence
- it excludes a repeated failed mechanism
- it separates an implementation blocker from mechanism failure
- it writes reusable typed memory from the result

Do not count:

- fluent proposal prose alone
- reviewer score alone
- metric rows without a parent anchor
- protocol-polluted results
- conclusions that do not change the next search step

## Pass Rule

Foundry passes only if all are true:

- `useful_signal_count >= 6/20`
- `useful_signal_count >= 1.5x old_loop`
- `duplicate_collision_rate <= old_loop`
- `implementation_blocker_rate <= old_loop`
- no protocol violation
- no claim-boundary violation

If Foundry does not pass, do not expand. Revise memory, policy, candidate
schema, or readiness gates first.

## Expansion Rule

Do not jump directly to 100/100, 100/200, or 200/200.

Allowed route:

1. 20/20 matched pilot.
2. 50/50 validation only if 20/20 shows a useful-signal advantage.
3. 100/100 only if 50/50 is stable.
4. 200/200 only if 100/100 shows stable advantage and the result remains
   attributable.
