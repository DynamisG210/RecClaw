# Architecture Branch Retrospective

## Decision

Do not merge `feat/architecture-exploration-integration` into `main`.

Keep it as an experimental architecture-lab prototype. The direction is useful,
but the branch directly expands default runtime permissions before Research
Core has shown a useful-signal advantage over the old loop.

## What Is Useful

The branch contains concepts that should be rebuilt later behind Research Core
gates:

- `mechanism_composition`
- `novelty_claim`
- `expected_failure_mode`
- `ablation_parent`
- `implementation_complexity`
- `base_architecture`
- runner-contract checks
- allowed implementation roots

These concepts improve proposal review and traceability. They are not, by
themselves, evidence that search quality improved.

## What Must Not Enter Main By Default

Do not cherry-pick broad default enablement:

- `Custom` as a default base model in `configs/action_space.yaml`
- `architecture_first.enabled=true`
- architecture formal-run quotas
- automatic `architecture_required` implementation
- sequence dataloaders or sequence protocol hooks
- multimodal feature dependencies
- Transformer parameters in the current general-rec action space
- RecBole core edits

## Retrospective Summary

Architecture openness is necessary for RecClaw's long-term research scope.
However, direct action-space expansion did not solve search quality and creates
protocol, attribution, and budget risk. Future architecture work must be
scheduled by Research Core and evaluated first as a gated lab, not as default
mainline behavior.
