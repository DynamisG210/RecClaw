# Method Change Space

Config-only experimentation is useful for establishing baselines, but it becomes too narrow once the repository has already covered obvious parameter sweeps. At that point, repeated config tuning mostly explores the same hypothesis class instead of opening new recommendation behaviors.

RecClaw therefore needs a compact method-level change taxonomy:

- `loss`: change the optimization target while keeping the training loop intact.
  - Example: add a margin term to BPR loss.
  - Example: add a popularity-aware regularization penalty.
- `sampler`: change how training negatives or candidate items are constructed.
  - Example: switch from uniform negative sampling to harder negatives.
  - Example: mix random negatives with exposure-aware negatives.
- `module`: change the model computation graph or aggregation rule.
  - Example: replace mean layer aggregation in LightGCN with learnable layer weights.
  - Example: add a gated fusion block on top of user and item embeddings.
- `feature`: add or transform input signals without changing the protocol.
  - Example: add item popularity as a side feature.
  - Example: add timestamp-derived recency features before training.
- `pipeline`: change the local orchestration around training and evaluation.
  - Example: add a scripted ablation pass after a candidate run.
  - Example: add a local crash triage step before result collection.

This taxonomy is the future action space for an agentic RecClaw system. It gives the agent explicit categories of repository changes instead of forcing every action into a config edit.

This week only one module-level candidate is scaffolded: `LightGCNLW`, a local LightGCN variant with learnable layer-weight aggregation.
