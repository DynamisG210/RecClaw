# Candidate Progress Log

This file is an agent-readable progress log for RecClaw candidate development.
It tracks candidate readiness and blockers. It is not a long-form experiment
report and does not record fabricated results.

Use candidate ids from `notes/candidate_library.md` and
`configs/candidate_registry.yaml`.

## Status Vocabulary

- `idea`
- `spec-ready`
- `implement-ready`
- `implemented`

## Entry Template

```yaml
candidate_id:
category:
base_model:
current_status:
existing_artifacts:
current_blocker:
next_action:
notes:
```

---

## Candidate Records

### cand_lightgcn_layer_weighted_agg

- candidate_id: cand_lightgcn_layer_weighted_agg
- category: Representation & Interaction
- base_model: LightGCN
- current_status: implemented
- existing_artifacts: `recclaw_ext/models/lightgcn_lw.py`, `configs/candidates/cand_lightgcn_layer_weighted_agg.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: local model scaffold exists, but current run path does not register or select `LightGCNLW`
- next_action: add local model discovery or a candidate-specific runtime path in a later integration task
- notes: no RecBole core change should be needed; do not treat this as a completed experiment result

### cand_bpr_margin_loss

- candidate_id: cand_bpr_margin_loss
- category: Objective & Optimization
- base_model: BPR
- current_status: implement-ready
- existing_artifacts: `recclaw_ext/losses/bpr_margin.py`, `configs/candidates/cand_bpr_margin_loss.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: local loss stub exists, but RecBole BPR does not consume `loss_type: BPR_MARGIN`
- next_action: wire the local loss through a local BPR extension or external training adapter
- notes: stub only; no training result has been produced

### cand_bpr_popularity_regularized

- candidate_id: cand_bpr_popularity_regularized
- category: Objective & Optimization
- base_model: BPR
- current_status: implement-ready
- existing_artifacts: `recclaw_ext/losses/bpr_popreg.py`, `configs/candidates/cand_bpr_popularity_regularized.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: popularity signal and loss integration are not connected to the BPR training path
- next_action: define how negative item popularity is passed into the local loss
- notes: local experimentation stub; keep the regularization simple until first runnable integration

### cand_bpr_hard_negative_mix

- candidate_id: cand_bpr_hard_negative_mix
- category: Bias & Sample Construction
- base_model: BPR
- current_status: implement-ready
- existing_artifacts: `recclaw_ext/samplers/mixed_negative.py`, `configs/candidates/cand_bpr_hard_negative_mix.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: mixed sampler stub is not connected to RecBole sampler or trainer flow
- next_action: decide whether to integrate through a local sampler wrapper or a local BPR candidate runner
- notes: does not implement online hard negative mining; current design is uniform plus popularity-biased mixing

### cand_lightgcn_small_embedding

- candidate_id: cand_lightgcn_small_embedding
- category: Efficiency & Serving Constraints
- base_model: LightGCN
- current_status: implement-ready
- existing_artifacts: `configs/candidates/cand_lightgcn_small_embedding.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: none at artifact level; candidate still needs a controlled run against the fixed LightGCN baseline
- next_action: run as a config-only LightGCN candidate when experiment execution resumes
- notes: no local code stub is needed

### cand_lightgcn_shallow_layers

- candidate_id: cand_lightgcn_shallow_layers
- category: Efficiency & Serving Constraints
- base_model: LightGCN
- current_status: implement-ready
- existing_artifacts: `configs/candidates/cand_lightgcn_shallow_layers.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: none at artifact level; candidate still needs a controlled run against the fixed LightGCN baseline
- next_action: run as a config-only LightGCN candidate when experiment execution resumes
- notes: no local code stub is needed

### cand_bpr_popularity_aware_negative

- candidate_id: cand_bpr_popularity_aware_negative
- category: Bias & Sample Construction
- base_model: BPR
- current_status: implement-ready
- existing_artifacts: `recclaw_ext/samplers/popularity_aware.py`, `configs/candidates/cand_bpr_popularity_aware_negative.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: sampler stub exists, but item popularity construction and sampler integration are not wired
- next_action: define the item-popularity vector source and connect it through a local sampling path
- notes: stub only; do not modify RecBole core for this step

### cand_bpr_long_tail_reweight

- candidate_id: cand_bpr_long_tail_reweight
- category: Bias & Sample Construction
- base_model: BPR
- current_status: implement-ready
- existing_artifacts: `recclaw_ext/losses/bpr_long_tail.py`, `configs/candidates/cand_bpr_long_tail_reweight.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: item weighting signal is not yet connected to BPR loss computation
- next_action: define tail-aware item weights and pass them into the local loss in a later integration task
- notes: stub only; keep weighting interpretable for the first runnable version

### cand_lightgcn_residual_layer_mix

- candidate_id: cand_lightgcn_residual_layer_mix
- category: Representation & Interaction
- base_model: LightGCN
- current_status: implement-ready
- existing_artifacts: `recclaw_ext/models/lightgcn_residual.py`, `configs/candidates/cand_lightgcn_residual_layer_mix.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: local model scaffold exists, but current run path does not register or select `LightGCNResidualMix`
- next_action: add local model discovery or a candidate-specific runtime path in a later integration task
- notes: no RecBole core change should be needed

### cand_lightgcn_rank_aware_loss

- candidate_id: cand_lightgcn_rank_aware_loss
- category: Objective & Optimization
- base_model: LightGCN
- current_status: implement-ready
- existing_artifacts: `recclaw_ext/losses/rank_aware.py`, `configs/candidates/cand_lightgcn_rank_aware_loss.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: rank-aware loss stub exists, but pair weights and LightGCN loss integration are not wired
- next_action: define the pair-weight signal and connect the loss through a local LightGCN extension
- notes: stub only; no metric-aligned training result has been produced
