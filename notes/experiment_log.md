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

## Current Layout Decision

- Candidate execution has three runner types: `config_only`, `model`, and `posthoc`.
- Training-time changes, including loss and sampler ideas, enter through `recclaw_ext.models`.
- Internal loss and sampler helpers live under `recclaw_ext.models._losses` and `recclaw_ext.models._samplers`.
- `recclaw_ext.losses`, `recclaw_ext.samplers`, `recclaw_ext.features`, and `recclaw_ext.rerank` have been removed.
- `posthoc` is reserved for trained-score adjustment flows such as popularity penalty or coverage boost.

## Entry Template

```yaml
candidate_id:
category:
base_model:
current_status:
runner_type:
wired:
entrypoint:
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
- runner_type: model
- wired: true
- entrypoint: `recclaw_ext.models:LightGCNLW`
- existing_artifacts: `recclaw_ext/models/lightgcn_lw.py`, `configs/candidates/cand_lightgcn_layer_weighted_agg.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: none at wiring level
- next_action: tune only under controlled, comparable baseline budgets
- notes: local model changes layer aggregation in `forward`

### cand_lightgcn_residual_layer_mix

- candidate_id: cand_lightgcn_residual_layer_mix
- category: Representation & Interaction
- base_model: LightGCN
- current_status: implemented
- runner_type: model
- wired: true
- entrypoint: `recclaw_ext.models:LightGCNResidualMix`
- existing_artifacts: `recclaw_ext/models/lightgcn_residual.py`, `configs/candidates/cand_lightgcn_residual_layer_mix.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: none at wiring level
- next_action: tune only under controlled, comparable baseline budgets
- notes: local model mixes ego embeddings into layer aggregation

### cand_bpr_margin_loss

- candidate_id: cand_bpr_margin_loss
- category: Objective & Optimization
- base_model: BPR
- current_status: implemented
- runner_type: model
- wired: true
- entrypoint: `recclaw_ext.models:BPRMargin`
- existing_artifacts: `recclaw_ext/models/bpr_margin.py`, `recclaw_ext/models/_losses.py`, `configs/candidates/cand_bpr_margin_loss.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: none at wiring level
- next_action: evaluate with a fair BPR baseline budget beyond 1-epoch smoke tests
- notes: `margin` is consumed by `BPRMargin.calculate_loss`

### cand_lightgcn_small_embedding

- candidate_id: cand_lightgcn_small_embedding
- category: Efficiency & Serving Constraints
- base_model: LightGCN
- current_status: implemented
- runner_type: config_only
- wired: true
- entrypoint: `recbole.model.general_recommender.lightgcn:LightGCN`
- existing_artifacts: `configs/candidates/cand_lightgcn_small_embedding.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: none at wiring level
- next_action: run only against a comparable LightGCN baseline budget
- notes: no local model code is needed

### cand_lightgcn_shallow_layers

- candidate_id: cand_lightgcn_shallow_layers
- category: Efficiency & Serving Constraints
- base_model: LightGCN
- current_status: implemented
- runner_type: config_only
- wired: true
- entrypoint: `recbole.model.general_recommender.lightgcn:LightGCN`
- existing_artifacts: `configs/candidates/cand_lightgcn_shallow_layers.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: none at wiring level
- next_action: run only against a comparable LightGCN baseline budget
- notes: no local model code is needed

### cand_bpr_hard_negative_mix

- candidate_id: cand_bpr_hard_negative_mix
- category: Bias & Sample Construction
- base_model: BPR
- current_status: implement-ready
- runner_type: model
- wired: false
- entrypoint: `recclaw_ext.models:BPRHardNegative`
- existing_artifacts: `recclaw_ext/models/_samplers.py`, `configs/candidates/cand_bpr_hard_negative_mix.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: model subclass does not exist yet
- next_action: implement `BPRHardNegative` and consume `hard_negative_ratio` inside `calculate_loss`
- notes: sampler logic should stay internal to the model-entry path

### cand_bpr_popularity_aware_negative

- candidate_id: cand_bpr_popularity_aware_negative
- category: Bias & Sample Construction
- base_model: BPR
- current_status: implement-ready
- runner_type: model
- wired: false
- entrypoint: `recclaw_ext.models:BPRPopularityAwareNegative`
- existing_artifacts: `recclaw_ext/models/_samplers.py`, `configs/candidates/cand_bpr_popularity_aware_negative.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: model subclass does not exist yet
- next_action: define item-popularity vector source and consume `popularity_alpha`
- notes: do not reintroduce a top-level sampler package

### cand_bpr_long_tail_reweight

- candidate_id: cand_bpr_long_tail_reweight
- category: Bias & Sample Construction
- base_model: BPR
- current_status: implement-ready
- runner_type: model
- wired: false
- entrypoint: `recclaw_ext.models:BPRLongTailReweight`
- existing_artifacts: `recclaw_ext/models/_losses.py`, `configs/candidates/cand_bpr_long_tail_reweight.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: model subclass does not exist yet
- next_action: implement model-level tail weights and consume `tail_weight_alpha`
- notes: loss helper exists; executable candidate still needs the model class

### cand_bpr_popularity_regularized

- candidate_id: cand_bpr_popularity_regularized
- category: Objective & Optimization
- base_model: BPR
- current_status: implement-ready
- runner_type: model
- wired: false
- entrypoint: `recclaw_ext.models:BPRPopularityRegularized`
- existing_artifacts: `recclaw_ext/models/_losses.py`, `configs/candidates/cand_bpr_popularity_regularized.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: model subclass does not exist yet
- next_action: implement `calculate_loss` and consume `lambda_pop`
- notes: loss helper exists; executable candidate still needs the model class

### cand_lightgcn_rank_aware_loss

- candidate_id: cand_lightgcn_rank_aware_loss
- category: Objective & Optimization
- base_model: LightGCN
- current_status: implement-ready
- runner_type: model
- wired: false
- entrypoint: `recclaw_ext.models:LightGCNRankAware`
- existing_artifacts: `recclaw_ext/models/_losses.py`, `configs/candidates/cand_lightgcn_rank_aware_loss.yaml`, `configs/candidate_registry.yaml`, `notes/candidate_library.md`
- current_blocker: model subclass does not exist yet
- next_action: implement rank-aware LightGCN loss path and consume `rank_weight_alpha`
- notes: loss helper exists; executable candidate still needs the model class
