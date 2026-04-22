# Candidate Library

## Purpose

This document is the current working candidate library for RecClaw's method-level experimentation stage.

Its purpose is to turn the method change space into a concrete pool of executable recommendation candidates. Each candidate is treated as a compact research unit with:

- a recommendation-specific problem,
- a hypothesis worth testing,
- a minimal feasible change,
- an implementation type,
- an expected gain,
- a key risk,
- and a current readiness status.

This document is intentionally narrower than the full method space. It does not try to cover every possible direction at once. Instead, it records the first batch of candidates that are most relevant to the current RecClaw stage and most compatible with the current baseline environment.

At this stage, the candidate library is centered on:

- `BPR`
- `LightGCN`
- lightweight rerank-style post-processing candidates

This library should be used as the immediate source for prioritization, specification, and near-term implementation.

---

## Status Definition

### idea
The direction is meaningful, but still broad. It is not yet specific enough for direct implementation.

### spec-ready
The recommendation problem, hypothesis, and minimal change are clear enough for prioritization and method design.

### implement-ready
The candidate is already specific enough that implementation can begin directly with limited additional design work.

### implemented
A code/config prototype or local scaffold already exists in the repository.

---

## Scope Note

This version of the candidate library intentionally focuses on the first five categories:

1. Bias & Sample Construction
2. Representation & Interaction
3. Objective & Optimization
4. Result Distribution Quality
5. Efficiency & Serving Constraints

The `Side Information & Cold Start` category is intentionally deferred in this version because it is not the current focus of near-term implementation.

---

# 1. Bias & Sample Construction

## cand_bpr_uniform_neg_baseline_note

- category: Bias & Sample Construction
- base_model: BPR
- rs_problem: the current default negative sampling setup needs to be explicitly recorded as the baseline reference for later method comparison
- hypothesis: the current uniform negative construction serves as the baseline supervision pattern against which later sampling improvements should be judged
- minimal_change: no method change; keep the current default setting as the reference point
- implementation_type: config
- expected_gain: clearer experimental grounding for later sampling-oriented candidate comparison
- risk: this candidate does not improve recommendation quality by itself and only serves as a baseline note
- status: spec-ready

## cand_bpr_hard_negative_mix

- category: Bias & Sample Construction
- base_model: BPR
- rs_problem: purely random negative samples are often too weak and provide insufficient top-k ranking pressure
- hypothesis: if BPR is trained with a mixed pool that includes harder negatives, the model may learn a sharper ranking boundary and improve top-k recommendation quality
- minimal_change: replace purely uniform negative sampling with a mixed negative strategy that combines random negatives and harder negatives
- implementation_type: sampler
- expected_gain: stronger pairwise ranking signal and better top-k discrimination
- risk: harder negatives may include noisy pseudo-negatives and may make optimization less stable
- status: implement-ready

### Minimal implementation path

- files to edit: `recclaw_ext/samplers/mixed_negative.py`, `configs/candidates/cand_bpr_hard_negative_mix.yaml`
- whether RecBole core changes are needed: no
- whether local extension is enough: yes for a local candidate stub; full trainer/sampler wiring can be deferred
- expected experiment comparison: compare the mixed-negative BPR candidate against the fixed BPR uniform-negative baseline on `ml-1m`

## cand_bpr_popularity_aware_negative

- category: Bias & Sample Construction
- base_model: BPR
- rs_problem: the negative sampling distribution does not match realistic exposure structure and may introduce sampling bias
- hypothesis: if negative sampling is adjusted according to item popularity, BPR may learn a less biased preference boundary and reduce exposure-related distortion
- minimal_change: modify the negative sampler so that item popularity affects negative selection probability
- implementation_type: sampler
- expected_gain: more realistic training negatives and reduced mismatch between training signal and recommendation environment
- risk: popularity-aware negatives may over-penalize head items and hurt overall relevance if designed too aggressively
- status: implement-ready

### Minimal implementation path

- files to edit: `recclaw_ext/samplers/popularity_aware.py`, `configs/candidates/cand_bpr_popularity_aware_negative.yaml`
- whether RecBole core changes are needed: no
- whether local extension is enough: yes for a local sampler scaffold; trainer/sampler integration can be deferred
- expected experiment comparison: compare popularity-aware BPR negatives against the fixed BPR uniform-negative baseline on `ml-1m`

## cand_bpr_long_tail_reweight

- category: Bias & Sample Construction
- base_model: BPR
- rs_problem: head items dominate training, while long-tail items receive too little effective optimization pressure
- hypothesis: if long-tail-related training cases are reweighted, the model may improve tail sensitivity and reduce head-dominated learning behavior
- minimal_change: add long-tail-aware weighting to samples or pairwise loss terms
- implementation_type: loss
- expected_gain: better sensitivity to tail items and more balanced learning pressure across the item space
- risk: excessive reweighting may hurt performance on frequently interacted items and lower the main ranking metric
- status: implement-ready

### Minimal implementation path

- files to edit: `recclaw_ext/losses/bpr_long_tail.py`, `configs/candidates/cand_bpr_long_tail_reweight.yaml`
- whether RecBole core changes are needed: no
- whether local extension is enough: yes for a local reweighting loss stub; BPR loss wiring can be deferred
- expected experiment comparison: compare long-tail reweighted BPR against the fixed BPR baseline on `ml-1m`

## cand_lightgcn_debiased_negative_sampling

- category: Bias & Sample Construction
- base_model: LightGCN
- rs_problem: in graph recommendation, uniform negatives are often too weak to teach fine-grained ranking boundaries
- hypothesis: if LightGCN uses stronger or more debiased negative construction, graph embeddings may separate relevant and irrelevant items more effectively
- minimal_change: replace the default negative construction with a stronger or more informative sampler for LightGCN training
- implementation_type: sampler
- expected_gain: sharper ranking signal in graph-based recommendation and stronger fine-grained discrimination
- risk: harder negatives may increase training instability or amplify noisy pseudo-negative effects
- status: implement-ready

### Minimal implementation path

- files to edit: `recclaw_ext/samplers/debiased_negative.py`, `configs/candidates/cand_lightgcn_debiased_negative_sampling.yaml`
- whether RecBole core changes are needed: no
- whether local extension is enough: yes for a local sampler scaffold; LightGCN sampler/trainer integration can be deferred
- expected experiment comparison: compare debiased-negative LightGCN against the fixed LightGCN baseline on `ml-1m`

---

# 2. Representation & Interaction

## cand_lightgcn_layer_weighted_agg

- category: Representation & Interaction
- base_model: LightGCN
- rs_problem: different propagation layers should not necessarily contribute through a simple uniform mean
- hypothesis: if LightGCN learns layer-wise aggregation weights, it may use shallow and deep collaborative information more effectively than uniform averaging
- minimal_change: replace uniform layer aggregation with learnable layer weights in the local LightGCN variant
- implementation_type: module
- expected_gain: more adaptive use of propagation depth and better top-k recommendation quality with limited architectural disruption
- risk: layer weighting may overfit or collapse toward a small subset of layers
- status: implemented

### Minimal implementation path

- files to edit: `recclaw_ext/models/lightgcn_lw.py`, `configs/candidates/cand_lightgcn_layer_weighted_agg.yaml`
- whether RecBole core changes are needed: no
- whether local extension is enough: yes; this candidate is based on the existing local `LightGCNLW` scaffold
- expected experiment comparison: compare `LightGCNLW` against the fixed LightGCN baseline on `ml-1m`

## cand_lightgcn_residual_layer_mix

- category: Representation & Interaction
- base_model: LightGCN
- rs_problem: multi-layer propagation information is fused too rigidly, which may weaken layer interaction and reduce representation quality
- hypothesis: if residual layer mixing is added, LightGCN may use propagation information more flexibly and preserve useful shallow signals better
- minimal_change: add residual-style mixing to the forward aggregation of LightGCN layer outputs
- implementation_type: module
- expected_gain: better layer information utilization and reduced over-smoothing risk
- risk: the residual path may dominate too strongly and weaken the benefit of propagation
- status: implement-ready

### Minimal implementation path

- files to edit: `recclaw_ext/models/lightgcn_residual.py`, `configs/candidates/cand_lightgcn_residual_layer_mix.yaml`
- whether RecBole core changes are needed: no
- whether local extension is enough: yes for a local model scaffold; model discovery/run wiring can be deferred
- expected experiment comparison: compare residual-mix LightGCN against the fixed LightGCN baseline on `ml-1m`

## cand_lightgcn_user_item_gate

- category: Representation & Interaction
- base_model: LightGCN
- rs_problem: user-item interaction remains too weak and too uniform after representation learning
- hypothesis: if a lightweight gate is introduced for user-item interaction, matching expressiveness may improve beyond simple fixed interaction behavior
- minimal_change: add a lightweight gating mechanism in the interaction head before final scoring
- implementation_type: module
- expected_gain: richer matching behavior and more selective user-item interaction
- risk: the gain may be limited on the current dataset, while the added module may complicate the model path
- status: idea

## cand_bpr_mlp_head

- category: Representation & Interaction
- base_model: BPR
- rs_problem: pure dot-product scoring is limited in expressive power and may miss nonlinear compatibility patterns
- hypothesis: if a minimal MLP head is added on top of user-item embeddings, BPR may capture nonlinear matching signals beyond simple inner product
- minimal_change: replace or augment the current scoring head with a very small MLP interaction head
- implementation_type: module
- expected_gain: stronger nonlinear matching expressiveness with a still-lightweight extension
- risk: the added head may make BPR less clean as a baseline and may not bring enough gain relative to added complexity
- status: idea

---

# 3. Objective & Optimization

## cand_bpr_margin_loss

- category: Objective & Optimization
- base_model: BPR
- rs_problem: the ranking boundary between positive and negative items is often not strong enough
- hypothesis: if BPR is made margin-aware, the model may enforce clearer positive-negative separation and improve ranking quality
- minimal_change: add a margin-aware reformulation or explicit margin term on top of the standard BPR loss
- implementation_type: loss
- expected_gain: stronger ranking boundary and potentially better NDCG@10
- risk: an overly strong margin may reduce optimization smoothness and destabilize training
- status: implement-ready

### Minimal implementation path

- files to edit: `recclaw_ext/losses/bpr_margin.py`, `configs/candidates/cand_bpr_margin_loss.yaml`
- whether RecBole core changes are needed: no
- whether local extension is enough: yes for a local loss stub; runtime selection from RecBole's stock BPR path can be wired later
- expected experiment comparison: compare the margin-aware BPR candidate against the fixed BPR baseline on `ml-1m`

## cand_bpr_popularity_regularized

- category: Objective & Optimization
- base_model: BPR
- rs_problem: popular items attract too much optimization pressure and dominate training
- hypothesis: if popularity-aware regularization is added, the model may reduce head dominance and learn more balanced preference signals
- minimal_change: add a popularity-aware regularization term to the BPR objective
- implementation_type: loss
- expected_gain: less head-item domination and more balanced ranking behavior
- risk: suppressing popularity too strongly may hurt relevance on genuinely useful head items
- status: implement-ready

### Minimal implementation path

- files to edit: `recclaw_ext/losses/bpr_popreg.py`, `configs/candidates/cand_bpr_popularity_regularized.yaml`
- whether RecBole core changes are needed: no
- whether local extension is enough: yes for a local loss stub; full model-loss wiring can be deferred
- expected experiment comparison: compare the popularity-regularized BPR candidate against the fixed BPR baseline on `ml-1m`

## cand_bpr_norm_constrained

- category: Objective & Optimization
- base_model: BPR
- rs_problem: embedding scale may drift during training and hurt stability or generalization
- hypothesis: if embedding norms are explicitly constrained, BPR may train more stably and produce more controlled representations
- minimal_change: add an embedding norm constraint or norm-aware regularization term
- implementation_type: loss
- expected_gain: improved optimization stability and reduced scale drift
- risk: excessive norm control may weaken useful representation flexibility
- status: implement-ready

### Minimal implementation path

- files to edit: `recclaw_ext/losses/bpr_norm.py`, `configs/candidates/cand_bpr_norm_constrained.yaml`
- whether RecBole core changes are needed: no
- whether local extension is enough: yes for a local norm-penalty loss stub; BPR loss wiring can be deferred
- expected experiment comparison: compare norm-constrained BPR against the fixed BPR baseline on `ml-1m`

## cand_lightgcn_aux_alignment_loss

- category: Objective & Optimization
- base_model: LightGCN
- rs_problem: graph propagation representation learning is optimized under a single objective and may not sufficiently coordinate layer behavior
- hypothesis: if an auxiliary alignment loss is introduced, LightGCN may learn more coherent propagation-layer representations
- minimal_change: add an auxiliary alignment loss across layer outputs or representation views during training
- implementation_type: loss
- expected_gain: better propagation consistency and potentially more stable graph representation learning
- risk: alignment pressure may over-constrain useful diversity across layers
- status: implement-ready

### Minimal implementation path

- files to edit: `recclaw_ext/losses/alignment.py`, `configs/candidates/cand_lightgcn_aux_alignment_loss.yaml`
- whether RecBole core changes are needed: no
- whether local extension is enough: yes for a local auxiliary loss stub; LightGCN training-loss wiring can be deferred
- expected experiment comparison: compare auxiliary-alignment LightGCN against the fixed LightGCN baseline on `ml-1m`

## cand_lightgcn_rank_aware_loss

- category: Objective & Optimization
- base_model: LightGCN
- rs_problem: the training objective is not closely enough aligned with NDCG-oriented top-k ranking behavior
- hypothesis: if a rank-aware objective is used, LightGCN may optimize more directly for top-k recommendation quality
- minimal_change: replace or augment the current training loss with a rank-aware ranking objective
- implementation_type: loss
- expected_gain: improved alignment between optimization behavior and evaluation target
- risk: rank-aware losses may be harder to optimize and more sensitive to design details
- status: implement-ready

### Minimal implementation path

- files to edit: `recclaw_ext/losses/rank_aware.py`, `configs/candidates/cand_lightgcn_rank_aware_loss.yaml`
- whether RecBole core changes are needed: no
- whether local extension is enough: yes for a local rank-aware loss stub; LightGCN loss wiring can be deferred
- expected experiment comparison: compare rank-aware LightGCN against the fixed LightGCN baseline on `ml-1m`

---

# 4. Result Distribution Quality

## cand_rerank_popularity_penalty

- category: Result Distribution Quality
- base_model: BPR / LightGCN
- rs_problem: recommendation results are too concentrated on head items
- hypothesis: if a popularity-based penalty is applied during reranking, the final recommendation distribution may become less head-concentrated
- minimal_change: apply a post-hoc popularity penalty in the rerank stage
- implementation_type: rerank
- expected_gain: broader recommendation distribution and reduced head-item concentration
- risk: the main relevance metric may drop if popularity is penalized too strongly
- status: spec-ready

### Minimal implementation path

- files to edit: `recclaw_ext/features/rerank_adjustments.py`, `configs/candidates/cand_rerank_popularity_penalty.yaml`
- whether RecBole core changes are needed: no
- whether local extension is enough: yes for a local score-adjustment stub; rerank evaluation flow is deferred
- expected experiment comparison: compare popularity-penalty reranking against the fixed BPR and LightGCN baselines on `ml-1m`

## cand_rerank_coverage_boost

- category: Result Distribution Quality
- base_model: BPR / LightGCN
- rs_problem: overall item coverage is low
- hypothesis: if a coverage-oriented rerank boost is added, the system may improve coverage with only limited precision loss
- minimal_change: add a rerank-stage score adjustment that boosts under-exposed items when relevance differences are small
- implementation_type: rerank
- expected_gain: improved item coverage and healthier output distribution
- risk: even mild reranking may still hurt NDCG@10 if the coverage pressure is not well controlled
- status: spec-ready

### Minimal implementation path

- files to edit: `recclaw_ext/features/rerank_adjustments.py`, `configs/candidates/cand_rerank_coverage_boost.yaml`
- whether RecBole core changes are needed: no
- whether local extension is enough: yes for a local score-adjustment stub; rerank evaluation flow is deferred
- expected experiment comparison: compare coverage-boost reranking against the fixed BPR and LightGCN baselines on `ml-1m`

---

# 5. Efficiency & Serving Constraints

## cand_lightgcn_small_embedding

- category: Efficiency & Serving Constraints
- base_model: LightGCN
- rs_problem: embedding dimension may be larger than necessary and increase model complexity without proportional gain
- hypothesis: if a lower-dimensional embedding is used, LightGCN may preserve near-baseline quality while reducing complexity
- minimal_change: reduce embedding dimension through config change only
- implementation_type: config
- expected_gain: lower representation cost and a clearer view of parameter redundancy
- risk: too small an embedding may sharply reduce ranking quality
- status: implement-ready

### Minimal implementation path

- files to edit: `configs/candidates/cand_lightgcn_small_embedding.yaml`
- whether RecBole core changes are needed: no
- whether local extension is enough: yes; this is a config-only LightGCN candidate
- expected experiment comparison: compare smaller-embedding LightGCN against the fixed LightGCN baseline on `ml-1m`

## cand_lightgcn_shallow_layers

- category: Efficiency & Serving Constraints
- base_model: LightGCN
- rs_problem: graph propagation depth may be unnecessarily high and add extra complexity
- hypothesis: if a shallower propagation structure is used, LightGCN may achieve similar effectiveness with lower complexity
- minimal_change: reduce propagation layer number through config change only
- implementation_type: config
- expected_gain: lower graph propagation cost and clearer quality-complexity tradeoff
- risk: useful multi-hop collaborative signal may be lost when the model becomes too shallow
- status: implement-ready

### Minimal implementation path

- files to edit: `configs/candidates/cand_lightgcn_shallow_layers.yaml`
- whether RecBole core changes are needed: no
- whether local extension is enough: yes; this is a config-only LightGCN candidate
- expected experiment comparison: compare shallow-layer LightGCN against the fixed LightGCN baseline on `ml-1m`

---

## Near-term shortlist

Among the candidates above, the strongest near-term implementation targets are:

- `cand_bpr_hard_negative_mix`
- `cand_bpr_popularity_aware_negative`
- `cand_bpr_long_tail_reweight`
- `cand_bpr_margin_loss`
- `cand_bpr_popularity_regularized`
- `cand_lightgcn_layer_weighted_agg`
- `cand_lightgcn_residual_layer_mix`
- `cand_lightgcn_rank_aware_loss`
- `cand_lightgcn_small_embedding`
- `cand_lightgcn_shallow_layers`

These candidates are currently the best bridge from method-space definition to executable method-level experimentation.
