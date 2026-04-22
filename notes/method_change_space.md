# Method Change Space

## Purpose

Once baseline reproduction and basic config sweeps are in place, recommendation research can no longer be driven by parameter tuning alone. Repeatedly changing learning rate, embedding size, or layer number may still produce local variation, but it does not substantially expand the hypothesis space. What RecClaw needs at this stage is not a larger set of knobs, but a clearer map of **where meaningful recommendation improvements come from**.

This document defines that map.

The goal of the method change space is to turn method-level experimentation from a vague notion of “editing some code” into a structured recommendation research space. Each category below should be understood in two ways at once:

- as a **family of recommendation problems** worth studying;
- as a **family of executable interventions** that can be realized through local changes in loss, sampling, modules, features, configs, or lightweight pipeline logic.

In other words, these categories are not merely code-edit locations. They are directions in recommendation research that also preserve executability.

---

## 1. Bias & Sample Construction

### What it covers

This category covers improvements that change **what training signal the model sees** and **how that signal is constructed**. In recommendation, observed interactions are not neutral supervision: they are shaped by exposure, popularity, sparsity, and missing feedback. As a result, a model may fail not because its architecture is too weak, but because the supervision it receives is systematically distorted.

Method changes in this category therefore focus on the construction of positives, negatives, sample weights, and effective training subsets. The core question is not “How do we make the model bigger?” but rather:

> Are we teaching the model with the right evidence?

### Target recommendation problems

This category primarily targets problems such as:

- **popularity bias**: head items dominate observation and optimization;
- **exposure bias**: training only reflects what users were shown, not what they could have preferred;
- **weak negatives**: uniformly sampled negatives are often too easy and provide limited ranking pressure;
- **sample imbalance**: different users and items contribute very unevenly to training;
- **pseudo-negative noise**: some unobserved items are treated as negatives even when they are potentially relevant;
- **long-tail undertraining**: rare items receive too little effective learning signal;
- **sequence truncation or alignment bias**: in temporal settings, the chosen construction procedure can distort what the model learns.

These are all cases where the bottleneck lies in the supervision process rather than purely in representation capacity.

### Typical implementation positions

In practice, this category most naturally lands in places that define or reshape the training signal:

- **sampler-level changes**  
  such as hard negative sampling, mixed negative pools, popularity-aware sampling, or exposure-sensitive negative construction;
- **loss-level changes**  
  such as sample reweighting, long-tail reweighting, or debiasing penalties tied to frequency or exposure proxies;
- **feature-level additions**  
  such as injecting item popularity or frequency buckets as auxiliary signals that help the model interpret biased observations;
- **pipeline-level preprocessing**  
  such as filtering, balancing, regrouping, or staging how samples are prepared before training;
- **config-level control**  
  such as switching sampling distributions or setting weighting coefficients.

So while this category is conceptually about recommendation bias, its executable form is usually some combination of **sampler + loss + lightweight data-facing logic**.

### Current-stage feasible directions

At the current stage, the most feasible directions are local training-time changes that preserve the current benchmark protocol and baseline environment. Strong directions include:

- **hard negative**
- **mixed negative**
- **popularity-aware negative**
- **long-tail reweight**
- **exposure-aware weighting**
- **frequency-bucket-based sample treatment**

These directions are especially suitable for pairwise baselines such as BPR, where recommendation quality is highly sensitive to negative construction and sample balance.

---

## 2. Representation & Interaction

### What it covers

This category covers improvements that change **how users, items, and their relations are represented and matched**. A recommender may receive reasonable supervision and still underperform because its representation mechanism is too rigid, too shallow, or too uniform in the way it propagates and combines information.

This category therefore focuses on the expressive side of recommendation: embedding formation, relational propagation, interaction structure, and scoring behavior. The core question is:

> Is the model expressive enough to capture the preference patterns that matter?

### Target recommendation problems

This category targets problems such as:

- **user-item matching too weak**: simple inner-product scoring may miss richer compatibility structure;
- **graph propagation too rigid**: different propagation depths may matter unequally, but the model aggregates them too mechanically;
- **multi-hop relation utilization insufficient**: higher-order collaborative signals exist but are not used effectively;
- **interaction function too simple**: the model cannot express more nuanced preference structure;
- **over-smoothed or under-differentiated embeddings**: deeper propagation may blur rather than sharpen useful distinctions;
- **single-path aggregation**: the model lacks adaptive control over what information should dominate.

These are failures of representation and interaction design rather than failures of sampling or objective definition.

### Typical implementation positions

This category most naturally appears in places where the model computes and combines representations:

- **module-level changes**  
  such as layer aggregation, residual propagation, gating, matching heads, or local interaction blocks;
- **feature-level extensions**  
  such as adding auxiliary representation channels or simple multi-view inputs;
- **loss-level auxiliary terms**  
  when representation stability or alignment needs extra optimization pressure;
- **config-level architectural switches**  
  such as enabling depth variants, residual structure, or lightweight interaction heads;
- **pipeline-level support**  
  mainly for ablation or staged testing rather than as the main source of change.

Compared with other categories, this one is the most direct bridge from research hypothesis to `module`-level local extension code.

### Current-stage feasible directions

At the current stage, feasible directions should remain local, interpretable, and compatible with existing baselines. Strong directions include:

- **LightGCN layer weighting**
- **residual propagation**
- **adaptive layer mixing**
- **shallow matching head**
- **gated user-item interaction**
- **lightweight representation stabilization variants**

This category is already a natural home for candidates like learnable layer-weighted LightGCN variants, and should remain one of the main frontiers for method-level experimentation.

---

## 3. Objective & Optimization

### What it covers

This category covers improvements that change **what the model is explicitly optimized to do** and **how optimization pressure is shaped**. In recommendation, a model can learn stably and still optimize the wrong surrogate. Good training loss behavior does not automatically imply strong top-k ranking behavior.

This category therefore focuses on objectives, regularizers, margin structure, and optimization constraints that better reflect recommendation goals. The core question is:

> Even if the model can represent useful patterns, is the training objective pushing it toward the right ranking behavior?

### Target recommendation problems

This category targets problems such as:

- **training objective not well aligned with NDCG or top-k ranking**;
- **ranking boundary weak**: positive and negative items are separated, but not with enough confidence or ranking sharpness;
- **popular items dominate training**: optimization repeatedly reinforces the head;
- **regularization too generic or too weak**: standard penalties do not control the failure modes that matter;
- **embedding norm drift**: unstable magnitude growth harms ranking or generalization;
- **metric-surrogate mismatch**: the loss captures only a rough approximation of the target evaluation behavior.

The common issue is not necessarily that the model is too small, but that optimization is misaligned with recommendation quality.

### Typical implementation positions

This category is most naturally realized through mechanisms that reshape optimization pressure:

- **loss-level changes**  
  such as margin-aware pairwise objectives, rank-aware surrogates, popularity-aware penalties, or tail-sensitive regularization;
- **module-level exposure of intermediate quantities**  
  when auxiliary objectives require access to layer outputs, norms, or side statistics;
- **config-level control**  
  such as margin values, regularization strengths, and auxiliary loss coefficients;
- **pipeline-level training variants**  
  for staged or ablation-based objective comparisons.

So while the research language here is about objective alignment, the executable core is usually **loss-centered**, with supporting module and config changes where necessary.

### Current-stage feasible directions

At the current stage, strong feasible directions include:

- **margin-aware BPR**
- **popularity-aware regularization**
- **rank-aware objective**
- **norm constraint**
- **auxiliary consistency terms**
- **tail-sensitive optimization variants**

This category is particularly valuable because it often yields meaningful method changes without requiring a large architectural rewrite.

---

## 4. Result Distribution Quality

### What it covers

This category covers improvements that target **the shape and quality of the final recommendation list as a distribution**, not only the relevance score of each item in isolation. A recommender can achieve acceptable ranking metrics while still producing lists that are overly concentrated, repetitive, or biased toward already dominant items.

This category therefore asks a broader question:

> What kind of recommendation list does the system produce, beyond whether each item is individually relevant?

### Target recommendation problems

This category targets problems such as:

- **coverage low**: too few unique items appear across recommendation outputs;
- **novelty low**: recommendations are accurate but overly mainstream;
- **results too concentrated**: exposure collapses toward a narrow set of head items;
- **long-tail exposure insufficient**: relevant tail items rarely surface;
- **diversity poor**: lists lack variety in what they recommend;
- **relevance-only training blind to output quality**: the model optimizes ranking but ignores distributional health.

These are recommendation-quality problems at the list and ecosystem level, not just at the single-item relevance level.

### Typical implementation positions

Because these problems appear at the output stage, this category is often realized through mechanisms closer to inference and post-processing:

- **pipeline- or rerank-level changes**  
  such as post-hoc penalties, boosts, or list-level adjustments that reshape the ranked output;
- **feature-level signals**  
  such as novelty, popularity, or exposure proxies used during reranking;
- **loss-level auxiliary objectives**  
  when broader exposure or less concentrated results are indirectly encouraged during training;
- **config-level tradeoff control**  
  such as balancing relevance against novelty, coverage, or diversity pressure.

So although this category belongs to the recommendation problem space, it often manifests as **rerank- or output-facing pipeline logic**, optionally supported by lightweight features or auxiliary losses.

### Current-stage feasible directions

At the current stage, the most feasible directions include:

- **rerank popularity penalty**
- **rerank coverage boost**
- **novelty/diversity tradeoff**
- **long-tail-sensitive reranking**
- **output concentration control**

These changes are important because they extend the method space beyond pure score maximization and make recommendation behavior more complete.

---

## 5. Efficiency & Serving Constraints

### What it covers

This category covers improvements motivated by **efficiency, compactness, and practical cost constraints**. In recommendation research, an accuracy gain is not automatically attractive if it comes with disproportionate computational cost. Even in an offline research setting, lightweight variants are valuable because they reveal which components of the current method are essential and which are redundant.

The central question is:

> Can we preserve most of the recommendation signal while using less depth, width, memory, or computation?

### Target recommendation problems

This category targets problems such as:

- **model too heavy**;
- **inference cost high**;
- **graph propagation too deep**: later hops may contribute little relative to their cost;
- **embedding dimension redundant**;
- **training overhead disproportionate to gain**;
- **unclear quality-cost tradeoff**: we do not yet know which complexity is actually necessary.

This is not merely an engineering concern. It is also a way to probe the true structure of useful recommendation behavior.

### Typical implementation positions

This category most often appears in changes that simplify or compress the effective method:

- **config-level reductions**  
  such as smaller embedding size, fewer layers, narrower hidden structure, or lighter variants of existing settings;
- **module-level simplification**  
  such as lightweight propagation or cheaper interaction heads;
- **pipeline-level comparison logic**  
  when reduced-cost variants are evaluated as structured ablations;
- **loss-level support**  
  only when compactness is encouraged indirectly through regularization.

So in executable terms, this category is usually led by **config + lightweight module changes**, rather than by heavy new modeling machinery.

### Current-stage feasible directions

At the current stage, strong feasible directions include:

- **small embedding**
- **shallow layers**
- **lightweight variants**
- **compact propagation**
- **parameter-efficient matching**

The value of this category is not only practical efficiency, but also sharper understanding of which parts of a recommender truly matter.

---

## 6. Side Information & Cold Start

### What it covers

This category covers improvements that use **auxiliary signals** or target **sparse-data regimes** such as new users, new items, and weak-history cases. Pure interaction-based recommendation is often effective in mature regions of the data, but much less reliable when historical signals are sparse or missing.

This category therefore focuses on methods that inject lightweight side information or bias corrections without requiring a complete change of benchmark setting. The core question is:

> How can the recommender behave more sensibly when interaction history alone is not enough?

### Target recommendation problems

This category targets problems such as:

- **new user**: insufficient history prevents stable preference estimation;
- **new item**: fresh items lack interaction evidence and are easily suppressed;
- **sparse interaction**: some entities remain poorly learned even when not strictly new;
- **side features underused**: simple but informative signals are available but ignored;
- **cold-item suppression**: low-history items are systematically disadvantaged;
- **activity heterogeneity**: users with very different interaction volumes may require different inductive treatment.

These are among the most recommendation-specific failure modes, and they are difficult to address through interaction structure alone.

### Typical implementation positions

This category most naturally appears in places that let the system inject simple auxiliary signals into recommendation behavior:

- **feature-level additions**  
  such as item popularity, item recency, user activity bucket, or cold-start-related coarse descriptors;
- **module-level fusion**  
  when side signals need lightweight integration into embeddings or scoring;
- **loss-level calibration**  
  when sparse or cold entities need additional protection during training;
- **pipeline-level feature construction**  
  for generating auxiliary statistics before training or inference;
- **config-level switches and weights**  
  for enabling and controlling these side channels.

So while the research question is about cold start and sparse regimes, the executable center is usually **feature-first**, optionally extended with simple fusion or calibration logic.

### Current-stage feasible directions

At the current stage, strong feasible directions include:

- **item popularity feature**
- **item recency feature**
- **user activity bucket**
- **cold-item bias init**
- **simple side-feature fusion**

This category is important not only because of its practical value, but because it anchors the method space in core recommendation-specific challenges rather than generic machine learning abstractions.

---

## Current-stage emphasis

Under the current RecClaw stage, the most immediate frontier for method-level expansion is:

1. **Bias & Sample Construction**
2. **Representation & Interaction**
3. **Objective & Optimization**

These three categories are the strongest short-term sources of executable recommendation-method candidates because they fit naturally on top of the current baseline environment and can produce meaningful changes without requiring a new evaluation protocol.

The following categories should also be expanded, but typically through lighter local methods:

4. **Result Distribution Quality**
5. **Efficiency & Serving Constraints**

Finally, the following category is essential for recommendation specificity and should be steadily developed through lightweight attachable signals:

6. **Side Information & Cold Start**

In practical terms:

- the first three categories are the current main battleground for method innovation;
- the middle two broaden the system from pure relevance toward healthier recommendation behavior and more realistic complexity;
- the last category ensures that the method space remains grounded in sparse-data and cold-start recommendation realities.

---

## Final statement

The method change space of RecClaw should be understood as a structured space of recommendation research interventions.

Its categories are not defined by where code happens to be edited. They are defined by where recommendation systems typically fail, and by what kinds of executable changes can address those failures.

A useful method space must therefore do both at once:

- preserve the language of recommendation problems,
- and remain concrete enough to map onto implementable candidates.

That is the purpose of these six categories:

1. to correct biased or weak supervision,
2. to strengthen representation and interaction,
3. to better align optimization with ranking goals,
4. to improve output-list distribution quality,
5. to reduce unnecessary complexity and serving burden,
6. to make the system more robust under sparse-data and cold-start conditions.

Any future candidate library, registry, or agent action space should be built on top of this view.