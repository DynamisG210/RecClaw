# RecClaw Program

This is the active operating manual for RecClaw. Treat it as the high-level
program contract for the agent, experiment runner, and human operator.

## Mission

RecClaw is an automated recommender-system research agent. Its goal is to
discover and validate useful recommendation algorithm mechanisms, not merely to
sample hyperparameters.

The core loop is:

```text
Observe -> Plan -> Propose Candidate -> Validate -> Implement -> Run -> Evaluate -> Reflect -> Remember
```

## Research Ability Line

The active search architecture now makes the research-ability line explicit in
`scripts/research_line.py`.

```text
Candidate Producers
-> Research Router
-> Development Experiment
-> Search Outcome
-> Search Memory
-> Meta-Research advisory
-> next Producer / Router / algorithm-policy context
```

Candidate Producers are role-specific proposal sources rather than one undivided
proposal stream:

- `mechanism_discovery`: LLM-oriented algorithmic mechanisms with novelty,
  ablation parent, and failure mode.
- `template_expansion`: deterministic expansion from wired high-potential
  parents.
- `repair_or_ablation`: one-axis repairs and structured ablations around
  promising families.
- `parameter_sanity`: bounded parameter-only checks around runnable parents.
- `high_risk_spec`: design-review-only research specs.

The Research Router scores proposals and runnable candidates by novelty,
expected information value, parent credibility, executable readiness, duplicate
signatures, blocker history, implementation complexity, and protocol-risk
signals. Router decisions are written into trial memory and
`results/research_routes.jsonl` for later replay.

Search Memory is rebuilt from `results/agent_memory.jsonl` each round. It tracks
family outcomes, producer outcomes, duplicate parameter/execution signatures,
blocker signatures, promising families, and frozen families. The
Meta-Research advisory is deliberately offline/advisory in this version: it can
suggest producer downweights, memory transformations, and router updates, but it
does not automatically promote a new strategy version into the runtime.

The current runnable method line is general recommendation on ML-1M with
RecBole, focused on BPR and LightGCN action-space exploration. Stronger RecBole
models may be used as external baselines, but they are not editable search bases
until they are explicitly added to the action space and runner.

## Non-Negotiable Protocol

- Dataset: ML-1M.
- Evaluation: RecBole full-sort validation/test.
- Split: random split with user grouping, fixed by `configs/task_ml1m.yaml`.
- Primary metric: `NDCG@10`.
- Do not change RecBole core source, data split, evaluation mode, or metric
  semantics during candidate search.
- Do not use `RecClaw_LabLog` as runtime input.

`RecClaw_LabLog` is for human reports and offline analysis only.

## Sources Of Truth

- `configs/action_space.yaml`: what the agent is allowed to change.
- `configs/search_policy.yaml`: how the current stage should search.
- `configs/reflection_policy.yaml`: how results become experience.
- `configs/candidate_proposal_schema.yaml`: proposal shape and safety contract.
- `configs/candidate_registry.yaml`: runnable or declared candidate catalog.
- `recclaw_ext/`: local implementations for model, loss, sampler, and graph
  propagation changes.

Human notes may explain the method, but they are not runtime truth unless a
script explicitly reads them.

## Candidate Contract

Every candidate must have:

- `candidate_id`
- `base_model`
- `runner_type`
- `entrypoint`
- `consumes`
- `wired`
- `status`

Use `wired: true` only when the entrypoint imports, the candidate config exists,
the consumed parameters have defaults, and `run_candidate.py` can execute it.

Candidate types:

- `config_only`: RecBole-native configuration changes.
- `model`: local model/loss/sampler/propagation implementation under
  `recclaw_ext/models/`.
- `posthoc`: score adjustment after model inference; currently not part of the
  default pilot path.

## Proposal Requirements

A useful RecClaw proposal should state:

- the recommendation problem it targets;
- the algorithmic mechanism;
- how it differs from the parent;
- expected failure mode;
- ablation parent;
- implementation complexity;
- exact local files needed if code is required.

Parameter-only proposals are allowed only as sanity checks or local refinement
after an algorithmic mechanism has shown promise.

## Implementation Boundary

Allowed:

- local model subclasses in `recclaw_ext/models/`;
- local helper losses and samplers;
- candidate YAML under `configs/candidates/`;
- registry/schema/search-policy updates needed to make a candidate runnable;
- isolated output directories under `results/`.

Forbidden:

- RecBole core edits;
- data leakage;
- changing task split or full-sort protocol;
- undeclared runtime parameters;
- writing formal smoke results into the main `results.csv`;
- using report artifacts as agent memory.

## Reflection And Memory

Full-memory runs may use:

- `results/agent_memory.jsonl`
- `results/results.csv`
- `results/candidate_search_tree.json`
- `results/experience_summary.md`
- `results/reflection_memory.jsonl`

No-history and random baselines must not read experience summary, agent memory,
reflection memory, or candidate lineage artifacts.

Reflection should distinguish:

- strong family;
- promising but unstable family;
- weak family;
- dead or unsafe family;
- repeated signature;
- post-validation ablation need.

A validated strong candidate should trigger seed validation, one-axis ablation,
and stability checks before unlimited sibling generation.

## Decision Rules

- `keep`: candidate improves the relevant baseline and is worth preserving.
- `revise`: candidate is near-useful, unstable, or mechanism-interesting but not
  enough to freeze as best.
- `discard`: candidate is weak or unpromising under the same protocol.
- `crash`: run failed or produced unusable output.

NDCG@10 is the primary decision metric. Recall, MRR, Hit, Precision,
ItemCoverage, and runtime are supporting evidence.

## Strong Baselines

Baseline models must be same-protocol and audited. Treat strong general models
as external references unless they are explicitly added to the action space and
runner path.

Do not claim RecClaw edits a model family that is not present in:

- `configs/action_space.yaml`
- `configs/candidate_proposal_schema.yaml`
- `scripts/run_candidate.py`

## Post-Discovery Workflow

When RecClaw discovers a strong new algorithm:

1. Run multi-seed validation.
2. Register the candidate in local runtime code.
3. Run controlled ablations for each mechanism component.
4. Compare against old RecClaw, tuned base model, random/no-history evidence,
   and strong general baselines.
5. Only then decide whether to continue search or freeze the family.

This keeps the agent from spending budget on endless local variants after it has
already found a strong family.

## Server Runs

Use isolated project/output directories. Bind to an explicitly selected GPU.
Do not inspect, kill, or modify other users' processes.

Safe acceleration for current formal runs:

```text
train_batch_size=2048
eval_batch_size=65536
worker=8
```

Changing training batch size creates a different optimization condition and
should not be mixed into formal comparisons without a separate label.

## Preflight

Before launching a formal pilot:

```bash
python3 scripts/analysis/lint_recclaw_space.py
python3 -m unittest discover -s tests
```

The linter should report zero errors and zero warnings.
