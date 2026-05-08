# Candidate Proposal Agent Guide

This note is an agent-facing guide for candidate proposal generation. It sits
between `scripts/agent.py` and `configs/candidate_registry.yaml`.

## Purpose

The agent should be able to propose new recommendation candidates, but every
proposal must be structured, checked, and routed before it is run or promoted
into the registry.

## Proposal Contract

All generated proposals must follow:

```text
configs/candidate_proposal_schema.yaml
```

Important fields:

- `proposal_type`: `tuning`, `algorithmic_variant`, or `research_spec`
- `runnable_level`: `parameter_only`, `config_only`, `code_required`, or `spec_only`
- `parent_candidate_id`: an existing registry candidate
- `consumes`: parameters used or introduced by the proposal
- `parameter_overrides`: concrete values for runnable tuning proposals
- `parameter_signature`: normalized duplicate guard, defined as
  `parent_candidate_id::{sorted parameter_overrides JSON}`. Protocol-only
  keys such as `seed` are excluded from the signature.
- `evaluation_plan`: recommended multi-seed validation plan before claiming a
  stable improvement
- `implementation_plan`: required in practice for `code_required` proposals
- `risk.recbole_core_change_required`: must not be true for normal candidates

## LLM Integration Point

The preferred final integration is inside `scripts/agent.py`.

The agent should send the LLM a compact context containing:

- `configs/candidate_proposal_schema.yaml`
- `configs/candidate_registry.yaml`
- recent rows from `results/agent_memory.jsonl`
- relevant excerpts from `notes/experiment_log.md`
- the requested proposal mode: `conservative`, `mixed`, or `explore`

The LLM should return either:

- a JSON array of proposal objects
- JSONL with one proposal object per line

The agent then writes proposals to:

```text
results/candidate_proposals.jsonl
```

`scripts/propose_candidate.py` remains useful as a stable heuristic fallback
and as a reference implementation for what valid proposals should look like.

## Validate Before Acting

Every proposal file must pass through:

```bash
python3 scripts/validate_candidate_proposal.py \
  --proposals results/candidate_proposals.jsonl
```

The validator also rejects duplicate runnable tuning proposals when the same
`parent_candidate_id + parameter_overrides` signature appears twice in the
proposal file. If `results/agent_memory.jsonl` exists, the default validation
also rejects signatures that have already been run. Use `--memory` to point at
another memory file.

Validation statuses:

- `accepted`: safe to run now
- `needs_review`: valid idea, but needs implementation, agent review, or design review
- `rejected`: invalid proposal; revise or drop it

Use `next_action` to route the proposal.

## Running Accepted Tuning Proposals

Do not use a separate proposal runner. Accepted runnable proposals can call the
existing candidate runner directly.

For an accepted proposal:

```json
{
  "parent_candidate_id": "cand_bpr_margin_loss",
  "parameter_overrides": {
    "margin": 0.1
  }
}
```

the agent should run:

```bash
python3 scripts/run_candidate.py cand_bpr_margin_loss --set margin=0.1
```

Only `proposal_type=tuning` with `runnable_level=parameter_only` or
`config_only` should be run this way. `code_required` proposals must first be
implemented under the allowed local extension paths.

For any accepted proposal that appears to improve the primary metric, repeat
the same `parameter_overrides` across the proposal's
`evaluation_plan.validation_seeds` before claiming a stable improvement. For
example:

```bash
python3 scripts/run_candidate.py cand_bpr_margin_loss --set margin=0.1 --set seed=2026
python3 scripts/run_candidate.py cand_bpr_margin_loss --set margin=0.1 --set seed=2027
python3 scripts/run_candidate.py cand_bpr_margin_loss --set margin=0.1 --set seed=2028
```

The current candidate-proposal layer records and validates the seed plan. The
actual scheduling of repeated seed runs belongs in the agent or benchmark
runner so that it can aggregate mean/std and avoid blocking normal proposal
generation.

## Boundaries

Normal candidate proposals must not require RecBole core changes. New method
ideas should be implemented under:

- `recclaw_ext/models/`
- `recclaw_ext/posthoc/`
- `configs/candidates/`

If an idea truly requires RecBole framework changes, treat it as a separate
framework-level design discussion, not as a normal candidate.
