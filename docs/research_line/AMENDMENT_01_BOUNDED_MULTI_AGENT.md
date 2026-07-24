# Amendment 01 — Bounded Independent Producer Agents

> - Amendment ID: `HELIX-ABC-001-AMENDMENT-01`
> - Date: 2026-07-24 (Asia/Shanghai)
> - Applies before: `M0`
> - Amended roadmap baseline SHA-256:
>   `fabea167c075e1ba6dda3de416c264a4e95cb8f30368285e8dfa9899d288f1b5`
> - Amendment status: `LOCAL_COMPLETE`
> - Implementation status: `NOT_STARTED`
> - Authority: `NONE`
> - Evidence class: `DEVELOPMENT_ONLY`
> - Formal acceptance: `false`
> - Implementation profile: `MINIMUM_SUFFICIENT_V1`

## 1. Purpose and scope

This amendment changes only the Research Line Producer execution model, its
resource-accounting contract, the Research Capability Quality Gate ordering,
and the affected milestone/freeze requirements.

The default Main target is now:

```text
BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
```

This is a bounded experiment treatment, not an open Agent Society and not a
claim that multi-agent execution is already implemented or superior.

No runtime source code, live LLM call, training, Canary, Pilot, Main Campaign,
Evidence Guard core change, commit, or push is authorized by this amendment.

## 2. Invariants preserved without change

The following remain frozen:

- A, B, and C use the same exact BL-ICF executable profile;
- A, B, and C use the same exact `CommonExecutionGuardV1`;
- A is Original + NullEvidencePort;
- B is Research Line + NullEvidencePort;
- C is the exact B Research treatment + EvidenceGuardPort;
- `B-A` estimates Research Capability;
- `C-B` estimates the independent Evidence Guard increment;
- `C-A` is descriptive overall Helix effect;
- Research Line owns Search Utility only;
- Evidence Guard owns development Evidence Authority only;
- Research Line cannot import or implement Evidence Guard semantics;
- one SearchRound contains one proposal-generation session, at most one
  ordinary execution, and exactly one typed feedback when opened;
- PRE block advances only within the same frozen slate;
- all Arms retain equal total Proposal, Token, wall-time, retry, execution, and
  GPU resource ceilings;
- Round, Execution Count, Token, and GPU Cost remain separate reporting axes;
- experiment state remains a single-writer SQLite WAL design;
- M0 through M8 remain serial;
- Hardening remains deferred.

## 3. Producer execution modes

`ProducerExecutionModeV1` is a closed enum:

```text
BATCHED_ROLE_PORTFOLIO_V1
NEUTRAL_MULTISAMPLE_CONTROL_V1
BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
```

All three modes:

- execute inside one `ProposalGenerationSessionV1` per Arm per opened round;
- consume the same frozen total resource envelope;
- use the same base model for controlled comparison;
- emit the same typed `CandidateProposalV2` / BL-ICF output contract;
- feed the same frozen Static Router during the agentization sub-gate;
- do not create extra SearchRounds, execution opportunities, or feedback.

### 3.1 Batched role portfolio

`BATCHED_ROLE_PORTFOLIO_V1` uses one physical LLM invocation whose response
contains four preassigned role slots.

It is a role-conditioned portfolio. It is not independent multi-agent
execution and must not be reported as a Multi-Agent success.

### 3.2 Neutral multisample control

`NEUTRAL_MULTISAMPLE_CONTROL_V1` uses a pre-frozen number of independent
physical invocations with:

- the same neutral research prompt;
- the same scoped context and memory projection;
- separate RNG streams and request IDs;
- no specialized Producer role instruction.

It isolates gains from repeated sampling or multiple physical calls. It is a
diagnostic control and is not eligible as the Main Research treatment.

### 3.3 Bounded independent Producer Agents

`BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1` uses four preassigned Producer
identities:

```text
mechanism_composer
lineage_refiner
falsification_designer
frontier_architect
```

Each Producer has:

- exactly one independent physical LLM invocation per active session slot;
- a fixed Producer identity assigned before invocation;
- a role-specific prompt;
- a scoped context projection;
- a scoped prior-round Search Memory and Mechanism Belief view;
- an independent RNG stream and request ID;
- request, response, context, and memory digests;
- a typed BL-ICF proposal output;
- complete lineage through compile, route, run, result, and memory.

No Producer may read another Producer's same-round request, response, scratch
state, or hidden chain of thought.

B and C must use the exact same:

- Producer execution mode;
- four Producer identities and order;
- base model and provider contract;
- token-allocation policy;
- prompt/context/memory projection policy;
- RNG derivation;
- dispatch and concurrency policy;
- Router and Meta checkpoint.

## 4. Explicit non-goals

Main V1 does not build an open Agent Society:

- no free Agent-to-Agent dialogue;
- no recursive debate;
- no dynamic role creation, deletion, or transfer;
- no heterogeneous base models;
- no online Critic;
- no free-form Implementer Agent;
- no Literature Agent;
- no agent-controlled Router, Meta, Runner, Evaluator, CommonExecutionGuard,
  or Evidence Guard.

Router, Meta, Runner, Evaluator, CommonExecutionGuard, and Evidence Guard
remain deterministic or versioned non-agentic typed components.

The non-discovery services remain:

```text
control_ablation_builder
repair_engineer
```

They are non-agentic typed services, receive no algorithm-discovery credit,
and cannot create additional LLM calls outside the frozen session plan.

## 5. ProposalGenerationSession and budget fairness

Every opened Arm-round creates exactly one `ProposalGenerationSessionV1`.

Arm A may use one Original invocation. Arms B and C may divide the same total
resource envelope across independent Producer invocations.

The following ceilings are equal across A, B, and C:

- total input tokens;
- total output tokens;
- total billed-token debit;
- total proposal count;
- wall time;
- retry debit;
- proposal-attempt debit;
- ordinary executions;
- CommonExecutionGuard validation;
- GPU device time;
- GPU cost.

Physical call count is not an equality constraint. Physical call count,
per-call latency, session latency, actual input/output/billed tokens, and
failures are treatment costs and must be reported.

A session cannot be refreshed, replaced, or reopened. Reaching any ceiling
stops subsequent child calls and closes the same session with typed
budget-exhausted or no-proposal status.

## 6. Research Capability Quality Gate ordering

`ResearchCapabilityQualityGateV1` executes in two ordered stages.

### 6.1 Agentization sub-gate

Under one frozen Static Router, compare:

```text
BATCHED_ROLE_PORTFOLIO_V1
NEUTRAL_MULTISAMPLE_CONTROL_V1
BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
```

The comparison must use the same:

- base model;
- total input/output/billed-token ceilings;
- proposal count;
- BL projection;
- candidate schema;
- fixture lineage;
- outcome-masked evaluator.

The generator and Router cannot observe evaluation outcomes. The evaluator
receives sealed mode-blinded candidate and cost records.

Required reports:

```text
schema_valid_rate
bl_compile_rate
semantic_uniqueness
mechanism_axis_coverage
falsification_completeness
parent_ablation_completeness
duplicate_rate
static_router_topk_utility
latency
physical_call_count
token_cost
```

The verdict is:

```text
PASS_INDEPENDENT_MULTI_AGENT
PASS_BATCHED_ONLY
FAIL
```

`PASS_INDEPENDENT_MULTI_AGENT` permits selection of the bounded independent
mode.

`PASS_BATCHED_ONLY` is an honest fallback. It is not a Multi-Agent success and
cannot automatically enter Main. Explicit user approval of the exact batched
treatment is required before M6 freeze.

`FAIL` stops M2.

### 6.2 Selected-mode Meta sub-gate

Only after selecting a non-FAIL Producer mode may the gate compare Static
Router against `VERSIONED_POLICY_UPDATE` on that same mode.

The Meta verdict remains:

```text
PASS_VERSIONED_META
PASS_STATIC_ONLY
FAIL
```

The Gate must record both verdicts, the selected Producer mode, and:

```text
PASS
USER_APPROVAL_REQUIRED_BEFORE_M6
FAIL
```

as the progression status.

## 7. Meta update boundary

Meta may update only at SearchRound boundaries and only:

- Producer token allocation;
- Producer mechanism-axis targeting;
- memory retrieval policy;
- Router priors and acquisition parameters.

Every update emits a policy digest and must replay deterministically.

Meta must not update:

- Producer execution mode;
- Producer identity or role count;
- base model;
- candidate schema;
- role prompt contract;
- total resource ceilings;
- Router Hard Gates;
- CommonExecutionGuard;
- EvidencePort or Fusion;
- Evidence Authority fields.

Meta validation must check:

- deterministic replay;
- round-boundary-only application;
- field-allowlist compliance;
- calibration;
- no Candidate-ID leakage;
- no collapse to one Producer;
- no collapse to parameter tuning or one mechanism family.

## 8. Developmental Mechanism Belief

The minimal `DevelopmentalMechanismBeliefV1` payload is:

```text
hypothesis_id
mechanism_axis
competing_hypotheses
predicted_outcome_signature
evidence_for
evidence_against
unresolved_confounds
next_discriminative_test
```

This payload:

- is `DEVELOPMENT_ONLY`;
- belongs to Search Memory;
- contains Search Utility observation references only;
- has no Evidence Authority;
- cannot write Claim Ceiling, Evidence Admission, Protocol Branch,
  accepted-evidence state, or ClaimRecord;
- cannot be written by Evidence Guard;
- is exposed to each Producer only through a role-scoped prior-round view.

## 9. Milestone consequences

### M0

Freeze the Producer mode enum, ProposalGenerationSession contract, total
resource envelope, physical-call cost records, and Mechanism Belief payload.
Do not call a real LLM or create future milestone stubs.

### M2

Implement and validate Producer modes under Static Router first. Select a
non-FAIL mode, then validate Meta on that mode.

### M5

Canary validates the selected physical call plan, total resource enforcement,
and physical-call/latency/token cost records. Canary does not select a mode
from outcomes.

### M6

Freeze the selected Producer mode and Meta/Static policy. A
`PASS_BATCHED_ONLY` path requires explicit user approval before freeze.

### M7

Execute only the exact frozen mode. Do not add dialogue, roles, agents, calls,
budget, or policy changes. Report actual physical calls, latency, and tokens.

## 10. Document-level acceptance and current boundary

The amendment is acceptable only if the amended roadmap still preserves:

- all three Arm definitions and contrasts;
- common BL-ICF and CommonExecutionGuard;
- B/C identity equality outside EvidencePort and Guard-derived trajectory;
- one session, at most one ordinary execution, and one feedback per opened
  SearchRound;
- same-slate traversal and no refill;
- equal total resource ceilings;
- Search Utility / Evidence Authority separation;
- deterministic Fusion and Guard isolation;
- single-writer SQLite state;
- M0-M8 serial execution;
- deferred Hardening.

Current findings:

```text
P0 = 0
P1 = 0
next_authorized_milestone = M0
implementation_status = NOT_STARTED
authority = NONE
evidence_class = DEVELOPMENT_ONLY
formal_acceptance = false
```
