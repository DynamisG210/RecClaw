# RecClaw M6I — Integrated State-Space and Cross-Arm Isolation Closure

## 0. Explicit authorization and current stop state

Open exactly one new recovery milestone:

`M6I — INTEGRATED_STATE_SPACE_AND_CROSS_ARM_ISOLATION_CLOSURE`

This is an explicit authorization to repair the V21 P0 and continue autonomously through one fresh
five-round-per-Arm Pilot only after the complete pre-Pilot closure in this document passes.

Do not create or run a new real Pilot immediately.

Permanently preserve:

```text
V19 root / state / seed / responses
V20 root / state / seed / responses
V21 root / state / seed / responses
V21_CROSS_ARM_CONTAMINATION_HARD_STOP.json
```

Current trusted incident facts:

```yaml
V19:
  completed_rounds: 8
  completed_training_claims: 8
  failure_class: META_NO_OBSERVATION_BOUNDARY_NOT_ADVANCED

V20:
  completed_rounds: 6
  completed_training_claims: 6
  failure_class: ACTIVE_TASK_WITHOUT_ROUTE_BOUNDARY_MISCLASSIFIED

V21:
  completed_rounds: 8
  completed_training_claims: 8
  broker_calls: 9
  failure_class: CONFIRMED_CROSS_ARM_RESEARCH_CALL_REUSE
  severity: P0
  exact_path:
    execution_order: A -> C -> B
    C_research_calls_reused_by_B: 4
    lineage_failure: C parent absent from B-private lineage
```

The next Pilot must not be used to discover another reachable state-machine or namespace branch that
could have been covered deterministically before Provider/GPU execution.

Authority remains:

```yaml
authority: NONE
evidence_class: DEVELOPMENT_ONLY
formal_acceptance: false
M7_started: false
M8_started: false
```

---

## 1. Core diagnosis

The repeated failures are not independent random runtime problems.

They show one common engineering defect:

> The orchestration path has been tested by examples and repaired through successive Pilot wrappers,
> but the full per-Arm round state machine, ownership model, cache identity and schedule-permutation
> space have not been closed as one executable contract.

Do not fix V21 by merely adding `arm` to `_research_calls`.

That may suppress the observed symptom while leaving:

- another shared cache or singleton;
- incomplete request identity;
- unsafe paired-call reuse;
- cross-Arm candidate instance reuse;
- task/route/observation branch gaps;
- order-dependent state transitions;
- wrapper-specific behavior that differs from Main.

The repair must close the complete reachable state space.

---

## 2. Scope

Audit and repair only the treatment-relevant runtime path:

```text
three-arm scheduler / opaque order
Proposal session
Original and Research broker calls
physical-call cache / logical-call records
Producer contexts and outputs
candidate instance identity
lineage index
Research task queue
Meta route and fast-boundary state
Guard/Fusion admission
Search Memory
frontier projections
round/resource/execution closure
```

Re-run conformance but do not redesign:

- BL-ICF compiler/materializer;
- training model implementations;
- Evidence Guard scientific core;
- dataset split;
- metric contract;
- 66-semantics profile;
- three-arm estimands.

Training Runtime V12 and gpu35 fixed canaries have already provided positive runtime evidence. Do not
repeat full training canaries unless source/runtime bytes change.

---

## 3. No more wrapper stacking

Do not create another Pilot-only wrapper that patches one branch around an unchanged ambiguous core.

Required action:

1. identify the canonical production state machine used by Pilot and future Main;
2. move the V19/V20/V21 boundary semantics into that core;
3. make Pilot and Main call the same implementation;
4. deprecate or make unreachable the V19/V20/V21 wrapper-only repairs;
5. prove there is one source of truth for:
   - Meta round opening;
   - route creation;
   - observation application;
   - no-observation closure;
   - active-task closure;
   - next-round authorization.

A wrapper may configure a versioned policy, but it may not define alternative scientific transition
semantics.

---

## 4. Ownership and mutable-state audit

Create:

```text
M6I_STATE_OWNERSHIP_MAP.json
M6I_MUTABLE_STATE_INVENTORY.json
M6I_CACHE_AND_IDENTITY_AUDIT.md
```

Inventory every mutable object reachable from the active campaign path:

- module-level dict/list/set;
- class-level mutable field;
- singleton;
- `lru_cache` or custom memoization;
- Broker response cache;
- logical-call registry;
- proposal/slate cache;
- Original call cache;
- Research call cache;
- RNG;
- Meta runtime and fast state;
- Research task queues;
- lineage indices;
- Search Memory;
- Guard evidence snapshots;
- candidate registry;
- artifact/state-store projections.

Classify each as exactly one of:

```text
COMMON_IMMUTABLE
EXPERIMENT_SHARED_APPEND_ONLY
ARM_PRIVATE
ROUND_LOCAL
EXPLICIT_PAIRED_CALL
FORBIDDEN_GLOBAL_MUTABLE
```

Any treatment-dependent mutable object without an explicit owner is a P0.

Default rule:

```text
Research / Meta / lineage / task / memory / candidate-instance state is ARM_PRIVATE.
```

---

## 5. Provider-call and cache identity model

Separate three identities:

```text
ProviderPhysicalCallId
ConsumerLogicalCallId
CandidateInstanceId
```

### 5.1 ProviderPhysicalCallId

May be shared only when the exact canonical request bytes and all behaviorally relevant identities
are equal:

```yaml
model_release_digest:
response_schema_digest:
temperature:
timeout_policy:
producer_role:
prompt_bytes_digest:
complete_context_digest:
memory_view_digest:
meta_fast_state_digest:
lineage_view_digest:
active_task_digest:
research_task_queue_digest:
round_index:
search_seed:
```

Do not use a handpicked subset such as:

```text
search_seed + round_index + memory_policy_digest
```

as a sufficient cache key.

### 5.2 ConsumerLogicalCallId

Must always bind:

```yaml
experiment_id:
opaque_arm_instance_id:
search_seed:
round_index:
producer_role:
provider_physical_call_id:
consumer_context_digest:
```

Even when B/C legitimately share one physical Provider response, they require separate consumer
records.

### 5.3 CandidateInstanceId

A semantic program digest may be common. A candidate instance may not.

Bind:

```yaml
opaque_arm_instance_id:
round_index:
producer_role:
semantic_program_digest:
local_parent_or_task_identity:
```

An Arm-private lineage parent ID may never appear as another Arm's local parent.

### 5.4 Explicit call-sharing policy

Implement `CallSharingPolicyV1`:

```text
ARM_PRIVATE
PAIRED_BC_EXACT_CONTEXT
COMMON_IMMUTABLE
```

Rules:

- A never consumes B/C Research calls.
- B/C physical-call sharing is permitted only when exact canonical request and context digests are
  byte-identical and the response is arm-neutral.
- If B/C Search Memory, Meta state, lineage, task or prompt differs, separate physical calls are
  mandatory.
- Shared response semantics are cloned into separate Arm-private candidate instances.
- No cross-Arm lineage or task reference may be cloned.
- Cache hit/miss decisions are written to an append-only audit record.

A simpler no-cross-Arm-cache implementation is acceptable if budget fairness and the frozen
experimental contract are requalified. Do not preserve unsafe sharing only to reduce token cost.

---

## 6. Canonical per-Arm round state machine

Create one explicit typed state machine.

Recommended states:

```text
ROUND_READY
PROPOSAL_PENDING
ROUTE_CREATED
ACTIVE_TASK_BOUND
CANDIDATE_SELECTED
EXECUTION_STARTED
RESULT_CLOSED
OBSERVATION_ADMITTED
OBSERVATION_WITHHELD
NO_EXECUTION
ROUND_TERMINAL
```

At each round exactly one proposal source path is valid:

```text
NORMAL_ROUTED_PROPOSAL
ACTIVE_BOUND_TASK
ORIGINAL_CONTROLLER_PATH
NO_PROPOSAL_TERMINAL
```

At each result boundary exactly one observation path is valid:

```text
ADMITTED_OBSERVATION
WITHHELD_OBSERVATION
DIAGNOSTIC_OR_ENGINEERING_ONLY
NO_OBSERVATION
```

The Meta boundary must advance exactly once for every terminal B/C round, regardless of whether:

- a route exists;
- an active task exists;
- an observation is admitted;
- an observation is withheld;
- execution never starts;
- execution fails;
- Guard blocks every candidate.

A terminal branch may not require an object that its proposal path never creates.

---

## 7. Exhaustive transition matrix

Build a machine-readable matrix and tests for at least:

### Proposal paths

```text
Original controller
normal Research route
validation task
matched-control task
ablation task
repair task
all PRE candidates blocked
empty/invalid slate
Provider failure before proposal
```

### Execution/result paths

```text
successful admitted result
REQUIRES_CONFIRMATION preliminary result
DIAGNOSTIC_ONLY
NOT_ADMISSIBLE
PROTOCOL_BRANCH
QUARANTINE / INCONCLUSIVE
common execution failure
resource-ceiling rejection
training failure
no execution
```

### Meta paths

```text
route + admitted observation
route + withheld observation
route + no execution
task + admitted observation
task + withheld/no observation
task + execution failure
no route + no task + typed terminal
```

For each matrix cell assert exact before/after changes to:

```text
Controller state
Meta state
Search Memory
lineage
task queue
Observed frontier
SearchEligible frontier
Confirmed frontier
resource ledger
round state
triplet barrier
```

No branch may remain described only in prose.

---

## 8. Arm-order permutation invariance

The V21 defect was exposed only under `A -> C -> B`.

Run every permutation:

```text
A-B-C
A-C-B
B-A-C
B-C-A
C-A-B
C-B-A
```

For fixed per-Arm inputs and deterministic fake outcomes, require:

```text
each Arm's final controller/meta/memory/lineage/task/frontier state
is invariant to inter-Arm execution order.
```

Shared append-only provider accounting may differ only in ordering, not in consumer semantics.

Also run at least 500 deterministic randomized multi-round schedules covering all permutations.

Any schedule-dependent Arm-local outcome is P0 unless explicitly frozen by the experiment design.

---

## 9. Static cross-Arm access audit

Perform AST/grep/runtime instrumentation over the active source projection.

Reject:

- module-level Research caches keyed without complete context;
- broker logical IDs without consumer Arm;
- path lookup of sibling Arm roots;
- Arm-private object references in another Arm's prompt, proposal, task or lineage;
- direct access to another Arm's controller/Meta/Search Memory/evidence DB;
- global mutable RNG shared between Arms;
- candidate IDs lacking Arm instance identity;
- task IDs lacking Arm instance identity.

Add an `owner_arm_instance_id` or equivalent private owner token to treatment-dependent runtime
objects and assert it at every read/write boundary.

---

## 10. Trace replay of V19, V20 and V21

Convert the three failure histories into immutable regression fixtures.

Required replays:

```text
V19: routed candidate, observation withheld, next Meta boundary
V20: matched-control active task, no route, next Meta boundary
V21: A-C-B order, divergent C/B contexts, no cross-Arm physical/logical reuse
```

The repaired core must:

- close V19 and V20 paths correctly;
- refuse unsafe V21 reuse before lineage resolution;
- produce separate B/C consumer records;
- preserve Arm-private lineage.

These are permanent regression tests, not version-specific one-off tests.

---

## 11. Zero-cost synthetic campaign qualification

Before any real API or GPU use, run the exact future Pilot orchestrator with:

- deterministic fake Broker;
- deterministic fake training/result adapter;
- real state store;
- real scheduler;
- real Meta state machine;
- real task queue;
- real Guard/Fusion mapping;
- real analysis projections.

Run:

```yaml
rounds_per_arm: 50
arm_orders: all_permutations_and_randomized
fault_injection:
  provider_failure: true
  resource_rejection: true
  execution_failure: true
  all_pre_blocked: true
  withheld_observation: true
  active_task_no_route: true
  validation_task: true
  diagnostic_and_protocol_branch: true
```

Pass requirements:

```yaml
scheduled_rounds: 150
terminal_rounds: 150
open_rounds: 0
duplicate_claims: 0
duplicate_feedback: 0
cross_arm_reads: 0
cross_arm_writes: 0
foreign_parent_refs: 0
unsafe_cache_hits: 0
meta_boundary_count_per_terminal_BC_round: 1
confirmed_frontier_from_search_rounds: 0
P0: 0
P1: 0
```

Run this qualification repeatedly under at least 100 randomized seeds.

---

## 12. Real Broker, no-training isolation probe

After the synthetic qualification passes, run a small treatment-free Provider probe with no RecBole
training:

1. one B/C pair with exact identical context;
2. one B/C pair with different memory/lineage/task context;
3. all six Arm order permutations.

Verify:

- identical contexts follow the frozen sharing policy;
- divergent contexts never share;
- each Arm gets separate consumer and candidate-instance records;
- no foreign lineage parent;
- credentials are removed;
- no Search Memory/Meta/Frontier side effects.

This probe is not a Pilot result and must not enter Search Memory or future policy training.

---

## 13. Independent pre-Pilot audit

Create:

```text
M6I_INDEPENDENT_AUDIT.md
M6I_STATE_MACHINE_COVERAGE.json
M6I_ARM_ORDER_INVARIANCE_REPORT.json
M6I_CALL_SHARING_CONFORMANCE.json
M6I_SYNTHETIC_50R_REPORT.json
M6I_PROVIDER_ISOLATION_PROBE.json
```

Require:

```yaml
P0: 0
P1: 0
state_machine_branch_coverage: complete_for_frozen_contract
all_six_arm_orders: PASS
randomized_order_runs: PASS
V19_V20_V21_replays: PASS
synthetic_50_round_per_arm: PASS
provider_isolation_probe: PASS
```

Green unit tests without these system-level reports are insufficient.

---

## 14. Fresh real Pilot

Only after M6I PASS:

1. create one new source checkpoint;
2. freeze one new contract/version;
3. use a fresh unused search seed;
4. use fresh root, DB, Broker calls, Search Memory, Meta state and evidence state;
5. run one five-round-per-Arm full-training Pilot;
6. independently audit all artifacts.

Do not reuse V19/V20/V21 calls or outcomes.

The real Pilot must verify:

```text
15/15 terminal rounds
all triplet barriers
successful BPR and LightGCN/compositional paths
safe provider call sharing
four Producer provenance
Arm-private lineage
task/validation lifecycle
Meta boundary exactly once
Guard path and no private leakage
Observed/SearchEligible/Confirmed separation
zero cross-Arm contamination
```

If GO with P0=0/P1=0, proceed directly to one frozen 50-round-per-Arm Effect Pilot with read-only
checkpoints at rounds 10, 20 and 50.

Do not run separate mutable 10-round and 20-round experiments.

---

## 15. Recovery policy

During M6I, solve ordinary implementation defects autonomously.

Do not stop after routine reports.

Stop only for:

- inability to define an unambiguous ownership model;
- required change to A/B/C treatment definitions;
- required change to frozen scientific estimands;
- held-out leakage;
- inability to eliminate cross-Arm state access;
- need to lower P0/P1 gates;
- destructive cleanup/push/release;
- three failed evidence-backed repair cycles for the same root-cause class.

No further real Pilot is authorized before the complete M6I audit passes.

---

## 16. Immediate instruction

Begin now:

```text
1. Freeze and verify the V21 hard-stop evidence.
2. Build the ownership/mutable-state/cache inventory.
3. Repair the canonical core, not another Pilot wrapper.
4. Implement explicit call-sharing and consumer identity.
5. Implement the full typed round state machine.
6. Run the exhaustive branch and six-permutation tests.
7. Replay V19/V20/V21.
8. Run the 50-round-per-Arm synthetic qualification under 100 randomized seeds.
9. Run the real Broker no-training isolation probe.
10. Perform an independent P0/P1 audit.
11. Only then freeze and run one fresh five-round full-training Pilot.
12. If that Pilot is GO, continue automatically to the frozen 50-round Effect Pilot.
```
