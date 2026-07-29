# M6I Cache and Identity Audit

- authority: `NONE`
- evidence class: `DEVELOPMENT_ONLY`
- active source files scanned: `76`
- mutable findings classified: `648`
- forbidden global mutable findings: `0`
- static P0 checks failed: `0`

## Static checks

| Check | Result | Evidence |
|---|---:|---|
| `RESEARCH_CACHE_BINDS_ARM_OWNER` | **PASS** | Research session identity includes the opaque owner digest. |
| `PROVIDER_CONTEXT_BINDS_ALL_REQUIRED_DIGESTS` | **PASS** | ProviderRequestContextV1 contains the complete frozen identity set. |
| `CONSUMER_LOGICAL_ID_BINDS_OPAQUE_ARM` | **PASS** | Every consumer logical identity contains the opaque Arm instance. |
| `CANDIDATE_INSTANCE_BINDS_OPAQUE_ARM` | **PASS** | Candidate instance identity is Arm-bound and rejects a foreign parent. |
| `TASK_IDS_BIND_OPAQUE_ARM` | **PASS** | Validation and matched-control task identities bind the Arm instance. |
| `NO_GLOBAL_MUTABLE_RANDOM` | **PASS** | No process-global seed mutation exists; randomized qualification uses function-local Random instances. |
| `ONE_CANONICAL_META_BOUNDARY` | **PASS** | All terminal Research rounds use the base orchestrator boundary adapter. |
| `META_WRAPPER_HAS_NO_ALTERNATIVE_BOUNDARY_OVERRIDE` | **PASS** | Meta Pilot config no longer defines transition semantics. |
| `ACTIVE_CALL_SHARING_POLICY_IS_ARM_PRIVATE` | **PASS** | The active broker defaults to no cross-Arm physical-call sharing. |
| `NO_SIBLING_ARM_ROOT_TRAVERSAL` | **PASS** | No active path computes a sibling Arm root. |

## Active sharing decision

`CallSharingPolicyV1.ARM_PRIVATE` is the active policy. B and C never
share a physical Provider call, even when semantic contexts are equal.
The explicit paired policy remains implemented only as an exact-context
contract and is not active for M6I qualification.

## Identity separation

- Provider physical identity: full Provider request/context preimage.
- Consumer logical identity: experiment + opaque Arm + round + role + physical identity.
- Candidate instance identity: opaque Arm + round + role + semantic program + local parent/task.

## Verdict

**PASS — P0=0 / P1=0.**
