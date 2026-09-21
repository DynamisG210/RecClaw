"""Real-broker, no-training M5 Canary built on the M4 neutral scheduler."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import sqlite3
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .compilation_cache import compile_campaign_program as compile_program
from recclaw_core.mechanism_space.canonical import deep_thaw

from recclaw_core.helix.contracts import CandidateEnvelope, RawResultEnvelope
from recclaw_core.helix.scientific_attribution import FusedSearchFeedbackV2

from .campaign_runtime import (
    CampaignRuntimeError,
    campaign_projection,
    campaign_runtime_profile,
    executable_mechanism,
    executable_mechanisms,
    execution_recipe_for_program,
    lineage_catalog_projection,
    program_from_proposal as campaign_program_from_proposal,
    root_parent_mechanism_id,
)
from .canary_broker import (
    CanaryBrokerCallV1,
    CanaryBrokerError,
    CodexCliCanaryBrokerV1,
    PostProviderSemanticRejectionV1,
    original_canary_prompt,
    research_canary_prompt,
)
from .canonical import canonical_value, sha256_digest
from .contracts import (
    ArmCode,
    ProducerExecutionModeV1,
    ResourceCeilingsV1,
    default_experiment_contract,
)
from .controllers import OriginalControllerV1, OriginalRuntimeAdapterV1
from .integrated_state_core import (
    ArmOwnerTokenV1,
    CanonicalParentBindingV1,
    CallSharingPolicyV1,
    CallSharingRegistryV1,
    CallSharingViolation,
    ParentBindingPolicyV1,
    ProviderRequestContextV1,
)
from .original_main import PinnedOriginalMainAdapterV1
from .precanary_orchestration import (
    ArmRoundResultV1,
    FakeProposalSessionV1,
    PreCanaryInvariantError,
    ThreeArmPreCanaryOrchestratorV1,
    probe_uid_isolation,
)
from .research_capability import (
    DISCOVERY_PRODUCERS,
    FixtureProducerBrokerV1,
    SearchMemoryWriterV1,
    StrongStaticRouterV1,
    VersionedMetaPolicyUpdaterV1,
    initial_research_policy,
)
from .research_contracts import (
    CandidateProposalV3,
    CandidateProposalV4,
    DiscriminativeExperimentPlanV1,
    DiscoveryCreditV1,
    ProducerCallRecordV1,
    ProducerSessionResultV1,
    ProposalIntentV1,
    SearchUtilityFeaturesV1,
)
from .research_science import (
    DeterministicRouterFeatureBuilderV1,
    LineageIndexV1,
    LineageRecordV1,
    matched_control_plan,
)
from .research_controller import ResearchLineControllerV1, ResearchRoundPlanV1


CANARY_SEARCH_SEED = 9011
CANARY_ROUNDS_PER_ARM = 3


def canary_budget() -> ResourceCeilingsV1:
    return ResourceCeilingsV1(
        total_input_tokens=60_000,
        total_output_tokens=20_000,
        total_billed_token_debit=80_000,
        total_proposal_count=4,
        wall_time_ms=1_200_000,
        retry_debit=0,
        proposal_attempt_debit=4,
        ordinary_executions=1,
        common_validation_count=4,
        gpu_device_time_ms=0,
        gpu_cost_microunits=0,
    )


@dataclass(frozen=True, slots=True)
class CanaryStoreContractV1:
    experiment_id: str
    arm_policies: tuple[Any, Any, Any]
    search_seeds: tuple[int, ...]
    scheduled_slots_per_arm_seed: int
    ordinary_execution_seed: int
    identity_digest: str

    @classmethod
    def create(cls) -> "CanaryStoreContractV1":
        base = default_experiment_contract()
        payload = {
            "arm_policies": [item.to_dict() for item in base.arm_policies],
            "authority": "NONE",
            "evidence_class": "DEVELOPMENT_ONLY",
            "experiment_id": "HELIX-ABC-DEVELOPMENT-CANARY-9011-V1",
            "formal_acceptance": False,
            "ordinary_execution_seed": base.ordinary_execution_seed,
            "scheduled_slots_per_arm_seed": CANARY_ROUNDS_PER_ARM,
            "search_seeds": [CANARY_SEARCH_SEED],
        }
        return cls(
            experiment_id=payload["experiment_id"],
            arm_policies=base.arm_policies,
            search_seeds=(CANARY_SEARCH_SEED,),
            scheduled_slots_per_arm_seed=CANARY_ROUNDS_PER_ARM,
            ordinary_execution_seed=base.ordinary_execution_seed,
            identity_digest=sha256_digest(payload),
        )


_TEMPLATE_BY_SIGNATURE = {
    ("CONSTRAINT_WEIGHTED", "SAMPLED_UNOBSERVED"): "ULTRAGCN",
    ("GEOMETRY_REGULARIZATION", "*"): "DIRECTAU",
    ("CONTRASTIVE_AUXILIARY", "*"): "SGL",
    ("MESSAGE_TRANSFORM_GRAPH", "*"): "NGCF",
    ("LIGHT_GRAPH_PROPAGATION", "*"): "LIGHTGCN",
    ("LATENT_FACTOR", "*"): "BPR_MF",
}

_AXIS_BY_TEMPLATE = {
    "BPR_MF": "objective",
    "DIRECTAU": "geometry",
    "LIGHTGCN": "propagation",
    "NGCF": "architecture",
    "SGL": "self_supervision",
    "ULTRAGCN": "architecture",
}


def _template_name(proposal: Mapping[str, Any]) -> str:
    recipe = proposal.get("recipe")
    if recipe is not None:
        recipe_name = str(recipe)
        if recipe_name in {"BPR_MF", "LIGHTGCN", "NGCF", "SGL"}:
            return recipe_name
        raise PreCanaryInvariantError(
            "proposal recipe is outside the closed executable set"
        )
    exact = (str(proposal["objective"]), str(proposal["sampler"]))
    if exact in _TEMPLATE_BY_SIGNATURE:
        return _TEMPLATE_BY_SIGNATURE[exact]
    for key in (
        (str(proposal["objective"]), "*"),
        (str(proposal["backbone"]), "*"),
    ):
        if key in _TEMPLATE_BY_SIGNATURE:
            return _TEMPLATE_BY_SIGNATURE[key]
    raise PreCanaryInvariantError(
        "proposal does not identify a supported executable BL-ICF recipe"
    )


def _executable_axis(proposal: Mapping[str, Any]) -> str:
    return _AXIS_BY_TEMPLATE[_template_name(proposal)]


def _load_templates(path: Path) -> dict[str, dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(item["anchor_name"]): item["program"]
        for item in document["fixtures"]
    }


def _program_from_proposal(
    proposal: Mapping[str, Any],
    templates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if "mechanism_id" in proposal:
        return campaign_program_from_proposal(proposal)
    template_name = _template_name(proposal)
    executable_axis = _executable_axis(proposal)
    program = copy.deepcopy(templates[template_name])
    payload = program["program_payload"]
    if "recipe" in proposal:
        executable_signature = f"recipe={template_name}"
    else:
        executable_signature = (
            f"recipe={template_name}; backbone={proposal['backbone']}; "
            f"objective={proposal['objective']}; sampler={proposal['sampler']}"
        )
    payload["research_question"] = (
        f"Evaluate the closed executable {template_name} recipe under the "
        "frozen development protocol. Non-executable proposal label retained "
        f"for provenance only: {proposal['candidate_label']}."
    )
    payload["core_hypothesis"] = (
        f"The {executable_axis} intervention represented exactly by "
        f"{executable_signature} may change the frozen ranking metric."
    )
    payload["mechanism_explanation"] = (
        f"Closed BL-ICF executable signature: {executable_signature}."
    )
    payload["failure_modes"] = [
        f"The executable {template_name} recipe may be neutral or harmful "
        "under the frozen budget and protocol."
    ]
    payload["expected_effects"] = {
        "coverage": "development Canary only",
        "efficiency": f"bounded runtime behavior for {template_name}",
        "relevance": f"bounded ranking change for {template_name}",
        "robustness": f"no claim beyond the exact {template_name} execution",
    }
    return program


@dataclass(slots=True)
class RealCanaryProposalBrokerV1:
    upstream: CodexCliCanaryBrokerV1
    template_path: Path
    research_controllers: dict[ArmCode, ResearchLineControllerV1]
    _research_calls: dict[str, tuple[CanaryBrokerCallV1, ...]]
    _search_feedback: dict[ArmCode, Mapping[str, Any]]
    _campaign_call_scopes: dict[str, tuple[str, ...]]
    _parent_binding_by_consumer: dict[str, CanonicalParentBindingV1]
    _research_session_wall_ms: dict[str, int]
    _provider_calls_by_round: dict[
        tuple[ArmCode, int, int], tuple[CanaryBrokerCallV1, ...]
    ]
    _provider_wall_time_by_round: dict[tuple[ArmCode, int, int], int]
    lineage_indexes: dict[ArmCode, LineageIndexV1]
    original_controller: Any
    _call_registry: CallSharingRegistryV1
    _arm_owners: dict[ArmCode, ArmOwnerTokenV1]
    _round_consumer_contexts: dict[tuple[ArmCode, int, int], dict[str, str]]
    _provider_by_consumer: dict[str, str]
    _context_by_consumer: dict[str, str]
    campaign_meta_runtime: Any | None = None
    producer_control_enabled: bool = True
    ordinary_execution_seed: int = 2026
    call_prefix: str = ""
    phase_name: str = "Canary"
    adaptive_memory: bool = False
    v13_mode: bool = False

    @classmethod
    def create(
        cls,
        *,
        upstream: CodexCliCanaryBrokerV1,
        template_path: Path,
        call_prefix: str = "",
        phase_name: str = "Canary",
        adaptive_memory: bool = False,
        campaign_meta_runtime: Any | None = None,
        producer_control_enabled: bool = True,
        ordinary_execution_seed: int = 2026,
        research_policy_override: Any | None = None,
        original_controller: Any | None = None,
        v13_mode: bool = False,
        call_sharing_policy: CallSharingPolicyV1 = (
            CallSharingPolicyV1.ARM_PRIVATE
        ),
    ) -> "RealCanaryProposalBrokerV1":
        def controller() -> ResearchLineControllerV1:
            policy = (
                research_policy_override
                if research_policy_override is not None
                else (
                    campaign_meta_runtime.control_policy
                    if campaign_meta_runtime is not None
                    else initial_research_policy()
                )
            )
            return ResearchLineControllerV1(
                producer_mode=(
                    ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
                ),
                policy=policy,
                broker=FixtureProducerBrokerV1(),
                router=StrongStaticRouterV1(),
                memory_writer=SearchMemoryWriterV1(
                    "DEVELOPMENT_ONLY/SEARCH_MEMORY"
                ),
            )

        return cls(
            upstream=upstream,
            template_path=template_path.resolve(),
            research_controllers={ArmCode.B: controller(), ArmCode.C: controller()},
            _research_calls={},
            _search_feedback={},
            _campaign_call_scopes={},
            _parent_binding_by_consumer={},
            _research_session_wall_ms={},
            _provider_calls_by_round={},
            _provider_wall_time_by_round={},
            lineage_indexes={
                ArmCode.A: LineageIndexV1(),
                ArmCode.B: LineageIndexV1(),
                ArmCode.C: LineageIndexV1(),
            },
            original_controller=(
                original_controller
                if original_controller is not None
                else OriginalRuntimeAdapterV1()
            ),
            _call_registry=CallSharingRegistryV1(
                policy=call_sharing_policy
            ),
            _arm_owners={},
            _round_consumer_contexts={},
            _provider_by_consumer={},
            _context_by_consumer={},
            campaign_meta_runtime=campaign_meta_runtime,
            producer_control_enabled=producer_control_enabled,
            ordinary_execution_seed=int(ordinary_execution_seed),
            call_prefix=call_prefix,
            phase_name=phase_name,
            adaptive_memory=adaptive_memory,
            v13_mode=v13_mode,
        )

    def bind_arm_instances(
        self,
        *,
        experiment_id: str,
        arm_to_instance: Mapping[ArmCode, str],
    ) -> None:
        """Bind Provider consumers to the scheduler's opaque Arm identities."""

        proposed = {
            arm: ArmOwnerTokenV1(
                experiment_id=str(experiment_id),
                arm=arm,
                opaque_arm_instance_id=str(arm_to_instance[arm]),
            )
            for arm in ArmCode
        }
        if len({item.opaque_arm_instance_id for item in proposed.values()}) != 3:
            raise PreCanaryInvariantError(
                "Broker requires three distinct opaque Arm instances"
            )
        if self._arm_owners and self._arm_owners != proposed:
            raise PreCanaryInvariantError(
                "Broker Arm ownership cannot be rebound"
            )
        for arm, owner in proposed.items():
            self.lineage_indexes[arm].bind_owner(
                owner.opaque_arm_instance_id
            )
        self._arm_owners = proposed

    def prepare_round_consumer_context(
        self,
        *,
        arm: ArmCode,
        search_seed: int,
        round_index: int,
        active_task_digest: str | None,
        research_task_queue_digest: str | None,
    ) -> None:
        """Freeze the scheduler-owned context before any Provider call."""

        self._round_consumer_contexts[
            (arm, int(search_seed), int(round_index))
        ] = {
            "active_task_digest": active_task_digest or "ABSENT",
            "research_task_queue_digest": (
                research_task_queue_digest or "ABSENT"
            ),
        }

    def _owner(self, arm: ArmCode) -> ArmOwnerTokenV1:
        owner = self._arm_owners.get(arm)
        if owner is None:
            # Direct unit-level broker exercises predate the neutral scheduler.
            # They remain isolated by a deterministic private owner, while every
            # campaign/Pilot path is rebound to its real opaque assignment.
            owner = ArmOwnerTokenV1(
                experiment_id="DIRECT_BROKER_EXERCISE",
                arm=arm,
                opaque_arm_instance_id=f"DIRECT-{arm.value}",
            )
            self._arm_owners[arm] = owner
            self.lineage_indexes[arm].bind_owner(
                owner.opaque_arm_instance_id
            )
        return owner

    def _model_release_digest(self) -> str:
        release = getattr(self.upstream, "release", None)
        digest = getattr(release, "release_digest", None)
        if isinstance(digest, str) and len(digest) == 64:
            return digest
        return sha256_digest(
            {
                "broker_class": type(self.upstream).__name__,
                "model": str(getattr(self.upstream, "model", "TEST_UPSTREAM")),
                "reasoning_effort": str(
                    getattr(self.upstream, "reasoning_effort", "UNSPECIFIED")
                ),
                "service_tier": str(
                    getattr(self.upstream, "service_tier", "UNSPECIFIED")
                ),
            }
        )

    def _response_schema_digest(self) -> str:
        for name in ("schema_file_sha256", "response_schema_digest"):
            value = getattr(self.upstream, name, None)
            if isinstance(value, str) and len(value) == 64:
                return value
        return sha256_digest(
            {
                "broker_class": type(self.upstream).__name__,
                "schema": "TEST_OR_LEGACY_SCHEMA",
            }
        )

    def _meta_fast_state_digest(self, arm: ArmCode) -> str:
        if arm is ArmCode.A:
            # Original control has no Research-Line Meta state.  Keep that
            # absence explicit instead of asking a B/C-only runtime to project A.
            return "ABSENT"
        if self.campaign_meta_runtime is None:
            return "ABSENT"
        projection = getattr(
            self.campaign_meta_runtime,
            "arm_private_context_digest",
            None,
        )
        if projection is None:
            return sha256_digest(
                {
                    "arm": arm.value,
                    "policy_bundle_digest": (
                        self.campaign_meta_runtime.policy_bundle_digest
                    ),
                }
            )
        return str(projection(arm))

    def _provider_context(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        search_seed: int,
        producer_role: str,
        prompt: str,
        ceilings: ResourceCeilingsV1,
        memory_view_digest: str,
        directive_set_digest: str | None,
    ) -> ProviderRequestContextV1:
        prepared = self._round_consumer_contexts.get(
            (arm, int(search_seed), int(round_index)),
            {
                "active_task_digest": "ABSENT",
                "research_task_queue_digest": "ABSENT",
            },
        )
        lineage_view_digest = self.lineage_indexes[arm].digest
        meta_fast_state_digest = self._meta_fast_state_digest(arm)
        prompt_bytes_digest = hashlib.sha256(
            prompt.encode("utf-8")
        ).hexdigest()
        complete_context_digest = sha256_digest(
            {
                "adaptive_memory": self.adaptive_memory,
                "arm_owner_digest": self._owner(arm).digest,
                "call_prefix": self.call_prefix,
                "ceilings": ceilings.to_dict(),
                "controller_policy_digest": (
                    self.research_controllers[arm].policy.digest
                    if arm in self.research_controllers
                    else "ORIGINAL_CONTROLLER"
                ),
                "directive_set_digest": directive_set_digest,
                "lineage_view_digest": lineage_view_digest,
                "memory_view_digest": memory_view_digest,
                "meta_fast_state_digest": meta_fast_state_digest,
                "phase_name": self.phase_name,
                "prepared_scheduler_context": prepared,
                "producer_role": producer_role,
                "round_index": round_index,
                "search_seed": search_seed,
                "v13_mode": self.v13_mode,
            }
        )
        return ProviderRequestContextV1(
            model_release_digest=self._model_release_digest(),
            response_schema_digest=self._response_schema_digest(),
            temperature=float(getattr(self.upstream, "temperature", 0.0)),
            timeout_policy_digest=sha256_digest(
                {
                    "retry_count": int(
                        getattr(self.upstream, "retry_count", 0)
                    ),
                    "timeout_ms": int(
                        getattr(self.upstream, "timeout_ms", 900_000)
                    ),
                }
            ),
            producer_role=producer_role,
            prompt_bytes_digest=prompt_bytes_digest,
            complete_context_digest=complete_context_digest,
            memory_view_digest=memory_view_digest,
            meta_fast_state_digest=meta_fast_state_digest,
            lineage_view_digest=lineage_view_digest,
            active_task_digest=prepared["active_task_digest"],
            research_task_queue_digest=prepared[
                "research_task_queue_digest"
            ],
            round_index=round_index,
            search_seed=search_seed,
            response_arm_neutral=True,
        )

    def _consumer_identity(
        self,
        *,
        arm: ArmCode,
        context: ProviderRequestContextV1,
        logical_prefix: str,
        logical_suffix: str = "",
    ) -> tuple[str, str]:
        physical, consumer, _decision = self._call_registry.register_request(
            owner=self._owner(arm),
            context=context,
        )
        logical_call_id = (
            f"{logical_prefix}{consumer.value}{logical_suffix}"
        )
        self._provider_by_consumer[logical_call_id] = physical.value
        self._context_by_consumer[
            logical_call_id
        ] = context.exact_request_digest
        return physical.value, logical_call_id

    def call_sharing_audit(self) -> dict[str, Any]:
        return self._call_registry.audit_projection()

    def _record_provider_usage(
        self,
        *,
        arm: ArmCode,
        search_seed: int,
        round_index: int,
        calls: Sequence[CanaryBrokerCallV1],
        wall_time_ms: int,
    ) -> None:
        key = (arm, int(search_seed), int(round_index))
        self._provider_calls_by_round[key] = tuple(calls)
        self._provider_wall_time_by_round[key] = max(
            int(wall_time_ms),
            max((item.latency_ms for item in calls), default=0),
        )

    def provider_usage_for_round(
        self,
        *,
        arm: ArmCode,
        search_seed: int,
        round_index: int,
    ) -> dict[str, Any]:
        key = (arm, int(search_seed), int(round_index))
        calls = self._provider_calls_by_round.get(key, ())
        return canonical_value(
            {
                "billed_tokens": sum(item.total_tokens for item in calls),
                "call_latencies_ms": [item.latency_ms for item in calls],
                "input_tokens": sum(item.input_tokens for item in calls),
                "output_tokens": sum(item.output_tokens for item in calls),
                "physical_call_count": len(calls),
                "proposal_count": sum(
                    len(tuple(item.response.get("proposals", ())))
                    for item in calls
                ),
                "response_digests": [
                    item.response_digest for item in calls
                ],
                "wall_time_ms": self._provider_wall_time_by_round.get(
                    key, 0
                ),
            }
        )

    def register_execution_candidate_instance(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        producer_role: str,
        semantic_program_digest: str,
        local_parent_or_task_identity: str | None,
    ) -> str:
        """Bind a selected execution to one Arm/round-local instance.

        Compiled candidate ids describe semantic content and may legitimately
        repeat.  Execution, task, lineage, and result identities must not use
        that content id as the mutable candidate instance.
        """

        return self._call_registry.register_candidate(
            owner=self._owner(arm),
            round_index=round_index,
            producer_role=producer_role,
            semantic_program_digest=semantic_program_digest,
            local_parent_or_task_identity=local_parent_or_task_identity,
        ).value

    @classmethod
    def create_v13(
        cls,
        *,
        upstream: Any,
        template_path: Path,
        repository_root: Path,
        search_seed: int,
        call_prefix: str = "",
        phase_name: str = "V13 Pilot",
        adaptive_memory: bool = True,
        campaign_meta_runtime: Any | None = None,
        producer_control_enabled: bool = True,
        ordinary_execution_seed: int = 2026,
        research_policy_override: Any | None = None,
    ) -> "RealCanaryProposalBrokerV1":
        return cls.create(
            upstream=upstream,
            template_path=template_path,
            call_prefix=call_prefix,
            phase_name=phase_name,
            adaptive_memory=adaptive_memory,
            campaign_meta_runtime=campaign_meta_runtime,
            producer_control_enabled=producer_control_enabled,
            ordinary_execution_seed=ordinary_execution_seed,
            research_policy_override=research_policy_override,
            original_controller=PinnedOriginalMainAdapterV1(
                repository_root=repository_root,
                search_seed=search_seed,
            ),
            v13_mode=True,
        )

    @property
    def bc_controller_identity_digest(self) -> str:
        identities = {
            item.identity_digest for item in self.research_controllers.values()
        }
        if len(identities) != 1:
            raise PreCanaryInvariantError("B/C controller identity diverged")
        return next(iter(identities))

    def _research_upstream_calls(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        search_seed: int,
        ceilings: ResourceCeilingsV1 | None = None,
    ) -> tuple[CanaryBrokerCallV1, ...]:
        if ceilings is None:
            return self._legacy_research_upstream_calls(
                arm=arm,
                round_index=round_index,
                search_seed=search_seed,
            )
        controller = self.research_controllers[arm]
        memory_summary = (
            dict(self._search_feedback.get(arm, {}))
            if self.adaptive_memory
            else {}
        )
        policy_projection = controller.policy.to_dict()
        meta_directives = (
            self.campaign_meta_runtime.producer_directives(
                arm=arm,
                round_index=round_index,
                memory_summary=memory_summary,
            )
            if (
                self.campaign_meta_runtime is not None
                and self.producer_control_enabled
            )
            else ()
        )
        directive_set_digest = (
            sha256_digest([item.to_dict() for item in meta_directives])
            if meta_directives
            else None
        )
        memory_policy_digest = sha256_digest(
            {
                "directives": directive_set_digest,
                "memory": memory_summary if self.adaptive_memory else {},
                "policy": policy_projection,
            }
        )
        prepared_context = self._round_consumer_contexts.get(
            (arm, int(search_seed), int(round_index)),
            {
                "active_task_digest": "ABSENT",
                "research_task_queue_digest": "ABSENT",
            },
        )
        key = sha256_digest(
            {
                "arm_owner_digest": self._owner(arm).digest,
                "ceilings": ceilings.to_dict(),
                "lineage_view_digest": self.lineage_indexes[arm].digest,
                "memory_policy_digest": memory_policy_digest,
                "meta_fast_state_digest": self._meta_fast_state_digest(arm),
                "prepared_scheduler_context": prepared_context,
                "round_index": round_index,
                "search_seed": search_seed,
                "session_identity": "RESEARCH_PROVIDER_SESSION_M6I_V1",
                "sharing_policy": self._call_registry.policy.value,
            }
        )
        if key not in self._research_calls:
            session_started_ns = time.monotonic_ns()
            memory_component = (
                f"-m{memory_policy_digest[:12]}"
                if self.adaptive_memory
                else ""
            )
            calls: list[CanaryBrokerCallV1] = []
            allocations = dict(controller.policy.producer_token_allocation)
            latest_lineage = self.lineage_indexes[arm].latest_success()
            default_lineage_root = (
                root_parent_mechanism_id(latest_lineage.mechanism_id)
                if latest_lineage is not None
                else ("LIGHTGCN" if round_index % 2 else "BPR_MF")
            )
            prioritized_axes: tuple[str, ...] = ()
            if not meta_directives:
                targeted_axes = tuple(
                    {
                        "regularization": "geometry",
                        "negative_sampling": "sampling",
                        "fusion": "message_transform",
                    }.get(axis, axis)
                    for axis in controller.policy.mechanism_axis_targeting
                )
                available_axes = tuple(
                    dict.fromkeys(
                        item.mechanism_axis
                        for item in executable_mechanisms()
                        if item.mechanism_id != default_lineage_root
                        and root_parent_mechanism_id(item.mechanism_id)
                        == default_lineage_root
                    )
                )
                prioritized_axes = tuple(
                    axis for axis in targeted_axes if axis in available_axes
                ) + tuple(
                    axis for axis in available_axes if axis not in targeted_axes
                )
            executed_mechanisms = tuple(
                str(item)
                for item in memory_summary.get(
                    "executed_mechanism_ids", ()
                )
            )
            meta_by_role = {
                item.producer_role: item for item in meta_directives
            }
            for role_index, role in enumerate(DISCOVERY_PRODUCERS):
                meta_directive = meta_by_role.get(role)
                exact_parent = (
                    latest_lineage
                    if role == "lineage_refiner"
                    else None
                )
                lineage_root = (
                    root_parent_mechanism_id(exact_parent.mechanism_id)
                    if exact_parent is not None
                    else (
                        meta_directive.lineage_root
                        if meta_directive is not None
                        else default_lineage_root
                    )
                )
                primary_axis = (
                    meta_directive.primary_axis
                    if meta_directive is not None
                    else prioritized_axes[
                        role_index % len(prioritized_axes)
                    ]
                )
                memory_query = (
                    meta_directive.memory_query
                    if meta_directive is not None
                    else "RECENT_ROLE_RELEVANT"
                )
                role_memory = dict(
                    memory_summary.get(
                        "prompt_feedback_projection",
                        {
                            "common_search_utility_slot": "ABSENT",
                            "research_task_slot": "ABSENT",
                        },
                    )
                )
                directive = {
                    "avoid_executed_semantics": True,
                    "lineage_root": lineage_root,
                    "memory_query": memory_query,
                    "parent_policy": (
                        "REQUIRE_EXACT_PRIOR_PARENT"
                        if exact_parent is not None
                        else "EXPLICIT_ROOT_REQUEST"
                        if role == "lineage_refiner"
                        else "OPTIONAL"
                    ),
                    "primary_axis": primary_axis,
                    "producer_role": role,
                    "versioned_policy_digest": controller.policy.digest,
                }
                if meta_directive is not None:
                    directive["learned_axis_score"] = (
                        meta_directive.learned_axis_score
                    )
                    directive["meta_directive_digest"] = (
                        meta_directive.digest
                    )
                is_control_slot = (
                    not self.v13_mode
                    and (
                        meta_directive.proposal_intent == "CONTROL"
                        if meta_directive is not None
                        else role == "falsification_designer"
                    )
                )
                if self.v13_mode and role == "falsification_designer":
                    directive.update(
                        {
                            "proposal_intent": "FALSIFICATION",
                            "scientific_role": (
                                "discriminative competing-hypothesis "
                                "experiment"
                            ),
                        }
                    )
                if is_control_slot:
                    primary_axis = executable_mechanism(
                        lineage_root
                    ).mechanism_axis
                    directive.update(
                        {
                            "primary_axis": primary_axis,
                            "proposal_intent": "CONTROL",
                            "required_mechanism_id": (
                                meta_directive.required_mechanism_id
                                if meta_directive is not None
                                else lineage_root
                            ),
                            "scientific_role": (
                                "matched parent control and competing "
                                "explanation anchor"
                            ),
                        }
                    )
                per_call_ceiling = min(
                    int(
                        getattr(
                            self.upstream,
                            "max_total_tokens_per_call",
                            ceilings.total_billed_token_debit,
                        )
                    ),
                    max(
                        1,
                        int(
                            ceilings.total_billed_token_debit
                            * float(
                                meta_directive.token_share
                                if meta_directive is not None
                                else allocations.get(role, 0.25)
                            )
                        ),
                    ),
                )
                catalog = lineage_catalog_projection(
                    root_mechanism_id=lineage_root,
                    targeted_axes=(primary_axis,),
                    executed_mechanism_ids=executed_mechanisms,
                    control_only=is_control_slot,
                )
                prompt = research_canary_prompt(
                    role=role,
                    round_index=round_index,
                    search_seed=search_seed,
                    phase_name=self.phase_name,
                    memory_summary=role_memory,
                    catalog_projection=catalog,
                    policy_directive=directive,
                    token_ceiling=per_call_ceiling,
                )
                provider_context = self._provider_context(
                    arm=arm,
                    round_index=round_index,
                    search_seed=search_seed,
                    producer_role=role,
                    prompt=prompt,
                    ceilings=ceilings,
                    memory_view_digest=sha256_digest(memory_summary),
                    directive_set_digest=directive_set_digest,
                )
                _physical_call_id, logical_call_id = (
                    self._consumer_identity(
                        arm=arm,
                        context=provider_context,
                        logical_prefix=f"{self.call_prefix}research-",
                        logical_suffix=f"-{role}",
                    )
                )
                self._campaign_call_scopes[logical_call_id] = tuple(
                    str(item["mechanism_id"])
                    for item in catalog["mechanisms"]
                )
                self._parent_binding_by_consumer[logical_call_id] = (
                    CanonicalParentBindingV1(
                        policy=ParentBindingPolicyV1(
                            str(directive["parent_policy"])
                        ),
                        runtime_parent_candidate_id=(
                            exact_parent.proposal_candidate_id
                            if exact_parent is not None
                            else None
                        ),
                    )
                )
                try:
                    call = self._upstream_call(
                        logical_call_id=logical_call_id,
                        proposal_generation_session_id=(
                            "research-consumer-session-v1:"
                            + sha256_digest(
                                {
                                    "arm_owner_digest": self._owner(arm).digest,
                                    "memory_component": memory_component,
                                    "round_index": round_index,
                                    "search_seed": search_seed,
                                }
                            )
                        ),
                        prompt=prompt,
                        expected_proposal_count=1,
                        max_total_tokens=per_call_ceiling,
                    )
                except CanaryBrokerError as error:
                    error.physical_call_count += len(calls)
                    error.input_tokens += sum(
                        item.input_tokens for item in calls
                    )
                    error.output_tokens += sum(
                        item.output_tokens for item in calls
                    )
                    error.billed_tokens += sum(
                        item.total_tokens for item in calls
                    )
                    error.wall_time_ms += sum(
                        item.latency_ms for item in calls
                    )
                    raise
                calls.append(call)
                self._record_provider_usage(
                    arm=arm,
                    search_seed=search_seed,
                    round_index=round_index,
                    calls=calls,
                    wall_time_ms=max(
                        0,
                        int(
                            (
                                time.monotonic_ns()
                                - session_started_ns
                            )
                            / 1_000_000
                        ),
                    ),
                )
            self._research_calls[key] = tuple(calls)
            self._research_session_wall_ms[key] = max(
                0,
                int((time.monotonic_ns() - session_started_ns) / 1_000_000),
            )
            self._record_provider_usage(
                arm=arm,
                search_seed=search_seed,
                round_index=round_index,
                calls=calls,
                wall_time_ms=self._research_session_wall_ms[key],
            )
        return self._research_calls[key]

    def _legacy_research_upstream_calls(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        search_seed: int,
    ) -> tuple[CanaryBrokerCallV1, ...]:
        memory_summary = (
            dict(self._search_feedback.get(arm, {}))
            if self.adaptive_memory
            else {}
        )
        memory_digest = sha256_digest(
            memory_summary if self.adaptive_memory else {}
        )
        key = sha256_digest(
            {
                "arm_owner_digest": self._owner(arm).digest,
                "lineage_view_digest": self.lineage_indexes[arm].digest,
                "memory_digest": memory_digest,
                "round_index": round_index,
                "search_seed": search_seed,
                "session_identity": "LEGACY_RESEARCH_PROVIDER_SESSION_M6I_V1",
            }
        )
        if key not in self._research_calls:
            memory_component = (
                f"-m{memory_digest[:12]}" if self.adaptive_memory else ""
            )
            calls: list[CanaryBrokerCallV1] = []
            prompt_memory = dict(
                memory_summary.get(
                    "prompt_feedback_projection",
                    {
                        "common_search_utility_slot": "ABSENT",
                        "research_task_slot": "ABSENT",
                    },
                )
            )
            for role in DISCOVERY_PRODUCERS:
                prompt = research_canary_prompt(
                    role=role,
                    round_index=round_index,
                    search_seed=search_seed,
                    phase_name=self.phase_name,
                    memory_summary=prompt_memory,
                )
                provider_context = self._provider_context(
                    arm=arm,
                    round_index=round_index,
                    search_seed=search_seed,
                    producer_role=role,
                    prompt=prompt,
                    ceilings=canary_budget(),
                    memory_view_digest=memory_digest,
                    directive_set_digest=None,
                )
                _physical_call_id, logical_call_id = (
                    self._consumer_identity(
                        arm=arm,
                        context=provider_context,
                        logical_prefix=f"{self.call_prefix}research-",
                        logical_suffix=f"-{role}",
                    )
                )
                try:
                    call = self._upstream_call(
                        logical_call_id=logical_call_id,
                        proposal_generation_session_id=(
                            "legacy-research-consumer-session-v1:"
                            + sha256_digest(
                                {
                                    "arm_owner_digest": self._owner(arm).digest,
                                    "memory_component": memory_component,
                                    "round_index": round_index,
                                    "search_seed": search_seed,
                                }
                            )
                        ),
                        prompt=prompt,
                        expected_proposal_count=1,
                    )
                except CanaryBrokerError as error:
                    error.physical_call_count += len(calls)
                    error.input_tokens += sum(item.input_tokens for item in calls)
                    error.output_tokens += sum(item.output_tokens for item in calls)
                    error.billed_tokens += sum(item.total_tokens for item in calls)
                    error.wall_time_ms += sum(item.latency_ms for item in calls)
                    raise
                calls.append(call)
            self._research_calls[key] = tuple(calls)
        return self._research_calls[key]

    def _upstream_call(
        self,
        *,
        logical_call_id: str,
        proposal_generation_session_id: str,
        prompt: str,
        expected_proposal_count: int,
        max_total_tokens: int | None = None,
    ) -> CanaryBrokerCallV1:
        def accepts_token_ceiling(callable_object: Any) -> bool:
            parameters = inspect.signature(callable_object).parameters.values()
            return any(
                item.name == "max_total_tokens"
                or item.kind is inspect.Parameter.VAR_KEYWORD
                for item in parameters
            )

        call_with_session = getattr(self.upstream, "call_with_session", None)
        if call_with_session is not None:
            kwargs = {
                "logical_call_id": logical_call_id,
                "proposal_generation_session_id": (
                    proposal_generation_session_id
                ),
                "prompt": prompt,
                "expected_proposal_count": expected_proposal_count,
            }
            if (
                max_total_tokens is not None
                and accepts_token_ceiling(call_with_session)
            ):
                kwargs["max_total_tokens"] = max_total_tokens
            return call_with_session(**kwargs)
        kwargs = {
            "logical_call_id": logical_call_id,
            "prompt": prompt,
            "expected_proposal_count": expected_proposal_count,
        }
        if (
            max_total_tokens is not None
            and accepts_token_ceiling(self.upstream.call)
        ):
            kwargs["max_total_tokens"] = max_total_tokens
        return self.upstream.call(**kwargs)

    def record_search_feedback(
        self,
        arm: ArmCode,
        feedback_projection: Mapping[str, Any],
        *,
        executed_mechanism_id: str | None,
        execution_succeeded: bool,
    ) -> None:
        projection = dict(feedback_projection)
        prior = dict(self._search_feedback.get(arm, {}))
        executed = list(prior.get("executed_mechanism_ids", ()))
        if (
            execution_succeeded
            and executed_mechanism_id
            and executed_mechanism_id not in executed
        ):
            executed.append(str(executed_mechanism_id))
        projection["executed_mechanism_ids"] = executed[-32:]
        self._search_feedback[arm] = canonical_value(projection)

    def matched_comparator_for(
        self,
        *,
        arm: ArmCode,
        proposal: CandidateProposalV4,
        protocol_digest: str,
        observation_seed: str,
    ) -> Any | None:
        return self.lineage_indexes[arm].matched_comparator(
            proposal,
            protocol_digest=protocol_digest,
            observation_seed=observation_seed,
        )

    def lineage_record_for(
        self,
        *,
        arm: ArmCode,
        proposal_candidate_id: str,
        protocol_digest: str,
    ) -> LineageRecordV1 | None:
        return self.lineage_indexes[arm].latest_for_candidate(
            proposal_candidate_id,
            protocol_digest=protocol_digest,
        )

    def record_lineage_outcome(
        self,
        *,
        arm: ArmCode,
        proposal: CandidateProposalV4,
        runtime_candidate_id: str,
        mechanism_program_digest: str,
        mechanism_semantics_digest: str,
        protocol_digest: str,
        observation_seed: str,
        run_status: str,
        normalized_metrics: Mapping[str, Any],
        result_digest: str,
        round_index: int,
    ) -> None:
        metric_name = next(
            (
                name
                for name in ("ndcg@10", "ndcg")
                if isinstance(normalized_metrics.get(name), (int, float))
            ),
            "ndcg@10",
        )
        metric = normalized_metrics.get(metric_name)
        self.lineage_indexes[arm].record(
            LineageRecordV1(
                proposal_candidate_id=proposal.candidate_id,
                runtime_candidate_id=runtime_candidate_id,
                mechanism_id=proposal.mechanism_id,
                mechanism_axis=proposal.mechanism_axis,
                mechanism_program_digest=mechanism_program_digest,
                mechanism_semantics_digest=mechanism_semantics_digest,
                parent_candidate_id=proposal.parent_candidate_id,
                protocol_digest=protocol_digest,
                observation_seed=observation_seed,
                run_status=run_status,
                metric_name=metric_name,
                metric_value=(
                    float(metric)
                    if isinstance(metric, (int, float))
                    else None
                ),
                result_digest=result_digest,
                round_index=round_index,
                mechanism_program=proposal.mechanism_program,
                owner_arm_instance_id=(
                    self._owner(arm).opaque_arm_instance_id
                ),
            )
        )

    def record_support_outcome(
        self,
        *,
        arm: ArmCode,
        proposal_candidate_id: str,
        runtime_candidate_id: str,
        mechanism_id: str,
        mechanism_axis: str,
        mechanism_program_digest: str,
        mechanism_semantics_digest: str,
        protocol_digest: str,
        observation_seed: str,
        run_status: str,
        normalized_metrics: Mapping[str, Any],
        result_digest: str,
        round_index: int,
        mechanism_program: Mapping[str, Any],
        parent_candidate_id: str | None = None,
    ) -> None:
        metric_name = next(
            (
                name
                for name in ("ndcg@10", "ndcg")
                if isinstance(normalized_metrics.get(name), (int, float))
            ),
            "ndcg@10",
        )
        metric = normalized_metrics.get(metric_name)
        self.lineage_indexes[arm].record(
            LineageRecordV1(
                proposal_candidate_id=proposal_candidate_id,
                runtime_candidate_id=runtime_candidate_id,
                mechanism_id=mechanism_id,
                mechanism_axis=mechanism_axis,
                mechanism_program_digest=mechanism_program_digest,
                mechanism_semantics_digest=mechanism_semantics_digest,
                parent_candidate_id=parent_candidate_id,
                protocol_digest=protocol_digest,
                observation_seed=observation_seed,
                run_status=run_status,
                metric_name=metric_name,
                metric_value=(
                    float(metric)
                    if isinstance(metric, (int, float))
                    else None
                ),
                result_digest=result_digest,
                round_index=round_index,
                mechanism_program=mechanism_program,
                owner_arm_instance_id=(
                    self._owner(arm).opaque_arm_instance_id
                ),
            )
        )

    def _proposal_from_call(
        self,
        *,
        arm: ArmCode,
        round_index: int = 1,
        session_id: str,
        role: str,
        call: CanaryBrokerCallV1,
        proposal: Mapping[str, Any],
    ) -> CandidateProposalV3 | CandidateProposalV4:
        program = campaign_program_from_proposal(proposal)
        compiled = compile_program(program)
        resolved_recipe = execution_recipe_for_program(program)
        mechanism = executable_mechanism(
            str(resolved_recipe["mechanism_id"])
        )
        utility = dict(proposal["utility_features"])
        cost = (
            0.6
            if len(mechanism.operator_ids) == 2
            or bool(
                set(mechanism.operator_ids)
                & {
                    "LGCN_AUX_ALIGNMENT",
                    "LGCN_DUAL_PATH",
                    "LGCN_EDGE_DROPOUT",
                }
            )
            else 0.3
        )
        intent = ProposalIntentV1(str(proposal["proposal_intent"]))
        discovery_credit = (
            DiscoveryCreditV1.DISCOVERY
            if self.v13_mode
            else (
                DiscoveryCreditV1.NON_DISCOVERY_CONTROL
                if intent is ProposalIntentV1.CONTROL
                else DiscoveryCreditV1.DISCOVERY
            )
        )
        lineage = self.lineage_indexes[arm]
        if self.v13_mode:
            binding = self._parent_binding_by_consumer.get(
                call.logical_call_id
            )
            if binding is None:
                expected_parent = (
                    lineage.latest_success()
                    if role == "lineage_refiner"
                    else None
                )
                binding = CanonicalParentBindingV1(
                    policy=(
                        ParentBindingPolicyV1.REQUIRE_EXACT_PRIOR_PARENT
                        if expected_parent is not None
                        else ParentBindingPolicyV1.EXPLICIT_ROOT_REQUEST
                        if role == "lineage_refiner"
                        else ParentBindingPolicyV1.OPTIONAL
                    ),
                    runtime_parent_candidate_id=(
                        expected_parent.proposal_candidate_id
                        if expected_parent is not None
                        else None
                    ),
                )
        else:
            binding = CanonicalParentBindingV1(
                policy=ParentBindingPolicyV1.OPTIONAL,
                runtime_parent_candidate_id=None,
            )
        parent_candidate_id = self._call_registry.resolve_candidate_parent(
            owner=self._owner(arm),
            binding=binding,
            provider_parent_candidate_id=proposal.get(
                "parent_candidate_id"
            ),
        )
        parent = (
            lineage.latest_for_candidate(str(parent_candidate_id))
            if parent_candidate_id is not None
            else None
        )
        if self.v13_mode and parent_candidate_id is not None and parent is None:
            raise PreCanaryInvariantError(
                "runtime-bound Research parent is absent from exact lineage"
            )
        if self.v13_mode and role == "lineage_refiner":
            expected_parent = lineage.latest_success()
            if expected_parent is None and parent_candidate_id is not None:
                raise PreCanaryInvariantError(
                    "root lineage request cannot bind a parent"
                )
            if expected_parent is not None and str(parent_candidate_id) != (
                expected_parent.proposal_candidate_id
            ):
                raise PreCanaryInvariantError(
                    "lineage_refiner did not bind its exact prior parent"
                )
        candidate_instance = self._call_registry.register_candidate(
            owner=self._owner(arm),
            round_index=round_index,
            producer_role=role,
            semantic_program_digest=str(
                compiled.mechanism_semantics_digest
            ),
            local_parent_or_task_identity=parent_candidate_id,
        )
        proposal_candidate_id = candidate_instance.value
        if not self.v13_mode:
            return CandidateProposalV3(
                candidate_id=proposal_candidate_id,
                producer_id=f"producer-{role}",
                producer_role=role,
                proposal_intent=intent,
                discovery_credit=discovery_credit,
                mechanism_id=mechanism.mechanism_id,
                mechanism_axis=mechanism.mechanism_axis,
                mechanism_program=program,
                candidate_label=str(proposal["candidate_label"]),
                mechanism_hypothesis=str(
                    proposal["mechanism_hypothesis"]
                ),
                competing_hypothesis=str(
                    proposal["competing_hypothesis"]
                ),
                predicted_outcome_signature=str(
                    proposal["predicted_outcome_signature"]
                ),
                failure_mode=str(proposal["failure_mode"]),
                utility_features=SearchUtilityFeaturesV1(
                    runnable_probability=1.0,
                    useful_signal=float(utility["useful_signal"]),
                    frontier_potential=float(
                        utility["frontier_potential"]
                    ),
                    information_gain=float(utility["information_gain"]),
                    cost=cost,
                    blocker_risk=0.05,
                ),
                parent_candidate_id=parent_candidate_id,
                assigned_before_call=True,
                post_hoc_relabel=False,
            )
        diagnostic = SearchUtilityFeaturesV1(
            runnable_probability=float(
                utility.get("runnable_probability", 0.5)
            ),
            useful_signal=float(utility["useful_signal"]),
            frontier_potential=float(utility["frontier_potential"]),
            information_gain=float(utility["information_gain"]),
            cost=float(utility.get("cost", cost)),
            blocker_risk=float(utility.get("blocker_risk", 0.5)),
        )
        derived_features, feature_evidence = (
            DeterministicRouterFeatureBuilderV1().build(
                compile_valid=bool(compiled.is_valid),
                handler_available=bool(mechanism.entrypoint),
                materializer_available=True,
                mechanism_id=mechanism.mechanism_id,
                mechanism_depth=(
                    lineage.mechanism_depth(
                        str(parent_candidate_id)
                        if parent_candidate_id is not None
                        else None
                    )
                    + int(parent_candidate_id is not None)
                ),
                estimated_cost=cost,
                semantics_digest=str(
                    compiled.mechanism_semantics_digest
                ),
                parent_available=(
                    parent is not None
                    or parent_candidate_id is None
                ),
                lineage=lineage,
                llm_diagnostic=diagnostic,
            )
        )
        protocol_digest = str(
            campaign_runtime_profile()["development_protocol_digest"]
        )
        root_program = campaign_program_from_proposal(
            {
                "mechanism_id": root_parent_mechanism_id(
                    mechanism.mechanism_id
                )
            }
        )
        root_report = compile_program(root_program)
        control_plan = matched_control_plan(
            lineage=lineage,
            primary_candidate_id=proposal_candidate_id,
            parent_candidate_id=(
                str(parent_candidate_id)
                if parent_candidate_id is not None
                else None
            ),
            changed_axis=mechanism.mechanism_axis,
            mechanism_hypothesis=str(proposal["mechanism_hypothesis"]),
            protocol_digest=protocol_digest,
            queued_comparator_candidate_id=str(root_report.candidate_id),
            queued_comparator_program_digest=str(
                root_report.mechanism_program_digest
            ),
            observation_seed=str(self.ordinary_execution_seed),
        )
        discriminative_plan = (
            DiscriminativeExperimentPlanV1(
                competing_hypotheses=(
                    str(proposal["mechanism_hypothesis"]),
                    str(proposal["competing_hypothesis"]),
                ),
                predicted_outcome_signature=str(
                    proposal["predicted_outcome_signature"]
                ),
                primary_candidate=proposal_candidate_id,
                matched_control_plan=control_plan,
                falsifier=str(proposal["failure_mode"]),
                next_decision_rule=(
                    "retain the mechanism explanation only if the exact "
                    "matched comparison has the predicted sign"
                ),
            )
            if role == "falsification_designer"
            else None
        )
        return CandidateProposalV4(
            candidate_id=proposal_candidate_id,
            producer_id=f"producer-{role}",
            producer_role=role,
            proposal_intent=intent,
            discovery_credit=discovery_credit,
            mechanism_id=mechanism.mechanism_id,
            mechanism_axis=mechanism.mechanism_axis,
            mechanism_program=program,
            candidate_label=str(proposal["candidate_label"]),
            mechanism_hypothesis=str(proposal["mechanism_hypothesis"]),
            competing_hypothesis=str(proposal["competing_hypothesis"]),
            predicted_outcome_signature=str(
                proposal["predicted_outcome_signature"]
            ),
            failure_mode=str(proposal["failure_mode"]),
            utility_features=derived_features,
            feature_evidence=feature_evidence,
            matched_control_plan=control_plan,
            discriminative_plan=discriminative_plan,
            parent_candidate_id=(
                str(parent_candidate_id)
                if parent_candidate_id is not None
                else None
            ),
            assigned_before_call=True,
            post_hoc_relabel=False,
        )

    def _typed_research_session(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        search_seed: int,
        ceilings: ResourceCeilingsV1,
        calls: Sequence[CanaryBrokerCallV1],
    ) -> ProducerSessionResultV1:
        session_id = (
            "producer-session-v1:"
            + sha256_digest(
                {
                    "arm_owner_digest": self._owner(arm).digest,
                    "call_prefix": self.call_prefix,
                    "round_index": round_index,
                    "search_seed": search_seed,
                }
            )
        )
        proposals: list[CandidateProposalV3 | CandidateProposalV4] = []
        call_records: list[ProducerCallRecordV1] = []
        for role, call in zip(DISCOVERY_PRODUCERS, calls, strict=True):
            raw = dict(call.response["proposals"][0])
            raw_program = campaign_program_from_proposal(raw)
            raw["mechanism_id"] = str(
                execution_recipe_for_program(raw_program)["mechanism_id"]
            )
            expected_intent = (
                ProposalIntentV1.FALSIFICATION
                if self.v13_mode and role == "falsification_designer"
                else (
                    ProposalIntentV1.CONTROL
                    if role == "falsification_designer"
                    else ProposalIntentV1.DISCOVERY
                )
            )
            if ProposalIntentV1(str(raw["proposal_intent"])) is not expected_intent:
                raise PreCanaryInvariantError(
                    "Producer response violated its preassigned scientific role"
                )
            if str(raw["mechanism_id"]) not in self._campaign_call_scopes.get(
                call.logical_call_id, ()
            ):
                raise PreCanaryInvariantError(
                    "Producer response is outside its preassigned mechanism scope"
                )
            typed = self._proposal_from_call(
                arm=arm,
                round_index=round_index,
                session_id=session_id,
                role=role,
                call=call,
                proposal=raw,
            )
            proposals.append(typed)
            call_records.append(
                ProducerCallRecordV1(
                    session_id=session_id,
                    mode=ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1,
                    physical_call_id=self._provider_by_consumer[
                        call.logical_call_id
                    ],
                    producer_id=typed.producer_id,
                    producer_role=role,
                    request_digest=call.request_digest,
                    response_digest=call.response_digest,
                    context_digest=self._context_by_consumer[
                        call.logical_call_id
                    ],
                    memory_digest=sha256_digest(
                        self._search_feedback.get(arm, {})
                    ),
                    prompt_digest=call.request_digest,
                    rng_digest=sha256_digest(
                        {"role": role, "search_seed": search_seed}
                    ),
                    candidate_ids=(typed.candidate_id,),
                    input_tokens=call.input_tokens,
                    output_tokens=call.output_tokens,
                    billed_tokens=call.total_tokens,
                    latency_ms=call.latency_ms,
                )
            )
        session_wall_time_ms = next(
            (
                self._research_session_wall_ms[key]
                for key, value in self._research_calls.items()
                if value == tuple(calls)
                and key in self._research_session_wall_ms
            ),
            sum(item.latency_ms for item in calls),
        )
        session_wall_time_ms = max(
            session_wall_time_ms,
            max((item.latency_ms for item in calls), default=0),
        )
        return ProducerSessionResultV1(
            session_id=session_id,
            mode=ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1,
            calls=tuple(call_records),
            proposals=tuple(proposals),
            total_resource_envelope_digest=sha256_digest(ceilings.to_dict()),
            base_model_ref=str(
                getattr(self.upstream, "model", "TEST_UPSTREAM")
            ),
            bl_projection_digest=str(campaign_projection()["projection_digest"]),
            candidate_schema_ref=(
                "CandidateProposalV4"
                if self.v13_mode
                else "CandidateProposalV3"
            ),
            proposal_count=len(proposals),
            physical_call_count=len(call_records),
            input_tokens=sum(item.input_tokens for item in calls),
            output_tokens=sum(item.output_tokens for item in calls),
            billed_tokens=sum(item.total_tokens for item in calls),
            session_latency_ms=session_wall_time_ms,
        )

    def generate(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        search_seed: int,
        drafts: Sequence[Mapping[str, Any]],
        ceilings: ResourceCeilingsV1,
    ) -> FakeProposalSessionV1:
        del drafts
        templates = _load_templates(self.template_path)
        if self.campaign_meta_runtime is None and not self.v13_mode:
            return self._legacy_generate(
                arm=arm,
                round_index=round_index,
                search_seed=search_seed,
                ceilings=ceilings,
                templates=templates,
            )
        if arm is ArmCode.A:
            call: CanaryBrokerCallV1 | None = None
            if self.original_controller.refresh_required(round_index):
                prompt = original_canary_prompt(
                    round_index=round_index,
                    search_seed=search_seed,
                    phase_name=self.phase_name,
                    catalog_projection=campaign_projection(),
                    original_state=self.original_controller.state_projection(),
                )
                provider_context = replace(
                    self._provider_context(
                        arm=arm,
                        round_index=round_index,
                        search_seed=search_seed,
                        producer_role="original_controller",
                        prompt=prompt,
                        ceilings=ceilings,
                        memory_view_digest=sha256_digest(
                            self.original_controller.state_projection()
                        ),
                        directive_set_digest=None,
                    ),
                    response_arm_neutral=False,
                )
                _physical_call_id, logical_call_id = (
                    self._consumer_identity(
                        arm=arm,
                        context=provider_context,
                        logical_prefix=f"{self.call_prefix}original-",
                    )
                )
                call = self._upstream_call(
                    logical_call_id=logical_call_id,
                    proposal_generation_session_id=(
                        "original-consumer-session-v1:"
                        + sha256_digest(
                            {
                                "arm_owner_digest": self._owner(arm).digest,
                                "round_index": round_index,
                                "search_seed": search_seed,
                            }
                        )
                    ),
                    prompt=prompt,
                    expected_proposal_count=4,
                    max_total_tokens=min(
                        ceilings.total_billed_token_debit,
                        int(
                            getattr(
                                self.upstream,
                                "max_total_tokens_per_call",
                                ceilings.total_billed_token_debit,
                            )
                        ),
                    ),
                )
                self._record_provider_usage(
                    arm=arm,
                    search_seed=search_seed,
                    round_index=round_index,
                    calls=(call,),
                    wall_time_ms=call.latency_ms,
                )
                try:
                    raw_proposals = tuple(
                        {
                            **dict(item),
                            "mechanism_id": str(
                                execution_recipe_for_program(
                                    campaign_program_from_proposal(item)
                                )["mechanism_id"]
                            ),
                        }
                        for item in call.response["proposals"]
                    )
                    if len(
                        {
                            str(item["mechanism_id"])
                            for item in raw_proposals
                        }
                    ) != len(raw_proposals):
                        raise PreCanaryInvariantError(
                            "Original refresh contains duplicate mechanisms"
                        )
                    self.original_controller.install_proposals(
                        round_index=round_index,
                        proposals=tuple(
                            {
                                **dict(item),
                                "mechanism_program": (
                                    campaign_program_from_proposal(item)
                                ),
                            }
                            for item in raw_proposals
                        ),
                    )
                except (
                    CallSharingViolation,
                    CampaignRuntimeError,
                    PreCanaryInvariantError,
                    ValueError,
                ) as error:
                    raise PostProviderSemanticRejectionV1(
                        failure_class=(
                            "ORIGINAL_RESPONSE_SEMANTIC_REJECTION"
                        ),
                        cause=error,
                    ) from error
            cached = self.original_controller.cached_proposals
            programs = tuple(
                dict(item["mechanism_program"]) for item in cached
            )
            return FakeProposalSessionV1(
                validation_programs=programs,
                ordered_programs=programs,
                selected_candidate_id="",
                physical_call_count=1 if call is not None else 0,
                input_tokens=call.input_tokens if call is not None else 0,
                output_tokens=call.output_tokens if call is not None else 0,
                billed_tokens=call.total_tokens if call is not None else 0,
                proposal_count=len(call.response["proposals"]) if call else 0,
                proposal_session_digest=sha256_digest(
                    {
                        "call": call.to_dict() if call is not None else None,
                        "mode": (
                            "ORIGINAL_REFRESH"
                            if call is not None
                            else "ORIGINAL_CACHED_SLATE"
                        ),
                        "original_controller": (
                            self.original_controller.state_projection()
                        ),
                    }
                ),
                route_trace_digest=None,
                research_plan=None,
                broker_call_latencies_ms=(
                    (call.latency_ms,) if call is not None else ()
                ),
                proposal_session_wall_time_ms=(
                    call.latency_ms if call is not None else 0
                ),
            )
        calls = self._research_upstream_calls(
            arm=arm,
            round_index=round_index,
            search_seed=search_seed,
            ceilings=ceilings,
        )
        try:
            session = self._typed_research_session(
                arm=arm,
                round_index=round_index,
                search_seed=search_seed,
                ceilings=ceilings,
                calls=calls,
            )
        except (
            CallSharingViolation,
            CampaignRuntimeError,
            PreCanaryInvariantError,
            ValueError,
        ) as error:
            raise PostProviderSemanticRejectionV1(
                failure_class="RESEARCH_RESPONSE_SEMANTIC_REJECTION",
                cause=error,
            ) from error
        return FakeProposalSessionV1(
            validation_programs=tuple(
                item.mechanism_program for item in session.proposals
            ),
            ordered_programs=tuple(
                item.mechanism_program for item in session.proposals
            ),
            selected_candidate_id="",
            physical_call_count=len(calls),
            input_tokens=sum(item.input_tokens for item in calls),
            output_tokens=sum(item.output_tokens for item in calls),
            billed_tokens=sum(item.total_tokens for item in calls),
            proposal_count=session.proposal_count,
            proposal_session_digest=sha256_digest(
                {
                    "campaign_meta_runtime": (
                        self.campaign_meta_runtime.policy_bundle_digest
                        if self.campaign_meta_runtime is not None
                        else None
                    ),
                    "typed_session_digest": session.digest,
                    "upstream_call_digests": [
                        item.response_digest for item in calls
                    ],
                }
            ),
            route_trace_digest=None,
            research_plan=None,
            research_proposals=session.proposals,
            producer_session=session,
            broker_call_latencies_ms=tuple(
                item.latency_ms for item in session.calls
            ),
            proposal_session_wall_time_ms=session.session_latency_ms,
        )

    def _legacy_generate(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        search_seed: int,
        ceilings: ResourceCeilingsV1,
        templates: Mapping[str, Mapping[str, Any]],
    ) -> FakeProposalSessionV1:
        if arm is ArmCode.A:
            call = self._upstream_call(
                logical_call_id=(
                    f"{self.call_prefix}original-{search_seed}-{round_index}"
                ),
                proposal_generation_session_id=(
                    f"{self.call_prefix}original-session-"
                    f"{search_seed}-{round_index}"
                ),
                prompt=original_canary_prompt(
                    round_index=round_index,
                    search_seed=search_seed,
                    phase_name=self.phase_name,
                ),
                expected_proposal_count=4,
            )
            programs = tuple(
                _program_from_proposal(item, templates)
                for item in call.response["proposals"]
            )
            fixture = tuple(
                {
                    "candidate_id": compile_program(program).candidate_id,
                    "original_score": float(4 - index),
                }
                for index, program in enumerate(programs)
            )
            original = OriginalControllerV1()
            proposals = original.propose(
                {
                    "execution_mode": "M0_FIXTURE_ONLY",
                    "fixture_proposals": fixture,
                },
                {"space": "BL_ICF_EXECUTABLE_PROFILE_V2"},
                {"proposal_count": 4},
            )
            selected = original.select(
                proposals,
                {"execution_mode": "M0_FIXTURE_ONLY"},
                {"proposal_count": 4},
            )
            return FakeProposalSessionV1(
                validation_programs=programs,
                ordered_programs=programs,
                selected_candidate_id=str(selected["candidate_id"]),
                physical_call_count=1,
                input_tokens=call.input_tokens,
                output_tokens=call.output_tokens,
                billed_tokens=call.total_tokens,
                proposal_count=4,
                proposal_session_digest=sha256_digest(
                    {
                        "call": call.to_dict(),
                        "mode": "ORIGINAL_SINGLE_INVOCATION",
                    }
                ),
                route_trace_digest=None,
                research_plan=None,
            )
        calls = self._legacy_research_upstream_calls(
            arm=arm,
            round_index=round_index,
            search_seed=search_seed,
        )
        typed_drafts = []
        for role, call in zip(DISCOVERY_PRODUCERS, calls, strict=True):
            proposal = dict(call.response["proposals"][0])
            expected_intent = (
                "FALSIFICATION"
                if role == "falsification_designer"
                else "DISCOVERY"
            )
            if proposal["proposal_intent"] != expected_intent:
                raise PreCanaryInvariantError(
                    "Producer response violated its frozen role intent"
                )
            typed_drafts.append(
                {
                    "mechanism_axis": _executable_axis(proposal),
                    "mechanism_program": _program_from_proposal(
                        proposal, templates
                    ),
                    "proposal_intent": proposal["proposal_intent"],
                    "utility_features": proposal["utility_features"],
                }
            )
        controller = self.research_controllers[arm]
        role_memory = {
            role: {
                "prior_round_digest": sha256_digest(
                    {
                        "controller_policy": controller.policy.digest,
                        "search_memory": (
                            controller.memory_writer.head.digest
                            if controller.memory_writer.head
                            else None
                        ),
                        "role": role,
                        "round": round_index - 1,
                    }
                ),
                "search_memory_digest": (
                    controller.memory_writer.head.digest
                    if controller.memory_writer.head
                    else None
                ),
            }
            for role in DISCOVERY_PRODUCERS
        }
        typed_session = controller.broker.dispatch(
            session_id=(
                f"{self.call_prefix or 'm5-'}research-"
                f"{search_seed}-{round_index}"
            ),
            mode=controller.producer_mode,
            drafts=typed_drafts,
            context={"round_index": round_index, "search_seed": search_seed},
            role_memory=role_memory,
            seed=search_seed + round_index,
            ceilings=ceilings,
            policy_projection=controller.policy.to_dict(),
        )
        route = controller.router.route(
            typed_session.proposals,
            policy_projection=controller.policy.to_dict(),
        )
        if route.selected_candidate_id is None:
            raise PreCanaryInvariantError("real Research route selected nothing")
        by_id = {item.candidate_id: item for item in typed_session.proposals}
        ordered = tuple(
            by_id[item].mechanism_program
            for item in route.ranked_candidate_ids
        )
        plan = ResearchRoundPlanV1(
            round_index=round_index,
            proposal_session_digest=typed_session.digest,
            route_trace_digest=route.digest,
            selected_candidate_id=route.selected_candidate_id,
            physical_call_count=len(calls),
            proposal_count=typed_session.proposal_count,
            ordinary_execution_opportunities=1,
            plan_status="SELECTED",
            policy_digest=controller.policy.digest,
        )
        return FakeProposalSessionV1(
            validation_programs=tuple(
                item.mechanism_program for item in typed_session.proposals
            ),
            ordered_programs=ordered,
            selected_candidate_id=route.selected_candidate_id,
            physical_call_count=len(calls),
            input_tokens=sum(item.input_tokens for item in calls),
            output_tokens=sum(item.output_tokens for item in calls),
            billed_tokens=sum(item.total_tokens for item in calls),
            proposal_count=typed_session.proposal_count,
            proposal_session_digest=sha256_digest(
                {
                    "typed_session_digest": typed_session.digest,
                    "upstream_call_digests": [
                        item.response_digest for item in calls
                    ],
                }
            ),
            route_trace_digest=route.digest,
            research_plan=plan,
            ordered_proposal_candidate_ids=tuple(
                route.ranked_candidate_ids
            ),
            research_proposals=typed_session.proposals,
        )

    def finalize_common_route(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        session: FakeProposalSessionV1,
        common_eligible_candidate_ids: Sequence[str],
    ) -> FakeProposalSessionV1:
        if self.campaign_meta_runtime is None and not self.v13_mode:
            return session
        eligible_runtime_ids = set(common_eligible_candidate_ids)
        if arm is ArmCode.A:
            actions = []
            by_runtime_id: dict[str, Mapping[str, Any]] = {}
            for proposal in self.original_controller.cached_proposals:
                program = dict(proposal["mechanism_program"])
                report = compile_program(program)
                runtime_id = str(report.candidate_id)
                if runtime_id not in eligible_runtime_ids:
                    continue
                mechanism = executable_mechanism(str(proposal["mechanism_id"]))
                original_priority = proposal.get("original_priority")
                if (
                    isinstance(
                        self.original_controller,
                        PinnedOriginalMainAdapterV1,
                    )
                    and original_priority not in {"high", "medium", "low"}
                ):
                    raise PreCanaryInvariantError(
                        "V13 Original proposal lacks its own priority"
                    )
                action = {
                    "base_model": mechanism.base_model_config,
                    "candidate_id": runtime_id,
                    "consumes": (),
                    "entrypoint": mechanism.entrypoint,
                    "family_id": mechanism.parent_mechanism_id
                    or mechanism.mechanism_id,
                    "mechanism_id": mechanism.mechanism_id,
                    "mechanism_semantics_digest": (
                        report.mechanism_semantics_digest
                    ),
                    "priority": original_priority or "high",
                    "runner_type": "model",
                    "status": "implemented",
                    "status_source": "COMMON_EXECUTION_GUARD_PASS",
                }
                actions.append(action)
                by_runtime_id[runtime_id] = program
            ranked = self.original_controller.rank(actions)
            if not ranked:
                raise PreCanaryInvariantError(
                    "Original has no Common-eligible campaign candidate"
                )
            ordered_ids = tuple(str(item["candidate_id"]) for item in ranked)
            return replace(
                session,
                ordered_programs=tuple(by_runtime_id[item] for item in ordered_ids),
                selected_candidate_id=ordered_ids[0],
                route_trace_digest=sha256_digest(
                    {
                        "controller": self.original_controller.identity_digest,
                        "ordered_runtime_candidate_ids": ordered_ids,
                    }
                ),
            )

        producer_session = session.producer_session
        if producer_session is None:
            raise PreCanaryInvariantError("Research session lacks typed lineage")
        filtered = tuple(
            proposal
            for proposal in producer_session.proposals
            if str(
                compile_program(deep_thaw(proposal.mechanism_program)).candidate_id
            )
            in eligible_runtime_ids
        )
        if not filtered:
            raise PreCanaryInvariantError(
                "Research slate has no Common-eligible campaign candidate"
            )
        controller = self.research_controllers[arm]
        if self.campaign_meta_runtime is None:
            route = controller.router.route(
                filtered,
                policy_projection=controller.policy.to_dict(),
            )
            proposal_order = tuple(route.ranked_candidate_ids)
            route_trace_digest = route.digest
        else:
            protocol_digest = str(
                campaign_runtime_profile()["development_protocol_digest"]
            )
            exact_parent_programs = {}
            for proposal in filtered:
                if (
                    isinstance(proposal, CandidateProposalV4)
                    and proposal.parent_candidate_id is not None
                ):
                    parent = self.lineage_indexes[arm].exact_parent(
                        proposal,
                        protocol_digest=protocol_digest,
                    )
                    if parent is None:
                        raise PreCanaryInvariantError(
                            "V13 route lost its exact prior-round parent"
                        )
                    exact_parent_programs[proposal.candidate_id] = (
                        parent.mechanism_program
                    )
            meta_route = self.campaign_meta_runtime.route_session(
                arm=arm,
                round_index=round_index,
                session=producer_session,
                proposals=filtered,
                exact_parent_programs=exact_parent_programs,
                static_router=controller.router,
                research_policy=controller.policy,
                search_memory_head_digest=(
                    controller.memory_writer.head.digest
                    if controller.memory_writer.head is not None
                    else None
                ),
            )
            proposal_order = tuple(meta_route.ranked_candidate_ids)
            route_trace_digest = meta_route.digest
        if not proposal_order:
            raise PreCanaryInvariantError("Research route selected nothing")
        by_proposal_id = {item.candidate_id: item for item in filtered}
        ordered_programs = tuple(
            by_proposal_id[item].mechanism_program for item in proposal_order
        )
        selected_runtime_id = str(
            compile_program(deep_thaw(ordered_programs[0])).candidate_id
        )
        plan = ResearchRoundPlanV1(
            round_index=round_index,
            proposal_session_digest=producer_session.digest,
            route_trace_digest=route_trace_digest,
            selected_candidate_id=proposal_order[0],
            physical_call_count=producer_session.physical_call_count,
            proposal_count=producer_session.proposal_count,
            ordinary_execution_opportunities=1,
            plan_status="SELECTED",
            policy_digest=controller.policy.digest,
        )
        return replace(
            session,
            ordered_programs=ordered_programs,
            ordered_proposal_candidate_ids=proposal_order,
            selected_candidate_id=selected_runtime_id,
            route_trace_digest=route_trace_digest,
            research_plan=plan,
        )


class RealCanaryOrchestratorV1(ThreeArmPreCanaryOrchestratorV1):
    def __init__(
        self,
        root: Path,
        *,
        broker: RealCanaryProposalBrokerV1,
    ) -> None:
        contract = CanaryStoreContractV1.create()
        super().__init__(
            root,
            assignment_nonce="M5-CANARY-9011-OPAQUE-V1",
            broker=broker,
            contract=contract,
            resource_ceilings=canary_budget(),
        )

    def _build_helix_raw(
        self,
        *,
        selected: CandidateEnvelope,
        opaque_instance_id: str,
        common_result: Any,
        observation_seed: str,
    ) -> RawResultEnvelope:
        return RawResultEnvelope(
            candidate_id=selected.candidate_id,
            opaque_arm_instance_id=opaque_instance_id,
            raw_result_digest=str(common_result.raw_output_digest),
            common_result_closure_digest=str(
                common_result.common_result_closure_digest
            ),
            observed_protocol={
                "protocol_id": "PROTO-ML1M-FULL-001",
                "profile_family": "OFFLINE_TOPN",
                "dataset": "ml-1m",
                "dataset_snapshot": "ml-1m-snapshot-001",
                "split": {
                    "strategy": "random_user_holdout",
                    "ratio": [0.8, 0.1, 0.1],
                },
                "training_sampling": {"mode": "uniform_negative"},
                "evaluation_candidate_universe": {"mode": "full_sort"},
                "candidate_policy": {"seen_items": "exclude"},
                "metric": {"name": "ndcg", "cutoff": 10},
                "training_procedure": {
                    "optimizer": "adam",
                    "max_epochs": 300,
                },
            },
            target_model="CandidateModel",
            comparator="LightGCN",
            seed_runs=(
                {
                    "seed_id": observation_seed,
                    "run_id": str(common_result.run_id),
                    "artifact_sha256": str(common_result.raw_output_digest),
                },
            ),
            observation_kind="INTERFACE_SMOKE",
            run_status="SMOKE_PASS",
            artifact_identity_status="EXACT",
            normalized_metrics={},
        )

    def _after_research_close(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        controller: ResearchLineControllerV1,
        feedback: FusedSearchFeedbackV2,
        source_proposal_candidate_id: str,
    ) -> None:
        del feedback, source_proposal_candidate_id
        controller.apply_meta_update(
            updater=VersionedMetaPolicyUpdaterV1(),
            completed_round_index=round_index,
            aggregate={
                "calibration_error": 0.1,
                "measured_axes": ("architecture",),
                "uncovered_axes": (
                    "geometry",
                    "message_transform",
                    "objective",
                    "propagation",
                    "sampling",
                    "self_supervision",
                ),
                "causal_followup_axes": ("architecture",),
                "axis_scores": {"architecture": -1.0},
                "producer_useful_rates": {
                    role: 0.5 for role in DISCOVERY_PRODUCERS
                },
            },
        )
        if arm not in {ArmCode.B, ArmCode.C}:
            raise PreCanaryInvariantError("Meta update escaped Research Arms")

    def run_canary(self) -> tuple[tuple[ArmRoundResultV1, ...], ...]:
        rounds = []
        for round_index in range(1, CANARY_ROUNDS_PER_ARM + 1):
            rounds.append(
                self.run_fake_triplet(
                    search_seed=CANARY_SEARCH_SEED,
                    round_index=round_index,
                    drafts=(),
                )
            )
        return tuple(rounds)

    def canary_audit(self) -> dict[str, Any]:
        connection = sqlite3.connect(self.store.db_path)
        try:
            round_count = int(
                connection.execute("SELECT COUNT(*) FROM rounds").fetchone()[0]
            )
            feedback_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM round_events "
                    "WHERE event_type='ROUND_CLOSED'"
                ).fetchone()[0]
            )
            execution_count = int(
                connection.execute(
                    "SELECT COALESCE(SUM(quantity),0) FROM resource_ledger "
                    "WHERE dimension='ORDINARY_EXECUTION'"
                ).fetchone()[0]
            )
            barriers = connection.execute(
                "SELECT round_index, closed_bitmap, next_index_authorized "
                "FROM triplet_barrier ORDER BY round_index"
            ).fetchall()
            budgets = connection.execute(
                """
                SELECT arm_code, round_index, dimension, SUM(quantity)
                FROM resource_ledger JOIN rounds USING(round_id)
                GROUP BY arm_code, round_index, dimension
                ORDER BY arm_code, round_index, dimension
                """
            ).fetchall()
        finally:
            connection.close()
        b = self.broker.research_controllers[ArmCode.B]
        c = self.broker.research_controllers[ArmCode.C]
        return {
            "barriers": [list(item) for item in barriers],
            "bc_controller_identity_equal": b.identity_digest == c.identity_digest,
            "bc_controller_policy_identity_equal": (
                b.policy.digest == c.policy.digest
            ),
            "broker_successful_upstream_calls": self.broker.upstream.call_count(),
            "budget_rows_digest": sha256_digest([list(item) for item in budgets]),
            "execution_count": execution_count,
            "feedback_count": feedback_count,
            "guard_call_count": self.guard_ledger.count(),
            "round_count": round_count,
            "state_store_integrity": self.store.integrity_report(),
        }


def environment_preflight(contract: Mapping[str, Any]) -> dict[str, Any]:
    dataset_root = Path(contract["dataset"]["root"])
    actual_dataset = {
        name: __import__("hashlib").sha256(
            (dataset_root / name).read_bytes()
        ).hexdigest()
        for name in contract["dataset"]["files"]
    }
    if actual_dataset != contract["dataset"]["files"]:
        raise PreCanaryInvariantError("dataset exact bytes do not match Canary contract")
    python = str(contract["runtime"]["python"])
    recbole_root = str(
        contract["runtime"].get("recbole_root", "/root/projects/RecBole")
    )
    probe_environment = dict(__import__("os").environ)
    probe_environment["PYTHONPATH"] = recbole_root
    probe = subprocess.run(
        [
            python,
            "-c",
            (
                "import json,torch,recbole,numpy,scipy;"
                "print(json.dumps({'python':__import__('sys').version.split()[0],"
                "'torch':torch.__version__,'cuda':torch.cuda.is_available(),"
                "'recbole':recbole.__version__,'numpy':numpy.__version__,"
                "'scipy':scipy.__version__},sort_keys=True))"
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=recbole_root,
        env=probe_environment,
    )
    runtime = json.loads(probe.stdout)
    if runtime != contract["runtime"]["versions"]:
        raise PreCanaryInvariantError("runtime versions differ from Canary contract")
    imported_root = subprocess.run(
        [python, "-c", "import recbole; print(recbole.__file__)"],
        check=True,
        capture_output=True,
        text=True,
        cwd=recbole_root,
        env=probe_environment,
    ).stdout.strip()
    if not Path(imported_root).resolve().is_relative_to(Path(recbole_root).resolve()):
        raise PreCanaryInvariantError("RecBole import escaped the frozen runtime root")
    codex = subprocess.run(
        [contract["broker"]["codex_executable"], "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    login_process = subprocess.run(
        [contract["broker"]["codex_executable"], "login", "status"],
        check=True,
        capture_output=True,
        text=True,
    )
    login = (login_process.stdout + login_process.stderr).strip()
    if codex != contract["broker"]["codex_cli_version"]:
        raise PreCanaryInvariantError("Codex CLI version mismatch")
    if login != "Logged in using ChatGPT":
        raise PreCanaryInvariantError("Codex CLI login state is unavailable")
    model_catalog = json.loads(
        Path(contract["broker"]["models_cache_path"]).read_text(encoding="utf-8")
    )
    if model_catalog.get("etag") != contract["broker"]["models_cache_etag"]:
        raise PreCanaryInvariantError("Codex model catalog ETag changed")
    available_models = {
        str(item["slug"]) for item in model_catalog.get("models", ())
    }
    if contract["broker"]["model"] not in available_models:
        raise PreCanaryInvariantError("frozen broker model is not advertised")
    subprocess.run(
        ["unshare", "--mount", "--fork", "/bin/true"],
        check=True,
        capture_output=True,
        text=True,
    )
    with tempfile.TemporaryDirectory() as raw:
        isolation_root = Path(raw)
        isolation_root.chmod(0o711)
        own = isolation_root / "own"
        sibling = isolation_root / "sibling"
        own.mkdir()
        sibling.mkdir()
        isolation = probe_uid_isolation(
            own_root=own, sibling_root=sibling, worker_uid=62101
        )
    if not all(isolation.values()):
        raise PreCanaryInvariantError("numeric UID isolation preflight failed")
    return {
        "broker_login": "CHATGPT",
        "codex_cli_version": codex,
        "dataset_files": actual_dataset,
        "model_available": contract["broker"]["model"],
        "mount_namespace": "PASS",
        "numeric_uid_isolation": isolation,
        "runtime_versions": runtime,
        "recbole_import_root": recbole_root,
        "verdict": "PASS",
    }


__all__ = [
    "CANARY_ROUNDS_PER_ARM",
    "CANARY_SEARCH_SEED",
    "RealCanaryOrchestratorV1",
    "RealCanaryProposalBrokerV1",
    "canary_budget",
    "environment_preflight",
]
