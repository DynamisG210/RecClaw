"""Real-broker, no-training M5 Canary built on the M4 neutral scheduler."""

from __future__ import annotations

import copy
import inspect
import json
import sqlite3
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from recclaw_core.mechanism_space import compile_program
from recclaw_core.mechanism_space.canonical import deep_thaw

from recclaw_core.helix.contracts import CandidateEnvelope, RawResultEnvelope
from recclaw_core.helix.scientific_attribution import FusedSearchFeedbackV2

from .campaign_runtime import (
    campaign_projection,
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
    DiscoveryCreditV1,
    ProducerCallRecordV1,
    ProducerSessionResultV1,
    ProposalIntentV1,
    SearchUtilityFeaturesV1,
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
    _research_calls: dict[tuple[int, int, str], tuple[CanaryBrokerCallV1, ...]]
    _search_feedback: dict[ArmCode, Mapping[str, Any]]
    _campaign_call_scopes: dict[str, tuple[str, ...]]
    original_controller: OriginalRuntimeAdapterV1
    campaign_meta_runtime: Any | None = None
    producer_control_enabled: bool = True
    call_prefix: str = ""
    phase_name: str = "Canary"
    adaptive_memory: bool = False

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
        research_policy_override: Any | None = None,
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
            original_controller=OriginalRuntimeAdapterV1(),
            campaign_meta_runtime=campaign_meta_runtime,
            producer_control_enabled=producer_control_enabled,
            call_prefix=call_prefix,
            phase_name=phase_name,
            adaptive_memory=adaptive_memory,
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
        key = (search_seed, round_index, memory_policy_digest)
        if key not in self._research_calls:
            memory_component = (
                f"-m{memory_policy_digest[:12]}"
                if self.adaptive_memory
                else ""
            )
            calls: list[CanaryBrokerCallV1] = []
            allocations = dict(controller.policy.producer_token_allocation)
            default_lineage_root = (
                "LIGHTGCN" if round_index % 2 else "BPR_MF"
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
                lineage_root = (
                    meta_directive.lineage_root
                    if meta_directive is not None
                    else default_lineage_root
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
                        "PREFER_EXACT_EXECUTED_PARENT"
                        if role
                        in {"lineage_refiner", "falsification_designer"}
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
                    meta_directive.proposal_intent == "CONTROL"
                    if meta_directive is not None
                    else role == "falsification_designer"
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
                logical_call_id = (
                    f"{self.call_prefix}research-"
                    f"{search_seed}-{round_index}{memory_component}-{role}"
                )
                self._campaign_call_scopes[logical_call_id] = tuple(
                    str(item["mechanism_id"])
                    for item in catalog["mechanisms"]
                )
                try:
                    call = self._upstream_call(
                        logical_call_id=logical_call_id,
                        proposal_generation_session_id=(
                            f"{self.call_prefix}research-session-"
                            f"{search_seed}-{round_index}{memory_component}"
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
            self._research_calls[key] = tuple(calls)
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
        memory_digest = (
            sha256_digest(memory_summary) if self.adaptive_memory else "shared"
        )
        key = (search_seed, round_index, memory_digest)
        if key not in self._research_calls:
            memory_component = (
                f"-m{memory_digest[:12]}" if self.adaptive_memory else ""
            )
            calls: list[CanaryBrokerCallV1] = []
            for role in DISCOVERY_PRODUCERS:
                try:
                    call = self._upstream_call(
                        logical_call_id=(
                            f"{self.call_prefix}research-"
                            f"{search_seed}-{round_index}{memory_component}-{role}"
                        ),
                        proposal_generation_session_id=(
                            f"{self.call_prefix}research-session-"
                            f"{search_seed}-{round_index}{memory_component}"
                        ),
                        prompt=research_canary_prompt(
                            role=role,
                            round_index=round_index,
                            search_seed=search_seed,
                            phase_name=self.phase_name,
                            memory_summary=memory_summary,
                        ),
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

    @staticmethod
    def _proposal_from_call(
        *,
        session_id: str,
        role: str,
        call: CanaryBrokerCallV1,
        proposal: Mapping[str, Any],
    ) -> CandidateProposalV3:
        program = campaign_program_from_proposal(proposal)
        compiled = compile_program(program)
        mechanism = executable_mechanism(str(proposal["mechanism_id"]))
        utility = dict(proposal["utility_features"])
        cost = 0.6 if mechanism.config.get("relative_cost") == "MEDIUM" else (
            0.6
            if mechanism.mechanism_id in {"NGCF", "SGL", "NCL"}
            else 0.3
        )
        intent = ProposalIntentV1(str(proposal["proposal_intent"]))
        discovery_credit = (
            DiscoveryCreditV1.NON_DISCOVERY_CONTROL
            if intent is ProposalIntentV1.CONTROL
            else DiscoveryCreditV1.DISCOVERY
        )
        proposal_identity = sha256_digest(
            {
                "session_id": session_id,
                "role": role,
                "response_digest": call.response_digest,
                "runtime_candidate_id": compiled.candidate_id,
            }
        )
        proposal_candidate_id = f"cand-{proposal_identity[:24]}"
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
            mechanism_hypothesis=str(proposal["mechanism_hypothesis"]),
            competing_hypothesis=str(proposal["competing_hypothesis"]),
            predicted_outcome_signature=str(
                proposal["predicted_outcome_signature"]
            ),
            failure_mode=str(proposal["failure_mode"]),
            utility_features=SearchUtilityFeaturesV1(
                runnable_probability=1.0,
                useful_signal=float(utility["useful_signal"]),
                frontier_potential=float(utility["frontier_potential"]),
                information_gain=float(utility["information_gain"]),
                cost=cost,
                blocker_risk=0.05,
            ),
            parent_candidate_id=proposal.get("parent_candidate_id"),
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
            f"{self.call_prefix or 'campaign-'}research-"
            f"{search_seed}-{round_index}"
        )
        proposals: list[CandidateProposalV3] = []
        call_records: list[ProducerCallRecordV1] = []
        for role, call in zip(DISCOVERY_PRODUCERS, calls, strict=True):
            raw = dict(call.response["proposals"][0])
            expected_intent = (
                ProposalIntentV1.CONTROL
                if role == "falsification_designer"
                else ProposalIntentV1.DISCOVERY
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
                    physical_call_id=call.logical_call_id,
                    producer_id=typed.producer_id,
                    producer_role=role,
                    request_digest=call.request_digest,
                    response_digest=call.response_digest,
                    context_digest=sha256_digest(
                        {"round_index": round_index, "search_seed": search_seed}
                    ),
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
            candidate_schema_ref="CandidateProposalV3",
            proposal_count=len(proposals),
            physical_call_count=len(call_records),
            input_tokens=sum(item.input_tokens for item in calls),
            output_tokens=sum(item.output_tokens for item in calls),
            billed_tokens=sum(item.total_tokens for item in calls),
            session_latency_ms=sum(item.latency_ms for item in calls),
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
        if self.campaign_meta_runtime is None:
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
                        catalog_projection=campaign_projection(),
                        original_state=self.original_controller.state_projection(),
                    ),
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
                raw_proposals = tuple(call.response["proposals"])
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
                            "mechanism_program": campaign_program_from_proposal(item),
                        }
                        for item in raw_proposals
                    ),
                )
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
            )
        calls = self._research_upstream_calls(
            arm=arm,
            round_index=round_index,
            search_seed=search_seed,
            ceilings=ceilings,
        )
        session = self._typed_research_session(
            arm=arm,
            round_index=round_index,
            search_seed=search_seed,
            ceilings=ceilings,
            calls=calls,
        )
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
                {"space": "BL_ICF_EXECUTABLE_PROFILE_V1"},
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
        if self.campaign_meta_runtime is None:
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
                action = {
                    "candidate_id": runtime_id,
                    "family_id": mechanism.parent_mechanism_id
                    or mechanism.mechanism_id,
                    "mechanism_id": mechanism.mechanism_id,
                    "mechanism_semantics_digest": (
                        report.mechanism_semantics_digest
                    ),
                    "priority": "high",
                    "status": "implemented",
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
            meta_route = self.campaign_meta_runtime.route_session(
                arm=arm,
                round_index=round_index,
                session=producer_session,
                proposals=filtered,
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
                    "seed_id": "2026",
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
                "mechanism_axis_gaps": (
                    "objective",
                    "propagation",
                    "self_supervision",
                    "geometry",
                ),
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
