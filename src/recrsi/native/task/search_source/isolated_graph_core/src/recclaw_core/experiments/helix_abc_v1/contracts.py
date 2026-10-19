"""Machine-readable M0 contracts for the three-arm Research Line experiment."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Mapping, Sequence

from .canonical import canonical_value, sha256_digest, validate_sha256


AUTHORITY = "NONE"
EVIDENCE_CLASS = "DEVELOPMENT_ONLY"
FORMAL_ACCEPTANCE = False
IMPLEMENTATION_PROFILE = "MINIMUM_SUFFICIENT_V1"

RESEARCH_FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "accepted_evidence",
        "admissible",
        "claim_ceiling",
        "claim_record",
        "claim_state",
        "cross_protocol_contamination",
        "evidence_admission",
        "evidence_authority",
        "evidence_use",
        "guard_reason_code",
        "guard_reason_codes",
        "permission_decision",
        "protocol_branch",
    }
)
RESEARCH_FORBIDDEN_FIELD_TOKENS = frozenset(
    "".join(character for character in name if character.isalnum())
    for name in RESEARCH_FORBIDDEN_FIELD_NAMES
)


class ArmCode(str, Enum):
    A = "A"
    B = "B"
    C = "C"


class ControllerKind(str, Enum):
    ORIGINAL = "OriginalControllerV1"
    RESEARCH_LINE = "ResearchLineControllerV1"


class EvidencePortKind(str, Enum):
    NULL = "NullEvidencePortV1"
    EVIDENCE_GUARD = "EvidenceGuardPortV1"


class ProducerExecutionModeV1(str, Enum):
    BATCHED_ROLE_PORTFOLIO_V1 = "BATCHED_ROLE_PORTFOLIO_V1"
    NEUTRAL_MULTISAMPLE_CONTROL_V1 = "NEUTRAL_MULTISAMPLE_CONTROL_V1"
    BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1 = (
        "BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1"
    )


class MetaPolicyModeV1(str, Enum):
    VERSIONED_POLICY_UPDATE = "VERSIONED_POLICY_UPDATE"
    STATIC_RESEARCH_ROUTER = "STATIC_RESEARCH_ROUTER"


class PhysicalInvocationPolicyV1(str, Enum):
    ONE_ORIGINAL_INVOCATION = "ONE_ORIGINAL_INVOCATION"
    BOUNDED_MODE_DEFINED_INVOCATIONS = "BOUNDED_MODE_DEFINED_INVOCATIONS"


class EvidenceStage(str, Enum):
    PRE = "PRE"
    POST = "POST"


class EvidenceAdjudicationStatus(str, Enum):
    NOT_ADJUDICATED = "NOT_ADJUDICATED"


class SearchFeedbackClass(str, Enum):
    BASELINE_RESULT = "BASELINE_RESULT"
    NO_SEARCH_UPDATE = "NO_SEARCH_UPDATE"


@dataclass(frozen=True, slots=True)
class ResourceCeilingsV1:
    """An explicit equal-total-resource envelope for one Arm-round session."""

    total_input_tokens: int
    total_output_tokens: int
    total_billed_token_debit: int
    total_proposal_count: int
    wall_time_ms: int
    retry_debit: int
    proposal_attempt_debit: int
    ordinary_executions: int
    common_validation_count: int
    gpu_device_time_ms: int
    gpu_cost_microunits: int

    def __post_init__(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{item.name} must be a non-negative integer")
        if self.ordinary_executions > 1:
            raise ValueError("one SearchRound permits at most one ordinary execution")

    def to_dict(self) -> dict[str, int]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class ArmPolicyV1:
    arm: ArmCode
    controller: ControllerKind
    research_line_enabled: bool
    producer_execution_mode: ProducerExecutionModeV1 | None
    meta_policy_mode: MetaPolicyModeV1 | None
    common_execution_guard_ref: str
    common_execution_guard_digest: str
    evidence_port: EvidencePortKind
    deterministic_fusion_ref: str
    deterministic_fusion_digest: str
    bl_icf_search_space_ref: str
    bl_icf_search_space_digest: str
    controller_policy_digest: str

    def __post_init__(self) -> None:
        for name in (
            "common_execution_guard_digest",
            "deterministic_fusion_digest",
            "bl_icf_search_space_digest",
            "controller_policy_digest",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if self.arm is ArmCode.A:
            if self.controller is not ControllerKind.ORIGINAL:
                raise ValueError("Arm A must use OriginalControllerV1")
            if self.research_line_enabled:
                raise ValueError("Arm A must not enable Research Line")
            if self.producer_execution_mode is not None or self.meta_policy_mode is not None:
                raise ValueError("Arm A cannot carry Research Producer or Meta policy")
            if self.evidence_port is not EvidencePortKind.NULL:
                raise ValueError("Arm A must use NullEvidencePortV1")
        else:
            if self.controller is not ControllerKind.RESEARCH_LINE:
                raise ValueError("Arms B/C must use ResearchLineControllerV1")
            if not self.research_line_enabled:
                raise ValueError("Arms B/C must enable Research Line")
            if self.producer_execution_mode is None or self.meta_policy_mode is None:
                raise ValueError("Arms B/C require frozen Producer and Meta modes")
            expected_port = (
                EvidencePortKind.NULL
                if self.arm is ArmCode.B
                else EvidencePortKind.EVIDENCE_GUARD
            )
            if self.evidence_port is not expected_port:
                raise ValueError(f"Arm {self.arm.value} has the wrong EvidencePort")

    def non_guard_projection(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("arm")
        payload.pop("evidence_port")
        return payload

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class ProposalGenerationSessionV1:
    """M0-only session contract; it contains no broker or LLM implementation."""

    session_id: str
    arm: ArmCode
    round_id: str
    resource_ceilings: ResourceCeilingsV1
    physical_invocation_policy: PhysicalInvocationPolicyV1
    producer_execution_mode: ProducerExecutionModeV1 | None
    session_count_for_round: int = 1
    llm_runtime_implemented: bool = False

    def __post_init__(self) -> None:
        if self.session_count_for_round != 1:
            raise ValueError("each opened Arm-round has exactly one proposal session")
        if self.llm_runtime_implemented:
            raise ValueError("M0 must not implement an LLM runtime")
        if self.arm is ArmCode.A:
            if (
                self.physical_invocation_policy
                is not PhysicalInvocationPolicyV1.ONE_ORIGINAL_INVOCATION
            ):
                raise ValueError("Arm A may later use one Original invocation")
            if self.producer_execution_mode is not None:
                raise ValueError("Arm A cannot use a Research Producer mode")
        else:
            if (
                self.physical_invocation_policy
                is not PhysicalInvocationPolicyV1.BOUNDED_MODE_DEFINED_INVOCATIONS
            ):
                raise ValueError("Arms B/C may later use bounded multiple invocations")
            if self.producer_execution_mode is None:
                raise ValueError("Arms B/C require a Producer mode")

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class EvidenceAdjudicationV1:
    candidate_id: str
    stage: EvidenceStage
    status: EvidenceAdjudicationStatus
    reason_codes: tuple[str, ...] = ("NULL_PORT",)

    def __post_init__(self) -> None:
        if self.status is not EvidenceAdjudicationStatus.NOT_ADJUDICATED:
            raise ValueError("M0 only implements NOT_ADJUDICATED Null-port output")
        object.__setattr__(self, "reason_codes", tuple(self.reason_codes))

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class FusedSearchFeedbackV1:
    candidate_id: str
    search_feedback_class: SearchFeedbackClass
    adjudication_status: EvidenceAdjudicationStatus
    raw_outcome_projection_digest: str | None
    frontier_eligibility: str
    fusion_policy_digest: str

    def __post_init__(self) -> None:
        validate_sha256(self.fusion_policy_digest, field_name="fusion_policy_digest")
        if self.raw_outcome_projection_digest is not None:
            validate_sha256(
                self.raw_outcome_projection_digest,
                field_name="raw_outcome_projection_digest",
            )
        if self.frontier_eligibility not in {"CURRENT_FRONTIER", "EXCLUDED"}:
            raise ValueError("frontier_eligibility is outside the M0 closed domain")

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class ExperimentContractV1:
    experiment_id: str
    implementation_profile: str
    authority: str
    evidence_class: str
    formal_acceptance: bool
    arm_policies: tuple[ArmPolicyV1, ArmPolicyV1, ArmPolicyV1]
    search_seeds: tuple[int, ...]
    scheduled_slots_per_arm_seed: int
    ordinary_execution_seed: int
    post_selection_stability_seeds: tuple[int, ...]
    dataset: str
    evaluation_protocol: str
    primary_metric: str
    proposal_sessions_per_open_round: int
    resource_ceiling_equality: str
    resource_units: tuple[str, ...]
    implementation_milestone: str = "M0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm_policies", tuple(self.arm_policies))
        object.__setattr__(self, "search_seeds", tuple(self.search_seeds))
        object.__setattr__(
            self,
            "post_selection_stability_seeds",
            tuple(self.post_selection_stability_seeds),
        )
        object.__setattr__(self, "resource_units", tuple(self.resource_units))
        if self.implementation_profile != IMPLEMENTATION_PROFILE:
            raise ValueError("M0 only supports MINIMUM_SUFFICIENT_V1")
        if (
            self.authority != AUTHORITY
            or self.evidence_class != EVIDENCE_CLASS
            or self.formal_acceptance is not FORMAL_ACCEPTANCE
        ):
            raise ValueError("M0 outputs are non-authoritative development evidence")
        if self.implementation_milestone != "M0":
            raise ValueError("this package implements M0 only")
        if tuple(policy.arm for policy in self.arm_policies) != (
            ArmCode.A,
            ArmCode.B,
            ArmCode.C,
        ):
            raise ValueError("arm_policies must contain exactly the ordered A/B/C tuple")
        if self.arm_policies[1].non_guard_projection() != self.arm_policies[
            2
        ].non_guard_projection():
            raise ValueError("Arms B/C must be identical outside EvidencePort")
        common_fields = (
            "common_execution_guard_ref",
            "common_execution_guard_digest",
            "deterministic_fusion_ref",
            "deterministic_fusion_digest",
            "bl_icf_search_space_ref",
            "bl_icf_search_space_digest",
        )
        for name in common_fields:
            if len({getattr(policy, name) for policy in self.arm_policies}) != 1:
                raise ValueError(f"{name} must be exact-equal across A/B/C")
        if self.search_seeds != (42, 43, 44):
            raise ValueError("M0 freezes search seeds [42,43,44]")
        if self.scheduled_slots_per_arm_seed != 50:
            raise ValueError("M0 freezes 50 scheduled slots per Arm/seed")
        if self.ordinary_execution_seed != 2026:
            raise ValueError("M0 freezes ordinary execution seed 2026")
        if self.post_selection_stability_seeds != (2026, 2027, 2028):
            raise ValueError("M0 freezes stability seeds [2026,2027,2028]")
        if (self.dataset, self.evaluation_protocol, self.primary_metric) != (
            "ML-1M",
            "frozen_full_sort",
            "NDCG@10",
        ):
            raise ValueError("M0 freezes ML-1M/frozen_full_sort/NDCG@10")
        if self.proposal_sessions_per_open_round != 1:
            raise ValueError("each opened round has exactly one proposal session")
        if self.resource_ceiling_equality != "EXACT_ACROSS_A_B_C":
            raise ValueError("resource ceilings must be exact-equal across A/B/C")

    @property
    def identity_digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExperimentContractV1":
        arm_policies = tuple(
            ArmPolicyV1(
                arm=ArmCode(item["arm"]),
                controller=ControllerKind(item["controller"]),
                research_line_enabled=bool(item["research_line_enabled"]),
                producer_execution_mode=(
                    ProducerExecutionModeV1(item["producer_execution_mode"])
                    if item.get("producer_execution_mode") is not None
                    else None
                ),
                meta_policy_mode=(
                    MetaPolicyModeV1(item["meta_policy_mode"])
                    if item.get("meta_policy_mode") is not None
                    else None
                ),
                common_execution_guard_ref=str(item["common_execution_guard_ref"]),
                common_execution_guard_digest=str(
                    item["common_execution_guard_digest"]
                ),
                evidence_port=EvidencePortKind(item["evidence_port"]),
                deterministic_fusion_ref=str(item["deterministic_fusion_ref"]),
                deterministic_fusion_digest=str(item["deterministic_fusion_digest"]),
                bl_icf_search_space_ref=str(item["bl_icf_search_space_ref"]),
                bl_icf_search_space_digest=str(item["bl_icf_search_space_digest"]),
                controller_policy_digest=str(item["controller_policy_digest"]),
            )
            for item in payload["arm_policies"]
        )
        if len(arm_policies) != 3:
            raise ValueError("arm_policies must contain exactly three rows")
        contract = cls(
            experiment_id=str(payload["experiment_id"]),
            implementation_profile=str(payload["implementation_profile"]),
            authority=str(payload["authority"]),
            evidence_class=str(payload["evidence_class"]),
            formal_acceptance=bool(payload["formal_acceptance"]),
            arm_policies=arm_policies,  # type: ignore[arg-type]
            search_seeds=tuple(int(item) for item in payload["search_seeds"]),
            scheduled_slots_per_arm_seed=int(
                payload["scheduled_slots_per_arm_seed"]
            ),
            ordinary_execution_seed=int(payload["ordinary_execution_seed"]),
            post_selection_stability_seeds=tuple(
                int(item) for item in payload["post_selection_stability_seeds"]
            ),
            dataset=str(payload["dataset"]),
            evaluation_protocol=str(payload["evaluation_protocol"]),
            primary_metric=str(payload["primary_metric"]),
            proposal_sessions_per_open_round=int(
                payload["proposal_sessions_per_open_round"]
            ),
            resource_ceiling_equality=str(payload["resource_ceiling_equality"]),
            resource_units=tuple(str(item) for item in payload["resource_units"]),
            implementation_milestone=str(payload.get("implementation_milestone", "")),
        )
        if canonical_value(payload) != contract.to_dict():
            raise ValueError("experiment contract payload must be closed and type-exact")
        return contract


def validate_equal_session_resource_ceilings(
    sessions: Sequence[ProposalGenerationSessionV1],
) -> None:
    if tuple(session.arm for session in sessions) != (ArmCode.A, ArmCode.B, ArmCode.C):
        raise ValueError("session comparison requires the ordered A/B/C tuple")
    if len({sha256_digest(session.resource_ceilings) for session in sessions}) != 1:
        raise ValueError("A/B/C ProposalGenerationSession ceilings must be exact-equal")
    if sessions[1].producer_execution_mode is not sessions[2].producer_execution_mode:
        raise ValueError("B/C must use the exact same Producer execution mode")


def validate_no_research_evidence_authority_fields(value: Any) -> None:
    """Reject Evidence Authority field names from Research-owned payloads."""

    if hasattr(value, "__dataclass_fields__"):
        names = set(value.__dataclass_fields__)
        forbidden = names & RESEARCH_FORBIDDEN_FIELD_NAMES
        if forbidden:
            raise ValueError(f"Research contract contains forbidden fields: {sorted(forbidden)}")
        for item in fields(value):
            validate_no_research_evidence_authority_fields(getattr(value, item.name))
        return
    if isinstance(value, Mapping):
        normalized = {
            "".join(
                character
                for character in str(key).lower()
                if character.isalnum()
            )
            for key in value
        }
        forbidden = normalized & RESEARCH_FORBIDDEN_FIELD_TOKENS
        if forbidden:
            raise ValueError(f"Research payload contains forbidden fields: {sorted(forbidden)}")
        for item in value.values():
            validate_no_research_evidence_authority_fields(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            validate_no_research_evidence_authority_fields(item)


def _policy_digest(label: str) -> str:
    return sha256_digest({"contract_label": label})


def default_experiment_contract() -> ExperimentContractV1:
    common_guard_ref = "recclaw.common-execution-guard.contract.v1"
    common_guard_digest = _policy_digest(common_guard_ref)
    fusion_ref = "recclaw.deterministic-fusion.m0.v1"
    fusion_digest = _policy_digest(fusion_ref)
    search_ref = "recclaw.bl-icf.mechanism-space.v1"
    search_digest = "fbe63260de6430537dd66b0724fe1beb0cf165ea47407ae18addd61a17dd1720"
    original_digest = _policy_digest(
        "OriginalControllerV1@2d8c881354e1b536a6c66d7dfbb977e0c5090e50"
    )
    research_digest = _policy_digest("ResearchLineControllerV1.m0-interface")

    def arm(
        code: ArmCode,
        controller: ControllerKind,
        research: bool,
        producer: ProducerExecutionModeV1 | None,
        meta: MetaPolicyModeV1 | None,
        port: EvidencePortKind,
        controller_digest: str,
    ) -> ArmPolicyV1:
        return ArmPolicyV1(
            arm=code,
            controller=controller,
            research_line_enabled=research,
            producer_execution_mode=producer,
            meta_policy_mode=meta,
            common_execution_guard_ref=common_guard_ref,
            common_execution_guard_digest=common_guard_digest,
            evidence_port=port,
            deterministic_fusion_ref=fusion_ref,
            deterministic_fusion_digest=fusion_digest,
            bl_icf_search_space_ref=search_ref,
            bl_icf_search_space_digest=search_digest,
            controller_policy_digest=controller_digest,
        )

    selected_mode = ProducerExecutionModeV1.BOUNDED_INDEPENDENT_PRODUCER_AGENTS_V1
    meta_mode = MetaPolicyModeV1.VERSIONED_POLICY_UPDATE
    return ExperimentContractV1(
        experiment_id="HELIX-ABC-001",
        implementation_profile=IMPLEMENTATION_PROFILE,
        authority=AUTHORITY,
        evidence_class=EVIDENCE_CLASS,
        formal_acceptance=FORMAL_ACCEPTANCE,
        arm_policies=(
            arm(
                ArmCode.A,
                ControllerKind.ORIGINAL,
                False,
                None,
                None,
                EvidencePortKind.NULL,
                original_digest,
            ),
            arm(
                ArmCode.B,
                ControllerKind.RESEARCH_LINE,
                True,
                selected_mode,
                meta_mode,
                EvidencePortKind.NULL,
                research_digest,
            ),
            arm(
                ArmCode.C,
                ControllerKind.RESEARCH_LINE,
                True,
                selected_mode,
                meta_mode,
                EvidencePortKind.EVIDENCE_GUARD,
                research_digest,
            ),
        ),
        search_seeds=(42, 43, 44),
        scheduled_slots_per_arm_seed=50,
        ordinary_execution_seed=2026,
        post_selection_stability_seeds=(2026, 2027, 2028),
        dataset="ML-1M",
        evaluation_protocol="frozen_full_sort",
        primary_metric="NDCG@10",
        proposal_sessions_per_open_round=1,
        resource_ceiling_equality="EXACT_ACROSS_A_B_C",
        resource_units=(
            "PROPOSAL_GENERATION_SESSION",
            "PHYSICAL_LLM_CALL_TREATMENT_COST",
            "INPUT_TOKEN",
            "OUTPUT_TOKEN",
            "BILLED_TOKEN_DEBIT",
            "PROPOSAL",
            "WALL_TIME_MS",
            "RETRY",
            "PROPOSAL_ATTEMPT",
            "ORDINARY_EXECUTION",
            "COMMON_VALIDATION",
            "GPU_DEVICE_TIME_MS",
            "GPU_COST_MICROUNITS",
        ),
    )
