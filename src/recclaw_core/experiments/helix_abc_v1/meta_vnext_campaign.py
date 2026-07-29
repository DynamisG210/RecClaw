"""NEXT_CAMPAIGN adapters for the frozen Meta V17 and V18 policies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from recclaw_core.mechanism_space.canonical import deep_thaw

from .canonical import canonical_value, sha256_digest, validate_sha256
from .campaign_runtime import (
    bl_icf_executable_profile_v2,
    executable_mechanism,
    executable_mechanisms,
    root_parent_mechanism_id,
)
from .contracts import ArmCode
from .meta_vnext import (
    CandidateMechanismDeltaV1,
    CandidatePoolVNextV1,
    FastResidualStateV1,
    MetaVNextRouteDecisionV1,
    MetaVNextRouterV1,
    PairwiseSlowPolicyV1,
    ResearchContextV1,
    SearchValueObservationV1,
    advance_fast_without_observation,
    initialize_fast_residual,
    materialize_candidate_pool,
    update_fast_residual,
)
from .meta_vnext.routing import CandidateRouteScoreV1
from .meta_vnext.routing import task_context_supported
from .meta_vnext.v18_support import (
    FAST_SUPPORT_ID_V18,
    ROUTER_POLICY_ID_V18,
    SLOW_PROJECTION_ID_V18,
    feature_support_sha256,
    load_feature_support,
    route_support_aware,
)
from .producer_opportunity import (
    PRODUCER_OPPORTUNITY_POLICY_DIGEST_V1,
    PRODUCER_OPPORTUNITY_POLICY_ID_V1,
    ProducerOpportunityDecisionV1,
    acquire_producer_opportunity,
)
from .research_capability import StrongStaticRouterV1, VersionedResearchPolicyV1
from .research_contracts import (
    CandidateProposalV2,
    CandidateProposalV3,
    CandidateProposalV4,
    DISCOVERY_PRODUCERS,
    ProducerSessionResultV1,
)


PROMOTION_DECISION_DIGEST_V17 = (
    "358dbaf87b9d7c030ec06c1a7e9229c0f5fc502b102b66fdffa40b31da3827f2"
)
POLICY_BUNDLE_DIGEST_V17 = (
    "20e3dd88d1e6d6f338ffc8b06d2e196762116c4779456dde43b1a2fe38918f61"
)
SOURCE_MANIFEST_DIGEST_V17 = (
    "f894450e242550550dae7acc222ce2ed1be36fedf26dfb42936e7489fbdd7223"
)
ROUTING_SOURCE_SHA256_V17 = (
    "1a05c64d978a696a6f6be6d7cfdd49e3841cd7d906c56a32f0523b48734e54dd"
)
ROUTING_SOURCE_SHA256_RUNTIME_REPAIR_V1 = (
    "7949eda0398050fb8633332e9b22489b9a24c855777d325fd6500027a3d84745"
)
META_V17_RUNTIME_REPAIR_DIGEST_V1 = sha256_digest(
    {
        "repair_id": "META_V17_SPARSE_FAST_RESPONSE_ROUNDS_V1",
        "policy_bundle_digest": POLICY_BUNDLE_DIGEST_V17,
        "source_manifest_digest": SOURCE_MANIFEST_DIGEST_V17,
        "original_routing_source_sha256": ROUTING_SOURCE_SHA256_V17,
        "repaired_routing_source_sha256": (
            ROUTING_SOURCE_SHA256_RUNTIME_REPAIR_V1
        ),
        "scope": "ALLOW_ORDERED_RESPONSE_HISTORY_GAPS_AFTER_NO_OBSERVATION",
    }
)
PROMOTION_DECISION_DIGEST_V18 = (
    "ce5783e7f39f29135422dff3d3fb4a377d3a11a1283677a0a29ba4517064223e"
)
POLICY_BUNDLE_DIGEST_V18 = (
    "2d42ac7b9ced6057d3e20e19f1f470338ed00b0fcbc70e790bdce7fa44706024"
)
CHECKPOINT_SHA256_V18 = (
    "0ef232905cb07abada89322f631908743b31b74527f36c59efc4c00f7b56fd9c"
)
PROMOTION_DECISION_DIGEST_V19 = (
    "5eff2259198fda7bac4e722451e1c3f2ea34e3c591e125d001cf45691ced5bf4"
)
POLICY_BUNDLE_DIGEST_V19 = (
    "95923b800e89c4c4bfb994b5aa8a16069ac8429af056b456f01b42af4ab33744"
)
CHECKPOINT_SHA256_V19 = (
    "27e3118e178e4ebd45f54dd9b0e7239af9a1997d021fc648a800166bc52e2173"
)
POLICY_BUNDLE_DIGEST_V20 = (
    "f3d52f0d2cb6994cd4feee657b89cc5738a280c5fb97ac93253daa8ac81fe972"
)
CHECKPOINT_SHA256_V20 = (
    "5ff049e563ac73a192ad657cdabcf11c014d58a55d95838e9bbb63a1c2b11227"
)
DEVELOPMENT_ACTIVATION_DECISION_DIGEST_V20 = (
    "4f57afd74a84bc29c9579ace4ad0a33e50b4ef8193dda03f9e4b275ffd7f0032"
)
V24_METHOD_DIAGNOSIS_SHA256 = (
    "9521e2ea0f21d9bf5e9714a7d82da8f7255a8f4bef2520382473fe4fea500145"
)


class MetaV17CampaignError(ValueError):
    """Raised when V17 cannot be applied at the frozen campaign boundary."""


def meta_v17_static_producer_policy() -> VersionedResearchPolicyV1:
    """Return the V17 Router with the pre-control-plane Producer policy."""

    return VersionedResearchPolicyV1(
        version=17,
        producer_token_allocation=tuple(
            (role, 0.25) for role in DISCOVERY_PRODUCERS
        ),
        mechanism_axis_targeting=(
            "architecture",
            "geometry",
            "message_transform",
            "objective",
            "propagation",
            "sampling",
            "self_supervision",
        ),
        memory_retrieval_policy="ROLE_SCOPED_PRIOR_ROUND_V1",
        router_priors=(
            ("runnable_probability", 0.5),
            ("useful_signal", 0.5),
        ),
        acquisition_parameters=(
            ("exploration_weight", 0.5),
            ("cost_weight", 0.5),
        ),
        predecessor_digest=None,
        meta_router_policy_digest=POLICY_BUNDLE_DIGEST_V17,
        meta_router_promotion_decision_digest=(
            PROMOTION_DECISION_DIGEST_V17
        ),
        promotion_decision_digest=PROMOTION_DECISION_DIGEST_V17,
        activation_boundary="NEXT_CAMPAIGN",
        control_mode="META_V17_STATIC_PRODUCER_V1",
    )


def meta_v17_research_control_policy() -> VersionedResearchPolicyV1:
    """Return the single B/C policy identity used by the V17 campaign.

    V17 remains the promoted candidate ranker.  The surrounding control
    policy makes its upstream Producer actuators explicit and content-bound;
    token shares remain neutral until a dedicated Producer-allocation
    qualification exists.
    """

    return VersionedResearchPolicyV1(
        version=18,
        producer_token_allocation=tuple(
            (role, 0.25) for role in DISCOVERY_PRODUCERS
        ),
        mechanism_axis_targeting=(
            "architecture",
            "geometry",
            "message_transform",
            "objective",
            "propagation",
            "sampling",
            "self_supervision",
        ),
        memory_retrieval_policy="ROLE_SCOPED_GAP_AWARE_V1",
        router_priors=(
            ("runnable_probability", 0.5),
            ("useful_signal", 0.5),
        ),
        acquisition_parameters=(
            ("exploration_weight", 0.5),
            ("cost_weight", 0.5),
        ),
        predecessor_digest=meta_v17_static_producer_policy().digest,
        meta_router_policy_digest=POLICY_BUNDLE_DIGEST_V17,
        meta_router_promotion_decision_digest=(
            PROMOTION_DECISION_DIGEST_V17
        ),
        promotion_decision_digest=None,
        activation_boundary="NEXT_CAMPAIGN",
        control_mode="PROMOTED_META_CONTROL_V1",
    )


def meta_v18_research_control_policy() -> VersionedResearchPolicyV1:
    """Bind the support-aware V18 ranker to the four discovery actuators."""

    return VersionedResearchPolicyV1(
        version=19,
        producer_token_allocation=tuple(
            (role, 0.25) for role in DISCOVERY_PRODUCERS
        ),
        mechanism_axis_targeting=(
            "architecture",
            "geometry",
            "message_transform",
            "objective",
            "propagation",
            "sampling",
            "self_supervision",
        ),
        memory_retrieval_policy="ROLE_SCOPED_GAP_AWARE_V1",
        router_priors=(
            ("runnable_probability", 0.5),
            ("useful_signal", 0.5),
        ),
        acquisition_parameters=(
            ("exploration_weight", 0.5),
            ("cost_weight", 0.5),
        ),
        predecessor_digest=meta_v17_research_control_policy().digest,
        meta_router_policy_digest=POLICY_BUNDLE_DIGEST_V18,
        meta_router_promotion_decision_digest=(
            PROMOTION_DECISION_DIGEST_V18
        ),
        promotion_decision_digest=PROMOTION_DECISION_DIGEST_V18,
        activation_boundary="NEXT_CAMPAIGN",
        control_mode="PROMOTED_META_V18_SUPPORT_AWARE_CONTROL",
    )


def meta_v19_research_control_policy() -> VersionedResearchPolicyV1:
    """Rebind V18 unchanged to the Provider-strict proposal schema."""

    parent = meta_v18_research_control_policy()
    return replace(
        parent,
        version=20,
        predecessor_digest=parent.digest,
        meta_router_policy_digest=POLICY_BUNDLE_DIGEST_V19,
        meta_router_promotion_decision_digest=(
            PROMOTION_DECISION_DIGEST_V19
        ),
        promotion_decision_digest=PROMOTION_DECISION_DIGEST_V19,
        control_mode="PROMOTED_META_V19_PROVIDER_STRICT_SCHEMA_REBIND",
    )


def meta_v20_research_control_policy() -> VersionedResearchPolicyV1:
    """Bind the V19 scorer to the development Producer opportunity policy."""

    parent = meta_v19_research_control_policy()
    return replace(
        parent,
        version=21,
        acquisition_parameters=(
            *parent.acquisition_parameters,
            ("producer_coverage_fraction", 0.5),
            ("producer_coverage_block_size", 8.0),
        ),
        predecessor_digest=parent.digest,
        meta_router_policy_digest=POLICY_BUNDLE_DIGEST_V20,
        meta_router_promotion_decision_digest=(
            DEVELOPMENT_ACTIVATION_DECISION_DIGEST_V20
        ),
        promotion_decision_digest=None,
        activation_boundary="NEXT_FRESH_CAMPAIGN",
        control_mode=(
            "DEVELOPMENT_META_V20_BLOCK8_PRODUCER_OPPORTUNITY"
        ),
    )


@dataclass(frozen=True, slots=True)
class MetaProducerDirectiveV1:
    producer_role: str
    lineage_root: str
    primary_axis: str
    proposal_intent: str
    memory_query: str
    token_share: float
    required_mechanism_id: str | None
    learned_axis_score: float
    control_policy_digest: str

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class MetaV17RouteResultV1:
    arm: str
    round_index: int
    mode: str
    pool_digest: str | None
    decision_digest: str | None
    ranked_candidate_ids: tuple[str, ...]
    selected_candidate_id: str
    static_champion_candidate_id: str
    selected_candidate_semantics_digest: str | None
    pool_candidate_semantics_digests: tuple[str, ...]
    pool_mechanism_axes: tuple[str, ...]
    round_semantic_collision_count: int
    task_context_supported: bool
    support_projection_digest: str | None = None
    fast_support_reason: str | None = None
    fast_supported: bool | None = None

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(slots=True)
class _ArmCampaignState:
    fast_state: FastResidualStateV1
    starting_metric: float | None = None
    best_metric: float | None = None
    recent_gain: float = 0.0
    executed_axes: tuple[str, ...] = ()
    blocker_count: int = 0
    semantic_collision_count: int = 0


@dataclass(frozen=True, slots=True)
class _BoundRound:
    route: MetaV17RouteResultV1
    pool: CandidatePoolVNextV1 | None
    decision: MetaVNextRouteDecisionV1 | None
    singleton_candidate: CandidateMechanismDeltaV1 | None = None


class MetaV17CampaignRuntimeV1:
    """Run the same promoted slow+fast policy in B and C.

    B/C have distinct private response state, but the policy checkpoint,
    routing configuration, update rule, starting semantic state, and budget
    context are identical. Evidence Guard data is not an input.
    """

    version_label = "V17"
    expected_policy_bundle_digest = POLICY_BUNDLE_DIGEST_V17
    promotion_decision_digest = PROMOTION_DECISION_DIGEST_V17
    source_manifest_digest = SOURCE_MANIFEST_DIGEST_V17
    runtime_repair_digest = META_V17_RUNTIME_REPAIR_DIGEST_V1
    activation_mode = "META_VNEXT_V17_SLOW_WITH_FAST_ONLY_IN_SUPPORT"

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        experiment_id: str,
        search_seed: int,
        scheduled_rounds: int,
        task_scale: float,
        task_density: float,
    ) -> None:
        checkpoint_path = checkpoint_path.resolve()
        checkpoint, policy, router = self._load_checkpoint(checkpoint_path)
        if scheduled_rounds < 1:
            raise MetaV17CampaignError("campaign requires a positive round count")
        if not 0.0 <= task_scale <= 1.0 or not 0.0 <= task_density <= 1.0:
            raise MetaV17CampaignError("task context must be normalized")
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = hashlib.sha256(
            checkpoint_path.read_bytes()
        ).hexdigest()
        self.experiment_id = str(experiment_id)
        self.search_seed = int(search_seed)
        self.scheduled_rounds = int(scheduled_rounds)
        self.task_scale = float(task_scale)
        self.task_density = float(task_density)
        self.policy = policy
        self.router = router
        self.checkpoint = checkpoint
        self.control_policy = self._research_control_policy()
        if (
            self.control_policy.meta_router_policy_digest
            != self.policy_bundle_digest
        ):
            raise MetaV17CampaignError(
                "Research control policy does not bind the active Meta router"
            )
        self._validate_routing_source()
        self._support_projections: dict[
            tuple[ArmCode, int], dict[str, Any]
        ] = {}
        self._states: dict[ArmCode, _ArmCampaignState] = {}
        self._bound_rounds: dict[tuple[ArmCode, int], _BoundRound] = {}
        self._producer_directives: dict[
            tuple[ArmCode, int], tuple[MetaProducerDirectiveV1, ...]
        ] = {}
        self._observation_records: list[dict[str, Any]] = []
        self._arm_instance_digests: dict[ArmCode, str] = {}
        self._search_seed_digest = sha256_digest(
            {
                "experiment_id": self.experiment_id,
                "search_seed": self.search_seed,
            }
        )

    def _load_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> tuple[dict[str, Any], PairwiseSlowPolicyV1, MetaVNextRouterV1]:
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        policy = PairwiseSlowPolicyV1.from_dict(checkpoint["policy"])
        router = MetaVNextRouterV1(**checkpoint["router"])
        bundle_digest = sha256_digest(
            {
                "schema": "recclaw.meta-vnext.policy-bundle.v1",
                "slow_policy_digest": policy.digest,
                "router_configuration_digest": router.configuration_digest,
            }
        )
        if (
            checkpoint["policy_bundle_digest"]
            != self.expected_policy_bundle_digest
            or bundle_digest != self.expected_policy_bundle_digest
            or checkpoint["slow_policy_digest"] != policy.digest
            or checkpoint["router_configuration_digest"]
            != router.configuration_digest
            or checkpoint["shadow_only"] is not True
            or checkpoint["activation_authority"] != "NONE"
        ):
            raise MetaV17CampaignError(
                f"{self.version_label} checkpoint identity is not exact"
            )
        return checkpoint, policy, router

    def _research_control_policy(self) -> VersionedResearchPolicyV1:
        return meta_v17_research_control_policy()

    def _validate_routing_source(self) -> None:
        routing_source_path = (
            Path(__file__).resolve().parent / "meta_vnext" / "routing.py"
        )
        routing_source_sha256 = hashlib.sha256(
            routing_source_path.read_bytes()
        ).hexdigest()
        if routing_source_sha256 != ROUTING_SOURCE_SHA256_RUNTIME_REPAIR_V1:
            raise MetaV17CampaignError(
                "Meta V17 runtime repair source identity is not exact"
            )
        self.routing_source_sha256 = routing_source_sha256

    @property
    def policy_bundle_digest(self) -> str:
        return self.expected_policy_bundle_digest

    def task_support_projection(self) -> dict[str, Any]:
        prototype_scales = tuple(
            float(row[2]) for row in self.policy.fast_response_prototypes
        )
        prototype_densities = tuple(
            float(row[3]) for row in self.policy.fast_response_prototypes
        )
        scale_bounds = (min(prototype_scales), max(prototype_scales))
        density_bounds = (min(prototype_densities), max(prototype_densities))
        return {
            "supported": (
                scale_bounds[0] <= self.task_scale <= scale_bounds[1]
                and density_bounds[0] <= self.task_density <= density_bounds[1]
            ),
            "task_density": self.task_density,
            "task_density_support": density_bounds,
            "task_scale": self.task_scale,
            "task_scale_support": scale_bounds,
        }

    def bind_instances(self, arm_to_instance: Mapping[ArmCode, str]) -> None:
        if self._states:
            raise MetaV17CampaignError("campaign instances are already bound")
        if set(arm_to_instance) != {ArmCode.A, ArmCode.B, ArmCode.C}:
            raise MetaV17CampaignError("campaign requires the exact A/B/C mapping")
        for arm in (ArmCode.B, ArmCode.C):
            opaque_digest = sha256_digest(
                {
                    "experiment_id": self.experiment_id,
                    "opaque_arm_instance_id": str(arm_to_instance[arm]),
                }
            )
            self._arm_instance_digests[arm] = opaque_digest
            self._states[arm] = _ArmCampaignState(
                fast_state=initialize_fast_residual(
                    opaque_arm_instance_digest=opaque_digest,
                    search_seed_digest=self._search_seed_digest,
                    policy=self.policy,
                )
            )
        if self.initial_semantic_state_projection(ArmCode.B) != (
            self.initial_semantic_state_projection(ArmCode.C)
        ):
            raise MetaV17CampaignError("B/C initial Meta state is not identical")

    def initial_semantic_state_projection(self, arm: ArmCode) -> dict[str, Any]:
        state = self._states[arm]
        return {
            "feature_names": state.fast_state.feature_names,
            "observed_responses": state.fast_state.observed_responses,
            "policy_bundle_digest": self.policy_bundle_digest,
            "round_boundary": state.fast_state.round_boundary,
            "update_rule_digest": state.fast_state.update_rule_digest,
            "visible_search_value_event_digests": (
                state.fast_state.visible_search_value_event_digests
            ),
        }

    def arm_private_context_digest(self, arm: ArmCode) -> str:
        """Digest only the selected Arm's mutable Meta context."""

        if arm not in {ArmCode.B, ArmCode.C} or arm not in self._states:
            raise MetaV17CampaignError(
                "Meta context is available only for a bound Research Arm"
            )
        state = self._states[arm]
        return sha256_digest(
            {
                "arm_instance_digest": self._arm_instance_digests[arm],
                "bound_rounds": [
                    {
                        "round_index": round_index,
                        "route": bound.route.to_dict(),
                    }
                    for (bound_arm, round_index), bound in sorted(
                        self._bound_rounds.items(),
                        key=lambda item: item[0][1],
                    )
                    if bound_arm is arm
                ],
                "fast_state_digest": state.fast_state.digest,
                "observation_records": [
                    item
                    for item in self._observation_records
                    if item["arm"] == arm.value
                ],
                "producer_directives": [
                    {
                        "round_index": round_index,
                        "directives": [
                            item.to_dict() for item in directives
                        ],
                    }
                    for (directive_arm, round_index), directives in sorted(
                        self._producer_directives.items(),
                        key=lambda item: item[0][1],
                    )
                    if directive_arm is arm
                ],
                "slow_state": {
                    "best_metric": state.best_metric,
                    "blocker_count": state.blocker_count,
                    "executed_axes": state.executed_axes,
                    "recent_gain": state.recent_gain,
                    "semantic_collision_count": (
                        state.semantic_collision_count
                    ),
                    "starting_metric": state.starting_metric,
                },
            }
        )

    def _research_context(
        self,
        arm: ArmCode,
        round_index: int,
    ) -> ResearchContextV1:
        state = self._states[arm]
        total_axes = max(1, len(state.executed_axes))
        axis_coverage = tuple(
            (
                axis,
                state.executed_axes.count(axis) / total_axes,
            )
            for axis in (
                "architecture",
                "geometry",
                "message_transform",
                "objective",
                "propagation",
                "sampling",
                "self_supervision",
            )
        )
        return ResearchContextV1(
            round_fraction=(round_index - 1) / self.scheduled_rounds,
            remaining_execution_fraction=(
                self.scheduled_rounds - round_index + 1
            )
            / self.scheduled_rounds,
            remaining_token_fraction=(
                self.scheduled_rounds - round_index + 1
            )
            / self.scheduled_rounds,
            remaining_gpu_fraction=(
                self.scheduled_rounds - round_index + 1
            )
            / self.scheduled_rounds,
            starting_frontier=state.starting_metric or 0.0,
            recent_frontier_gain=max(-1.0, min(1.0, state.recent_gain)),
            stagnation_fraction=(
                0.0
                if not state.executed_axes
                else min(
                    1.0,
                    sum(
                        1
                        for record in self._observation_records
                        if record["arm"] == arm.value
                        and float(record["frontier_value"]) <= 0.0
                    )
                    / len(state.executed_axes),
                )
            ),
            axis_coverage=axis_coverage,
            exact_duplicate_count=state.semantic_collision_count,
            near_duplicate_count=0,
            blocker_count=state.blocker_count,
            lineage_depth=round_index - 1,
            task_scale=self.task_scale,
            task_density=self.task_density,
        )

    def _learned_axis_score(
        self,
        *,
        axis: str,
        coverage: float,
    ) -> float:
        task_scale, task_density = self._axis_task_context()
        coefficients = dict(
            zip(
                self.policy.feature_names,
                self.policy.coefficients,
                strict=True,
            )
        )
        score = (
            float(coefficients.get(f"axis_{axis}", 0.0))
            + float(
                coefficients.get(f"axis_{axis}_x_task_scale", 0.0)
            )
            * task_scale
            + float(
                coefficients.get(f"axis_{axis}_x_task_density", 0.0)
            )
            * task_density
            + float(coefficients.get("axis_coverage_gap", 0.0))
            * (1.0 - coverage)
        )
        return round(score, 15)

    def _axis_task_context(self) -> tuple[float, float]:
        return self.task_scale, self.task_density

    def _directive_sort_key(
        self,
        row: tuple[float, float, int, str, str],
    ) -> tuple[Any, ...]:
        return (-row[0], row[1], -row[2], row[3], row[4])

    def producer_directives(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        memory_summary: Mapping[str, Any],
    ) -> tuple[MetaProducerDirectiveV1, ...]:
        """Plan four upstream research roles before any Producer invocation."""

        if arm not in {ArmCode.B, ArmCode.C} or arm not in self._states:
            raise MetaV17CampaignError(
                "Producer control is limited to bound Research Arms"
            )
        key = (arm, round_index)
        if key in self._producer_directives:
            return self._producer_directives[key]
        state = self._states[arm]
        if round_index != state.fast_state.round_boundary + 1:
            raise MetaV17CampaignError(
                "Producer control requires the next contiguous round"
            )
        executed = {
            str(item)
            for item in memory_summary.get("executed_mechanism_ids", ())
        }
        coverage_counts = {
            axis: state.executed_axes.count(axis)
            for axis in self.control_policy.mechanism_axis_targeting
        }
        denominator = max(1, len(state.executed_axes))
        pair_rows: list[tuple[float, float, int, str, str]] = []
        for root in ("LIGHTGCN", "BPR_MF"):
            by_axis: dict[str, list[str]] = {}
            for mechanism in executable_mechanisms():
                if (
                    mechanism.mechanism_id == root
                    or root_parent_mechanism_id(mechanism.mechanism_id)
                    != root
                ):
                    continue
                by_axis.setdefault(mechanism.mechanism_axis, []).append(
                    mechanism.mechanism_id
                )
            for axis, mechanism_ids in by_axis.items():
                coverage = coverage_counts.get(axis, 0) / denominator
                unexecuted_count = sum(
                    item not in executed for item in mechanism_ids
                )
                pair_rows.append(
                    (
                        self._learned_axis_score(
                            axis=axis,
                            coverage=coverage,
                        ),
                        coverage,
                        unexecuted_count,
                        root,
                        axis,
                    )
                )
        if len(pair_rows) < 3:
            raise MetaV17CampaignError(
                "Meta control requires at least three executable root-axis pairs"
            )
        ordered = sorted(
            pair_rows,
            key=self._directive_sort_key,
        )
        composer = next(
            (
                row
                for row in ordered
                if row[4]
                != executable_mechanism(row[3]).mechanism_axis
            ),
            ordered[0],
        )
        control_axis = executable_mechanism(composer[3]).mechanism_axis
        lineage = next(
            (
                row
                for row in ordered
                if row[3] == composer[3]
                and row[4] not in {composer[4], control_axis}
            ),
            next(row for row in ordered if row != composer),
        )
        frontier = next(
            (
                row
                for row in ordered
                if row[3] != composer[3]
                and row[4]
                not in {composer[4], lineage[4], control_axis}
            ),
            next(
                row
                for row in ordered
                if (row[3], row[4])
                not in {
                    (composer[3], composer[4]),
                    (lineage[3], lineage[4]),
                }
            ),
        )
        role_rows = {
            "mechanism_composer": (
                composer,
                "DISCOVERY",
                "RECENT_ROLE_RELEVANT",
                None,
            ),
            "lineage_refiner": (
                lineage,
                "DISCOVERY",
                "MATCH_PRIMARY_AXIS",
                None,
            ),
            "falsification_designer": (
                (
                    self._learned_axis_score(
                        axis=executable_mechanism(composer[3]).mechanism_axis,
                        coverage=0.0,
                    ),
                    0.0,
                    1,
                    composer[3],
                    executable_mechanism(composer[3]).mechanism_axis,
                ),
                "CONTROL",
                "CONFLICT_OR_UNRESOLVED",
                composer[3],
            ),
            "frontier_architect": (
                frontier,
                "DISCOVERY",
                "UNDERCOVERED_AXES",
                None,
            ),
        }
        allocations = dict(
            self.control_policy.producer_token_allocation
        )
        result = tuple(
            MetaProducerDirectiveV1(
                producer_role=role,
                lineage_root=str(role_rows[role][0][3]),
                primary_axis=str(role_rows[role][0][4]),
                proposal_intent=str(role_rows[role][1]),
                memory_query=str(role_rows[role][2]),
                token_share=float(allocations[role]),
                required_mechanism_id=role_rows[role][3],
                learned_axis_score=float(role_rows[role][0][0]),
                control_policy_digest=self.control_policy.digest,
            )
            for role in DISCOVERY_PRODUCERS
        )
        if (
            len({item.producer_role for item in result}) != 4
            or sum(item.token_share for item in result) != 1.0
        ):
            raise MetaV17CampaignError(
                "Meta Producer control violates the fixed four-role budget"
            )
        self._producer_directives[key] = result
        return result

    def _route_pool(
        self,
        *,
        pool: CandidatePoolVNextV1,
        state: _ArmCampaignState,
    ) -> tuple[MetaVNextRouteDecisionV1, bool, dict[str, Any]]:
        supported = task_context_supported(
            pool.eligible_candidates[0],
            self.policy,
        )
        if supported != self.task_support_projection()["supported"]:
            raise MetaV17CampaignError("Meta task-support projections diverged")
        if supported:
            decision = self.router.route(
                pool,
                self.policy,
                fast_state=state.fast_state,
                shadow_mode=False,
            )
        else:
            slow_scores = tuple(
                CandidateRouteScoreV1(
                    candidate_id=item.candidate_id,
                    candidate_semantics_digest=(
                        item.candidate_semantics_digest
                    ),
                    slow_score=round(self.policy.score(item), 15),
                    fast_correction=0.0,
                    uncertainty=0.0,
                    final_score=round(self.policy.score(item), 15),
                )
                for item in pool.eligible_candidates
            )
            selected = max(
                enumerate(slow_scores),
                key=lambda item: (item[1].final_score, -item[0]),
            )[1]
            decision = MetaVNextRouteDecisionV1(
                pool_digest=pool.digest,
                policy_digest=self.policy.digest,
                fast_state_digest=None,
                shadow_mode=False,
                scored_candidates=slow_scores,
                selected_candidate_id=selected.candidate_id,
                selected_candidate_semantics_digest=(
                    selected.candidate_semantics_digest
                ),
            )
        return (
            decision,
            supported,
            {
                "task_context_supported": supported,
                "fast_supported": supported,
                "fast_support_reason": (
                    "FAST_SUPPORTED"
                    if supported
                    else "ACTUAL_TASK_CONTEXT_OUT_OF_SUPPORT"
                ),
            },
        )

    def _route_mode(self, fast_supported: bool) -> str:
        return (
            "META_VNEXT_V17_SLOW_PLUS_FAST"
            if fast_supported
            else "META_VNEXT_V17_SLOW_ONLY_FAST_OUT_OF_SUPPORT"
        )

    def _acquire_candidate_order(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        parent_ranked_candidate_ids: tuple[str, ...],
        proposal_by_id: Mapping[
            str,
            CandidateProposalV2
            | CandidateProposalV3
            | CandidateProposalV4,
        ],
        parent_decision_digest: str | None,
    ) -> tuple[tuple[str, ...], str | None]:
        del arm, round_index, proposal_by_id, parent_decision_digest
        return parent_ranked_candidate_ids, None

    def _singleton_route_mode(self) -> str:
        return "STATIC_SINGLETON_INSUFFICIENT_META_POOL"

    def _fast_observation_supported(
        self,
        *,
        candidate: CandidateMechanismDeltaV1,
        route: MetaV17RouteResultV1,
    ) -> bool:
        del candidate
        return (
            route.fast_supported
            if route.fast_supported is not None
            else route.task_context_supported
        )

    def route_session(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        session: ProducerSessionResultV1,
        proposals: Sequence[
            CandidateProposalV2 | CandidateProposalV3 | CandidateProposalV4
        ]
        | None = None,
        exact_parent_programs: Mapping[str, Mapping[str, Any]] | None = None,
        static_router: StrongStaticRouterV1,
        research_policy: VersionedResearchPolicyV1,
        search_memory_head_digest: str | None = None,
    ) -> MetaV17RouteResultV1:
        if arm not in {ArmCode.B, ArmCode.C} or arm not in self._states:
            raise MetaV17CampaignError("V17 routing is limited to bound Research Arms")
        state = self._states[arm]
        if round_index != state.fast_state.round_boundary + 1:
            raise MetaV17CampaignError("Meta round boundary is not contiguous")
        key = (arm, round_index)
        if key in self._bound_rounds:
            raise MetaV17CampaignError("Meta route is create-once per Arm-round")
        eligibility_router = StrongStaticRouterV1(
            runnable_floor=static_router.runnable_floor,
            utility_floor=static_router.utility_floor,
            blocker_ceiling=static_router.blocker_ceiling,
            cost_ceiling=static_router.cost_ceiling,
            slate_ceiling=max(
                static_router.slate_ceiling,
                len(session.proposals),
            ),
        )
        route_proposals = tuple(proposals or session.proposals)
        static_trace = eligibility_router.route(
            route_proposals,
            policy_projection=research_policy.to_dict(),
        )
        ranked_static = static_trace.ranked_candidate_ids
        if not ranked_static:
            raise MetaV17CampaignError("Research pool has no runnable candidate")
        round_semantic_collision_count = sum(
            item.reason.value == "SEMANTIC_DUPLICATE"
            for item in static_trace.decisions
        )
        state.semantic_collision_count += round_semantic_collision_count
        static_champion_id = ranked_static[0]
        proposal_by_id = {
            proposal.candidate_id: proposal for proposal in route_proposals
        }
        if len(ranked_static) == 1:
            acquired, acquisition_digest = self._acquire_candidate_order(
                arm=arm,
                round_index=round_index,
                parent_ranked_candidate_ids=tuple(ranked_static),
                proposal_by_id=proposal_by_id,
                parent_decision_digest=None,
            )
            result = MetaV17RouteResultV1(
                arm=arm.value,
                round_index=round_index,
                mode=self._singleton_route_mode(),
                pool_digest=None,
                decision_digest=acquisition_digest,
                ranked_candidate_ids=acquired,
                selected_candidate_id=acquired[0],
                static_champion_candidate_id=static_champion_id,
                selected_candidate_semantics_digest=None,
                pool_candidate_semantics_digests=(),
                pool_mechanism_axes=(),
                round_semantic_collision_count=round_semantic_collision_count,
                task_context_supported=bool(
                    self.task_support_projection()["supported"]
                ),
            )
            self._bound_rounds[key] = _BoundRound(
                route=result,
                pool=None,
                decision=None,
            )
            return result
        parent_programs: dict[str, Mapping[str, Any]] = {}
        if all(
            isinstance(proposal_by_id[item], CandidateProposalV2)
            for item in ranked_static
        ):
            legacy_parent = deep_thaw(
                proposal_by_id[static_champion_id].mechanism_program
            )
            parent_programs = {
                item: legacy_parent for item in ranked_static
            }
        else:
            for candidate_id in ranked_static:
                proposal = proposal_by_id[candidate_id]
                exact_parent = dict(exact_parent_programs or {}).get(
                    candidate_id
                )
                if exact_parent is not None:
                    parent_programs[candidate_id] = deep_thaw(exact_parent)
                    continue
                explicit_parent = (
                    proposal_by_id.get(proposal.parent_candidate_id)
                    if proposal.parent_candidate_id is not None
                    else None
                )
                if explicit_parent is not None:
                    parent_programs[candidate_id] = deep_thaw(
                        explicit_parent.mechanism_program
                    )
                    continue
                if (
                    isinstance(proposal, CandidateProposalV4)
                    and proposal.parent_candidate_id is not None
                ):
                    raise MetaV17CampaignError(
                        "V13 declared parent is absent from exact lineage"
                    )
                mechanism_id = getattr(proposal, "mechanism_id", None)
                if mechanism_id is not None:
                    mechanism = executable_mechanism(str(mechanism_id))
                    root_parent = root_parent_mechanism_id(
                        mechanism.mechanism_id
                    )
                    if root_parent != mechanism.mechanism_id:
                        parent_programs[candidate_id] = deep_thaw(
                            executable_mechanism(
                                root_parent
                            ).mechanism_program
                        )
                        continue
                parent_programs[candidate_id] = deep_thaw(
                    proposal.mechanism_program
                )
        context = self._research_context(arm, round_index)
        pool = materialize_candidate_pool(
            pool_id=(
                f"{self.experiment_id}:{arm.value}:{self.search_seed}:"
                f"{round_index}"
            ),
            proposals=route_proposals,
            parent_programs=parent_programs,
            research_context=context,
            producer_invocation_digests=tuple(
                call.digest for call in session.calls
            ),
            pre_round_state_digest=sha256_digest(
                {
                    "fast_state_digest": state.fast_state.digest,
                    "research_context_digest": context.digest,
                    "search_memory_head": search_memory_head_digest,
                }
            ),
            candidate_order_policy_digest=sha256_digest(
                {
                    "policy": (
                        f"META_VNEXT_{self.version_label}_"
                        "FINAL_SCORE_DESC_STABLE_POOL_ORDER"
                    ),
                    "policy_bundle_digest": self.policy_bundle_digest,
                }
            ),
            static_router=eligibility_router,
            policy_projection=research_policy.to_dict(),
            lineage_depths={
                candidate_id: round_index - 1 for candidate_id in ranked_static
            },
        )
        decision, fast_supported, support_projection = self._route_pool(
            pool=pool,
            state=state,
        )
        self._support_projections[(arm, round_index)] = canonical_value(
            support_projection
        )
        score_by_id = {
            item.candidate_id: item.final_score
            for item in decision.scored_candidates
        }
        pool_order = {
            item.candidate_id: index
            for index, item in enumerate(pool.eligible_candidates)
        }
        score_ranked = tuple(
            sorted(
                score_by_id,
                key=lambda candidate_id: (
                    -score_by_id[candidate_id],
                    pool_order[candidate_id],
                ),
            )
        )
        if score_ranked[0] != decision.selected_candidate_id:
            raise MetaV17CampaignError("Meta decision and executable order diverged")
        ranked, acquisition_digest = self._acquire_candidate_order(
            arm=arm,
            round_index=round_index,
            parent_ranked_candidate_ids=score_ranked,
            proposal_by_id=proposal_by_id,
            parent_decision_digest=decision.digest,
        )
        result = MetaV17RouteResultV1(
            arm=arm.value,
            round_index=round_index,
            mode=self._route_mode(fast_supported),
            pool_digest=pool.digest,
            decision_digest=acquisition_digest or decision.digest,
            ranked_candidate_ids=ranked,
            selected_candidate_id=ranked[0],
            static_champion_candidate_id=static_champion_id,
            selected_candidate_semantics_digest=next(
                item.candidate_semantics_digest
                for item in pool.eligible_candidates
                if item.candidate_id == ranked[0]
            ),
            pool_candidate_semantics_digests=tuple(
                item.candidate_semantics_digest
                for item in pool.eligible_candidates
            ),
            pool_mechanism_axes=tuple(
                item.primary_mechanism_axis
                for item in pool.eligible_candidates
            ),
            round_semantic_collision_count=round_semantic_collision_count,
            task_context_supported=bool(
                support_projection.get(
                    "task_context_supported",
                    fast_supported,
                )
            ),
            support_projection_digest=(
                support_projection.get("digest")
                or sha256_digest(support_projection)
            ),
            fast_support_reason=support_projection.get(
                "fast_support_reason"
            ),
            fast_supported=fast_supported,
        )
        self._bound_rounds[key] = _BoundRound(
            route=result,
            pool=pool,
            decision=decision,
        )
        return result

    def record_observation(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        candidate_id: str,
        runtime_candidate_id: str | None = None,
        run_status: str,
        ndcg: float | None,
        wall_time_ms: int,
        source_search_utility_event_digest: str,
    ) -> None:
        self.record_round_boundary(
            arm=arm,
            round_index=round_index,
            proposal_source="NORMAL_ROUTED_PROPOSAL",
            observation_path="ADMITTED_OBSERVATION",
            candidate_id=candidate_id,
            runtime_candidate_id=runtime_candidate_id,
            run_status=run_status,
            ndcg=ndcg,
            wall_time_ms=wall_time_ms,
            source_search_utility_event_digest=(
                source_search_utility_event_digest
            ),
        )

    def record_round_boundary(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        proposal_source: str,
        observation_path: str,
        candidate_id: str | None,
        runtime_candidate_id: str | None,
        run_status: str,
        ndcg: float | None,
        wall_time_ms: int,
        source_search_utility_event_digest: str,
    ) -> None:
        """Advance the Meta boundary exactly once for every terminal B/C round."""

        validate_sha256(
            source_search_utility_event_digest,
            field_name="source_search_utility_event_digest",
        )
        valid_sources = {
            "NORMAL_ROUTED_PROPOSAL",
            "ACTIVE_BOUND_TASK",
            "NO_PROPOSAL_TERMINAL",
        }
        valid_observations = {
            "ADMITTED_OBSERVATION",
            "WITHHELD_OBSERVATION",
            "DIAGNOSTIC_OR_ENGINEERING_ONLY",
            "NO_OBSERVATION",
        }
        if proposal_source not in valid_sources:
            raise MetaV17CampaignError(
                "Meta boundary has an invalid Research proposal source"
            )
        if observation_path not in valid_observations:
            raise MetaV17CampaignError(
                "Meta boundary has an invalid observation path"
            )
        if any(
            record["arm"] == arm.value
            and record["round_index"] == round_index
            for record in self._observation_records
        ):
            raise MetaV17CampaignError("Meta boundary is create-once")
        bound = self._bound_rounds.get((arm, round_index))
        if (
            proposal_source == "NORMAL_ROUTED_PROPOSAL"
            and bound is None
        ):
            raise MetaV17CampaignError(
                "normal Research boundary lacks its frozen route"
            )
        if (
            proposal_source
            in {"ACTIVE_BOUND_TASK", "NO_PROPOSAL_TERMINAL"}
            and bound is not None
        ):
            raise MetaV17CampaignError(
                "task/no-proposal boundary cannot consume a Meta route"
            )
        if (
            observation_path == "ADMITTED_OBSERVATION"
            and proposal_source == "NORMAL_ROUTED_PROPOSAL"
        ):
            if candidate_id is None:
                raise MetaV17CampaignError(
                    "admitted observation requires a candidate"
                )
            if bound is None:
                raise MetaV17CampaignError(
                    "routed admitted observation lacks its frozen route"
                )
            self._record_admitted_observation(
                arm=arm,
                round_index=round_index,
                candidate_id=candidate_id,
                runtime_candidate_id=runtime_candidate_id,
                run_status=run_status,
                ndcg=ndcg,
                wall_time_ms=wall_time_ms,
                source_search_utility_event_digest=(
                    source_search_utility_event_digest
                ),
                proposal_source=proposal_source,
                observation_path=observation_path,
            )
            return
        state = self._states[arm]
        state.fast_state = advance_fast_without_observation(
            state.fast_state,
            opaque_arm_instance_digest=self._arm_instance_digests[arm],
            search_seed_digest=self._search_seed_digest,
            slow_policy_digest=self.policy.digest,
            round_boundary=round_index,
        )
        if run_status not in {"SUCCESS", "COMPLETED", "SMOKE_PASS"}:
            state.blocker_count += 1
        state.recent_gain = 0.0
        self._observation_records.append(
            canonical_value(
                {
                    "arm": arm.value,
                    "candidate_id": candidate_id,
                    "decision_digest": (
                        bound.route.decision_digest
                        if bound is not None
                        else None
                    ),
                    "fast_state_after_digest": state.fast_state.digest,
                    "frontier_value": 0.0,
                    "mode": (
                        bound.route.mode if bound is not None else "NO_ROUTE"
                    ),
                    "observation_digest": None,
                    "observation_path": observation_path,
                    "pool_digest": (
                        bound.route.pool_digest
                        if bound is not None
                        else None
                    ),
                    "proposal_source": proposal_source,
                    "round_index": round_index,
                    "run_status": run_status,
                    "runtime_candidate_id": runtime_candidate_id,
                    "selected_axis": "none",
                    "source_boundary_digest": (
                        source_search_utility_event_digest
                    ),
                    "wall_time_ms": int(wall_time_ms),
                }
            )
        )

    def _record_admitted_observation(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        candidate_id: str,
        runtime_candidate_id: str | None = None,
        run_status: str,
        ndcg: float | None,
        wall_time_ms: int,
        source_search_utility_event_digest: str,
        proposal_source: str,
        observation_path: str,
    ) -> None:
        validate_sha256(
            source_search_utility_event_digest,
            field_name="source_search_utility_event_digest",
        )
        key = (arm, round_index)
        if key not in self._bound_rounds:
            raise MetaV17CampaignError("Meta observation lacks its frozen route")
        if any(
            record["arm"] == arm.value
            and record["round_index"] == round_index
            for record in self._observation_records
        ):
            raise MetaV17CampaignError("Meta observation is create-once")
        bound = self._bound_rounds[key]
        if candidate_id not in bound.route.ranked_candidate_ids:
            raise MetaV17CampaignError("observed candidate was outside the frozen slate")
        state = self._states[arm]
        success = run_status == "SUCCESS" and ndcg is not None
        frontier = 0.0
        discriminative = 0.0
        selected_axis = "other"
        if success:
            metric = float(ndcg)
            if state.starting_metric is None:
                state.starting_metric = metric
            frontier = max(-1.0, min(1.0, metric - state.starting_metric))
            state.recent_gain = (
                metric - state.best_metric
                if state.best_metric is not None
                else 0.0
            )
            state.best_metric = (
                metric
                if state.best_metric is None
                else max(state.best_metric, metric)
            )
        else:
            state.blocker_count += 1
            state.recent_gain = 0.0
        candidate: CandidateMechanismDeltaV1 | None = None
        if bound.pool is not None:
            candidate = next(
                item
                for item in bound.pool.eligible_candidates
                if item.candidate_id == candidate_id
            )
            selected_axis = candidate.primary_mechanism_axis
            if candidate.ablation_or_falsification:
                discriminative = min(0.05, abs(frontier))
        state.executed_axes = (*state.executed_axes, selected_axis)
        if (
            candidate is None
            or not success
            or not self._fast_observation_supported(
                candidate=candidate,
                route=bound.route,
            )
        ):
            state.fast_state = advance_fast_without_observation(
                state.fast_state,
                opaque_arm_instance_digest=self._arm_instance_digests[arm],
                search_seed_digest=self._search_seed_digest,
                slow_policy_digest=self.policy.digest,
                round_boundary=round_index,
            )
            observation_digest = None
        else:
            observation = SearchValueObservationV1(
                candidate_semantics_digest=candidate.candidate_semantics_digest,
                source_search_utility_event_digest=(
                    source_search_utility_event_digest
                ),
                frontier_value=frontier,
                discriminative_value=discriminative,
                normalized_cost=max(
                    0.0,
                    min(1.0, float(wall_time_ms) / 900_000.0),
                ),
                blocker_loss=0.0,
                round_boundary=round_index,
            )
            state.fast_state = update_fast_residual(
                state.fast_state,
                policy=self.policy,
                candidate=candidate,
                observation=observation,
                opaque_arm_instance_digest=self._arm_instance_digests[arm],
                search_seed_digest=self._search_seed_digest,
                round_boundary=round_index,
            )
            observation_digest = observation.digest
        self._observation_records.append(
            canonical_value(
                {
                    "arm": arm.value,
                    "candidate_id": candidate_id,
                    "decision_digest": bound.route.decision_digest,
                    "fast_state_after_digest": state.fast_state.digest,
                    "frontier_value": frontier,
                    "mode": bound.route.mode,
                    "observation_digest": observation_digest,
                    "observation_path": observation_path,
                    "pool_digest": bound.route.pool_digest,
                    "proposal_source": proposal_source,
                    "round_index": round_index,
                    "run_status": run_status,
                    "runtime_candidate_id": runtime_candidate_id,
                    "selected_axis": selected_axis,
                    "wall_time_ms": int(wall_time_ms),
                }
            )
        )

    def audit_projection(self) -> dict[str, Any]:
        return canonical_value(
            {
                "activation_boundary": "NEXT_CAMPAIGN",
                "activation_mode": self.activation_mode,
                "checkpoint_sha256": self.checkpoint_sha256,
                "control_policy_digest": self.control_policy.digest,
                "experiment_id": self.experiment_id,
                "policy_bundle_digest": self.policy_bundle_digest,
                "promotion_decision_digest": self.promotion_decision_digest,
                "runtime_repair_digest": self.runtime_repair_digest,
                "runtime_routing_source_sha256": self.routing_source_sha256,
                "routes": [
                    bound.route.to_dict()
                    for _, bound in sorted(
                        self._bound_rounds.items(),
                        key=lambda item: (
                            item[0][1],
                            item[0][0].value,
                        ),
                    )
                ],
                "producer_directives": [
                    {
                        "arm": arm.value,
                        "round_index": round_index,
                        "directives": [
                            item.to_dict() for item in directives
                        ],
                        "directive_set_digest": sha256_digest(
                            [item.to_dict() for item in directives]
                        ),
                    }
                    for (arm, round_index), directives in sorted(
                        self._producer_directives.items(),
                        key=lambda item: (
                            item[0][1],
                            item[0][0].value,
                        ),
                    )
                ],
                "observations": self._observation_records,
                "source_manifest_digest": self.source_manifest_digest,
                "support_projections": [
                    {
                        "arm": arm.value,
                        "round_index": round_index,
                        "projection": projection,
                    }
                    for (arm, round_index), projection in sorted(
                        self._support_projections.items(),
                        key=lambda item: (
                            item[0][1],
                            item[0][0].value,
                        ),
                    )
                ],
                "task_support": self.task_support_projection(),
                "states": {
                    arm.value: {
                        "best_metric": state.best_metric,
                        "blocker_count": state.blocker_count,
                        "executed_axes": state.executed_axes,
                        "fast_state_digest": state.fast_state.digest,
                        "observed_response_count": len(
                            state.fast_state.observed_responses
                        ),
                        "observed_response_rounds": tuple(
                            item[0]
                            for item in state.fast_state.observed_responses
                        ),
                        "round_boundary": state.fast_state.round_boundary,
                        "semantic_collision_count": (
                            state.semantic_collision_count
                        ),
                    }
                    for arm, state in sorted(
                        self._states.items(),
                        key=lambda item: item[0].value,
                    )
                },
            }
        )


class MetaV18CampaignRuntimeV1(MetaV17CampaignRuntimeV1):
    """Use the V17 learned coefficients through the V18 support contract."""

    version_label = "V18"
    expected_checkpoint_id = "META_VNEXT_V18_SUPPORT_AWARE"
    expected_checkpoint_sha256 = CHECKPOINT_SHA256_V18
    expected_policy_bundle_digest = POLICY_BUNDLE_DIGEST_V18
    promotion_decision_digest = PROMOTION_DECISION_DIGEST_V18
    source_manifest_digest = sha256_digest(
        {
            "parent_source_manifest_digest": SOURCE_MANIFEST_DIGEST_V17,
            "candidate_contract": "CandidateProposalV4",
            "executable_profile_digest": (
                "d483faa471c3f26d321daa89c6946ab67a11a95d459d2d5c095c9da419ee5da1"
            ),
            "feature_support_sha256": (
                "5065cd66f8673b65b4f123bbbabff2404c7df2ed7f32c07a7e3a62b802a47768"
            ),
        }
    )
    runtime_repair_digest = sha256_digest(
        {
            "policy_id": ROUTER_POLICY_ID_V18,
            "slow_projection": SLOW_PROJECTION_ID_V18,
            "fast_support": FAST_SUPPORT_ID_V18,
            "parent_runtime_repair_digest": META_V17_RUNTIME_REPAIR_DIGEST_V1,
        }
    )
    activation_mode = (
        "META_VNEXT_V18_SUPPORT_PROJECTED_SLOW_WITH_EXACT_SUPPORT_FAST"
    )

    def _load_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> tuple[dict[str, Any], PairwiseSlowPolicyV1, MetaVNextRouterV1]:
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != (
            self.expected_checkpoint_sha256
        ):
            raise MetaV17CampaignError(
                f"{self.version_label} checkpoint bytes are not exact"
            )
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        parent_path = checkpoint_path.parent / str(
            checkpoint["parent_checkpoint_resource"]
        )
        if hashlib.sha256(parent_path.read_bytes()).hexdigest() != checkpoint[
            "parent_checkpoint_sha256"
        ]:
            raise MetaV17CampaignError(
                f"{self.version_label} parent checkpoint bytes drifted"
            )
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        policy = PairwiseSlowPolicyV1.from_dict(parent["policy"])
        router = MetaVNextRouterV1(**parent["router"])
        parent_bundle = sha256_digest(
            {
                "schema": "recclaw.meta-vnext.policy-bundle.v1",
                "slow_policy_digest": policy.digest,
                "router_configuration_digest": router.configuration_digest,
            }
        )
        profile_digest = bl_icf_executable_profile_v2()["profile_digest"]
        bundle = sha256_digest(
            {
                "schema": "recclaw.meta-vnext.policy-bundle.v2",
                "parent_slow_policy_digest": policy.digest,
                "router_configuration_digest": router.configuration_digest,
                "feature_schema_digest": checkpoint["feature_schema_digest"],
                "candidate_contract": checkpoint["candidate_contract"],
                "executable_profile_digest": profile_digest,
                "feature_support_sha256": feature_support_sha256(),
                "slow_projection": checkpoint["slow_projection"],
                "fast_support": checkpoint["fast_support"],
            }
        )
        if (
            checkpoint.get("schema")
            != "recclaw.meta-vnext-policy-checkpoint.v2"
            or checkpoint.get("checkpoint_id")
            != self.expected_checkpoint_id
            or checkpoint.get("candidate_contract") != "CandidateProposalV4"
            or checkpoint.get("executable_profile_digest") != profile_digest
            or checkpoint.get("feature_support_sha256")
            != feature_support_sha256()
            or checkpoint.get("slow_projection") != SLOW_PROJECTION_ID_V18
            or checkpoint.get("fast_support") != FAST_SUPPORT_ID_V18
            or checkpoint.get("parent_policy_bundle_digest")
            != POLICY_BUNDLE_DIGEST_V17
            or parent_bundle != POLICY_BUNDLE_DIGEST_V17
            or checkpoint.get("parent_slow_policy_digest") != policy.digest
            or checkpoint.get("router_configuration_digest")
            != router.configuration_digest
            or checkpoint.get("policy_bundle_digest") != bundle
            or bundle != self.expected_policy_bundle_digest
            or checkpoint.get("shadow_only") is not False
            or checkpoint.get("activation_boundary") != "NEXT_CAMPAIGN"
            or checkpoint.get("pilot_outcomes_used") is not False
        ):
            raise MetaV17CampaignError(
                f"{self.version_label} checkpoint identity is not exact"
            )
        return checkpoint, policy, router

    def _research_control_policy(self) -> VersionedResearchPolicyV1:
        return meta_v18_research_control_policy()

    def _axis_task_context(self) -> tuple[float, float]:
        support = self.task_support_projection()
        scale = min(
            support["task_scale_support"][1],
            max(support["task_scale_support"][0], self.task_scale),
        )
        density = min(
            support["task_density_support"][1],
            max(support["task_density_support"][0], self.task_density),
        )
        return float(scale), float(density)

    def _directive_sort_key(
        self,
        row: tuple[float, float, int, str, str],
    ) -> tuple[Any, ...]:
        return (row[1], -row[2], -row[0], row[3], row[4])

    def producer_directives(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        memory_summary: Mapping[str, Any],
    ) -> tuple[MetaProducerDirectiveV1, ...]:
        directives = super().producer_directives(
            arm=arm,
            round_index=round_index,
            memory_summary=memory_summary,
        )
        result = tuple(
            replace(
                item,
                proposal_intent="FALSIFICATION",
                required_mechanism_id=None,
            )
            if item.producer_role == "falsification_designer"
            else item
            for item in directives
        )
        self._producer_directives[(arm, round_index)] = result
        return result

    def _route_pool(
        self,
        *,
        pool: CandidatePoolVNextV1,
        state: _ArmCampaignState,
    ) -> tuple[MetaVNextRouteDecisionV1, bool, dict[str, Any]]:
        decision, support = route_support_aware(
            pool=pool,
            policy=self.policy,
            router=self.router,
            fast_state=state.fast_state,
            actual_task_scale=self.task_scale,
            actual_task_density=self.task_density,
            shadow_mode=False,
        )
        projection = support.to_dict()
        projection["digest"] = support.digest
        return decision, support.fast_supported, projection

    def _route_mode(self, fast_supported: bool) -> str:
        return (
            "META_VNEXT_V18_SUPPORT_PROJECTED_SLOW_PLUS_FAST"
            if fast_supported
            else (
                "META_VNEXT_V18_SUPPORT_PROJECTED_SLOW_ONLY_"
                "FAST_OUT_OF_SUPPORT"
            )
        )

    def _fast_observation_supported(
        self,
        *,
        candidate: CandidateMechanismDeltaV1,
        route: MetaV17RouteResultV1,
    ) -> bool:
        support = load_feature_support()
        return (
            bool(route.fast_supported)
            and candidate.primary_mechanism_axis
            in set(support["calibrated_mechanism_axes"])
        )

    def task_support_projection(self) -> dict[str, Any]:
        projection = super().task_support_projection()
        slow_scale, slow_density = self._axis_task_context_from_bounds(
            projection
        )
        return {
            **projection,
            "slow_projected_task_density": slow_density,
            "slow_projected_task_scale": slow_scale,
            "slow_projection": SLOW_PROJECTION_ID_V18,
            "fast_support": FAST_SUPPORT_ID_V18,
        }

    def _axis_task_context_from_bounds(
        self,
        support: Mapping[str, Any],
    ) -> tuple[float, float]:
        scale = min(
            support["task_scale_support"][1],
            max(support["task_scale_support"][0], self.task_scale),
        )
        density = min(
            support["task_density_support"][1],
            max(support["task_density_support"][0], self.task_density),
        )
        return float(scale), float(density)

    def audit_projection(self) -> dict[str, Any]:
        result = super().audit_projection()
        result["checkpoint_id"] = self.checkpoint["checkpoint_id"]
        result["parent_checkpoint_sha256"] = self.checkpoint[
            "parent_checkpoint_sha256"
        ]
        result["feature_support_sha256"] = feature_support_sha256()
        result["executable_profile_digest"] = self.checkpoint[
            "executable_profile_digest"
        ]
        return canonical_value(result)


class MetaV19CampaignRuntimeV1(MetaV18CampaignRuntimeV1):
    """V18 policy with only its Provider-strict transport identity rebound."""

    version_label = "V19"
    expected_checkpoint_id = (
        "META_VNEXT_V19_PROVIDER_STRICT_SCHEMA_REBIND"
    )
    expected_checkpoint_sha256 = CHECKPOINT_SHA256_V19
    expected_policy_bundle_digest = POLICY_BUNDLE_DIGEST_V19
    promotion_decision_digest = PROMOTION_DECISION_DIGEST_V19
    source_manifest_digest = sha256_digest(
        {
            "parent_source_manifest_digest": (
                MetaV18CampaignRuntimeV1.source_manifest_digest
            ),
            "candidate_contract": "CandidateProposalV4",
            "executable_profile_digest": (
                "f748b4b4b3103eadf2c3c262c14c4b779893255f2318bd99ebbf9f9eeee47536"
            ),
            "transport_rebind": (
                "PROVIDER_STRICT_SCHEMA_REQUIRED_ALL_OBJECT_PROPERTIES"
            ),
        }
    )
    runtime_repair_digest = sha256_digest(
        {
            "parent_runtime_repair_digest": (
                MetaV18CampaignRuntimeV1.runtime_repair_digest
            ),
            "transport_rebind": (
                "PROVIDER_STRICT_SCHEMA_REQUIRED_ALL_OBJECT_PROPERTIES"
            ),
        }
    )
    activation_mode = (
        "META_VNEXT_V19_PROVIDER_STRICT_SCHEMA_REBIND_V18_POLICY"
    )

    def _research_control_policy(self) -> VersionedResearchPolicyV1:
        return meta_v19_research_control_policy()


class MetaV20CampaignRuntimeV1(MetaV19CampaignRuntimeV1):
    """V19 scoring plus a pre-frozen, Arm-private Producer opportunity layer."""

    version_label = "V20"
    expected_checkpoint_id = (
        "META_VNEXT_V20_PRODUCER_OPPORTUNITY_ACQUISITION"
    )
    expected_checkpoint_sha256 = CHECKPOINT_SHA256_V20
    expected_policy_bundle_digest = POLICY_BUNDLE_DIGEST_V20
    promotion_decision_digest = (
        DEVELOPMENT_ACTIVATION_DECISION_DIGEST_V20
    )
    source_manifest_digest = sha256_digest(
        {
            "parent_source_manifest_digest": (
                MetaV19CampaignRuntimeV1.source_manifest_digest
            ),
            "producer_opportunity_policy_digest": (
                PRODUCER_OPPORTUNITY_POLICY_DIGEST_V1
            ),
            "v24_method_diagnosis_sha256": (
                V24_METHOD_DIAGNOSIS_SHA256
            ),
        }
    )
    runtime_repair_digest = sha256_digest(
        {
            "parent_runtime_repair_digest": (
                MetaV19CampaignRuntimeV1.runtime_repair_digest
            ),
            "producer_opportunity_policy_digest": (
                PRODUCER_OPPORTUNITY_POLICY_DIGEST_V1
            ),
        }
    )
    activation_mode = (
        "DEVELOPMENT_META_V20_V19_SCORE_WITH_BLOCK8_PRODUCER_OPPORTUNITY"
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._producer_opportunity_history: dict[
            ArmCode, tuple[str, ...]
        ] = {
            ArmCode.B: (),
            ArmCode.C: (),
        }
        self._producer_opportunity_decisions: dict[
            tuple[ArmCode, int], ProducerOpportunityDecisionV1
        ] = {}

    def _load_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> tuple[dict[str, Any], PairwiseSlowPolicyV1, MetaVNextRouterV1]:
        if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != (
            self.expected_checkpoint_sha256
        ):
            raise MetaV17CampaignError(
                "V20 checkpoint bytes are not exact"
            )
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        parent_path = checkpoint_path.parent / str(
            checkpoint["parent_checkpoint_resource"]
        )
        if hashlib.sha256(parent_path.read_bytes()).hexdigest() != (
            CHECKPOINT_SHA256_V19
        ):
            raise MetaV17CampaignError(
                "V20 parent V19 checkpoint bytes drifted"
            )
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        slow_path = parent_path.parent / str(
            parent["parent_checkpoint_resource"]
        )
        if hashlib.sha256(slow_path.read_bytes()).hexdigest() != parent[
            "parent_checkpoint_sha256"
        ]:
            raise MetaV17CampaignError(
                "V20 inherited slow-policy checkpoint bytes drifted"
            )
        slow_parent = json.loads(slow_path.read_text(encoding="utf-8"))
        policy = PairwiseSlowPolicyV1.from_dict(slow_parent["policy"])
        router = MetaVNextRouterV1(**slow_parent["router"])
        profile_digest = bl_icf_executable_profile_v2()["profile_digest"]
        bundle = sha256_digest(
            {
                "schema": "recclaw.meta-vnext.policy-bundle.v3",
                "parent_policy_bundle_digest": POLICY_BUNDLE_DIGEST_V19,
                "parent_checkpoint_sha256": CHECKPOINT_SHA256_V19,
                "coefficient_action": (
                    "INHERIT_EXACT_V19_NO_COEFFICIENT_CHANGE"
                ),
                "producer_opportunity_policy_digest": (
                    PRODUCER_OPPORTUNITY_POLICY_DIGEST_V1
                ),
                "candidate_contract": "CandidateProposalV4",
                "executable_profile_digest": profile_digest,
                "v24_method_diagnosis_sha256": (
                    V24_METHOD_DIAGNOSIS_SHA256
                ),
            }
        )
        expected_opportunity = {
            "block_size": 8,
            "coverage_role_count": 4,
            "policy_digest": PRODUCER_OPPORTUNITY_POLICY_DIGEST_V1,
            "policy_id": PRODUCER_OPPORTUNITY_POLICY_ID_V1,
            "score_source": "EXACT_PARENT_META_V19_ORDER",
            "selection": (
                "FORCE_EACH_AVAILABLE_ROLE_ONCE_PER_BLOCK_BEFORE_SCORE_ONLY"
            ),
        }
        if (
            checkpoint.get("schema")
            != "recclaw.meta-vnext-policy-checkpoint.v3"
            or checkpoint.get("checkpoint_id")
            != self.expected_checkpoint_id
            or checkpoint.get("authority") != "NONE"
            or checkpoint.get("formal_acceptance") is not False
            or checkpoint.get("candidate_contract")
            != "CandidateProposalV4"
            or checkpoint.get("executable_profile_digest")
            != profile_digest
            or checkpoint.get("parent_checkpoint_sha256")
            != CHECKPOINT_SHA256_V19
            or checkpoint.get("parent_policy_bundle_digest")
            != POLICY_BUNDLE_DIGEST_V19
            or parent.get("policy_bundle_digest")
            != POLICY_BUNDLE_DIGEST_V19
            or checkpoint.get("coefficient_action")
            != "INHERIT_EXACT_V19_NO_COEFFICIENT_CHANGE"
            or checkpoint.get("producer_opportunity_policy")
            != expected_opportunity
            or checkpoint.get("policy_bundle_digest") != bundle
            or bundle != self.expected_policy_bundle_digest
            or checkpoint.get("development_activation_decision_digest")
            != DEVELOPMENT_ACTIVATION_DECISION_DIGEST_V20
            or checkpoint.get("activation_boundary")
            != "NEXT_FRESH_CAMPAIGN"
            or checkpoint.get("pilot_outcomes_used") is not True
            or checkpoint.get("outcome_use_scope")
            != (
                "V24_DIAGNOSTIC_ONLY_NO_THRESHOLD_METRIC_SEED_"
                "OR_RESULT_STATE_REUSE"
            )
            or checkpoint.get("success_metric_or_threshold_changed")
            is not False
            or checkpoint.get("v24_method_diagnosis_sha256")
            != V24_METHOD_DIAGNOSIS_SHA256
            or checkpoint.get("shadow_only") is not False
        ):
            raise MetaV17CampaignError(
                "V20 checkpoint identity is not exact"
            )
        return checkpoint, policy, router

    def _research_control_policy(self) -> VersionedResearchPolicyV1:
        return meta_v20_research_control_policy()

    def _acquire_candidate_order(
        self,
        *,
        arm: ArmCode,
        round_index: int,
        parent_ranked_candidate_ids: tuple[str, ...],
        proposal_by_id: Mapping[
            str,
            CandidateProposalV2
            | CandidateProposalV3
            | CandidateProposalV4,
        ],
        parent_decision_digest: str | None,
    ) -> tuple[tuple[str, ...], str | None]:
        decision = acquire_producer_opportunity(
            parent_ranked_candidate_ids=parent_ranked_candidate_ids,
            parent_decision_digest=(
                parent_decision_digest
                or sha256_digest(
                    {
                        "mode": "STATIC_SINGLETON",
                        "ranked_candidate_ids": (
                            parent_ranked_candidate_ids
                        ),
                    }
                )
            ),
            producer_role_by_candidate_id={
                candidate_id: proposal_by_id[
                    candidate_id
                ].producer_role
                for candidate_id in parent_ranked_candidate_ids
            },
            prior_selected_roles=(
                self._producer_opportunity_history[arm]
            ),
        )
        self._producer_opportunity_history[arm] = (
            *self._producer_opportunity_history[arm],
            decision.selected_producer_role,
        )
        self._producer_opportunity_decisions[(arm, round_index)] = decision
        return decision.ranked_candidate_ids, decision.digest

    def _route_mode(self, fast_supported: bool) -> str:
        parent = super()._route_mode(fast_supported)
        return f"{parent}_V20_BLOCK8_PRODUCER_OPPORTUNITY"

    def _singleton_route_mode(self) -> str:
        return (
            "STATIC_SINGLETON_V20_PRODUCER_OPPORTUNITY_RECORDED"
        )

    def arm_private_context_digest(self, arm: ArmCode) -> str:
        return sha256_digest(
            {
                "parent_context_digest": (
                    super().arm_private_context_digest(arm)
                ),
                "producer_opportunity_history": (
                    self._producer_opportunity_history[arm]
                ),
                "producer_opportunity_decisions": [
                    {
                        "round_index": round_index,
                        "decision": decision.to_dict(),
                    }
                    for (decision_arm, round_index), decision in sorted(
                        self._producer_opportunity_decisions.items(),
                        key=lambda item: item[0][1],
                    )
                    if decision_arm is arm
                ],
            }
        )

    def audit_projection(self) -> dict[str, Any]:
        result = super().audit_projection()
        result.update(
            {
                "development_activation_not_promotion": True,
                "producer_opportunity_policy_digest": (
                    PRODUCER_OPPORTUNITY_POLICY_DIGEST_V1
                ),
                "producer_opportunity_policy_id": (
                    PRODUCER_OPPORTUNITY_POLICY_ID_V1
                ),
                "producer_opportunity_decisions": [
                    {
                        "arm": arm.value,
                        "round_index": round_index,
                        "decision": decision.to_dict(),
                    }
                    for (arm, round_index), decision in sorted(
                        self._producer_opportunity_decisions.items(),
                        key=lambda item: (
                            item[0][1],
                            item[0][0].value,
                        ),
                    )
                ],
                "v24_method_diagnosis_sha256": (
                    V24_METHOD_DIAGNOSIS_SHA256
                ),
            }
        )
        return canonical_value(result)


__all__ = [
    "MetaV17CampaignError",
    "MetaV17CampaignRuntimeV1",
    "MetaV18CampaignRuntimeV1",
    "MetaV19CampaignRuntimeV1",
    "MetaV20CampaignRuntimeV1",
    "MetaProducerDirectiveV1",
    "MetaV17RouteResultV1",
    "META_V17_RUNTIME_REPAIR_DIGEST_V1",
    "POLICY_BUNDLE_DIGEST_V17",
    "POLICY_BUNDLE_DIGEST_V18",
    "POLICY_BUNDLE_DIGEST_V19",
    "POLICY_BUNDLE_DIGEST_V20",
    "PROMOTION_DECISION_DIGEST_V17",
    "PROMOTION_DECISION_DIGEST_V18",
    "PROMOTION_DECISION_DIGEST_V19",
    "SOURCE_MANIFEST_DIGEST_V17",
    "meta_v17_research_control_policy",
    "meta_v17_static_producer_policy",
    "meta_v18_research_control_policy",
    "meta_v19_research_control_policy",
    "meta_v20_research_control_policy",
]
