"""Materialize parent-relative mechanism features before outcomes are visible."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any, Mapping, Sequence

from recclaw_core.mechanism_space import compile_program
from recclaw_core.mechanism_space.canonical import deep_thaw

from ..canonical import canonical_value, sha256_digest
from ..research_capability import StrongStaticRouterV1
from ..research_contracts import CandidateProposalV2, ProposalIntentV1
from .contracts import (
    MECHANISM_AXIS_FEATURES_V1,
    PRIMITIVE_CLASS_FEATURES_V1,
    RANK_FEATURE_NAMES_V1,
    CandidateMechanismDeltaV1,
    CandidatePoolVNextV1,
    ChangeClassV1,
    ResearchContextV1,
)


class MetaVNextFeatureError(ValueError):
    """Raised when a mechanism delta cannot be materialized honestly."""


FEATURE_SCHEMA_DIGEST_V1 = sha256_digest(
    {
        "schema": "recclaw.meta-vnext.rank-feature-schema.v1",
        "feature_names": RANK_FEATURE_NAMES_V1,
        "identity_features": [],
        "outcome_features": [],
        "evidence_authority_features": [],
    }
)

_COST_ORDINAL = {"LOW": 0.0, "MEDIUM": 0.5, "HIGH": 1.0}


def _program_payload(program: Mapping[str, Any]) -> dict[str, Any]:
    thawed = deep_thaw(program)
    payload = thawed.get("program_payload")
    if not isinstance(payload, Mapping):
        raise MetaVNextFeatureError("mechanism program lacks program_payload")
    return dict(payload)


def _components_by_id(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    components = payload.get("components", ())
    if not isinstance(components, (list, tuple)):
        raise MetaVNextFeatureError("program components must be a sequence")
    result: dict[str, dict[str, Any]] = {}
    for component in components:
        if (
            not isinstance(component, Mapping)
            or "component_id" not in component
            or "slot_id" not in component
        ):
            raise MetaVNextFeatureError("program component lacks identity or slot")
        component_id = str(component["component_id"])
        if component_id in result:
            raise MetaVNextFeatureError("component IDs must be unique")
        result[component_id] = dict(component)
    return result


def _parameter_change_count(left: Any, right: Any) -> int:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = set(left) | set(right)
        return sum(
            _parameter_change_count(left.get(key), right.get(key))
            for key in keys
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        common = min(len(left), len(right))
        return (
            sum(_parameter_change_count(left[index], right[index]) for index in range(common))
            + abs(len(left) - len(right))
        )
    return int(canonical_value(left) != canonical_value(right))


def _multiset_delta(
    candidate_values: Sequence[str],
    parent_values: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    candidate = Counter(str(item) for item in candidate_values)
    parent = Counter(str(item) for item in parent_values)
    added = tuple(sorted((candidate - parent).elements()))
    removed = tuple(sorted((parent - candidate).elements()))
    return added, removed


def _secondary_axes(changed_slots: Sequence[str], primary_axis: str) -> tuple[str, ...]:
    inferred: set[str] = set()
    for raw_slot in changed_slots:
        slot = str(raw_slot).upper()
        if "OBJECTIVE" in slot:
            inferred.add("objective")
        if "EMBEDDING" in slot:
            inferred.add("geometry")
        if any(token in slot for token in ("PROPAGATION", "RELATION", "FUSION")):
            inferred.add("propagation")
        if any(token in slot for token in ("MESSAGE", "ENCODER")):
            inferred.add("message_transform")
        if "SAMPLER" in slot:
            inferred.add("sampling")
        if "TRAINING" in slot:
            inferred.add("optimization")
    inferred.discard(str(primary_axis))
    return tuple(sorted(inferred))


def _bounded_count(value: int, scale: float) -> float:
    return min(1.0, max(0.0, float(value) / float(scale)))


def _primitive_class(primitive_id: str) -> str:
    value = str(primitive_id).lower()
    if value.startswith(("objective.", "loss.")):
        return "objective"
    if value.startswith("embedding."):
        return "embedding"
    if value.startswith(("encoder.", "message.")):
        return "encoder_message"
    if value.startswith(("propagation.", "relation.", "fusion.")):
        return "propagation"
    if value.startswith(("ssl.", "augmentation.")):
        return "self_supervision"
    if value.startswith(("sampler.", "sampling.")):
        return "sampling"
    if value.startswith(("regularizer.", "regularization.")):
        return "regularization"
    if value.startswith(("training.", "optimizer.")):
        return "training"
    return "other"


def _classify_change(
    *,
    candidate_program_digest: str,
    parent_program_digest: str,
    parameter_change_count: int,
    added_primitives: Sequence[str],
    removed_primitives: Sequence[str],
    replaced_slots: Sequence[str],
    architecture_operator_delta: int,
    custom_component_delta: int,
    candidate_construction_mode: str,
    parent_construction_mode: str,
    total_slot_count: int,
) -> ChangeClassV1:
    if candidate_program_digest == parent_program_digest:
        return ChangeClassV1.CONTROL
    structural_count = (
        len(added_primitives) + len(removed_primitives) + len(replaced_slots)
    )
    if (
        structural_count == 0
        and architecture_operator_delta == 0
        and custom_component_delta == 0
        and parameter_change_count > 0
    ):
        return ChangeClassV1.PARAMETER_TUNING_ONLY
    architecture_rewrite = (
        architecture_operator_delta != 0
        or custom_component_delta != 0
        or candidate_construction_mode != parent_construction_mode
        or len(replaced_slots) > max(1, total_slot_count // 2)
    )
    if architecture_rewrite:
        return ChangeClassV1.ARCHITECTURE_REWRITE
    return ChangeClassV1.MECHANISM_CHANGE


def materialize_candidate_pool(
    *,
    pool_id: str,
    proposals: Sequence[CandidateProposalV2],
    parent_programs: Mapping[str, Mapping[str, Any]],
    research_context: ResearchContextV1,
    producer_invocation_digests: Sequence[str],
    pre_round_state_digest: str,
    candidate_order_policy_digest: str,
    static_router: StrongStaticRouterV1,
    policy_projection: Mapping[str, Any] | None = None,
    lineage_depths: Mapping[str, int] | None = None,
) -> CandidatePoolVNextV1:
    """Freeze the eligible runtime pool and its pre-outcome feature matrix."""

    source_proposals = tuple(proposals)
    eligibility_router = replace(
        static_router,
        slate_ceiling=max(static_router.slate_ceiling, len(source_proposals)),
    )
    eligibility_trace = eligibility_router.route(
        source_proposals, policy_projection
    )
    assessments = tuple(
        (
            decision,
            static_router.score(proposal.utility_features, policy_projection)
            if decision.allowed
            else None,
        )
        for proposal, decision in zip(
            source_proposals, eligibility_trace.decisions, strict=True
        )
    )
    lineage = dict(lineage_depths or {})
    candidates: list[CandidateMechanismDeltaV1] = []
    rejected: list[str] = []
    assessment_projection: list[dict[str, Any]] = []

    for proposal, (decision, static_score) in zip(
        source_proposals, assessments, strict=True
    ):
        assessment_projection.append(
            {"decision": decision.to_dict(), "static_score": static_score}
        )
        if not decision.allowed:
            rejected.append(proposal.candidate_id)
            continue
        parent_program = parent_programs.get(proposal.candidate_id)
        if parent_program is None:
            raise MetaVNextFeatureError(
                f"eligible candidate {proposal.candidate_id} lacks its parent program"
            )
        parent_report = compile_program(parent_program)
        if not parent_report.is_valid:
            raise MetaVNextFeatureError("parent mechanism program must compile")
        if parent_report.mechanism_semantics_digest is None:
            raise MetaVNextFeatureError("parent semantics digest is unavailable")
        if decision.mechanism_semantics_digest is None or static_score is None:
            raise MetaVNextFeatureError("eligible static assessment is incomplete")

        candidate_program = deep_thaw(proposal.mechanism_program)
        candidate_payload = _program_payload(candidate_program)
        parent_payload = _program_payload(parent_program)
        candidate_components = _components_by_id(candidate_payload)
        parent_components = _components_by_id(parent_payload)
        candidate_program_digest = sha256_digest(candidate_program)
        parent_program_digest = sha256_digest(parent_program)

        candidate_primitives = [
            str(item.get("primitive_id", ""))
            for item in candidate_components.values()
        ]
        parent_primitives = [
            str(item.get("primitive_id", "")) for item in parent_components.values()
        ]
        added_primitives, removed_primitives = _multiset_delta(
            candidate_primitives, parent_primitives
        )
        common_components = set(candidate_components) & set(parent_components)
        replaced_slots = tuple(
            sorted(
                {
                    str(candidate_components[component_id]["slot_id"])
                    for component_id in common_components
                    if candidate_components[component_id].get("primitive_id")
                    != parent_components[component_id].get("primitive_id")
                }
            )
        )
        parameter_change_count = sum(
            _parameter_change_count(
                candidate_components[component_id].get("parameters", {}),
                parent_components[component_id].get("parameters", {}),
            )
            for component_id in common_components
            if candidate_components[component_id].get("primitive_id")
            == parent_components[component_id].get("primitive_id")
        )
        added_slots = {
            str(candidate_components[component_id]["slot_id"])
            for component_id in set(candidate_components) - set(parent_components)
        }
        removed_slots = {
            str(parent_components[component_id]["slot_id"])
            for component_id in set(parent_components) - set(candidate_components)
        }
        parameter_slots = {
            str(candidate_components[component_id]["slot_id"])
            for component_id in common_components
            if candidate_components[component_id].get("primitive_id")
            == parent_components[component_id].get("primitive_id")
            and candidate_components[component_id].get("parameters", {})
            != parent_components[component_id].get("parameters", {})
        }
        actual_changed_slots = (
            added_slots | removed_slots | set(replaced_slots) | parameter_slots
        )
        declared_roles = {
            str(item.get("slot_id")): str(item.get("change_role", "SUPPORT"))
            for item in candidate_payload.get("changed_slots", ())
            if isinstance(item, Mapping)
        }
        core_changed_slots = tuple(
            sorted(
                slot
                for slot in actual_changed_slots
                if declared_roles.get(slot) == "CORE"
            )
        )
        support_changed_slots = tuple(
            sorted(set(actual_changed_slots) - set(core_changed_slots))
        )
        architecture_operator_delta = len(
            candidate_payload.get("architecture_operators", ())
        ) - len(parent_payload.get("architecture_operators", ()))
        custom_component_delta = len(candidate_payload.get("custom_components", ())) - len(
            parent_payload.get("custom_components", ())
        )
        construction_mode = str(candidate_payload.get("construction_mode", "UNKNOWN"))
        parent_construction_mode = str(
            parent_payload.get("construction_mode", "UNKNOWN")
        )
        total_slot_count = len(
            {
                str(item["slot_id"])
                for item in (*candidate_components.values(), *parent_components.values())
            }
        )
        change_class = _classify_change(
            candidate_program_digest=candidate_program_digest,
            parent_program_digest=parent_program_digest,
            parameter_change_count=parameter_change_count,
            added_primitives=added_primitives,
            removed_primitives=removed_primitives,
            replaced_slots=replaced_slots,
            architecture_operator_delta=architecture_operator_delta,
            custom_component_delta=custom_component_delta,
            candidate_construction_mode=construction_mode,
            parent_construction_mode=parent_construction_mode,
            total_slot_count=total_slot_count,
        )
        structural_delta = (
            len(added_primitives) + len(removed_primitives) + len(replaced_slots)
        )
        intervention_magnitude = min(
            1.0,
            (
                float(parameter_change_count)
                + 2.0 * float(structural_delta)
                + 2.0 * abs(float(architecture_operator_delta))
                + 2.0 * abs(float(custom_component_delta))
            )
            / max(1.0, 3.0 * float(total_slot_count)),
        )
        cost = candidate_payload.get("estimated_cost", {})
        compute_class = str(cost.get("relative_training_compute", "MEDIUM")).upper()
        memory_class = str(cost.get("relative_memory", "MEDIUM")).upper()
        matched_control = change_class is ChangeClassV1.CONTROL
        ablation_or_falsification = (
            proposal.proposal_intent is ProposalIntentV1.FALSIFICATION
        )
        static = proposal.utility_features
        axis_coverage = research_context.coverage_for(proposal.mechanism_axis)
        class_features = {
            ChangeClassV1.CONTROL: (1.0, 0.0, 0.0, 0.0),
            ChangeClassV1.PARAMETER_TUNING_ONLY: (0.0, 1.0, 0.0, 0.0),
            ChangeClassV1.MECHANISM_CHANGE: (0.0, 0.0, 1.0, 0.0),
            ChangeClassV1.ARCHITECTURE_REWRITE: (0.0, 0.0, 0.0, 1.0),
        }[change_class]
        axis_key = (
            proposal.mechanism_axis
            if proposal.mechanism_axis in MECHANISM_AXIS_FEATURES_V1
            else "other"
        )
        axis_features = tuple(
            float(axis_key == axis) for axis in MECHANISM_AXIS_FEATURES_V1
        )
        primitive_class_delta = Counter(
            _primitive_class(item) for item in added_primitives
        )
        primitive_class_delta.subtract(
            _primitive_class(item) for item in removed_primitives
        )
        primitive_class_features = tuple(
            max(
                -1.0,
                min(
                    1.0,
                    float(primitive_class_delta.get(name, 0))
                    / max(1.0, float(total_slot_count)),
                ),
            )
            for name in PRIMITIVE_CLASS_FEATURES_V1
        )
        duplicate_pressure = _bounded_count(
            research_context.exact_duplicate_count
            + research_context.near_duplicate_count,
            4.0,
        )
        blocker_pressure = _bounded_count(
            research_context.blocker_count,
            4.0,
        )
        budget_pressure = 1.0 - min(
            float(research_context.remaining_execution_fraction),
            float(research_context.remaining_token_fraction),
        )
        gpu_pressure = 1.0 - float(
            research_context.remaining_gpu_fraction
        )
        lineage_depth = _bounded_count(
            lineage.get(proposal.candidate_id, research_context.lineage_depth),
            8.0,
        )
        rank_values = (
            float(static_score),
            float(static.runnable_probability),
            float(static.useful_signal),
            float(static.frontier_potential),
            float(static.information_gain),
            float(static.cost),
            float(static.blocker_risk),
            *class_features,
            *axis_features,
            _bounded_count(len(core_changed_slots), max(1.0, total_slot_count)),
            _bounded_count(len(support_changed_slots), max(1.0, total_slot_count)),
            _bounded_count(structural_delta, max(1.0, total_slot_count)),
            _bounded_count(parameter_change_count, max(1.0, total_slot_count)),
            max(-1.0, min(1.0, float(architecture_operator_delta))),
            max(-1.0, min(1.0, float(custom_component_delta))),
            intervention_magnitude,
            float(matched_control),
            float(ablation_or_falsification),
            _COST_ORDINAL.get(compute_class, 0.5),
            _COST_ORDINAL.get(memory_class, 0.5),
            *primitive_class_features,
            1.0 - float(axis_coverage),
            intervention_magnitude * float(research_context.stagnation_fraction),
            class_features[3] * float(research_context.stagnation_fraction),
            float(ablation_or_falsification)
            * float(research_context.stagnation_fraction),
            float(static.cost) * budget_pressure,
            _COST_ORDINAL.get(compute_class, 0.5) * gpu_pressure,
            intervention_magnitude * duplicate_pressure,
            float(static.blocker_risk) * blocker_pressure,
            lineage_depth,
            lineage_depth * float(research_context.round_fraction),
            *(
                value * float(research_context.task_scale)
                for value in axis_features
            ),
            *(
                value * float(research_context.task_density)
                for value in axis_features
            ),
        )
        candidates.append(
            CandidateMechanismDeltaV1(
                candidate_id=proposal.candidate_id,
                candidate_semantics_digest=decision.mechanism_semantics_digest,
                mechanism_program_digest=candidate_program_digest,
                parent_program_digest=parent_program_digest,
                parent_semantics_digest=parent_report.mechanism_semantics_digest,
                producer_id=proposal.producer_id,
                producer_role=proposal.producer_role,
                proposal_intent=proposal.proposal_intent.value,
                mechanism_family_id=str(candidate_program.get("family_id", "UNKNOWN")),
                primary_mechanism_axis=proposal.mechanism_axis,
                secondary_mechanism_axes=_secondary_axes(
                    actual_changed_slots, proposal.mechanism_axis
                ),
                change_class=change_class,
                construction_mode=construction_mode,
                core_changed_slots=core_changed_slots,
                support_changed_slots=support_changed_slots,
                added_primitives=added_primitives,
                removed_primitives=removed_primitives,
                replaced_slots=replaced_slots,
                parameter_change_count=parameter_change_count,
                architecture_operator_delta=architecture_operator_delta,
                custom_component_delta=custom_component_delta,
                intervention_magnitude=intervention_magnitude,
                lineage_depth=lineage.get(
                    proposal.candidate_id, research_context.lineage_depth
                ),
                matched_control=matched_control,
                ablation_or_falsification=ablation_or_falsification,
                estimated_compute_class=compute_class,
                estimated_memory_class=memory_class,
                static_utility_features=static,
                strong_static_score=float(static_score),
                route_eligibility="ELIGIBLE",
                hard_gate_reason=decision.reason.value,
                rank_features=tuple(zip(RANK_FEATURE_NAMES_V1, rank_values, strict=True)),
            )
        )

    if len(candidates) < 2:
        raise MetaVNextFeatureError(
            "Meta VNext requires at least two eligible candidates per pool"
        )
    return CandidatePoolVNextV1(
        pool_id=str(pool_id),
        source_candidate_pool_digest=sha256_digest(
            [item.to_dict() for item in source_proposals]
        ),
        hard_gate_assessment_digest=sha256_digest(assessment_projection),
        pre_round_state_digest=pre_round_state_digest,
        producer_invocation_digests=tuple(producer_invocation_digests),
        candidate_order_policy_digest=candidate_order_policy_digest,
        research_context=research_context,
        candidates=tuple(candidates),
        rejected_candidate_ids=tuple(rejected),
    )
