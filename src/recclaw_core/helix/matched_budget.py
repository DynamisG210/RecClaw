"""Pre-outcome matched-budget contract for Research versus Full Helix.

The Helix treatment is scientifically useful only if it improves how a fixed
set of experiment opportunities is allocated.  This module makes that claim
auditable: both arms reference one shared execution contract and Helix may
only redirect ordinary metric rounds to exact-candidate evidence tasks.  It
cannot add rounds, worker attempts, Provider calls, token budget, or heldout
access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
)


MATCHED_BUDGET_SCHEMA_V31 = "recclaw.helix.matched-budget-experiment.v31"
MATCHED_BUDGET_AUDIT_SCHEMA_V31 = "recclaw.helix.matched-budget-audit.v31"

STRICT_SEARCH_SPACE_ID = "BL_ICF_MECHANISM_SPACE_V1"
STRICT_SEARCH_SPACE_DIGEST = (
    "fbe63260de6430537dd66b0724fe1beb0cf165ea47407ae18addd61a17dd1720"
)
ORDERED_PRIMITIVE_IDS_COUNT = 264
ORDERED_PRIMITIVE_IDS_DIGEST_ALGORITHM = (
    "SHA256_UTF8_LF_TERMINATED_ID_LIST_V1"
)
ORDERED_PRIMITIVE_IDS_DIGEST = (
    "46d484b8030cc0bf242dc21f46b431b9ae0943637a5096ee5ea689954817533b"
)

ONLINE_METRIC_SOURCE = "BEST_VALID_RESULT"
ONLINE_PARTITION_ROLE = "DEVELOPMENT_VALIDATION"
DEVELOPMENT_PROTOCOL_REF_V31 = "recclaw.campaign.ml1m-full-sort.v1"
DEVELOPMENT_PROTOCOL_DIGEST_V31 = (
    "7f623fd953001f999e8b5d2657749f6a3ca86c7be5410a48bb8281241a258bbe"
)
HELDOUT_ACCESS_POLICY = "POST_SELECTION_ONLY"
PRODUCER_ROLES_V31 = (
    "mechanism_composer",
    "lineage_refiner",
    "falsification_designer",
    "frontier_architect",
)
PROPOSAL_LANES_V31 = ("BL_ICF", "BL_ICF_PROGRAM", "OPEN_SPEC")
PRODUCER_LANE_BY_ROLE_V31 = {
    role: "BL_ICF_PROGRAM" for role in PRODUCER_ROLES_V31
}
REGULAR_PROPOSAL_TOTAL_TOKEN_CEILING_V31 = 9_000
STRICT_PROGRAM_TOTAL_TOKEN_CEILING_V31 = 64_000
PROPOSAL_OUTPUT_TOKEN_CEILING_V31 = 6_000
IMPLEMENTER_TOTAL_TOKEN_CEILING_V31 = 20_000
IMPLEMENTER_OUTPUT_TOKEN_CEILING_V31 = 20_000
PROVIDER_TOKEN_ACCOUNTING_V31 = {
    "successful_attempt_debit": "ACTUAL_BILLED_TOKENS",
    "missing_usage_debit": "TRACE_SPECIFIC_TRANSPORT_TOTAL_CEILING",
    "comparison_report": "ACTUAL_AND_CONSERVATIVE_DEBIT_WITH_CALL_COUNTS",
}
PAIRED_ROUND_SCHEDULE_V31 = (
    "ODD_RESEARCH_THEN_HELIX_EVEN_HELIX_THEN_RESEARCH"
)
PROFILE_SOURCE_SCHEMA_V31 = "recclaw.research-line.profile-source.v1"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: Any, *, name: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be an object")
        return {}
    return value


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True, slots=True)
class MatchedBudgetAuditV31:
    ready: bool
    errors: tuple[str, ...]
    manifest_digest: str
    shared_contract_digest: str | None
    research_arm_id: str | None
    helix_arm_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": MATCHED_BUDGET_AUDIT_SCHEMA_V31,
                "ready": self.ready,
                "errors": self.errors,
                "manifest_digest": self.manifest_digest,
                "shared_contract_digest": self.shared_contract_digest,
                "research_arm_id": self.research_arm_id,
                "helix_arm_id": self.helix_arm_id,
                "cost_interpretation": (
                    "HELIX_ACTIONS_REPLACE_ORDINARY_METRIC_ROUNDS"
                ),
                "online_learning_authority": "RESEARCH_LINE_ONLY",
                "heldout_learning_authority": "NONE",
            }
        )


def audit_matched_budget_contract_v31(
    manifest: Mapping[str, Any],
) -> MatchedBudgetAuditV31:
    """Audit a two-arm V31 contract without reading outcomes.

    The function deliberately returns a complete negative receipt instead of
    failing on the first defect.  A launcher or preflight may require
    ``ready`` and persist the receipt before either arm starts.
    """

    errors: list[str] = []
    value = canonical_value(dict(manifest))
    if value.get("schema") != MATCHED_BUDGET_SCHEMA_V31:
        errors.append("schema must be the V31 matched-budget schema")
    if value.get("status") != "PREFLIGHT_READY":
        errors.append("manifest status must be PREFLIGHT_READY")

    shared = _mapping(
        value.get("shared_contract"), name="shared_contract", errors=errors
    )
    computed_shared_digest = sha256_digest(shared) if shared else None
    declared_shared_digest = value.get("shared_contract_digest")
    if not _is_sha256(declared_shared_digest):
        errors.append("shared_contract_digest must be a lowercase SHA256")
    elif computed_shared_digest != declared_shared_digest:
        errors.append("shared_contract_digest does not bind shared_contract")

    search = _mapping(shared.get("search_space"), name="search_space", errors=errors)
    exact_search_fields = {
        "search_space_id": STRICT_SEARCH_SPACE_ID,
        "search_space_digest": STRICT_SEARCH_SPACE_DIGEST,
        "ordered_primitive_ids_count": ORDERED_PRIMITIVE_IDS_COUNT,
        "ordered_primitive_ids_digest_algorithm": (
            ORDERED_PRIMITIVE_IDS_DIGEST_ALGORITHM
        ),
        "ordered_primitive_ids_digest": ORDERED_PRIMITIVE_IDS_DIGEST,
        "fixed_fallback": False,
    }
    for name, expected in exact_search_fields.items():
        if search.get(name) != expected:
            errors.append(f"search_space.{name} must equal {expected!r}")
    profile_ref = _mapping(
        search.get("profile_ref"), name="search_space.profile_ref", errors=errors
    )
    if profile_ref.get("status") != "FROZEN_BEFORE_ARM_LAUNCH":
        errors.append("search-space profile must be frozen before arm launch")
    for name in ("profile_id", "profile_kind", "profile_digest"):
        if not isinstance(profile_ref.get(name), str) or not profile_ref.get(name):
            errors.append(f"search_space.profile_ref.{name} is required")
    if profile_ref and not _is_sha256(profile_ref.get("profile_digest")):
        errors.append("search_space.profile_ref.profile_digest must be a SHA256")

    dataset = _mapping(
        shared.get("dataset_contract"), name="dataset_contract", errors=errors
    )
    if not isinstance(dataset.get("dataset_id"), str) or not dataset.get("dataset_id"):
        errors.append("dataset_contract.dataset_id is required")
    if not isinstance(dataset.get("split"), str) or not dataset.get("split"):
        errors.append("dataset_contract.split is required")
    for name in (
        "dataset_snapshot_digest",
        "search_partition_manifest_digest",
        "train_file_digest",
        "development_validation_file_digest",
    ):
        if not _is_sha256(dataset.get(name)):
            errors.append(f"dataset_contract.{name} is required")

    metric = _mapping(
        shared.get("online_metric_contract"),
        name="online_metric_contract",
        errors=errors,
    )
    if metric.get("metric") != "NDCG@10":
        errors.append("online metric must be NDCG@10")
    if metric.get("source") != ONLINE_METRIC_SOURCE:
        errors.append("online metric source must be development validation")
    if metric.get("partition_role") != ONLINE_PARTITION_ROLE:
        errors.append("online partition role must be DEVELOPMENT_VALIDATION")
    if metric.get("adaptive_reuse") != "ALLOWED_WITHIN_CAMPAIGN":
        errors.append("development validation reuse policy is not explicit")
    if metric.get("protocol_ref") != DEVELOPMENT_PROTOCOL_REF_V31:
        errors.append("online protocol ref must equal the frozen V31 protocol")
    if metric.get("protocol_digest") != DEVELOPMENT_PROTOCOL_DIGEST_V31:
        errors.append("online protocol digest must equal the frozen V31 protocol")

    runtime_release = _mapping(
        shared.get("runtime_release"), name="runtime_release", errors=errors
    )
    if runtime_release.get("status") != "FROZEN_BEFORE_ARM_LAUNCH":
        errors.append("common runtime release must be frozen before arm launch")
    for name in (
        "release_digest",
        "worker_source_digest",
        "metric_parser_source_digest",
        "environment_digest",
        "target_runtime_receipt_sha256",
        "target_runtime_receipt_digest",
    ):
        if not _is_sha256(runtime_release.get(name)):
            errors.append(f"runtime_release.{name} is required")
    if runtime_release.get("online_metric_source") != ONLINE_METRIC_SOURCE:
        errors.append("runtime release does not emit development validation")
    if runtime_release.get("online_partition_role") != ONLINE_PARTITION_ROLE:
        errors.append("runtime release exposes the wrong online partition")
    if runtime_release.get("target_host") != "gpu41":
        errors.append("runtime release must bind target_host gpu41")
    if runtime_release.get("native_gpu_id") != 2:
        errors.append("runtime release must bind native gpu_id 2")
    if runtime_release.get("cuda_visible_devices") != "MUST_BE_UNSET":
        errors.append("runtime release must forbid CUDA_VISIBLE_DEVICES remapping")
    if runtime_release.get("dataloader_workers") != 0:
        errors.append("runtime release must bind gpu41 dataloader_workers=0")
    if not isinstance(runtime_release.get("target_runtime_receipt_path"), str):
        errors.append("runtime release target receipt path is required")

    source_release = _mapping(
        shared.get("campaign_source_release"),
        name="campaign_source_release",
        errors=errors,
    )
    if source_release.get("status") != "FROZEN_BEFORE_ARM_LAUNCH":
        errors.append("common campaign source archive must be frozen before arm launch")
    if source_release.get("release_id") != "HELIX_V31_MATCHED_SOURCE_V1":
        errors.append("campaign source release_id must be HELIX_V31_MATCHED_SOURCE_V1")
    if source_release.get("archive_format") != "TAR_GZIP":
        errors.append("campaign source archive format must be TAR_GZIP")
    if not isinstance(source_release.get("archive_ref"), str) or not source_release.get(
        "archive_ref"
    ):
        errors.append("campaign source archive_ref is required")
    if not _is_sha256(source_release.get("archive_sha256")):
        errors.append("campaign source archive_sha256 is required")

    heldout = _mapping(
        shared.get("outer_heldout_contract"),
        name="outer_heldout_contract",
        errors=errors,
    )
    if heldout.get("access_policy") != HELDOUT_ACCESS_POLICY:
        errors.append("outer heldout must be post-selection only")
    if heldout.get("online_feedback_allowed") is not False:
        errors.append("outer heldout feedback must be disabled online")
    if heldout.get("status") != "PRECOMMITTED_UNREAD":
        errors.append("outer heldout must be precommitted and unread")
    if not isinstance(heldout.get("manifest_ref"), str) or not heldout.get(
        "manifest_ref"
    ):
        errors.append("outer heldout manifest_ref is required")
    if not _is_sha256(heldout.get("manifest_sha256")):
        errors.append("outer heldout manifest_sha256 is required")
    if not _is_sha256(heldout.get("manifest_digest")):
        errors.append("outer heldout manifest_digest is required")

    execution = _mapping(
        shared.get("execution_budget"), name="execution_budget", errors=errors
    )
    round_count = execution.get("metric_round_count")
    if not _positive_int(round_count):
        errors.append("execution_budget.metric_round_count must be positive")
        round_count = 0
    if not _positive_int(execution.get("max_attempts_per_round")):
        errors.append("execution_budget.max_attempts_per_round must be positive")
    if not _positive_int(execution.get("worker_ceiling_seconds")):
        errors.append("execution_budget.worker_ceiling_seconds must be positive")
    if not _positive_int(execution.get("epochs_requested")):
        errors.append("execution_budget.epochs_requested must be positive")
    schedule = execution.get("ordered_seed_schedule")
    if (
        not isinstance(schedule, Sequence)
        or isinstance(schedule, (str, bytes))
        or len(schedule) != round_count
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in schedule)
        or len(set(schedule)) != len(schedule)
    ):
        errors.append("ordered seed schedule must contain one unique integer per round")

    formal = _mapping(
        shared.get("formal_execution"),
        name="formal_execution",
        errors=errors,
    )
    if formal.get("strategy") != PAIRED_ROUND_SCHEDULE_V31:
        errors.append("formal execution must use balanced paired-round scheduling")
    if formal.get("max_rounds_this_invocation") != 1:
        errors.append("formal execution must advance at most one round per invocation")
    if formal.get("max_progress_skew") != 1:
        errors.append("formal execution progress skew must be bounded to one round")
    if formal.get("catch_up_lagging_arm_before_new_pair") is not True:
        errors.append("formal execution must catch up the lagging arm first")
    if formal.get("stop_on_unpaired_progress") is not True:
        errors.append("formal execution must stop on unpaired progress")
    if formal.get("concurrent_gpu_workers") != 1:
        errors.append("formal execution must use one GPU worker at a time")
    formal_root = formal.get("formal_root")
    if not isinstance(formal_root, str) or not formal_root.strip():
        errors.append("formal execution root is required")
    seed_source = _mapping(
        formal.get("observation_seed_source"),
        name="formal_execution.observation_seed_source",
        errors=errors,
    )
    if not isinstance(seed_source.get("ref"), str) or not seed_source.get("ref"):
        errors.append("formal observation seed source ref is required")
    if not _is_sha256(seed_source.get("sha256")):
        errors.append("formal observation seed source SHA256 is required")
    if seed_source.get("count") != round_count:
        errors.append("formal observation seed source count must equal metric rounds")
    profile_source = _mapping(
        formal.get("research_profile_source"),
        name="formal_execution.research_profile_source",
        errors=errors,
    )
    if profile_source.get("schema") != PROFILE_SOURCE_SCHEMA_V31:
        errors.append("formal Research profile source schema is invalid")
    for name in ("ref", "source_ref"):
        if not isinstance(profile_source.get(name), str) or not profile_source.get(name):
            errors.append(f"formal Research profile source {name} is required")
    for name in ("sha256", "source_digest"):
        if not _is_sha256(profile_source.get(name)):
            errors.append(f"formal Research profile source {name} is required")

    provider = _mapping(
        shared.get("provider_budget"), name="provider_budget", errors=errors
    )
    if not _is_sha256(provider.get("config_digest")):
        errors.append("provider_budget.config_digest is required")
    roles = provider.get("producer_roles")
    if (
        not isinstance(roles, Sequence)
        or isinstance(roles, (str, bytes))
        or tuple(roles) != PRODUCER_ROLES_V31
    ):
        errors.append("provider budget must bind the ordered four Producer roles")
    if "producer_token_ceiling_each" in provider:
        errors.append(
            "ambiguous producer_token_ceiling_each is forbidden; bind lane ceilings"
        )
    lanes_by_role = _mapping(
        provider.get("producer_lane_by_role"),
        name="provider_budget.producer_lane_by_role",
        errors=errors,
    )
    if dict(lanes_by_role) != PRODUCER_LANE_BY_ROLE_V31:
        errors.append("all four Producer roles must use BL_ICF_PROGRAM")
    lane_ceilings = _mapping(
        provider.get("proposal_lane_transport_ceilings"),
        name="provider_budget.proposal_lane_transport_ceilings",
        errors=errors,
    )
    expected_lane_ceilings = {
        "BL_ICF": {
            "total_tokens": REGULAR_PROPOSAL_TOTAL_TOKEN_CEILING_V31,
            "output_tokens": PROPOSAL_OUTPUT_TOKEN_CEILING_V31,
        },
        "BL_ICF_PROGRAM": {
            "total_tokens": STRICT_PROGRAM_TOTAL_TOKEN_CEILING_V31,
            "output_tokens": PROPOSAL_OUTPUT_TOKEN_CEILING_V31,
        },
        "OPEN_SPEC": {
            "total_tokens": REGULAR_PROPOSAL_TOTAL_TOKEN_CEILING_V31,
            "output_tokens": PROPOSAL_OUTPUT_TOKEN_CEILING_V31,
        },
    }
    if dict(lane_ceilings) != expected_lane_ceilings:
        errors.append("proposal lane transport ceilings do not match V31 runtime")
    implementer_ceiling = _mapping(
        provider.get("implementer_transport_ceiling"),
        name="provider_budget.implementer_transport_ceiling",
        errors=errors,
    )
    if dict(implementer_ceiling) != {
        "total_tokens": IMPLEMENTER_TOTAL_TOKEN_CEILING_V31,
        "output_tokens": IMPLEMENTER_OUTPUT_TOKEN_CEILING_V31,
    }:
        errors.append("implementer transport ceiling does not match V31 runtime")
    token_accounting = _mapping(
        provider.get("token_accounting"),
        name="provider_budget.token_accounting",
        errors=errors,
    )
    if dict(token_accounting) != PROVIDER_TOKEN_ACCOUNTING_V31:
        errors.append("Provider token accounting does not bind V31 debit semantics")
    if not _positive_int(provider.get("provider_timeout_seconds")):
        errors.append("Provider timeout must be positive")

    baseline = _mapping(
        shared.get("development_baseline"),
        name="development_baseline",
        errors=errors,
    )
    if baseline.get("status") != "FROZEN_BEFORE_ARM_LAUNCH":
        errors.append("development baseline must be frozen before arm launch")
    for name in ("plan_ref", "receipt_ref"):
        if not isinstance(baseline.get(name), str) or not baseline.get(name):
            errors.append(f"development baseline {name} is required")
    for name in ("plan_digest", "receipt_sha256", "receipt_digest"):
        if not _is_sha256(baseline.get(name)):
            errors.append(f"development baseline {name} is required")
    score = baseline.get("ndcg_at_10")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not 0.0 <= float(score) <= 1.0
    ):
        errors.append("development baseline NDCG@10 must be in [0, 1]")

    arms = value.get("arms")
    research: Mapping[str, Any] | None = None
    helix: Mapping[str, Any] | None = None
    if not isinstance(arms, Sequence) or isinstance(arms, (str, bytes)):
        errors.append("arms must contain exactly Research and Full Helix")
        arms = ()
    for arm in arms:
        if not isinstance(arm, Mapping):
            errors.append("each arm must be an object")
            continue
        if arm.get("arm_kind") == "RESEARCH_LINE":
            research = arm
        elif arm.get("arm_kind") == "FULL_HELIX":
            helix = arm
        else:
            errors.append("unknown arm_kind")
        if arm.get("shared_contract_digest") != computed_shared_digest:
            errors.append("each arm must bind the exact shared contract")
    if len(arms) != 2 or research is None or helix is None:
        errors.append("arms must contain one Research Line and one Full Helix")

    if research is not None:
        treatment = _mapping(
            research.get("treatment"), name="research treatment", errors=errors
        )
        if treatment.get("kind") != "NONE":
            errors.append("Research arm treatment must be NONE")
        if treatment.get("max_allocation_actions") != 0:
            errors.append("Research arm cannot allocate Helix evidence actions")
    if helix is not None:
        treatment = _mapping(
            helix.get("treatment"), name="Helix treatment", errors=errors
        )
        if treatment.get("kind") != "FRONTIER_EVIDENCE_ALLOCATOR_V31":
            errors.append("Helix treatment must be the V31 evidence allocator")
        max_actions = treatment.get("max_allocation_actions")
        if (
            not isinstance(max_actions, int)
            or isinstance(max_actions, bool)
            or max_actions < 0
            or max_actions > round_count
        ):
            errors.append("Helix allocation cap must fit inside metric rounds")
        if treatment.get("budget_semantics") != (
            "SUBSET_OF_FROZEN_METRIC_OPPORTUNITIES"
        ):
            errors.append("Helix evidence work must replace ordinary rounds")
        if treatment.get("guard_provider_calls") != 0:
            errors.append("deterministic Guard must not add Provider calls")
        if treatment.get("research_update_authority") != "RESEARCH_LINE_ONLY":
            errors.append("Helix treatment cannot shadow Research learning")

    return MatchedBudgetAuditV31(
        ready=not errors,
        errors=tuple(sorted(set(errors))),
        manifest_digest=sha256_digest(value),
        shared_contract_digest=computed_shared_digest,
        research_arm_id=(str(research.get("arm_id")) if research else None),
        helix_arm_id=(str(helix.get("arm_id")) if helix else None),
    )


def require_matched_budget_contract_v31(
    manifest: Mapping[str, Any],
) -> MatchedBudgetAuditV31:
    audit = audit_matched_budget_contract_v31(manifest)
    if not audit.ready:
        raise ValueError("; ".join(audit.errors))
    return audit


__all__ = [
    "DEVELOPMENT_PROTOCOL_DIGEST_V31",
    "DEVELOPMENT_PROTOCOL_REF_V31",
    "HELDOUT_ACCESS_POLICY",
    "MATCHED_BUDGET_AUDIT_SCHEMA_V31",
    "MATCHED_BUDGET_SCHEMA_V31",
    "MatchedBudgetAuditV31",
    "ONLINE_METRIC_SOURCE",
    "ONLINE_PARTITION_ROLE",
    "PAIRED_ROUND_SCHEDULE_V31",
    "PROFILE_SOURCE_SCHEMA_V31",
    "ORDERED_PRIMITIVE_IDS_COUNT",
    "ORDERED_PRIMITIVE_IDS_DIGEST",
    "ORDERED_PRIMITIVE_IDS_DIGEST_ALGORITHM",
    "IMPLEMENTER_OUTPUT_TOKEN_CEILING_V31",
    "IMPLEMENTER_TOTAL_TOKEN_CEILING_V31",
    "PRODUCER_LANE_BY_ROLE_V31",
    "PRODUCER_ROLES_V31",
    "PROPOSAL_LANES_V31",
    "PROPOSAL_OUTPUT_TOKEN_CEILING_V31",
    "PROVIDER_TOKEN_ACCOUNTING_V31",
    "REGULAR_PROPOSAL_TOTAL_TOKEN_CEILING_V31",
    "STRICT_PROGRAM_TOTAL_TOKEN_CEILING_V31",
    "STRICT_SEARCH_SPACE_DIGEST",
    "STRICT_SEARCH_SPACE_ID",
    "audit_matched_budget_contract_v31",
    "require_matched_budget_contract_v31",
]
