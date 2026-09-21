"""Durable Evidence Guard bridge for the standalone Research campaign."""

from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    COMMON_DATASET,
    COMMON_EVALUATOR,
    COMMON_SPLIT,
    DEVELOPMENT_EVALUATOR,
    DEVELOPMENT_SPLIT,
    P4_SPARSE_SPECTRAL_EVALUATOR,
    P4_SPARSE_SPECTRAL_SPLIT,
    ExperimentBindingV1,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    SearchCandidateBindingV1,
)
from recclaw_core.research_line.portfolio import PortfolioCandidateV2

from .contracts import (
    CandidateEnvelope,
    GuardContext,
    PortStatus,
    RawResultEnvelope,
)
from .guard_adapter import EvidenceGuardPortV1, eligible_raw_result
from .frontier_allocation import (
    AllocationActionV31,
    AllocationClosureV31,
    FrontierEvidenceAllocatorV31,
    HelixAllocationPolicyV31,
)
from .ledger import EvidenceGuardLedgerWriterV1
from .scientific_attribution import (
    ValueOfInformationHelixAdmissionV30,
    EvidenceSummaryV1,
    ResearchTaskV1,
    SearchUtilityEventV2,
)


BRIDGE_SCHEMA = "recclaw.helix.standalone-evidence-bridge.v31"
ACTION_CONTRACT_SCHEMA = "recclaw.helix.action-contract.v1"
PROTOCOL_ATTESTATION_SCHEMA = "recclaw.helix.protocol-attestation.v1"
CLAIM_CONTROL_SCHEMA = "recclaw.helix.claim-control.v1"


def _mechanism_isolation_question(binding: SearchCandidateBindingV1) -> Mapping[str, Any] | None:
    """Retain a declared ambiguity as research context, never an action trigger."""

    proposal = binding.proposal
    spec = getattr(proposal, "spec", None)
    program = proposal.mechanism_program
    payload = program.get("program_payload", {}) if isinstance(program, Mapping) else {}
    axes = sorted({str(row["slot_id"]) for row in payload.get("changed_slots", ()) if isinstance(row, Mapping) and row.get("slot_id")})
    competing = getattr(spec, "competing_explanation", None)
    intervention = getattr(spec, "mechanism_off_definition", None)
    if len(axes) < 2 or not competing or not intervention:
        return None
    return canonical_value({
        "candidate_semantic_digest": binding.mechanism_semantics_digest,
        "changed_slots": axes,
        "question": competing,
        "intervention": intervention,
        "prediction": getattr(spec, "falsifier", None),
        "evidence_limit": "MULTI_AXIS_OBSERVATION_DOES_NOT_ISOLATE_COMPONENT_EFFECT",
    })


def _mechanism_research_feedback(
    binding: SearchCandidateBindingV1,
) -> Mapping[str, Any] | None:
    """Ground the next research decision in this proposal's actual hypothesis."""

    spec = getattr(binding.proposal, "spec", None)
    if spec is None:
        return None
    declared = {}
    for field in ("mechanism_change", "competing_explanation", "falsifier"):
        value = getattr(spec, field, None)
        if isinstance(value, str) and value.strip():
            declared[field] = value.strip()[:900]
    if not declared:
        return None
    return canonical_value({
        "candidate_id": binding.proposal.candidate_id,
        "declared_hypothesis": declared,
        "interpretation_scope": (
            "Measured candidate performance informs search immediately; declared "
            "explanations are hypotheses, not isolated component effects. Compare "
            "related variants to decide what to retain or change next."
        ),
    })


T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}

# These checks are part of the frozen scientific protocol.  Failures in the
# remaining attestation checks are provenance/binding defects, not evidence
# that the protocol itself changed.
PROTOCOL_ATTESTATION_CHECKS = frozenset(
    {
        "dataset",
        "dataset_manifest",
        "split",
        "evaluator",
        "seed",
        "epochs",
        "comparator",
        "execution_role",
    }
)


def next_eligible_seed(
    schedule: tuple[str, ...],
    *,
    current_seed: str,
    verified_seed_ids: set[str] | frozenset[str],
) -> str | None:
    """Return only a strictly later, frozen-schedule seed for this attempt."""

    try:
        current_index = schedule.index(str(current_seed))
    except ValueError as error:
        raise ValueError(
            "current observation seed is outside the frozen validation schedule"
        ) from error
    return next(
        (
            seed
            for seed in schedule[current_index + 1 :]
            if seed not in verified_seed_ids
        ),
        None,
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _current_attempt_class(
    *,
    attestation: Mapping[str, Any],
    raw_result: Mapping[str, Any],
    identity_matches: bool,
    raw_is_eligible: bool,
) -> str:
    """Classify the current attempt without laundering failure into science."""

    failed_checks = {
        str(item) for item in attestation.get("failed_checks", ())
    }
    if failed_checks.intersection(PROTOCOL_ATTESTATION_CHECKS):
        return "PROTOCOL_DRIFT"
    if not identity_matches:
        return "INVALID"
    if raw_result.get("run_status") != "SUCCESS":
        return "ENGINEERING_FAILURE"
    if attestation.get("status") != "EXACT" or not raw_is_eligible:
        return "INVALID"
    return "VALID_METRIC"


def _observed_protocol(
    registered: Mapping[str, Any],
    *,
    dataset: Any,
    split: Any,
    evaluator: Any,
    expected_split: str,
    expected_evaluator: Mapping[str, Any],
    dataset_manifest_digest: Any,
    dataset_anchor_matches: bool,
) -> dict[str, Any]:
    """Project runtime facts into the Guard's closed protocol vocabulary."""

    observed = dict(registered)
    if dataset != COMMON_DATASET:
        observed["dataset"] = str(dataset)
    if split != expected_split:
        observed["split"] = {"runtime_binding": str(split)}
    if not isinstance(evaluator, Mapping) or canonical_value(dict(evaluator)) != dict(
        expected_evaluator
    ):
        observed["evaluation_candidate_universe"] = {
            "runtime_binding": canonical_value(evaluator)
            if isinstance(evaluator, Mapping)
            else str(evaluator)
        }
    if not dataset_anchor_matches:
        observed["dataset_snapshot"] = str(dataset_manifest_digest)
    return canonical_value(observed)


def standalone_guard_protocol(*, epochs: int, protocol_digest: str) -> dict[str, Any]:
    """Return the exact development protocol presented to Evidence Guard."""

    return canonical_value(
        {
            "protocol_id": "PROTO-ML1M-RESEARCH-STANDALONE-V1",
            "profile_family": "OFFLINE_TOPN",
            "dataset": "ml-1m",
            "dataset_snapshot": protocol_digest,
            "split": {
                "strategy": "sha256_seeded_within_user",
                "ratio": [0.8, 0.1, 0.1],
                "online_partition": "DEVELOPMENT_VALIDATION",
                "heldout_access": "POST_SELECTION_ONLY",
            },
            "training_sampling": {"mode": "mechanism_program_defined"},
            "evaluation_candidate_universe": {"mode": "full_sort"},
            "candidate_policy": {"seen_items": "exclude"},
            "metric": {
                "name": "ndcg",
                "cutoff": 10,
                "source": "BEST_VALID_RESULT",
            },
            "training_procedure": {
                "optimizer": "adam",
                "max_epochs": epochs,
                "early_stopping_patience": 10,
            },
        }
    )


def _write_once_or_verify(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != payload:
            raise ValueError("durable common closure identity drift")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError("durable common closure identity drift")
    finally:
        temporary.unlink(missing_ok=True)


class StandaloneEvidenceGuardBridgeV1:
    """Adapt Research recipes/results to PRE/POST Evidence Guard contracts.

    PRE rejections are returned to the Research attempt scheduler before the
    physical runner is called. POST first seals the common scientific closure,
    then commits the Guard event and exposes only compact Fusion feedback.
    """

    def __init__(
        self,
        *,
        run_root: Path,
        campaign_id: str,
        protocol: Mapping[str, Any],
        comparator: str,
        comparator_ndcg_at_10: float,
        expected_dataset_manifest_digest: str,
        expected_split: str = COMMON_SPLIT,
        expected_evaluator: Mapping[str, Any] = COMMON_EVALUATOR,
        required_seed_count: int = 3,
        validation_seed_schedule: tuple[str, ...] = (),
        minimum_effect_delta: float = 0.0,
        replication_trigger_delta: float = 0.0,
        max_allocation_actions: int | None = None,
        max_open_allocation_actions: int = 1,
    ) -> None:
        self.run_root = Path(run_root).resolve()
        self.campaign_id = str(campaign_id)
        self.protocol = canonical_value(dict(protocol))
        self.comparator = str(comparator)
        self.comparator_ndcg_at_10 = float(comparator_ndcg_at_10)
        if not math.isfinite(self.comparator_ndcg_at_10):
            raise ValueError("comparator_ndcg_at_10 must be finite")
        if not _is_sha256(expected_dataset_manifest_digest):
            raise ValueError(
                "expected_dataset_manifest_digest must be a lowercase SHA256"
            )
        self.expected_dataset_manifest_digest = expected_dataset_manifest_digest
        supported_bindings = {
            (P4_SPARSE_SPECTRAL_SPLIT, sha256_digest(P4_SPARSE_SPECTRAL_EVALUATOR)),
            (
                COMMON_SPLIT,
                sha256_digest(COMMON_EVALUATOR),
            ),
            (
                DEVELOPMENT_SPLIT,
                sha256_digest(DEVELOPMENT_EVALUATOR),
            ),
        }
        normalized_evaluator = canonical_value(dict(expected_evaluator))
        if (str(expected_split), sha256_digest(normalized_evaluator)) not in supported_bindings:
            raise ValueError("unsupported Evidence Guard evaluator/split binding")
        self.expected_split = str(expected_split)
        self.expected_evaluator = normalized_evaluator
        self.minimum_effect_delta = float(minimum_effect_delta)
        if not math.isfinite(self.minimum_effect_delta):
            raise ValueError("minimum_effect_delta must be finite")
        self.replication_trigger_delta = float(replication_trigger_delta)
        if (
            not math.isfinite(self.replication_trigger_delta)
            or self.replication_trigger_delta < 0.0
        ):
            raise ValueError(
                "replication_trigger_delta must be finite and non-negative"
            )
        self.required_seed_count = int(required_seed_count)
        if self.required_seed_count < 1:
            raise ValueError("required_seed_count must be positive")
        self.validation_seed_schedule = tuple(
            str(seed) for seed in validation_seed_schedule
        )
        if len(self.validation_seed_schedule) < self.required_seed_count:
            raise ValueError(
                "validation_seed_schedule must cover required_seed_count"
            )
        self.opaque_arm_instance_id = sha256_digest(
            {"campaign_id": self.campaign_id, "arm": "FULL_HELIX_C"}
        )
        claim_id = f"{self.campaign_id}:local-improvement"
        self.context = GuardContext(
            claim={
                "claim_id": claim_id,
                "protocol_id": self.protocol["protocol_id"],
                "claim_kind": "LOCAL_IMPROVEMENT",
                "target_model": "CANDIDATE_SPECIFIC",
                "comparator": self.comparator,
                "metric": "ndcg",
                "required_seed_count": self.required_seed_count,
                "scope": {
                    "dataset": "ml-1m",
                    "frozen_comparator_ndcg_at_10": (
                        self.comparator_ndcg_at_10
                    ),
                    "minimum_effect_delta": self.minimum_effect_delta,
                },
            },
            protocol=self.protocol,
            current_evidence={
                "snapshot_id": f"{self.campaign_id}:empty",
                "claim_id": claim_id,
                "protocol_id": self.protocol["protocol_id"],
                "observation_ids": [],
            },
        )
        self.ledger = EvidenceGuardLedgerWriterV1(
            self.run_root / "evidence_guard" / "private_ledger"
        )
        if self.ledger.db_path.is_file():
            self._close_out_of_schedule_reserved_replications()
        allocation_limit = (
            len(self.validation_seed_schedule)
            if max_allocation_actions is None
            else max_allocation_actions
        )
        self.allocation_policy = HelixAllocationPolicyV31(
            max_actions=allocation_limit,
            max_open_actions=max_open_allocation_actions,
            replication_trigger_delta=self.replication_trigger_delta,
        )
        self.allocator = FrontierEvidenceAllocatorV31(
            policy=self.allocation_policy,
            ledger=self.ledger,
        )
        self.port = EvidenceGuardPortV1(
            context=self.context,
            ledger=self.ledger,
            opaque_arm_instance_id=self.opaque_arm_instance_id,
        )
        self.admission = ValueOfInformationHelixAdmissionV30()

    def _close_out_of_schedule_reserved_replications(self) -> None:
        validation_seeds = set(self.validation_seed_schedule)
        for allocation in self.ledger.allocation_actions():
            if (
                allocation.get("status") != "RESERVED"
                or allocation.get("action") != AllocationActionV31.REPLICATE.value
                or allocation.get("target") in validation_seeds
            ):
                continue
            decision = allocation.get("decision")
            if not isinstance(decision, Mapping):
                continue
            target = str(allocation["target"])
            closure = canonical_value(
                {
                    "action_id": allocation["action_id"],
                    "task_id": None,
                    "task_status": "CLOSED",
                    "candidate_id": decision.get("candidate_id"),
                    "candidate_semantic_digest": decision.get(
                        "candidate_semantic_digest"
                    ),
                    "mechanism_program_digest": decision.get(
                        "mechanism_program_digest"
                    ),
                    "allocation_target": target,
                    "required_seed_or_control": target,
                    "current_attempt_class": "NON_METRIC_RESUME_MIGRATION",
                    "evidence_count": None,
                    "scientific_conclusion_strength": "NOT_EVALUATED",
                    "closure_reason": "VALIDATION_SCHEDULE_MIGRATED_OUT",
                    "metric_bearing_evidence": False,
                }
            )
            self.ledger.close_allocation_action(
                action_id=str(allocation["action_id"]),
                closure_status=AllocationClosureV31.CANCELLED.value,
                closure=closure,
            )

    @property
    def identity_digest(self) -> str:
        return sha256_digest(
            {
                "schema": BRIDGE_SCHEMA,
                "campaign_id": self.campaign_id,
                "opaque_arm_instance_id": self.opaque_arm_instance_id,
                "protocol": self.protocol,
                "comparator": self.comparator,
                "comparator_ndcg_at_10": self.comparator_ndcg_at_10,
                "expected_dataset_manifest_digest": (
                    self.expected_dataset_manifest_digest
                ),
                "expected_split": self.expected_split,
                "expected_evaluator": self.expected_evaluator,
                "required_seed_count": self.required_seed_count,
                "validation_seed_schedule": self.validation_seed_schedule,
                "minimum_effect_delta": self.minimum_effect_delta,
                "replication_trigger_delta": self.replication_trigger_delta,
                "allocation_policy": self.allocation_policy.to_dict(),
                "guard_port_identity": self.port.identity_digest,
                "admission_policy_digest": self.admission.policy_digest,
            }
        )

    def adjust_portfolio_information(
        self,
        candidates: Sequence[PortfolioCandidateV2],
    ) -> tuple[PortfolioCandidateV2, ...]:
        """Add bounded, non-blocking Helix VOI before Research routing.

        Research remains the sole router.  Helix only contributes a candidate-
        level information estimate from closed evidence, executability, and
        cost; it never removes a candidate or changes the discovery budget.
        """

        pool = tuple(candidates)
        if not pool:
            return pool
        budget = self.ledger.allocation_budget_snapshot(
            max_actions=self.allocation_policy.max_actions
        )
        if int(budget["remaining_actions"]) <= 0:
            return pool
        evidence_counts: dict[str, int] = {}
        for observation in self.ledger.evidence_snapshot().observations:
            key = observation.candidate_semantic_digest
            evidence_counts[key] = evidence_counts.get(key, 0) + 1
        minimum_cost = min(item.predicted_gpu_seconds for item in pool)
        raw_scores: list[float] = []
        for item in pool:
            evidence_fraction = min(
                1.0,
                evidence_counts.get(item.semantic_digest, 0)
                / self.required_seed_count,
            )
            signal = 0.6 * item.information_value + 0.4 * max(
                0.0, item.frontier_gain
            )
            raw_scores.append(
                (1.0 - evidence_fraction)
                * item.valid_seal_probability
                * (1.0 - item.correlated_compute_risk)
                * signal
                * math.sqrt(minimum_cost / item.predicted_gpu_seconds)
            )
        scale = max(raw_scores)
        if scale <= 0.0:
            return pool
        return tuple(
            replace(
                item,
                information_value=max(
                    item.information_value,
                    min(
                        1.0,
                        0.7 * item.information_value
                        + 0.3 * (raw_score / scale),
                    ),
                ),
            )
            for item, raw_score in zip(pool, raw_scores, strict=True)
        )

    def _remaining_metric_opportunities(self, observation_seed: str) -> int:
        current_schedule_index = self.validation_seed_schedule.index(
            str(observation_seed)
        )
        return len(self.validation_seed_schedule) - current_schedule_index - 1

    def _action_contract(
        self,
        *,
        recipe: Mapping[str, Any],
        binding: SearchCandidateBindingV1,
        observation_seed: str,
    ) -> dict[str, Any]:
        evaluator = recipe.get("evaluator")
        checks = {
            "binding_identity": (
                recipe.get("capability_ref") == binding.capability_ref
                and recipe.get("capability_digest") == binding.capability_digest
                and recipe.get("mechanism_semantics_digest")
                == binding.mechanism_semantics_digest
            ),
            "dataset": recipe.get("dataset") == COMMON_DATASET,
            "split": recipe.get("split") == self.expected_split,
            "evaluator": (
                isinstance(evaluator, Mapping)
                and canonical_value(dict(evaluator)) == self.expected_evaluator
            ),
            "candidate_role": recipe.get("execution_role") == "CANDIDATE",
            "model": recipe.get("model") is not None,
            "mechanism_program_digest": self._has_mechanism_program_digest(
                recipe, binding
            ),
            "observation_seed": bool(str(observation_seed)),
        }
        failed = tuple(sorted(name for name, passed in checks.items() if not passed))
        return canonical_value(
            {
                "schema": ACTION_CONTRACT_SCHEMA,
                "candidate_id": binding.proposal.candidate_id,
                "candidate_semantic_digest": binding.mechanism_semantics_digest,
                "binding_digest": binding.digest,
                "execution_recipe_digest": sha256_digest(recipe),
                "registered_protocol_digest": sha256_digest(self.protocol),
                "guard_bridge_identity_digest": self.identity_digest,
                "observation_seed": str(observation_seed),
                "checks": checks,
                "failed_checks": failed,
                "status": "LEGAL" if not failed else "ILLEGAL",
            }
        )

    def _protocol_attestation(
        self,
        *,
        recipe: Mapping[str, Any],
        binding: SearchCandidateBindingV1,
        candidate_run: Mapping[str, Any],
        observation_seed: str,
    ) -> dict[str, Any]:
        raw_binding = candidate_run.get("experiment_binding")
        try:
            execution = ExperimentBindingV1.from_canonical_dict(raw_binding)
        except (TypeError, ValueError) as error:
            return canonical_value(
                {
                    "schema": PROTOCOL_ATTESTATION_SCHEMA,
                    "status": "MISMATCH",
                    "candidate_id": binding.proposal.candidate_id,
                    "checks": {"experiment_binding": False},
                    "failed_checks": ("experiment_binding",),
                    "dataset_manifest_digest": None,
                    "expected_dataset_manifest_digest": (
                        self.expected_dataset_manifest_digest
                    ),
                    "observed_protocol": self.protocol,
                    "diagnostic": type(error).__name__,
                }
            )

        manifest_digest = execution.dataset_manifest_digest
        anchor_matches = (
            manifest_digest == self.expected_dataset_manifest_digest
        )
        evaluator = execution.evaluator
        expected_epochs = self.protocol["training_procedure"]["max_epochs"]
        checks = {
            "experiment_binding": True,
            "binding_digest": (
                candidate_run.get("experiment_binding_digest") == execution.digest
                and candidate_run.get("binding_digest") == execution.digest
            ),
            "execution_recipe_digest": (
                execution.execution_recipe_digest == sha256_digest(recipe)
                and candidate_run.get("execution_recipe_digest")
                == execution.execution_recipe_digest
            ),
            "candidate_binding": (
                execution.capability_ref == binding.capability_ref
                and execution.capability_digest == binding.capability_digest
                and execution.mechanism_id == binding.proposal.mechanism_id
                and execution.profile_ref == recipe.get("profile_ref")
                and execution.profile_digest == recipe.get("profile_digest")
                and execution.entrypoint == recipe.get("entrypoint")
                and execution.entrypoint_source_sha256
                == recipe.get("entrypoint_source_sha256")
                and execution.candidate_source_content_digest
                == recipe.get("candidate_source_content_digest")
                and execution.config == recipe.get("config")
            ),
            "dataset": execution.dataset == COMMON_DATASET,
            "dataset_manifest": _is_sha256(manifest_digest) and anchor_matches,
            "split": execution.split == self.expected_split,
            "evaluator": canonical_value(dict(evaluator)) == self.expected_evaluator,
            "model": execution.model == str(recipe.get("model")),
            "seed": str(execution.seed) == str(observation_seed),
            "epochs": execution.epochs == expected_epochs,
            "comparator": (
                execution.comparator_ref == recipe.get("comparator_ref")
                and execution.comparator_digest == recipe.get("comparator_digest")
            ),
            "execution_role": execution.execution_role == "CANDIDATE",
        }
        failed = tuple(sorted(name for name, passed in checks.items() if not passed))
        observed = _observed_protocol(
            self.protocol,
            dataset=execution.dataset,
            split=execution.split,
            evaluator=evaluator,
            expected_split=self.expected_split,
            expected_evaluator=self.expected_evaluator,
            dataset_manifest_digest=manifest_digest,
            dataset_anchor_matches=anchor_matches,
        )
        return canonical_value(
            {
                "schema": PROTOCOL_ATTESTATION_SCHEMA,
                "status": "EXACT" if not failed else "MISMATCH",
                "candidate_id": binding.proposal.candidate_id,
                "checks": checks,
                "failed_checks": failed,
                "dataset_manifest_digest": manifest_digest,
                "expected_dataset_manifest_digest": (
                    self.expected_dataset_manifest_digest
                ),
                "experiment_binding_digest": execution.digest,
                "observed_protocol": observed,
            }
        )

    def _mechanism_axis(
        self,
        *,
        recipe: Mapping[str, Any],
        binding: SearchCandidateBindingV1,
    ) -> str:
        proposal = binding.proposal.to_dict()
        return str(
            recipe.get("mechanism_axis")
            or proposal.get("mechanism_axis")
            or recipe.get("mechanism_id")
            or "UNKNOWN_AXIS"
        )

    def _evidence_summary(
        self,
        *,
        candidate: CandidateEnvelope,
        current_seed: str,
        current_attestation: Mapping[str, Any],
        current_raw_result: RawResultEnvelope,
        recipe: Mapping[str, Any],
        binding: SearchCandidateBindingV1,
    ) -> EvidenceSummaryV1:
        if current_seed not in self.validation_seed_schedule:
            raise ValueError(
                "current observation seed is outside the frozen validation schedule"
            )
        protocol_digest = sha256_digest(self.protocol)
        mechanism_program_digest = self._mechanism_program_digest(recipe, binding)
        indexed_results = self.ledger.raw_results_for_identity(
            candidate_semantic_digest=candidate.candidate_semantic_digest,
            mechanism_program_digest=mechanism_program_digest,
            protocol_digest=protocol_digest,
            comparator_identity=candidate.comparator,
        )
        attempt_results = self.ledger.raw_result_attempts_for_identity(
            candidate_semantic_digest=candidate.candidate_semantic_digest,
            mechanism_program_digest=mechanism_program_digest,
            protocol_digest=protocol_digest,
            comparator_identity=candidate.comparator,
        )
        raw_results = tuple(
            payload
            for _digest, payload in sorted(
                {
                    sha256_digest(payload): payload
                    for payload in (*attempt_results, *indexed_results)
                }.items()
            )
        )
        verified: dict[str, float] = {}
        invalid: set[str] = set()
        protocol_drift: set[str] = set()
        engineering_failure: set[str] = set()
        metric_name = str(self.context.to_dict()["claim"]["metric"])

        for raw in raw_results:
            metrics = raw.get("normalized_metrics")
            value = metrics.get(metric_name) if isinstance(metrics, Mapping) else None
            raw_is_eligible = eligible_raw_result(
                raw,
                candidate=candidate,
                expected_protocol=self.protocol,
                metric_name=metric_name,
                require_candidate_id=False,
            )
            for seed_run in raw.get("seed_runs", ()):
                if not isinstance(seed_run, Mapping) or seed_run.get("seed_id") is None:
                    continue
                seed_id = str(seed_run["seed_id"])
                if raw_is_eligible:
                    verified[seed_id] = float(value)
                    continue
                invalid.add(seed_id)
                if raw.get("observed_protocol") != self.protocol:
                    protocol_drift.add(seed_id)
                else:
                    engineering_failure.add(seed_id)

        ordered_seeds = tuple(sorted(verified))
        deltas = tuple(
            verified[seed] - self.comparator_ndcg_at_10 for seed in ordered_seeds
        )
        evidence_count = len(ordered_seeds)
        mean_delta = sum(deltas) / len(deltas) if deltas else None
        current_payload = current_raw_result.to_dict()
        current_is_eligible = eligible_raw_result(
            current_payload,
            candidate=candidate,
            expected_protocol=self.protocol,
            metric_name=metric_name,
        )
        identity_matches = (
            current_payload.get("candidate_id") == candidate.candidate_id
            and current_payload.get("opaque_arm_instance_id")
            == candidate.opaque_arm_instance_id
            and current_payload.get("target_model") == candidate.target_model
            and current_payload.get("comparator") == candidate.comparator
        )
        current_attempt_class = _current_attempt_class(
            attestation=current_attestation,
            raw_result=current_payload,
            identity_matches=identity_matches,
            raw_is_eligible=current_is_eligible,
        )

        # Scientific state is cumulative over currently verified seeds.  A
        # failed retry is a current-attempt diagnostic and must not roll this
        # state back or replace the missing-seed set.
        if evidence_count == 0:
            state = "INCONCLUSIVE"
        elif evidence_count < self.required_seed_count:
            state = (
                "PRELIMINARY_POSITIVE"
                if mean_delta is not None
                and mean_delta > self.replication_trigger_delta
                else "PRELIMINARY_NONPOSITIVE"
            )
        elif all(delta > self.minimum_effect_delta for delta in deltas):
            state = "SUPPORTED"
        elif mean_delta is not None and mean_delta > self.minimum_effect_delta:
            state = "REPLICATED_INCONCLUSIVE"
        else:
            state = "REFUTED"

        remaining_validation_seeds = tuple(
            seed
            for seed in self.validation_seed_schedule
            if seed not in verified
        )
        next_seed = next_eligible_seed(
            self.validation_seed_schedule,
            current_seed=current_seed,
            verified_seed_ids=set(verified),
        )
        if len(deltas) > 1:
            dispersion = statistics.stdev(deltas)
            standard_error = dispersion / math.sqrt(len(deltas))
            critical = T_CRITICAL_95.get(len(deltas) - 1, 1.96)
            descriptive_interval = (
                float(mean_delta - critical * standard_error),
                float(mean_delta + critical * standard_error),
            )
        else:
            dispersion = None
            standard_error = None
            descriptive_interval = None
        if deltas and all(delta > 0 for delta in deltas):
            sign_consistency = "ALL_POSITIVE"
        elif deltas and all(delta <= 0 for delta in deltas):
            sign_consistency = "ALL_NONPOSITIVE"
        elif deltas:
            sign_consistency = "MIXED_SIGNS"
        else:
            sign_consistency = "NO_VERIFIED_SEEDS"
        verified_set = set(verified)
        invalid_ids = tuple(sorted(invalid - verified_set))
        protocol_drift -= verified_set
        engineering_failure -= verified_set
        return EvidenceSummaryV1(
            candidate_id=candidate.candidate_id,
            candidate_semantic_digest=candidate.candidate_semantic_digest,
            mechanism_program_digest=(
                mechanism_program_digest
                if self._has_mechanism_program_digest(recipe, binding)
                else None
            ),
            protocol_digest=protocol_digest,
            comparator_identity=candidate.comparator,
            mechanism_axis=self._mechanism_axis(recipe=recipe, binding=binding),
            verified_seed_ids=ordered_seeds,
            invalid_seed_ids=invalid_ids,
            protocol_drift_seed_ids=tuple(sorted(protocol_drift)),
            engineering_failure_seed_ids=tuple(sorted(engineering_failure)),
            missing_seed_ids=remaining_validation_seeds,
            comparator_deltas=deltas,
            mean_comparator_delta=mean_delta,
            dispersion=dispersion,
            standard_error=standard_error,
            descriptive_t_interval_95=descriptive_interval,
            sign_consistency=sign_consistency,
            scientific_conclusion_strength=state,
            current_attempt_class=current_attempt_class,
            required_seed_count=self.required_seed_count,
            evidence_count=evidence_count,
            minimum_effect_delta=self.minimum_effect_delta,
            protocol_status=(
                "PROTOCOL_BRANCH"
                if current_attempt_class == "PROTOCOL_DRIFT"
                else "CURRENT_PROTOCOL"
            ),
            next_eligible_seed=next_seed,
        )

    def _search_utility_event(
        self,
        *,
        candidate: CandidateEnvelope,
        binding: SearchCandidateBindingV1,
        candidate_run: Mapping[str, Any],
        observation_seed: str,
        summary: EvidenceSummaryV1,
    ) -> SearchUtilityEventV2:
        blocker = {
            "PROTOCOL_DRIFT": "PROTOCOL_DRIFT",
            "ENGINEERING_FAILURE": "ENGINEERING_FAILURE",
            "INVALID": "INVALID_EVIDENCE",
        }.get(summary.current_attempt_class, "NONE")
        return SearchUtilityEventV2(
            candidate_semantic_digest=candidate.candidate_semantic_digest,
            candidate_id=candidate.candidate_id,
            mechanism_axis=summary.mechanism_axis,
            common_outcome_class=summary.conclusion_strength,
            runnable_observation=(
                "RUNNABLE"
                if candidate_run.get("exit_status") == "SUCCESS"
                else "NOT_RUNNABLE"
            ),
            comparator_delta=(
                summary.mean_comparator_delta
                if summary.mean_comparator_delta is not None
                and blocker == "NONE"
                else "NOT_AVAILABLE"
            ),
            metric_contract_digest=sha256_digest(self.protocol["metric"]),
            resource_cost_projection=(
                candidate_run.get("resource_telemetry", {})
                if isinstance(candidate_run.get("resource_telemetry", {}), Mapping)
                else {}
            ),
            typed_blocker_class=blocker,
            observation_seed=str(observation_seed),
            candidate_value=(
                self.comparator_ndcg_at_10 + summary.mean_comparator_delta
                if summary.mean_comparator_delta is not None and blocker == "NONE"
                else None
            ),
        )

    def _source_control_attribution(
        self,
        *,
        active_task: Mapping[str, Any] | None,
        candidate: CandidateEnvelope,
        summary: EvidenceSummaryV1,
        observation_seed: str,
    ) -> Mapping[str, Any] | None:
        """Project one exact Guard control back to its source mechanism.

        The source candidate-local evidence remains untouched.  This record is
        a separate, development-only, descriptive attribution because the
        frozen one-metric-per-round schedule yields a future-seed control, not
        a same-seed experimental pair.
        """

        if not isinstance(active_task, Mapping):
            return None
        operation = str(active_task.get("operation", ""))
        if operation not in {"MATCHED_CONTROL", "MECHANISM_OFF"}:
            return None
        metadata = active_task.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("guard_source") != "EVIDENCE_GUARD":
            return None
        binding_checks = {
            "candidate_semantic_digest": (
                active_task.get("candidate_semantic_digest")
                == candidate.candidate_semantic_digest
            ),
            "mechanism_program_digest": (
                active_task.get("mechanism_program_digest")
                == candidate.mechanism_program_digest
            ),
        }
        failed_binding_checks = tuple(
            name for name, passed in binding_checks.items() if not passed
        )
        if failed_binding_checks:
            raise ValueError(
                "active Guard control task does not match executed binding: "
                + ",".join(failed_binding_checks)
            )

        source_summary = metadata.get("guard_source_evidence_summary")
        if not isinstance(source_summary, Mapping):
            raise ValueError("active Guard control task lacks source evidence summary")
        source_semantic = metadata.get("frontier_candidate_semantic_digest")
        source_program = metadata.get("frontier_mechanism_program_digest")
        source_candidate_id = metadata.get("frontier_candidate_id")
        source_axis = metadata.get("guard_source_mechanism_axis")
        if not (
            isinstance(source_candidate_id, str)
            and isinstance(source_semantic, str)
            and _is_sha256(source_semantic)
            and isinstance(source_program, str)
            and _is_sha256(source_program)
            and isinstance(source_axis, str)
            and source_axis
        ):
            raise ValueError("active Guard control task has incomplete source identity")
        if source_program == candidate.mechanism_program_digest:
            raise ValueError("Guard control task cannot attribute a mechanism to itself")
        if (
            source_summary.get("candidate_id") != source_candidate_id
            or source_summary.get("candidate_semantic_digest") != source_semantic
            or source_summary.get("mechanism_program_digest") != source_program
            or source_summary.get("protocol_digest") != summary.protocol_digest
            or source_summary.get("comparator_identity") != summary.comparator_identity
        ):
            raise ValueError("active Guard control source summary identity drifted")
        source_state = str(source_summary.get("scientific_conclusion_strength", ""))
        if source_state != "REPLICATED_INCONCLUSIVE":
            raise ValueError("Guard control source is not replicated-inconclusive")

        source_mean = source_summary.get("mean_comparator_delta")
        control_mean = summary.mean_comparator_delta
        source_count = source_summary.get("evidence_count")
        if (
            not isinstance(source_mean, (int, float))
            or isinstance(source_mean, bool)
            or not math.isfinite(float(source_mean))
            or not isinstance(source_count, int)
            or isinstance(source_count, bool)
            or source_count < 1
        ):
            raise ValueError("Guard control source summary lacks finite evidence")

        valid_control = (
            summary.current_attempt_class == "VALID_METRIC"
            and control_mean is not None
            and summary.evidence_count > 0
        )
        incremental_delta: float | None = None
        if valid_control:
            incremental_delta = float(source_mean) - float(control_mean)
            if incremental_delta > self.minimum_effect_delta:
                attribution_state = "DESCRIPTIVE_SUPPORT"
            elif incremental_delta <= 0.0:
                attribution_state = "DESCRIPTIVE_REFUTATION"
            else:
                attribution_state = "DESCRIPTIVE_UNRESOLVED"
            confidence_weight = min(
                1.0,
                float(summary.evidence_count) / float(self.required_seed_count),
            )
        else:
            attribution_state = "CONTROL_INVALID_NO_SCIENCE_UPDATE"
            confidence_weight = 0.0

        relation = metadata.get("guard_control_relation")
        relation = dict(relation) if isinstance(relation, Mapping) else {}
        attribution_identity_digest = sha256_digest(
            {
                "source_candidate_semantic_digest": source_semantic,
                "source_mechanism_program_digest": source_program,
                "control_candidate_semantic_digest": candidate.candidate_semantic_digest,
                "control_mechanism_program_digest": candidate.mechanism_program_digest,
                "protocol_digest": summary.protocol_digest,
                "comparator_identity": summary.comparator_identity,
                "operation": operation,
            }
        )
        attribution = canonical_value(
            {
                "schema": "recclaw.helix.source-control-attribution.v1",
                "attribution_identity_digest": attribution_identity_digest,
                "source_candidate_id": source_candidate_id,
                "source_candidate_semantic_digest": source_semantic,
                "source_mechanism_program_digest": source_program,
                "source_mechanism_axis": source_axis,
                "source_candidate_local_state": source_state,
                "source_verified_seed_ids": tuple(
                    source_summary.get("verified_seed_ids", ())
                ),
                "source_evidence_count": source_count,
                "source_mean_comparator_delta": float(source_mean),
                "control_candidate_id": candidate.candidate_id,
                "control_candidate_semantic_digest": candidate.candidate_semantic_digest,
                "control_mechanism_program_digest": candidate.mechanism_program_digest,
                "control_verified_seed_ids": summary.verified_seed_ids,
                "control_evidence_count": summary.evidence_count,
                "control_mean_comparator_delta": control_mean,
                "control_observation_seed": str(observation_seed),
                "operation": operation,
                "declared_relation": relation,
                "pairing_class": "UNPAIRED_FUTURE_SEED",
                "incremental_delta": incremental_delta,
                "minimum_effect_delta": self.minimum_effect_delta,
                "attribution_state": attribution_state,
                "confidence_weight": confidence_weight,
                "evidence_class": "DEVELOPMENT_ONLY",
                "formal_claim_authority": "NONE",
                "source_candidate_evidence_mutated": False,
            }
        )
        persisted = False
        stored: Mapping[str, Any] | None = None
        if valid_control:
            combined_count = source_count + summary.evidence_count
            prior = tuple(
                item
                for item in self.ledger.control_history()
                if item["control_identity_digest"] == attribution_identity_digest
            )
            prior_state = max(
                prior,
                key=lambda item: int(item["evidence_count"]),
                default=None,
            )
            if prior_state is None or combined_count > int(prior_state["evidence_count"]):
                stored = self.ledger.upsert_control_state(
                    control_identity_digest=attribution_identity_digest,
                    evidence_count=combined_count,
                    state=attribution,
                )
                persisted = True
            else:
                stored = prior_state["state"]
        return canonical_value(
            {
                **dict(attribution),
                "state_persisted": persisted,
                "stored_state": stored,
            }
        )

    @staticmethod
    def _resolved_mechanism_program_digest(
        recipe: Mapping[str, Any], binding: SearchCandidateBindingV1
    ) -> str | None:
        supplied = recipe.get("mechanism_program_digest")
        if _is_sha256(supplied):
            return str(supplied)
        proposal_value = (
            binding.proposal.to_dict()
            if callable(getattr(binding.proposal, "to_dict", None))
            else None
        )
        bound = (
            proposal_value.get("mechanism_program_digest")
            if isinstance(proposal_value, Mapping)
            else None
        )
        if _is_sha256(bound):
            return str(bound)
        return None

    @classmethod
    def _mechanism_program_digest(
        cls, recipe: Mapping[str, Any], binding: SearchCandidateBindingV1
    ) -> str:
        resolved = cls._resolved_mechanism_program_digest(recipe, binding)
        if resolved is not None:
            return resolved
        # CandidateEnvelope still needs a closed digest for PRE diagnostics,
        # but this sentinel is never treated as a program identity.  The
        # action contract below blocks execution and POST when the real
        # Research binding did not provide its digest.
        return "0" * 64

    @classmethod
    def _has_mechanism_program_digest(
        cls, recipe: Mapping[str, Any], binding: SearchCandidateBindingV1
    ) -> bool:
        return cls._resolved_mechanism_program_digest(recipe, binding) is not None

    def _candidate(
        self,
        *,
        recipe: Mapping[str, Any],
        binding: SearchCandidateBindingV1,
        observation_seed: str,
    ) -> CandidateEnvelope:
        return CandidateEnvelope(
            candidate_id=binding.proposal.candidate_id,
            candidate_semantic_digest=binding.mechanism_semantics_digest,
            opaque_arm_instance_id=self.opaque_arm_instance_id,
            common_status="COMMON_PASS",
            mechanism_program_digest=self._mechanism_program_digest(recipe, binding),
            common_plan_digest=sha256_digest(
                {
                    "recipe": recipe,
                    "binding": binding.canonical_dict(),
                    "observation_seed": observation_seed,
                }
            ),
            action_family="RUN_OFFLINE_TOPN",
            planned_protocol=self.protocol,
            target_model=str(recipe["model"]),
            comparator=self.comparator,
            seed_ids=(str(observation_seed),),
            purpose="development comparison under the frozen Research protocol",
        )

    def pre_run(
        self,
        *,
        recipe: Mapping[str, Any],
        binding: SearchCandidateBindingV1,
        observation_seed: str,
    ) -> Mapping[str, Any]:
        action_contract = self._action_contract(
            recipe=recipe,
            binding=binding,
            observation_seed=observation_seed,
        )
        candidate = self._candidate(
            recipe=recipe,
            binding=binding,
            observation_seed=observation_seed,
        )
        adjudication = self.port.pre_run(candidate)
        if action_contract["status"] == "LEGAL":
            # Evidence Guard is advisory before execution.  It may annotate
            # claim or evidence concerns, but it cannot veto an otherwise
            # executable Research candidate.
            adjudication = replace(adjudication, status=PortStatus.ALLOW)
        else:
            # Only concrete execution-binding violations remain blocking.
            adjudication = replace(
                adjudication,
                status=PortStatus.BLOCK,
                reason_codes=tuple(
                    sorted(
                        {
                            *adjudication.reason_codes,
                            *(
                                "HELIX_ACTION_CONTRACT_" + name.upper()
                                for name in action_contract["failed_checks"]
                            ),
                        }
                    )
                ),
            )
        selection_action = self.admission.admit_pre(adjudication)
        return canonical_value(
            {
                "schema": BRIDGE_SCHEMA,
                **adjudication.to_dict(),
                "candidate_envelope_digest": candidate.digest,
                "action_contract": action_contract,
                "fusion": {
                    "selection_action": selection_action,
                    "policy_digest": self.admission.policy_digest,
                },
                "control_binding_required": not self._has_mechanism_program_digest(
                    recipe, binding
                ),
            }
        )

    def post_run(
        self,
        *,
        recipe: Mapping[str, Any],
        binding: SearchCandidateBindingV1,
        candidate_run: Mapping[str, Any],
        closure: Any,
        observation_seed: str,
        active_task: Mapping[str, Any] | None = None,
        remaining_metric_opportunities: int | None = None,
        validation_task: ResearchTaskV1 | None = None,
        control_task: ResearchTaskV1 | None = None,
        protocol_branch_task: ResearchTaskV1 | None = None,
    ) -> Mapping[str, Any]:
        closure_value = closure.to_dict()
        closure_digest = closure.digest
        closure_path = (
            self.run_root
            / "evidence_guard"
            / "common_closures"
            / f"{closure_digest}.json"
        )
        _write_once_or_verify(
            closure_path,
            {
                "schema": "recclaw.helix.durable-common-closure.v1",
                "closure_digest": closure_digest,
                "closure": closure_value,
            },
        )
        candidate = self._candidate(
            recipe=recipe,
            binding=binding,
            observation_seed=observation_seed,
        )
        action_contract = self._action_contract(
            recipe=recipe,
            binding=binding,
            observation_seed=observation_seed,
        )
        if not self._has_mechanism_program_digest(recipe, binding):
            return canonical_value(
                {
                    "schema": BRIDGE_SCHEMA,
                    "candidate_id": candidate.candidate_id,
                    "candidate_envelope_digest": candidate.digest,
                    "status": "BLOCK",
                    "outcome_class": "CONTROL_BINDING_INCOMPLETE",
                    "evidence_use": "CONTROL_BINDING_INCOMPLETE",
                    "claim_ceiling": "NO_CLAIM_UPDATE",
                    "action_contract": action_contract,
                    "control_binding_required": True,
                    "control_binding_status": "AWAIT_RESEARCH_INTERPRETER",
                    "task_intent": {
                        "status": "INCOMPLETE",
                        "requested_control_kind": "NONE",
                        "reason": "REAL_MECHANISM_PROGRAM_DIGEST_REQUIRED",
                    },
                    "fusion": {
                        "selection_action": "NEXT_FROM_SAME_SLATE",
                        "policy_digest": self.admission.policy_digest,
                    },
                    "formal_claim_authority": "NONE",
                }
            )
        attestation = self._protocol_attestation(
            recipe=recipe,
            binding=binding,
            candidate_run=candidate_run,
            observation_seed=observation_seed,
        )
        metrics = candidate_run.get("metrics")
        ndcg = metrics.get("ndcg@10") if isinstance(metrics, Mapping) else None
        normalized_metrics: dict[str, float] = {}
        if (
            isinstance(ndcg, (int, float))
            and not isinstance(ndcg, bool)
            and math.isfinite(float(ndcg))
        ):
            normalized_metrics["ndcg"] = float(ndcg)
        raw_result_digest = sha256_digest(candidate_run)
        raw = RawResultEnvelope(
            candidate_id=binding.proposal.candidate_id,
            opaque_arm_instance_id=self.opaque_arm_instance_id,
            raw_result_digest=raw_result_digest,
            common_result_closure_digest=closure_digest,
            observed_protocol=attestation["observed_protocol"],
            target_model=str(recipe["model"]),
            comparator=self.comparator,
            seed_runs=(
                {
                    "seed_id": str(observation_seed),
                    "run_id": str(
                        candidate_run.get("run_id")
                        or candidate_run.get("experiment_binding_ref")
                        or raw_result_digest
                    ),
                    "artifact_sha256": raw_result_digest,
                },
            ),
            observation_kind=(
                "METRIC_EVALUATION" if normalized_metrics else "RUNTIME_OBSERVATION"
            ),
            run_status=str(candidate_run.get("exit_status") or "RUNTIME_FAILURE"),
            artifact_identity_status=(
                "EXACT" if attestation["status"] == "EXACT" else "INCOMPLETE"
            ),
            normalized_metrics=normalized_metrics,
        )
        adjudication = self.port.post_run(raw)
        summary = self._evidence_summary(
            candidate=candidate,
            current_seed=observation_seed,
            current_attestation=attestation,
            current_raw_result=raw,
            recipe=recipe,
            binding=binding,
        )
        source_control_attribution = self._source_control_attribution(
            active_task=active_task,
            candidate=candidate,
            summary=summary,
            observation_seed=observation_seed,
        )
        search_event = self._search_utility_event(
            candidate=candidate,
            binding=binding,
            candidate_run=candidate_run,
            observation_seed=observation_seed,
            summary=summary,
        )
        fusion, compact = self.admission.admit_post(
            adjudication=adjudication,
            search_utility_event=search_event,
            validation_task=validation_task,
            matched_control_task=control_task,
            protocol_branch_task=protocol_branch_task,
            evidence_summary=summary,
        )
        projection = fusion.control_projection
        if projection is None:
            raise RuntimeError("V30 did not return a control projection for Evidence Guard evidence")
        if validation_task is not None or control_task is not None:
            # These validated V1 inputs request evidence work. The Research
            # interpreter owns the executable V2 task, campaign protocol and
            # allocation identity; never substitute the request for that task.
            fusion = replace(fusion, research_task=None)
        isolation_question = (
            _mechanism_isolation_question(binding)
            if summary.current_attempt_class == "VALID_METRIC" and source_control_attribution is None
            else None
        )
        allocation_closure = self.allocator.close_active_action(
            active_task=active_task,
            summary=summary.to_dict(),
        )
        allocation_summary = {
            **summary.to_dict(),
            "discriminative_question": isolation_question,
            "research_feedback": (
                _mechanism_research_feedback(binding)
                if summary.current_attempt_class == "VALID_METRIC"
                else None
            ),
            "remaining_metric_opportunities": (
                remaining_metric_opportunities if remaining_metric_opportunities is not None
                else self._remaining_metric_opportunities(observation_seed)
            ),
        }
        allocation_decision = self.allocator.decide(
            summary=allocation_summary,
            requested_control_kind=projection.requested_control_kind,
            # Discovery learns from every result without automatically buying
            # another experiment. Explicit validation callers retain their
            # existing task/budget path; a control result cannot spawn a chain.
            allow_new_action=(
                source_control_attribution is None
                and (
                    validation_task is not None
                    and summary.scientific_conclusion_strength == "PRELIMINARY_POSITIVE"
                    or control_task is not None
                    and summary.scientific_conclusion_strength == "REPLICATED_INCONCLUSIVE"
                )
            ),
        )
        allocation_action = allocation_decision.action.value
        control_identity_digest = sha256_digest(
            {
                "candidate_semantic_digest": candidate.candidate_semantic_digest,
                "mechanism_program_digest": candidate.mechanism_program_digest,
                "protocol_digest": summary.protocol_digest,
                "comparator": candidate.comparator,
            }
        )
        control_state = canonical_value(
            {
                "schema": CLAIM_CONTROL_SCHEMA,
                "control_identity_digest": control_identity_digest,
                "candidate_id": candidate.candidate_id,
                "candidate_semantic_digest": candidate.candidate_semantic_digest,
                "mechanism_program_digest": candidate.mechanism_program_digest,
                "protocol_digest": summary.protocol_digest,
                "comparator": candidate.comparator,
                "evidence_count": summary.evidence_count,
                "evidence_summary": allocation_summary,
                "control_projection": projection.to_dict(),
                "fused_feedback": fusion.to_dict(),
                "task_binding_status": projection.control_binding_status,
                "validation_task_digest": (
                    validation_task.digest if validation_task is not None else None
                ),
                "control_task_digest": (
                    control_task.digest if control_task is not None else None
                ),
            }
        )
        prior_controls = tuple(
            item
            for item in self.ledger.control_history()
            if item["control_identity_digest"] == control_identity_digest
        )
        prior_control = max(
            prior_controls,
            key=lambda item: int(item["evidence_count"]),
            default=None,
        )
        scientific_state_advanced = (
            summary.current_attempt_class == "VALID_METRIC"
            and (
                prior_control is None
                or summary.evidence_count > int(prior_control["evidence_count"])
            )
        )
        if scientific_state_advanced:
            stored_control = self.ledger.upsert_control_state(
                control_identity_digest=control_identity_digest,
                evidence_count=summary.evidence_count,
                state=control_state,
            )
        else:
            # Failed/invalid retries are retained in guard_calls and the
            # decision returned below, but cannot rewrite a scientific state
            # at the same evidence count.
            stored_control = (
                prior_control["state"] if prior_control is not None else None
            )
        return canonical_value(
            {
                "schema": BRIDGE_SCHEMA,
                **adjudication.to_dict(),
                "comparator_delta": adjudication.comparator_delta,
                "raw_result_envelope_digest": raw.digest,
                "common_closure_path": str(closure_path),
                "protocol_attestation": attestation,
                "evidence_summary": allocation_summary,
                "control_projection": projection.to_dict(),
                "claim_transition": {
                    "state": summary.conclusion_strength,
                    "scientific_conclusion_strength": summary.scientific_conclusion_strength,
                    "current_attempt_class": summary.current_attempt_class,
                    "verified_seeds": summary.verified_seed_ids,
                    "invalid_seeds": summary.invalid_seed_ids,
                    "missing_seeds": summary.missing_seed_ids,
                    "evidence_count": summary.evidence_count,
                    "required_seed_count": summary.required_seed_count,
                    "mean_comparator_delta": summary.mean_comparator_delta,
                    "dispersion": summary.dispersion,
                    "standard_error": summary.standard_error,
                    "descriptive_t_interval_95": summary.descriptive_t_interval_95,
                    "sign_consistency": summary.sign_consistency,
                    "conclusion_strength": summary.scientific_conclusion_strength,
                    "descriptive_only": True,
                    "formal_inference": False,
                    "adaptive_valid": False,
                },
                "validation_directive": {
                    "action": (
                        "CONTROL_RESULT_CONSUMED_RETURN_TO_EXPLORATION"
                        if source_control_attribution is not None
                        else "NEXT_UNSEEN_SEED_SAME_BUDGET"
                        if allocation_decision.action
                        is AllocationActionV31.REPLICATE
                        else projection.next_seed_allocation
                        if allocation_decision.action
                        is AllocationActionV31.CONTROL
                        else "NO_ADDITIONAL_EVIDENCE_ALLOCATED"
                    ),
                    "next_seed": (
                        allocation_decision.target
                        if allocation_decision.action
                        is AllocationActionV31.REPLICATE
                        else None
                    ),
                    "missing_seed_count": len(summary.missing_seed_ids),
                    "task_binding_required": (
                        projection.task_binding_required and allocation_action != "NOOP"
                    ),
                    "control_binding_required": (
                        projection.control_binding_required
                        and allocation_decision.reason == "CONTROL_BINDING_REQUIRED"
                    ),
                    "budget_class": "SAME_BUDGET_EVIDENCE_VALIDATION",
                },
                "research_update_mode": (
                    "PRESERVE_NATIVE_RESEARCH"
                    if summary.current_attempt_class == "VALID_METRIC"
                    else "DIAGNOSTIC_ONLY"
                ),
                "allocation_decision": {
                    **allocation_decision.to_dict(),
                    "action": allocation_action,
                    "decision_relevant": allocation_action != "NOOP",
                    "replication_trigger_delta": self.replication_trigger_delta,
                    "support_effect_delta": self.minimum_effect_delta,
                },
                "allocation_closure": allocation_closure,
                "promotion_gate": {
                    "decision": projection.development_promotion,
                    "artifact_promotion_allowed": (
                        summary.conclusion_strength == "SUPPORTED"
                        and summary.current_attempt_class == "VALID_METRIC"
                    ),
                    "trusted_incumbent_update_allowed": (
                        summary.conclusion_strength == "SUPPORTED"
                        and summary.current_attempt_class == "VALID_METRIC"
                        and projection.development_promotion
                        == "ALLOW_DEVELOPMENT_PROMOTION"
                    ),
                    "formal_claim_authority": projection.formal_claim_authority,
                },
                "router_feedback": {
                    "protocol_risk": 1.0 if summary.current_attempt_class == "PROTOCOL_DRIFT" else 0.0,
                    "evidence_confidence": projection.confidence_weight,
                    "replication_status": summary.conclusion_strength,
                    "validation_debt": len(summary.missing_seed_ids),
                    "memory_update": projection.memory_update,
                    "evidence_class": projection.evidence_class,
                },
                "fusion": fusion.to_dict(),
                "compact_feedback": compact.to_dict() if compact is not None else None,
                "control_state": stored_control,
                "control_state_persisted": scientific_state_advanced,
                "source_control_attribution": source_control_attribution,
            }
        )

    def reserve_bound_control(
        self,
        *,
        response: Mapping[str, Any],
        control_task: Mapping[str, Any],
        observation_seed: str,
    ) -> Mapping[str, Any]:
        """Reserve a CONTROL only after Research supplies an exact binding."""

        allocation = response.get("allocation_decision")
        if not isinstance(allocation, Mapping):
            raise ValueError("control reservation lacks an allocation intent")
        if allocation.get("action") == AllocationActionV31.CONTROL.value:
            return canonical_value(dict(response))
        if allocation.get("reason") != "CONTROL_BINDING_REQUIRED":
            raise ValueError("control reservation was not requested by the allocator")
        summary = response.get("evidence_summary")
        projection = response.get("control_projection")
        metadata = control_task.get("metadata")
        if not isinstance(summary, Mapping) or not isinstance(projection, Mapping):
            raise ValueError("control reservation lacks scientific evidence")
        if not isinstance(metadata, Mapping):
            raise ValueError("control reservation lacks task provenance")
        for task_field, summary_field in (
            ("frontier_candidate_semantic_digest", "candidate_semantic_digest"),
            ("frontier_mechanism_program_digest", "mechanism_program_digest"),
        ):
            if metadata.get(task_field) != summary.get(summary_field):
                raise ValueError(f"bound control {task_field} drift")
        target = control_task.get("required_seed_or_control")
        if not isinstance(target, str) or not target:
            raise ValueError("bound control lacks an executable target")
        requested_control_kind = projection.get("requested_control_kind")
        if requested_control_kind not in {"MATCHED_CONTROL", "MECHANISM_OFF"}:
            raise ValueError("bound control kind is outside the closed set")
        decision = self.allocator.decide(
            summary=dict(summary),
            requested_control_kind=str(requested_control_kind),
            bound_control_target=target,
        )
        if decision.action is not AllocationActionV31.CONTROL:
            raise RuntimeError("exact control binding did not reserve an action")
        directive = response.get("validation_directive")
        directive_value = dict(directive) if isinstance(directive, Mapping) else {}
        return canonical_value(
            {
                **dict(response),
                "allocation_decision": {
                    **decision.to_dict(),
                    "replication_trigger_delta": self.replication_trigger_delta,
                    "support_effect_delta": self.minimum_effect_delta,
                },
                "validation_directive": {
                    **directive_value,
                    "action": projection.get("next_seed_allocation"),
                    "next_seed": None,
                },
            }
        )

    def close(self) -> None:
        self.ledger.close()

    def control_states(self) -> tuple[dict[str, Any], ...]:
        return self.ledger.control_states()

    def control_history(self) -> tuple[dict[str, Any], ...]:
        return self.ledger.control_history()

    def allocation_actions(self) -> tuple[dict[str, Any], ...]:
        return self.ledger.allocation_actions()

    def allocation_budget_snapshot(self) -> Mapping[str, Any]:
        return self.ledger.allocation_budget_snapshot(
            max_actions=self.allocation_policy.max_actions
        )


# Full Helix V31 owns evidence adjudication, value-of-information admission,
# frontier allocation, control/replication closure, and feedback fusion around
# the Research Line proposal/implementation engine.  Keep the historical
# EvidenceGuard name only as a source-compatible alias for old replay tests.
StandaloneFullHelixBridgeV31 = StandaloneEvidenceGuardBridgeV1
StandaloneEvidenceGuardBridgeV31 = StandaloneEvidenceGuardBridgeV1


__all__ = [
    "BRIDGE_SCHEMA",
    "ACTION_CONTRACT_SCHEMA",
    "CLAIM_CONTROL_SCHEMA",
    "PROTOCOL_ATTESTATION_SCHEMA",
    "StandaloneEvidenceGuardBridgeV1",
    "StandaloneEvidenceGuardBridgeV31",
    "StandaloneFullHelixBridgeV31",
    "standalone_guard_protocol",
]
