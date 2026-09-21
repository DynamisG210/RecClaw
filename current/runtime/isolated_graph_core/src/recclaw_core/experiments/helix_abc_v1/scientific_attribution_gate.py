"""Cross-component scientific-attribution gate for the V13 Pilot boundary.

This module does not implement another search or evidence policy.  It checks
evidence produced by the already-typed Original, Research, Helix, Meta, and
analysis paths before a V13 contract may be frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .canonical import canonical_value, sha256_digest


REQUIRED_DISPOSITIONS_V13 = (
    "DEVELOPMENT_EVIDENCE_USE_ALLOWED",
    "REQUIRES_CONFIRMATION",
    "DIAGNOSTIC_ONLY",
    "NOT_ADMISSIBLE",
    "PROTOCOL_BRANCH",
    "QUARANTINE_POST",
    "GUARD_INCONCLUSIVE",
    "ALL_PRE_BLOCKED",
    "PRE_CONTRACT_FAILURE",
    "COMMON_EXECUTION_FAILURE",
)

_EXPECTED_DISPOSITIONS: Mapping[str, Mapping[str, Any]] = {
    "DEVELOPMENT_EVIDENCE_USE_ALLOWED": {
        "controller_changed": True,
        "search_memory_changed": True,
        "meta_changed": True,
        "observed_frontier_changed": True,
        "frontier_eligibility": "SEARCH_ELIGIBLE",
        "queue_task_type": None,
    },
    "REQUIRES_CONFIRMATION": {
        "controller_changed": True,
        "search_memory_changed": True,
        "meta_changed": True,
        "observed_frontier_changed": True,
        "frontier_eligibility": "SEARCH_ELIGIBLE_PRELIMINARY",
        "queue_task_type": "VALIDATE_SAME_CANDIDATE",
    },
    "DIAGNOSTIC_ONLY": {
        "controller_changed": False,
        "search_memory_changed": False,
        "meta_changed": False,
        "observed_frontier_changed": True,
        "frontier_eligibility": "EXCLUDED",
        "queue_task_type": None,
    },
    "NOT_ADMISSIBLE": {
        "controller_changed": False,
        "search_memory_changed": False,
        "meta_changed": False,
        "observed_frontier_changed": True,
        "frontier_eligibility": "EXCLUDED",
        "queue_task_type": None,
    },
    "PROTOCOL_BRANCH": {
        "controller_changed": True,
        "search_memory_changed": True,
        "meta_changed": False,
        "observed_frontier_changed": True,
        "frontier_eligibility": "EXCLUDED",
        "queue_task_type": "PROTOCOL_BRANCH_DIAGNOSTIC",
    },
    "QUARANTINE_POST": {
        "controller_changed": False,
        "search_memory_changed": False,
        "meta_changed": False,
        "observed_frontier_changed": True,
        "frontier_eligibility": "EXCLUDED",
        "queue_task_type": None,
    },
    "GUARD_INCONCLUSIVE": {
        "controller_changed": False,
        "search_memory_changed": False,
        "meta_changed": False,
        "observed_frontier_changed": True,
        "frontier_eligibility": "EXCLUDED",
        "queue_task_type": None,
    },
    "ALL_PRE_BLOCKED": {
        "controller_changed": False,
        "search_memory_changed": False,
        "meta_changed": False,
        "observed_frontier_changed": False,
        "frontier_eligibility": "EXCLUDED",
        "queue_task_type": None,
    },
    "PRE_CONTRACT_FAILURE": {
        "controller_changed": False,
        "search_memory_changed": False,
        "meta_changed": False,
        "observed_frontier_changed": False,
        "frontier_eligibility": "EXCLUDED",
        "queue_task_type": None,
    },
    "COMMON_EXECUTION_FAILURE": {
        "controller_changed": False,
        "search_memory_changed": False,
        "meta_changed": False,
        "observed_frontier_changed": True,
        "frontier_eligibility": "EXCLUDED",
        "queue_task_type": None,
    },
}


@dataclass(frozen=True, slots=True)
class AttributionDispositionObservationV13:
    disposition: str
    controller_changed: bool
    search_memory_changed: bool
    meta_changed: bool
    observed_frontier_changed: bool
    frontier_eligibility: str
    confirmed_frontier_changed: bool
    queue_task_type: str | None
    prompt_projection_keys: tuple[str, ...]
    guard_private_prompt_field_count: int
    evidence_snapshot_delta: int

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class AttributionGateFindingV13:
    severity: str
    check: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class PilotScientificAttributionGateResultV13:
    verdict: str
    p0: int
    p1: int
    p2: int
    disposition_digests: tuple[tuple[str, str], ...]
    checked_invariants: tuple[str, ...]
    findings: tuple[AttributionGateFindingV13, ...]
    evidence_digest: str

    @property
    def digest(self) -> str:
        return sha256_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


class PilotScientificAttributionGateV13:
    """Evaluate the frozen V13 attribution invariants from runtime evidence."""

    _PROMPT_KEYS = (
        "common_search_utility_slot",
        "research_task_slot",
    )
    _RESEARCH_CHECKS = (
        "four_producer_identities_and_provenance",
        "falsification_is_discovery_credit",
        "control_repair_credit_separated",
        "exact_prior_round_lineage",
        "missing_parent_rejected",
        "matched_comparator_belief",
        "utility_floor_enforced",
        "runtime_derived_utility_features",
        "no_guard_private_research_input",
        "meta_v18_promoted_and_supported",
        "no_search_collapse",
    )
    _ORIGINAL_CHECKS = (
        "direct_pinned_main_source",
        "full_differential_trace_equal",
        "v13_runtime_uses_golden_path",
        "priority_from_original_response",
        "status_from_common_guard",
        "legacy_adapter_unreachable",
    )
    _ANALYSIS_CHECKS = (
        "ordinary_one_seed_not_confirmed",
        "guard_ineligible_excluded",
        "preliminary_only_in_declared_projection",
        "confirmed_requires_frozen_evaluator",
        "pilot_effect_not_computed",
    )
    _E2E_CHECKS = (
        "all_three_arms_closed",
        "one_execution_opportunity_per_arm",
        "same_common_profile",
        "same_resource_ceilings",
        "b_c_non_guard_identity_equal",
        "b_c_guard_is_only_treatment_difference",
        "arm_private_roots_disjoint",
        "v18_meta_active_for_b_c",
        "no_provider_call",
        "no_training_execution",
    )

    @staticmethod
    def _boolean_findings(
        section: str,
        required: Sequence[str],
        evidence: Mapping[str, Any],
    ) -> list[AttributionGateFindingV13]:
        findings: list[AttributionGateFindingV13] = []
        for name in required:
            if evidence.get(name) is not True:
                findings.append(
                    AttributionGateFindingV13(
                        severity="P1",
                        check=f"{section}.{name}",
                        detail="required runtime evidence is absent or false",
                    )
                )
        return findings

    def evaluate(
        self,
        *,
        dispositions: Sequence[AttributionDispositionObservationV13],
        research_evidence: Mapping[str, Any],
        original_evidence: Mapping[str, Any],
        analysis_evidence: Mapping[str, Any],
        fake_e2e_evidence: Mapping[str, Any],
    ) -> PilotScientificAttributionGateResultV13:
        findings: list[AttributionGateFindingV13] = []
        by_name = {item.disposition: item for item in dispositions}
        if set(by_name) != set(REQUIRED_DISPOSITIONS_V13):
            missing = sorted(set(REQUIRED_DISPOSITIONS_V13) - set(by_name))
            extra = sorted(set(by_name) - set(REQUIRED_DISPOSITIONS_V13))
            findings.append(
                AttributionGateFindingV13(
                    severity="P1",
                    check="disposition_matrix.coverage",
                    detail=f"missing={missing}; extra={extra}",
                )
            )
        for name in REQUIRED_DISPOSITIONS_V13:
            observation = by_name.get(name)
            if observation is None:
                continue
            expected = _EXPECTED_DISPOSITIONS[name]
            for field, value in expected.items():
                if getattr(observation, field) != value:
                    findings.append(
                        AttributionGateFindingV13(
                            severity="P1",
                            check=f"disposition_matrix.{name}.{field}",
                            detail=(
                                f"expected {value!r}, observed "
                                f"{getattr(observation, field)!r}"
                            ),
                        )
                    )
            if observation.confirmed_frontier_changed:
                findings.append(
                    AttributionGateFindingV13(
                        severity="P0",
                        check=f"disposition_matrix.{name}.confirmed_frontier",
                        detail="development observation changed Confirmed Frontier",
                    )
                )
            if observation.prompt_projection_keys != self._PROMPT_KEYS:
                findings.append(
                    AttributionGateFindingV13(
                        severity="P1",
                        check=f"disposition_matrix.{name}.prompt_projection",
                        detail="prompt projection is not the closed two-slot contract",
                    )
                )
            if observation.guard_private_prompt_field_count:
                findings.append(
                    AttributionGateFindingV13(
                        severity="P0",
                        check=f"disposition_matrix.{name}.prompt_privacy",
                        detail="Guard-private data reached a Research prompt",
                    )
                )
            expected_snapshot_delta = (
                0
                if name
                in {
                    "ALL_PRE_BLOCKED",
                    "PRE_CONTRACT_FAILURE",
                    "COMMON_EXECUTION_FAILURE",
                }
                else 1
            )
            if observation.evidence_snapshot_delta != expected_snapshot_delta:
                findings.append(
                    AttributionGateFindingV13(
                        severity="P1",
                        check=f"disposition_matrix.{name}.evidence_snapshot",
                        detail=(
                            f"expected delta {expected_snapshot_delta}, observed "
                            f"{observation.evidence_snapshot_delta}"
                        ),
                    )
                )

        findings.extend(
            self._boolean_findings(
                "research", self._RESEARCH_CHECKS, research_evidence
            )
        )
        findings.extend(
            self._boolean_findings(
                "original", self._ORIGINAL_CHECKS, original_evidence
            )
        )
        findings.extend(
            self._boolean_findings(
                "analysis", self._ANALYSIS_CHECKS, analysis_evidence
            )
        )
        findings.extend(
            self._boolean_findings(
                "fake_e2e", self._E2E_CHECKS, fake_e2e_evidence
            )
        )
        p0 = sum(item.severity == "P0" for item in findings)
        p1 = sum(item.severity == "P1" for item in findings)
        p2 = sum(item.severity == "P2" for item in findings)
        evidence = {
            "analysis": canonical_value(analysis_evidence),
            "dispositions": [item.to_dict() for item in dispositions],
            "fake_e2e": canonical_value(fake_e2e_evidence),
            "original": canonical_value(original_evidence),
            "research": canonical_value(research_evidence),
        }
        checked = tuple(
            [f"disposition_matrix.{name}" for name in REQUIRED_DISPOSITIONS_V13]
            + [f"research.{name}" for name in self._RESEARCH_CHECKS]
            + [f"original.{name}" for name in self._ORIGINAL_CHECKS]
            + [f"analysis.{name}" for name in self._ANALYSIS_CHECKS]
            + [f"fake_e2e.{name}" for name in self._E2E_CHECKS]
        )
        return PilotScientificAttributionGateResultV13(
            verdict="PASS" if p0 == 0 and p1 == 0 else "FAIL",
            p0=p0,
            p1=p1,
            p2=p2,
            disposition_digests=tuple(
                (name, by_name[name].digest)
                for name in REQUIRED_DISPOSITIONS_V13
                if name in by_name
            ),
            checked_invariants=checked,
            findings=tuple(findings),
            evidence_digest=sha256_digest(evidence),
        )


__all__ = [
    "AttributionDispositionObservationV13",
    "AttributionGateFindingV13",
    "PilotScientificAttributionGateResultV13",
    "PilotScientificAttributionGateV13",
    "REQUIRED_DISPOSITIONS_V13",
]
