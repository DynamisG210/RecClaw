"""Q1 DEVELOPMENT_ONLY Idea/OpenSpec quality calibration contracts.

The module adds one enriched projection to the existing OpenSpec path and a
small, outcome-blind A/B contract.  Runtime orchestration is intentionally
kept separate from these offline contracts.
"""

from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

from .canonical import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
)
from .fresh_r1 import (
    AVAILABLE_DEPENDENCIES,
    BUDGET_LIMITS,
    MODEL,
    PROTOCOL_REQUIREMENTS,
    _materialize_and_qualify,
    _shared_policy,
    _write_new_json,
    bounded_provider_call,
    render_implementation_prompt,
)
from .fresh_r2 import (
    _r2_bindings,
    _r2_environment,
    build_active_r2_profile,
    build_r1_registry,
    derive_fresh_r2_proposal_schema,
    load_registered_r1_artifacts,
    public_active_profile_catalog,
)
from .innovation_spine import build_shared_implementer_request
from .open_spec import project_open_producer_draft, resolve_capability
from .quality_calibration import _structural_calibration_unit_check
from .resource_scheduling import (
    PROBE_EPOCHS,
    PROBE_TIMEOUT_SECONDS,
    Q0R_V2_PREFIX_CONTRACT_SHA256,
    structural_features,
)
from .v4_response_contract import validate_v4_response_contract
from .vnext_contracts import (
    CapabilityResolutionResultV1,
    OpenResearchSpecV1,
    QualificationStatusV1,
    RealizationModeV1,
)


Q1_MODEL = MODEL
Q1_PROPOSAL_TOKEN_CEILING = 12_000
Q1_IMPLEMENTATION_TOKEN_CEILING = 20_000
Q1_CANDIDATE_SLOTS = ("diagnosis", "frontier")
Q1_SLOT_MODES = {
    "diagnosis": "DIAGNOSIS_DRIVEN",
    "frontier": "FRONTIER_HYPOTHESIS",
}
Q1_BRANCH = "feat/research-line-idea-openspec-quality"
Q1_BASE_COMMIT = "278391ab47c78508af211978045ad5573d6fc135"
Q1_BASE_PARENT = "5372da07829c92d73b530855fded95de4f57059b"
Q1_BASE_TREE = "4ef40aa98ddb3ddd1020c2a8ffc7b0dee0752edf"
Q0R2_CANONICAL_SHA256 = (
    "3db80f559e474687f70cdc7aa0eeb0487c4b12fb7053444cfcc35a5d1252f4be"
)
Q0R2_PHYSICAL_SHA256 = (
    "f8ec201e6a98dd9fb9eea4b95f680ea6eb3f5939235d14e4ddf86f7cca362c56"
)
Q1_RUN_IDENTITY = "q1-idea-openspec-quality-v1-interface-fix1"
Q1_CONTEXT_REF = "recclaw-q1-accepted-research-context-v1"
Q1_PROPOSAL_SEEDS = {"diagnosis": 55011, "frontier": 55012}
Q1_QUALIFICATION_SEED = 55101
Q1_SLOT_ROLES = {
    "diagnosis": "falsification_designer",
    "frontier": "frontier_architect",
}
Q1_CONDITIONAL_SENTINEL_PROMPT = (
    "Return exactly one JSON object matching the supplied schema. Use schema "
    "recclaw.q1-provider-contract-sentinel.v1 and exactly one proposals item "
    "with sentinel_mode A and sentinel_observation PRESENT. This is a synthetic "
    "transport-contract sentinel; do not add prose or any other fields."
)


class IdeaQualityError(RuntimeError):
    """Raised when the Q1 fairness or pre-outcome contract is violated."""


def derive_enriched_proposal_schema() -> dict[str, Any]:
    """Extend the current real Provider schema with only Q1 scientific fields."""

    schema = copy.deepcopy(derive_fresh_r2_proposal_schema())
    proposal = schema["properties"]["proposals"]["items"]
    properties = proposal["properties"]
    additions = {
        "idea_mode": {
            "enum": ["DIAGNOSIS_DRIVEN", "FRONTIER_HYPOTHESIS"],
            "type": "string",
        },
        "research_question": {"minLength": 1, "type": "string"},
        "observed_failure_mode": {"type": ["string", "null"]},
        "closest_parent": {"minLength": 1, "type": "string"},
        "minimal_testable_wedge": {"minLength": 1, "type": "string"},
        "causal_chain": {
            "items": {"minLength": 1, "type": "string"},
            "minItems": 1,
            "type": "array",
        },
        "discriminative_predictions": {
            "items": {"minLength": 1, "type": "string"},
            "minItems": 1,
            "type": "array",
        },
        "mechanism_off_definition": {"minLength": 1, "type": "string"},
        "resource_hypothesis": {"minLength": 1, "type": "string"},
        "realization_mode": {
            "enum": ["PARENT_PRESERVING", "NON_NESTED"],
            "type": "string",
        },
    }
    overlap = set(properties) & set(additions)
    if overlap:
        raise IdeaQualityError(f"enriched schema duplicates existing fields: {overlap}")
    properties.update(additions)
    proposal["required"] = list(proposal["required"]) + list(additions)
    proposal["allOf"] = [
        {
            "if": {
                "properties": {"idea_mode": {"const": "DIAGNOSIS_DRIVEN"}},
                "required": ["idea_mode"],
            },
            "then": {
                "properties": {
                    "observed_failure_mode": {"minLength": 1, "type": "string"}
                }
            },
        },
        {
            "if": {
                "properties": {"idea_mode": {"const": "FRONTIER_HYPOTHESIS"}},
                "required": ["idea_mode"],
            },
            "then": {
                "properties": {
                    "observed_failure_mode": {"enum": [None, "NOT_OBSERVED"]}
                }
            },
        },
    ]
    schema["title"] = "RecClaw Q1 Enriched OpenSpec Proposal Response"
    jsonschema.validators.validator_for(schema).check_schema(schema)
    return canonical_value(schema)


def derive_q1_conditional_contract_sentinel_schema() -> dict[str, Any]:
    """Return a tiny non-research schema isolating Q1 conditional composition."""

    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Q1 Provider Conditional Contract Sentinel",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema": {
                "type": "string",
                "enum": ["recclaw.q1-provider-contract-sentinel.v1"],
            },
            "proposals": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "sentinel_mode": {
                            "type": "string",
                            "enum": ["A", "B"],
                        },
                        "sentinel_observation": {
                            "type": ["string", "null"],
                        },
                    },
                    "required": ["sentinel_mode", "sentinel_observation"],
                    "allOf": [
                        {
                            "if": {
                                "properties": {
                                    "sentinel_mode": {"const": "A"}
                                },
                                "required": ["sentinel_mode"],
                            },
                            "then": {
                                "properties": {
                                    "sentinel_observation": {
                                        "type": "string",
                                        "minLength": 1,
                                    }
                                }
                            },
                        },
                        {
                            "if": {
                                "properties": {
                                    "sentinel_mode": {"const": "B"}
                                },
                                "required": ["sentinel_mode"],
                            },
                            "then": {
                                "properties": {
                                    "sentinel_observation": {
                                        "enum": [None, "NOT_OBSERVED"]
                                    }
                                }
                            },
                        },
                    ],
                },
            },
        },
        "required": ["schema", "proposals"],
    }
    jsonschema.validators.validator_for(schema).check_schema(schema)
    return canonical_value(schema)


def q1_provider_contract_static_matrix() -> dict[str, Any]:
    """Describe the bounded structural hypotheses behind the observed HTTP 400."""

    baseline = derive_fresh_r2_proposal_schema()
    enriched = derive_enriched_proposal_schema()
    sentinel = derive_q1_conditional_contract_sentinel_schema()
    baseline_proposal = baseline["properties"]["proposals"]["items"]
    enriched_proposal = enriched["properties"]["proposals"]["items"]
    sentinel_proposal = sentinel["properties"]["proposals"]["items"]
    return canonical_value(
        {
            "schema": "recclaw.research-line.q1-provider-contract-matrix.v1",
            "observed": {
                "baseline_http_statuses": [200, 200],
                "enriched_http_statuses": [400, 400],
                "provider_error_body_persisted": False,
            },
            "invariants": {
                "model": Q1_MODEL,
                "tools": [],
                "token_ceiling": Q1_PROPOSAL_TOKEN_CEILING,
                "strict": True,
                "temperature": 0.0,
                "response_format": "json_schema",
            },
            "schemas": {
                "baseline": {
                    "canonical_bytes": len(canonical_json_bytes(baseline)),
                    "proposal_properties": len(baseline_proposal["properties"]),
                    "proposal_required": len(baseline_proposal["required"]),
                    "conditional_composition": False,
                },
                "enriched": {
                    "canonical_bytes": len(canonical_json_bytes(enriched)),
                    "proposal_properties": len(enriched_proposal["properties"]),
                    "proposal_required": len(enriched_proposal["required"]),
                    "conditional_composition": True,
                    "unique_keywords": ["allOf", "const", "if", "then"],
                },
                "sentinel": {
                    "canonical_bytes": len(canonical_json_bytes(sentinel)),
                    "proposal_properties": len(sentinel_proposal["properties"]),
                    "proposal_required": len(sentinel_proposal["required"]),
                    "conditional_composition": True,
                    "research_semantics": False,
                },
            },
            "hypotheses": [
                {
                    "id": "H_CONDITIONAL_COMPOSITION_UNSUPPORTED",
                    "prediction": "the tiny conditional sentinel returns HTTP 400",
                    "single_probe_budget": 1,
                },
                {
                    "id": "H_SCHEMA_SIZE_OR_PROPERTY_COUNT",
                    "prediction": "the tiny conditional sentinel is accepted",
                    "single_probe_budget": 0,
                },
            ],
            "decision_rule": (
                "HTTP_400 confirms the conditional-composition boundary independent "
                "of schema size/property count; HTTP_200 leaves the root cause unresolved"
            ),
            "research_candidate_generation_allowed": False,
            "held_out_reads": 0,
        }
    )


def build_q1_ab_contract() -> dict[str, Any]:
    """Freeze the smallest symmetric baseline-vs-enriched comparison."""

    return canonical_value(
        {
            "schema": "recclaw.research-line.q1-idea-quality-ab-contract.v1",
            "development_only": True,
            "model": Q1_MODEL,
            "tools": [],
            "proposal_token_ceiling": Q1_PROPOSAL_TOKEN_CEILING,
            "implementation_token_ceiling": Q1_IMPLEMENTATION_TOKEN_CEILING,
            "candidate_slots": list(Q1_CANDIDATE_SLOTS),
            "slot_modes": Q1_SLOT_MODES,
            "candidate_count_per_arm": len(Q1_CANDIDATE_SLOTS),
            "selection_budget_per_arm": 1,
            "inputs_symmetric_except_contract_enrichment": True,
            "origin_blind": True,
            "selection_frozen_before_implementation_or_qualification": True,
            "selection_features": [
                "discriminative_value",
                "mechanism_off_executability",
                "parent_clarity",
                "q0r2_resource_feasibility",
                "scientific_testability",
            ],
            "forbidden_selection_features": [
                "implementation_success",
                "ndcg_or_other_effect_metric",
                "qualification_result",
            ],
            "selection_rule": (
                "highest total pre-outcome score, then diagnosis before frontier, "
                "then lexical spec digest"
            ),
            "held_out_reads": 0,
            "permanent_mode_ratio": None,
        }
    )


def _contract_instruction(arm: str, slot: str) -> str:
    if arm == "baseline":
        return (
            "Use only the current OpenSpec fields in the supplied response schema. "
            "For the diagnosis slot, bind the hypothesis to the accepted context gap. "
            "For the frontier slot, pose one open structural recommender hypothesis."
        )
    if arm == "enriched":
        return (
            "Use the enriched OpenSpec fields. Set idea_mode to "
            f"{Q1_SLOT_MODES[slot]}. State research_question, closest_parent, "
            "minimal_testable_wedge, causal_chain, a prediction that distinguishes the "
            "named competing explanation, mechanism_off_definition, resource_hypothesis, "
            "and realization_mode. DIAGNOSIS_DRIVEN must cite a real accepted failure or "
            "gap from context; FRONTIER_HYPOTHESIS must use null or NOT_OBSERVED and must "
            "not invent a failure. Parent-preserving is preferred only when truthful."
        )
    raise IdeaQualityError(f"unknown Q1 contract arm: {arm}")


def render_q1_producer_prompt(
    template: str,
    *,
    arm: str,
    slot: str,
    role: str,
    seed: int,
    context: Mapping[str, Any],
    profile_catalog: Sequence[Mapping[str, str]],
    protocol_ref: str,
    protocol_digest: str,
    context_ref: str,
    context_digest: str,
    profile_ref: str,
    profile_digest: str,
) -> str:
    """Render paired prompts without revealing comparison origin or outcomes."""

    if slot not in Q1_CANDIDATE_SLOTS:
        raise IdeaQualityError(f"unknown Q1 slot: {slot}")
    replacements = {
        "{{LOGICAL_SLOT_ID}}": slot,
        "{{PROPOSAL_SEED}}": str(seed),
        "{{PRODUCER_ROLE}}": role,
        "{{CONTRACT_INSTRUCTION}}": _contract_instruction(arm, slot),
        "{{RESEARCH_CONTEXT_JSON}}": json.dumps(
            context, sort_keys=True, separators=(",", ":")
        ),
        "{{PROFILE_CATALOG_JSON}}": json.dumps(
            profile_catalog, sort_keys=True, separators=(",", ":")
        ),
        "{{PROTOCOL_REF}}": protocol_ref,
        "{{PROTOCOL_DIGEST}}": protocol_digest,
        "{{CONTEXT_REF}}": context_ref,
        "{{CONTEXT_DIGEST}}": context_digest,
        "{{PROFILE_REF}}": profile_ref,
        "{{PROFILE_DIGEST}}": profile_digest,
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    if re.search(r"\{\{[A-Z][A-Z0-9_]*\}\}", rendered):
        raise IdeaQualityError("Q1 Producer prompt has unresolved placeholder")
    forbidden = (
        "side_a",
        "side_b",
        "expected winner",
        "implementation success",
        "candidate_ndcg",
        "observed_ndcg",
        "qualification result",
    )
    if any(token in rendered.lower() for token in forbidden):
        raise IdeaQualityError("Q1 Producer prompt leaked origin or outcome")
    return rendered


def score_preoutcome_testability(
    spec: OpenResearchSpecV1,
    *,
    q0r2_resource_feasible: bool,
    outcome_features: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score only scientific clarity and resource feasibility before outcomes."""

    if outcome_features:
        raise ValueError("outcome or implementation features are forbidden in Q1 selection")
    features = {
        "scientific_testability": int(bool(spec.falsifier and spec.expected_evidence)),
        "parent_clarity": int(bool(spec.closest_parent or spec.matched_control_requirement)),
        "discriminative_value": int(
            bool(spec.discriminative_predictions or spec.competing_explanation)
        ),
        "mechanism_off_executability": int(
            bool(spec.mechanism_off_definition or spec.matched_control_requirement)
        ),
        "q0r2_resource_feasibility": int(q0r2_resource_feasible),
    }
    return canonical_value(
        {
            "features": features,
            "total": sum(features.values()),
            "outcome_fields_consumed": [],
            "spec_digest": spec.digest,
            "score_digest": sha256_digest(features),
        }
    )


def build_research_context(repo_root: Path) -> dict[str, Any]:
    """Project accepted evidence without held-out data or effect authority."""

    q0r2_path = (
        repo_root
        / "docs/research_line/vnext/Q0R2_RESOURCE_ADMISSION_CANONICAL_RECEIPT.json"
    )
    if bytes_sha256(q0r2_path.read_bytes()) != Q0R2_CANONICAL_SHA256:
        raise IdeaQualityError("accepted Q0R2 receipt byte drift")
    q0r2 = json.loads(q0r2_path.read_text(encoding="utf-8"))
    if (
        q0r2.get("status") != "PASS"
        or q0r2.get("held_out_reads") != 0
        or q0r2.get("effect_untouched", {}).get("mechanism_effect_updates") != 0
        or q0r2.get("evaluation", {}).get("q1_allowed") is not True
    ):
        raise IdeaQualityError("accepted Q0R2 context contract drift")
    results = q0r2["fresh_results"]
    return canonical_value(
        {
            "schema": "recclaw.research-line.q1-accepted-context.v1",
            "evidence_class": "DEVELOPMENT_ONLY",
            "accepted_q0r2_receipt_sha256": Q0R2_CANONICAL_SHA256,
            "accepted_q0r2_physical_sha256": Q0R2_PHYSICAL_SHA256,
            "observed_research_gap": (
                "Accepted frontier and known-good packages were qualified but resource-"
                "deferred without mechanism-effect updates; earlier fresh realizations "
                "also lacked an explicit isolatable wedge and executable mechanism-off."
            ),
            "resource_observations": {
                "matched_control": {
                    "exit_status": results["matched_bpr_control"]["exit_status"],
                    "deadline_seconds": results["matched_bpr_control"][
                        "deadline_seconds"
                    ],
                },
                "parent_equivalent": {
                    "exit_status": results["parent_equivalent_null"]["exit_status"],
                    "deadline_seconds": results["parent_equivalent_null"][
                        "deadline_seconds"
                    ],
                },
                "frontier": {
                    "exit_status": results["frontier_candidate"]["exit_status"],
                    "resource_disposition": results["frontier_candidate"][
                        "missingness"
                    ]["resource_disposition"],
                },
                "known_good": {
                    "exit_status": results["known_good_reference"]["exit_status"],
                    "resource_disposition": results["known_good_reference"][
                        "missingness"
                    ]["resource_disposition"],
                },
            },
            "allowed_dependencies": list(AVAILABLE_DEPENDENCIES),
            "budget_limits": BUDGET_LIMITS,
            "protocol_requirements": list(PROTOCOL_REQUIREMENTS),
            "effect_fields_consumed": [],
            "mechanism_effect_updates": 0,
            "held_out_reads": 0,
        }
    )


def _provider_usage(groups: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, int]:
    attempts = [attempt for group in groups for attempt in group]
    successes = [row for row in attempts if row.get("status") == "SUCCESS"]
    return {
        "physical_calls": len(attempts),
        "successful_calls": len(successes),
        "retries": max(0, len(attempts) - len(groups)),
        "input_tokens": sum(int(row.get("input_tokens", 0)) for row in successes),
        "output_tokens": sum(int(row.get("output_tokens", 0)) for row in successes),
        "billed_tokens": sum(int(row.get("billed_tokens", 0)) for row in successes),
        "provider_wall_time_ms": sum(
            int(row.get("latency_ms", 0)) for row in attempts
        ),
    }


def _q1_unit_check(spec: OpenResearchSpecV1):
    def factory(evidence: dict[str, Any]):
        base = _structural_calibration_unit_check(evidence)

        def check(model: Any, config: Any, dataset: Any) -> None:
            base(model, config, dataset)
            if spec.realization_mode is None:
                evidence["realization_conformance"] = "LEGACY_BASELINE_NOT_DECLARED"
                return
            if spec.realization_mode is RealizationModeV1.NON_NESTED:
                evidence["realization_conformance"] = "PASS_NON_NESTED_DIRECT"
                evidence["mechanism_off_execution"] = "MATCHED_PARENT_PACKAGE_REQUIRED"
                return
            import torch

            from recbole.data.interaction import Interaction
            from recbole.model.general_recommender.bpr import BPR

            switch = getattr(model, "set_mechanism_enabled", None)
            if not callable(switch):
                raise AssertionError(
                    "PARENT_PRESERVING realization lacks set_mechanism_enabled"
                )
            baseline = BPR(config, dataset).to(config["device"])
            with torch.no_grad():
                baseline.user_embedding.weight.copy_(model.user_embedding.weight)
                baseline.item_embedding.weight.copy_(model.item_embedding.weight)
            pair = Interaction(
                {
                    model.USER_ID: torch.tensor([1, 2], device=config["device"]),
                    model.ITEM_ID: torch.tensor([1, 2], device=config["device"]),
                    model.NEG_ITEM_ID: torch.tensor([2, 3], device=config["device"]),
                }
            )
            users = Interaction(
                {model.USER_ID: torch.tensor([1, 2], device=config["device"])}
            )
            switch(False)
            model.eval()
            baseline.eval()
            with torch.no_grad():
                predict_delta = float(
                    torch.max(
                        torch.abs(model.predict(pair) - baseline.predict(pair))
                    ).item()
                )
                full_sort_delta = float(
                    torch.max(
                        torch.abs(
                            model.full_sort_predict(users)
                            - baseline.full_sort_predict(users)
                        )
                    ).item()
                )
            model.train()
            baseline.train()
            candidate_loss = model.calculate_loss(pair)
            baseline_loss = baseline.calculate_loss(pair)
            candidate_values = (
                candidate_loss if isinstance(candidate_loss, tuple) else (candidate_loss,)
            )
            baseline_values = (
                baseline_loss if isinstance(baseline_loss, tuple) else (baseline_loss,)
            )
            loss_delta = float(
                torch.abs(
                    sum(value.reshape(()) for value in candidate_values).detach()
                    - sum(value.reshape(()) for value in baseline_values).detach()
                ).item()
            )
            switch(True)
            if max(predict_delta, full_sort_delta, loss_delta) > 1e-7:
                raise AssertionError(
                    "PARENT_PRESERVING mechanism-off does not recover parent behavior"
                )
            evidence.update(
                {
                    "realization_conformance": "PASS_PARENT_PRESERVING",
                    "mechanism_off_execution": "PASS",
                    "mechanism_off_predict_max_abs_delta": predict_delta,
                    "mechanism_off_full_sort_max_abs_delta": full_sort_delta,
                    "mechanism_off_loss_abs_delta": loss_delta,
                }
            )

        return check

    return factory


def project_q1_resource_admission(
    *,
    spec: OpenResearchSpecV1,
    resolution: Any,
    qualification_status: str,
    candidate_package_digest: str,
    source_path: Path,
) -> dict[str, Any]:
    """Admit one qualified package to a Q0R2-style resource-only probe plan."""

    source = structural_features(source_path)
    eligible = (
        qualification_status == "PASS"
        and resolution.protocol_compatible
        and resolution.dependency_compatible
        and resolution.budget_compatible
    )
    schedule = []
    if eligible:
        schedule.append(
            {
                "ordinal": 1,
                "action": "Q0R2_FIXED_BATCH_PREFIX_RESOURCE_PROBE",
                "candidate_package_digest": candidate_package_digest,
                "epochs": PROBE_EPOCHS,
                "deadline_seconds": PROBE_TIMEOUT_SECONDS,
                "prefix_contract_sha256": Q0R_V2_PREFIX_CONTRACT_SHA256,
                "effect_update_allowed": False,
            }
        )
    return canonical_value(
        {
            "schema": "recclaw.research-line.q1-resource-only-admission.v1",
            "status": "RESOURCE_PROBE_ADMITTED" if schedule else "RESOURCE_DEFERRED",
            "resource_hypothesis": spec.resource_hypothesis,
            "source_features": source,
            "schedule": schedule,
            "outcome_fields_consumed": [],
            "effect_fields_consumed": [],
            "mechanism_effect_updates": 0,
            "resource_censor_updates_effect": False,
            "held_out_reads": 0,
        }
    )


def run_idea_quality(repo_root: Path, *, run_root: Path) -> dict[str, Any]:
    """Execute the one-shot real Q1 Producer-to-resource-plan chain."""

    repo_root = repo_root.resolve()
    run_root = run_root.resolve()
    if run_root.exists():
        raise IdeaQualityError(f"Q1 root already exists: {run_root}")
    started_ns = time.monotonic_ns()
    context = build_research_context(repo_root)
    artifacts, _r1_receipt = load_registered_r1_artifacts(repo_root)
    registry = build_r1_registry(artifacts)
    _current, _manifest, _next, _build, active = build_active_r2_profile(registry)
    catalog = public_active_profile_catalog(active, artifacts, seed=55000)
    ab_contract = build_q1_ab_contract()
    context_digest = sha256_digest(context)
    bindings = canonical_value(
        {
            **_r2_bindings(active),
            "context_ref": Q1_CONTEXT_REF,
            "context_digest": context_digest,
        }
    )
    environment = _r2_environment(active)
    resources = Path(__file__).resolve().parent / "resources"
    producer_template_path = resources / "idea_quality_producer_prompt_v1.txt"
    implementer_template_path = resources / "idea_quality_implementer_prompt_v1.txt"
    implementation_schema_path = resources / "fresh_r1_implementation_response_v1.schema.json"
    tool_policy_path = resources / "fresh_open_spec_tool_policy_v1.json"
    baseline_schema = derive_fresh_r2_proposal_schema()
    enriched_schema = derive_enriched_proposal_schema()

    run_root.mkdir(parents=True)
    _write_new_json(run_root / "AB_CONTRACT_BEFORE_PROVIDER.json", ab_contract)
    _write_new_json(run_root / "RESEARCH_CONTEXT.json", context)
    schema_paths = {
        "baseline": run_root / "contracts/baseline.schema.json",
        "enriched": run_root / "contracts/enriched.schema.json",
    }
    _write_new_json(schema_paths["baseline"], baseline_schema)
    _write_new_json(schema_paths["enriched"], enriched_schema)

    pools: dict[str, list[dict[str, Any]]] = {"baseline": [], "enriched": []}
    live: dict[tuple[str, str], tuple[OpenResearchSpecV1, Any]] = {}
    proposal_attempt_groups: list[Sequence[Mapping[str, Any]]] = []
    template = producer_template_path.read_text(encoding="utf-8")
    for arm in ("baseline", "enriched"):
        schema = baseline_schema if arm == "baseline" else enriched_schema
        for slot in Q1_CANDIDATE_SLOTS:
            prompt = render_q1_producer_prompt(
                template,
                arm=arm,
                slot=slot,
                role=Q1_SLOT_ROLES[slot],
                seed=Q1_PROPOSAL_SEEDS[slot],
                context=context,
                profile_catalog=catalog,
                protocol_ref=active.protocol_ref,
                protocol_digest=active.protocol_digest,
                context_ref=Q1_CONTEXT_REF,
                context_digest=context_digest,
                profile_ref=active.profile_ref,
                profile_digest=active.profile_digest,
            )
            call = bounded_provider_call(
                call_root=run_root / "provider/proposals" / arm / slot,
                schema_path=schema_paths[arm],
                logical_call_id=f"{Q1_RUN_IDENTITY}:{slot}:proposal",
                session_id=f"{Q1_RUN_IDENTITY}:{slot}:paired-proposal-session",
                prompt=prompt,
                token_ceiling=Q1_PROPOSAL_TOKEN_CEILING,
                maximum_physical_attempts=1,
            )
            proposal_attempt_groups.append(call.attempts)
            if call.call is None:
                pools[arm].append(
                    {
                        "slot": slot,
                        "stage": "PROPOSAL_PROVIDER_FAILURE",
                        "failure": call.failure,
                        "provider_attempts": call.attempts,
                    }
                )
                continue
            validate_v4_response_contract(call.call.response, provider_schema=schema)
            draft = call.call.response["proposals"][0]
            if draft["producer_role"] != Q1_SLOT_ROLES[slot]:
                raise IdeaQualityError("Provider changed paired Producer role")
            if arm == "enriched" and draft["idea_mode"] != Q1_SLOT_MODES[slot]:
                raise IdeaQualityError("Provider changed frozen enriched idea mode")
            spec, facts = project_open_producer_draft(draft, bindings=bindings)
            resolution = resolve_capability(
                spec, resolution_facts=facts, environment=environment
            )
            feasible = resolution.resolution in {
                CapabilityResolutionResultV1.SEARCH_READY,
                CapabilityResolutionResultV1.INNOVATION_REQUIRED,
            }
            score = score_preoutcome_testability(
                spec, q0r2_resource_feasible=feasible
            )
            record = canonical_value(
                {
                    "slot": slot,
                    "producer_role": Q1_SLOT_ROLES[slot],
                    "proposal_seed": Q1_PROPOSAL_SEEDS[slot],
                    "provider_attempts": call.attempts,
                    "proposal_response_digest": call.call.response_digest,
                    "research_spec": spec.canonical_dict(),
                    "resolution_facts": facts,
                    "resolution": resolution.canonical_dict(),
                    "preoutcome_score": score,
                    "manual_candidate_patches": 0,
                    "stage": "OPENSPEC_FROZEN",
                }
            )
            pools[arm].append(record)
            live[(arm, slot)] = (spec, resolution)

    if any(len(rows) != len(Q1_CANDIDATE_SLOTS) for rows in pools.values()) or any(
        row.get("stage") != "OPENSPEC_FROZEN"
        for rows in pools.values()
        for row in rows
    ):
        raise IdeaQualityError("Q1 candidate pool did not freeze completely")
    selected: dict[str, dict[str, Any]] = {}
    for arm, rows in pools.items():
        selected[arm] = sorted(
            rows,
            key=lambda row: (
                -int(row["preoutcome_score"]["total"]),
                Q1_CANDIDATE_SLOTS.index(str(row["slot"])),
                str(row["research_spec"]["schema"]),
                sha256_digest(row["research_spec"]),
            ),
        )[0]
    frozen_selection = canonical_value(
        {
            "schema": "recclaw.research-line.q1-frozen-selection.v1",
            "candidate_pools": pools,
            "selected": {
                arm: {
                    "slot": row["slot"],
                    "spec_digest": sha256_digest(row["research_spec"]),
                    "score": row["preoutcome_score"],
                }
                for arm, row in selected.items()
            },
            "selection_rule": ab_contract["selection_rule"],
            "implementation_or_qualification_outcomes_present_when_written": 0,
            "outcome_fields_consumed": [],
            "held_out_reads": 0,
        }
    )
    frozen_selection_digest = _write_new_json(
        run_root / "FROZEN_SELECTION_BEFORE_IMPLEMENTATION.json", frozen_selection
    )

    implementer_template = implementer_template_path.read_text(encoding="utf-8")
    policy = _shared_policy(
        bytes_sha256(implementer_template_path.read_bytes()),
        bytes_sha256(tool_policy_path.read_bytes()),
    )
    implementation_attempt_groups: list[Sequence[Mapping[str, Any]]] = []
    selected_results: dict[str, dict[str, Any]] = {}
    for arm in ("baseline", "enriched"):
        slot = str(selected[arm]["slot"])
        spec, resolution = live[(arm, slot)]
        request = build_shared_implementer_request(spec, policy=policy)
        prompt = render_implementation_prompt(implementer_template, request)
        call = bounded_provider_call(
            call_root=run_root / "provider/implementations" / arm,
            schema_path=implementation_schema_path,
            logical_call_id=f"{Q1_RUN_IDENTITY}:{slot}:implementation",
            session_id=f"{Q1_RUN_IDENTITY}:paired-origin-blind-implementation-session",
            prompt=prompt,
            token_ceiling=Q1_IMPLEMENTATION_TOKEN_CEILING,
            maximum_physical_attempts=1,
        )
        implementation_attempt_groups.append(call.attempts)
        if call.call is None:
            selected_results[arm] = {
                "stage": "IMPLEMENTATION_PROVIDER_FAILURE",
                "failure": call.failure,
                "implementation_provider_attempts": call.attempts,
            }
            continue
        materialized, qualification, behavior = _materialize_and_qualify(
            repo_root=repo_root,
            side_root=run_root / "selected" / arm,
            slot_id=slot,
            seed=Q1_QUALIFICATION_SEED,
            spec=spec,
            implementation=call.call.response["proposals"][0],
            implementation_prompt_digest=bytes_sha256(
                implementer_template_path.read_bytes()
            ),
            tool_policy_digest=bytes_sha256(tool_policy_path.read_bytes()),
            run_identity=Q1_RUN_IDENTITY,
            unit_check_factory=_q1_unit_check(spec),
        )
        candidate_root = (
            run_root
            / "selected"
            / arm
            / "candidates"
            / slot
            / str(materialized.shared_request["blind_candidate_id"])
        )
        source_path = candidate_root / "recclaw_ext/candidate.py"
        qualification_payload = {
            **qualification.to_dict(),
            "behavioral_evidence": behavior,
        }
        _write_new_json(
            run_root / "selected" / arm / "qualification.json",
            qualification_payload,
        )
        _write_new_json(
            run_root / "selected" / arm / "candidate_package.json",
            materialized.to_dict(),
        )
        resource_admission = project_q1_resource_admission(
            spec=spec,
            resolution=resolution,
            qualification_status=qualification.receipt.status.value,
            candidate_package_digest=materialized.package.digest,
            source_path=source_path,
        )
        _write_new_json(
            run_root / "selected" / arm / "resource_admission.json",
            resource_admission,
        )
        selected_results[arm] = canonical_value(
            {
                "slot": slot,
                "spec_ref": spec.spec_id,
                "spec_digest": spec.digest,
                "resolution": resolution.resolution.value,
                "resolution_digest": resolution.digest,
                "implementation_provider_attempts": call.attempts,
                "fresh_provider_implementation": True,
                "candidate_root": str(candidate_root),
                "candidate_package_digest": materialized.package.digest,
                "candidate_source_tree_digest": materialized.package.source_tree_digest,
                "entrypoint": materialized.package.executable_entrypoint,
                "qualification_status": qualification.receipt.status.value,
                "qualification_receipt_digest": qualification.receipt.digest,
                "behavioral_evidence": behavior,
                "resource_admission": resource_admission,
                "manual_candidate_patches": 0,
                "effect_update_count": 0,
                "stage": (
                    "RESOURCE_PLAN_ADMITTED"
                    if resource_admission["schedule"]
                    else "RESOURCE_DEFERRED"
                ),
            }
        )

    proposal_usage = _provider_usage(proposal_attempt_groups)
    implementation_usage = _provider_usage(implementation_attempt_groups)
    enriched_modes = sorted(
        row["research_spec"]["idea_mode"] for row in pools["enriched"]
    )
    enriched_selected = selected_results.get("enriched", {})
    gates = {
        "function_real_and_runnable": all(
            row.get("qualification_status") == QualificationStatusV1.PASS.value
            for row in selected_results.values()
        ),
        "end_to_end_result_chain_real_and_valid": all(
            bool(row.get("resource_admission", {}).get("schedule"))
            for row in selected_results.values()
        ),
        "serves_open_algorithm_research_target": (
            enriched_modes
            == ["DIAGNOSIS_DRIVEN", "FRONTIER_HYPOTHESIS"]
            and bool(enriched_selected.get("behavioral_evidence"))
        ),
        "no_fixed_66_tuning_static_wrapper_fallback_mock_or_smoke_substitution": all(
            row.get("fresh_provider_implementation") is True
            and row.get("manual_candidate_patches") == 0
            and row.get("behavioral_evidence", {}).get("probe_status", "").startswith(
                "PASS"
            )
            for row in selected_results.values()
        ),
    }
    retry_count = proposal_usage["retries"] + implementation_usage["retries"]
    status = "PASS_DEVELOPMENT_ONLY" if all(gates.values()) else "HARD_BLOCK"
    receipt = canonical_value(
        {
            "schema": "recclaw.research-line.q1-idea-openspec-quality-canonical-receipt.v1",
            "status": status,
            "development_only": True,
            "input_identity": {
                "accepted_q0r2_commit": Q1_BASE_COMMIT,
                "accepted_q0r2_parent": Q1_BASE_PARENT,
                "accepted_q0r2_tree": Q1_BASE_TREE,
                "accepted_q0r2_receipt_sha256": Q0R2_CANONICAL_SHA256,
                "branch": Q1_BRANCH,
                "source_files": {
                    path.relative_to(repo_root).as_posix(): bytes_sha256(path.read_bytes())
                    for path in (
                        Path(__file__),
                        producer_template_path,
                        implementer_template_path,
                    )
                },
            },
            "ab_contract": ab_contract,
            "research_context_digest": context_digest,
            "frozen_selection_sha256": frozen_selection_digest,
            "selected_results": selected_results,
            "proposal_provider_usage": proposal_usage,
            "implementation_provider_usage": implementation_usage,
            "retry_count": retry_count,
            "manual_candidate_patches": 0,
            "preoutcome_interface_fix": {
                "count": 1,
                "failure_phase": "PROMPT_RENDER_BEFORE_ANY_PROVIDER_CALL",
                "failure_type": "FALSE_POSITIVE_PLACEHOLDER_DETECTION",
                "fix": (
                    "match only unresolved uppercase template tokens instead of "
                    "rejecting double braces inside real research/profile context"
                ),
                "abandoned_root": (
                    "/tmp/recclaw_q1_idea_quality_278391a_20260802_01/run"
                ),
                "candidate_or_selection_changed": False,
                "provider_calls_before_fix": 0,
                "mechanism_negative_evidence": False,
            },
            "held_out_reads": 0,
            "resource_censored_count": 0,
            "resource_censored_mechanism_effect_updates": 0,
            "effect_update_count": 0,
            "architecture_quality_signal": {
                "baseline_current_schema_candidate_count": len(pools["baseline"]),
                "enriched_candidate_count": len(pools["enriched"]),
                "enriched_modes": enriched_modes,
                "enriched_selected_has_explicit_wedge": bool(
                    selected["enriched"]["research_spec"].get(
                        "minimal_testable_wedge"
                    )
                ),
                "enriched_selected_has_executable_mechanism_off": (
                    enriched_selected.get("behavioral_evidence", {}).get(
                        "mechanism_off_execution"
                    )
                    in {"PASS", "MATCHED_PARENT_PACKAGE_REQUIRED"}
                ),
                "scientific_effect": "NOT_ADJUDICATED",
            },
            "evaluation": {
                "gates": gates,
                "formal_scientific_experiment": False,
                "scientific_effect_claim": False,
                "formal_acceptance_self_approved": False,
                "q2_continuation_allowed": status == "PASS_DEVELOPMENT_ONLY",
            },
            "scientific_effect_claim": False,
            "wall_time_ms": max(1, (time.monotonic_ns() - started_ns) // 1_000_000),
        }
    )
    receipt_digest = _write_new_json(run_root / "Q1_CANONICAL_RECEIPT.json", receipt)
    return canonical_value(
        {
            "status": status,
            "canonical_receipt_sha256": receipt_digest,
            "selected_enriched_package_digest": enriched_selected.get(
                "candidate_package_digest"
            ),
            "selected_enriched_resource_plan_nonempty": bool(
                enriched_selected.get("resource_admission", {}).get("schedule")
            ),
            "held_out_reads": 0,
            "effect_update_count": 0,
        }
    )
