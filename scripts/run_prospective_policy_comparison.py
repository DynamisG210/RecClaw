#!/usr/bin/env python3
"""Freeze, run, select, and package the Q4 prospective policy comparison."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    bytes_sha256,
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.fresh_r1 import (  # noqa: E402
    MODEL,
    _write_new_json,
    bounded_provider_call,
)
from recclaw_core.experiments.helix_abc_v1.fresh_r2 import (  # noqa: E402
    _r2_bindings,
    _r2_environment,
    build_active_r2_profile,
    build_r1_registry,
    load_registered_r1_artifacts,
    public_active_profile_catalog,
)
from recclaw_core.experiments.helix_abc_v1.idea_quality import (  # noqa: E402
    Q1_CONTEXT_REF,
    Q1_PROPOSAL_TOKEN_CEILING,
    build_research_context,
    derive_enriched_proposal_schema,
    score_preoutcome_testability,
)
from recclaw_core.experiments.helix_abc_v1.open_meta import (  # noqa: E402
    STATIC_IDEA_POLICY_DIGEST_V1,
    STATIC_IDEA_POLICY_REF_V1,
)
from recclaw_core.experiments.helix_abc_v1.open_meta_f1 import (  # noqa: E402
    ALLOWED_DIRECTIONS,
)
from recclaw_core.experiments.helix_abc_v1.open_spec import (  # noqa: E402
    project_open_producer_draft,
    resolve_capability,
)
from recclaw_core.experiments.helix_abc_v1.prospective_policy_comparison import (  # noqa: E402
    POLICY_ORDER,
    ProspectivePolicyComparisonError,
    build_arm_manifest,
    compute_four_metrics,
    read_object,
    select_current_f1,
    select_outcome_aware,
    select_static,
    verify_file,
    verify_semantic_digest,
)
from recclaw_core.experiments.helix_abc_v1.v4_response_contract import (  # noqa: E402
    validate_v4_response_contract,
)
from recclaw_core.experiments.helix_abc_v1.vnext_contracts import (  # noqa: E402
    CapabilityResolutionResultV1,
)


CAMPAIGN_ID = "q4-prospective-policy-comparison-20260803-01"
ACCEPTED_SOAK_COMMIT = "63e44341bd1da982f00a9db7de2be7e779c7a63d"
ACCEPTED_SOAK_TREE = "af8803eabaa89f56c5d53afbf60bc201187d5556"
ACCEPTED_F1_EVIDENCE_COMMIT = "1d3223104871bac6820ab1a550461bd142591a02"
ACCEPTED_F1_EVIDENCE_TREE = "6ef8660105c99c13f4ef8b359a817ab6d2a120c0"
F1_FILE_SHA256 = {
    "policy/versioned_policy.json": "1bc607a8f0d4e2f266ae471b79c29c0251fdcf8ec37697c127875d6e38dfe88c",
    "activation/activation.json": "7ad0d44da4d5357a36908a82b8edbe775ef347aedf0665e22d111be0da32969a",
    "selection/selection.json": "c4fc92b5b53c1194cf9089ff09155ee578625324c9a625fbcfcc5cab750f44e6",
}
F1_POLICY_DIGEST = "bde1af2801cf2de15de5034fff6f5f39afcb9bb161cb7fd23ec88eb51e4314ca"
F1_ACTIVATION_DIGEST = "71894903a3b0320ddd06e4072e13770329dd0a98e7cb5df04748b2ae6ae4ef0f"
OUTCOME_POLICY_FILE_SHA256 = "105296b6fc2db7f2059f1a52ce309e8f583c27486542a42c7d62000c0ffe98a8"
OUTCOME_ACTIVATION_FILE_SHA256 = "a03c840062d0f70cb4c1a39d0afcc1efe52aadee3910d6aae8e43ff71f63a7e5"
OUTCOME_POLICY_DIGEST = "1827be7da1792c8044454db04f6335c72cd1bae793e95ff5b76cf63f8c4a1e39"
OUTCOME_ACTIVATION_DIGEST = "788250783b3d6a257d740b220d07c07723ea64773597bb37a60ea5f0e5c85dbf"
SOAK_INPUT_SHA256 = {
    "docs/research_line/vnext/Q4_MULTIROUND_SOAK_CANONICAL_RECEIPT.json": "8c86853f6676fa03da317d0211e27f747299f9df851530f68b0c02ae5034cad3",
    "results/research_line/q4_multiround_soak_20260803_01/Q4_PHYSICAL_RECEIPT.json": "b71c3a1215e9af01e5802d7362ea87384ba5565254fd2c74e5224a35cc49b438",
    "results/research_line/q4_multiround_soak_20260803_01/Q4_MULTIROUND_SOAK_PACKAGE.json": "631b0f7f28dc5567d776526f51f6a2c009ae02d4105f47bc49be7fb69cf38102",
}
PROPOSAL_SLOTS = (
    ("slot-01", "mechanism_composer", "FRONTIER_HYPOTHESIS", 57011),
    ("slot-02", "lineage_refiner", "FRONTIER_HYPOTHESIS", 57012),
    ("slot-03", "falsification_designer", "DIAGNOSIS_DRIVEN", 57013),
    ("slot-04", "frontier_architect", "FRONTIER_HYPOTHESIS", 57014),
)
OUTCOME_AWARE_SELECTION_SEED = 56331
COMMON_EXECUTION = canonical_value(
    {
        "provider_model": "gpt-5.4",
        "provider_model_digest": "2a7b79b0151aa44a0abee17adc0e18df1c07d8d15d7affa989c3b3afb6bee0a0",
        "provider_endpoint_digest": "810326f35f8f2efce5f4ca6f73d8df3300f60b4a7f3367fdab3ad73ec3363fcc",
        "proposal_prompt_contract": "ONE_COMMON_ORIGIN_BLIND_HIGH_CHANGE_OPEN_IDEA_CONTRACT",
        "proposal_schema": "Q1_ENRICHED_PROVIDER_SCHEMA",
        "proposal_token_ceiling": Q1_PROPOSAL_TOKEN_CEILING,
        "proposal_candidate_count": len(PROPOSAL_SLOTS),
        "implementation_token_ceiling": 20_000,
        "qualification_seed": 54301,
        "resource_probe_seed": 54102,
        "training_seed": 54303,
        "resource_probe_epochs": 3,
        "training_epochs": 100,
        "resource_deadline_seconds": 300,
        "training_deadline_seconds": 900,
        "deadline_seconds": 900,
        "engineering_watchdog_seconds": 10_800,
        "budget_seconds": 7_200,
        "resource_prefix_contract_sha256": "c87190d7a1a0b997f2f513c8bc7605a5e9c7d2cf6ec3ad6cf3c6f2c7351e446c",
        "data_digest": "99213591aa3344b023e2d07f99f9f970fcdf49e80ef0122db29e89660d8fdf98",
        "config_digest": "80453dd68a7fb47af90e89674f0f8d349110526f4245d8e4004da145fd0bc54e",
        "retries": 0,
        "held_out_reads": 0,
        "denominator_rule": "FULL_PROVIDER_POOL_ALWAYS_RETAINED_SELECTED_DENOMINATOR_ONE_PER_POLICY",
        "missingness_rule": "MISSING_IS_NEVER_ZERO_NEGATIVE_OR_EFFECT_EVIDENCE",
        "outcome_interpretation": "DEVELOPMENT_ONLY_SINGLE_SEED_INCONCLUSIVE_NO_SCIENTIFIC_EFFECT_CLAIM",
    }
)


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write(path: Path, value: Mapping[str, Any]) -> str:
    return _write_new_json(path, canonical_value(value))


def _copy_exact(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(source.read_bytes())


def _snapshot_policy_inputs(repo_root: Path, output_root: Path, f1_root: Path) -> dict[str, Any]:
    head = _git(repo_root, "rev-parse", "HEAD")
    tree = _git(repo_root, "rev-parse", "HEAD^{tree}")
    if head != ACCEPTED_SOAK_COMMIT or tree != ACCEPTED_SOAK_TREE:
        raise ProspectivePolicyComparisonError("prospective worktree start identity drift")
    if _git(repo_root, "rev-parse", f"{ACCEPTED_F1_EVIDENCE_COMMIT}^{{tree}}") != ACCEPTED_F1_EVIDENCE_TREE:
        raise ProspectivePolicyComparisonError("accepted F1 evidence commit drift")
    for relative, expected in SOAK_INPUT_SHA256.items():
        verify_file(repo_root / relative, expected, label=relative)
    f1_snapshot_root = output_root / "policy_bindings/current_f1"
    for relative, expected in F1_FILE_SHA256.items():
        source = f1_root / relative
        verify_file(source, expected, label=f"accepted F1 {relative}")
        _copy_exact(source, f1_snapshot_root / relative)
    outcome_source_root = (
        repo_root
        / "results/research_line/q4_multiround_soak_20260803_01/round_03/policy_update"
    )
    outcome_snapshot_root = output_root / "policy_bindings/outcome_aware"
    outcome_files = {
        "versioned_policy.json": OUTCOME_POLICY_FILE_SHA256,
        "active_policy.json": OUTCOME_ACTIVATION_FILE_SHA256,
    }
    for relative, expected in outcome_files.items():
        source = outcome_source_root / relative
        verify_file(source, expected, label=f"accepted outcome-aware {relative}")
        _copy_exact(source, outcome_snapshot_root / relative)
    f1_policy = read_object(f1_snapshot_root / "policy/versioned_policy.json")
    f1_activation = read_object(f1_snapshot_root / "activation/activation.json")
    verify_semantic_digest(
        f1_policy, digest_field="policy_digest", expected=F1_POLICY_DIGEST
    )
    if (
        f1_activation.get("activation_digest") != F1_ACTIVATION_DIGEST
        or f1_activation.get("policy_digest") != F1_POLICY_DIGEST
        or f1_activation.get("held_out_reads") != 0
    ):
        raise ProspectivePolicyComparisonError("accepted F1 activation binding drift")
    outcome_policy = read_object(outcome_snapshot_root / "versioned_policy.json")
    outcome_activation = read_object(outcome_snapshot_root / "active_policy.json")
    verify_semantic_digest(
        outcome_policy,
        digest_field="policy_digest",
        expected=OUTCOME_POLICY_DIGEST,
    )
    verify_semantic_digest(
        outcome_activation,
        digest_field="activation_digest",
        expected=OUTCOME_ACTIVATION_DIGEST,
    )
    return canonical_value(
        {
            "accepted_soak_commit": head,
            "accepted_soak_tree": tree,
            "accepted_f1_evidence_commit": ACCEPTED_F1_EVIDENCE_COMMIT,
            "accepted_f1_evidence_tree": ACCEPTED_F1_EVIDENCE_TREE,
            "f1_files": F1_FILE_SHA256,
            "f1_policy_digest": F1_POLICY_DIGEST,
            "f1_activation_digest": F1_ACTIVATION_DIGEST,
            "outcome_policy_file_sha256": OUTCOME_POLICY_FILE_SHA256,
            "outcome_activation_file_sha256": OUTCOME_ACTIVATION_FILE_SHA256,
            "outcome_policy_digest": OUTCOME_POLICY_DIGEST,
            "outcome_activation_digest": OUTCOME_ACTIVATION_DIGEST,
            "soak_inputs": SOAK_INPUT_SHA256,
        }
    )


def freeze(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise ProspectivePolicyComparisonError(f"output root exists: {output_root}")
    output_root.mkdir(parents=True)
    bindings = _snapshot_policy_inputs(repo_root, output_root, args.f1_root.resolve())
    authority = canonical_value(
        {
            "schema": "recclaw.research-line.q4-prospective-authority-matrix.v1",
            "STATIC": {
                "consumer": "open_meta.run_static_idea_policy",
                "policy_ref": STATIC_IDEA_POLICY_REF_V1,
                "policy_digest": STATIC_IDEA_POLICY_DIGEST_V1,
                "allowed_selection_features": ["OPAQUE_CANONICAL_SPEC_IDENTITY"],
                "selection_rule": "EXISTING_STATIC_HIGH_CHANGE_IDENTITY_ORDER",
                "selection_probability": "DETERMINISTIC_POINT_MASS",
            },
            "CURRENT_F1": {
                "consumer": "prospective select_current_f1 projection of accepted F1 direction order",
                "policy_digest": F1_POLICY_DIGEST,
                "activation_digest": F1_ACTIVATION_DIGEST,
                "allowed_selection_features": ["ACCEPTED_F1_PRODUCER_DIRECTION_ORDER", "CANDIDATE_ID_TIE_BREAK"],
                "selection_probability": "DETERMINISTIC_POINT_MASS",
                "post_outcome_revision": False,
            },
            "OUTCOME_AWARE": {
                "consumer": "open_meta_q3.build_q3_acquisition_manifest",
                "policy_digest": OUTCOME_POLICY_DIGEST,
                "activation_digest": OUTCOME_ACTIVATION_DIGEST,
                "allowed_selection_features": ["THREE_ACCEPTED_HEADS", "PREFROZEN_SCIENTIFIC_FALSIFIABILITY"],
                "exploration_probability": 0.15,
                "exploration_seed": OUTCOME_AWARE_SELECTION_SEED,
                "real_sampling": True,
            },
            "forbidden_for_all": [
                "CANDIDATE_ORIGIN",
                "MECHANICAL_QUALIFICATION",
                "RESOURCE_OBSERVATION",
                "IMPLEMENTATION_OBSERVATION",
                "DEVELOPMENT_OUTCOME",
                "HELD_OUT_DATA",
            ],
            "historical_authority": "ONLY_EACH_ACCEPTED_POLICY_SNAPSHOT_MAY_USE_ITS_OWN_FROZEN_HISTORY",
            "held_out_reads": 0,
            "development_only": True,
        }
    )
    metrics = canonical_value(
        {
            "schema": "recclaw.research-line.q4-prospective-metric-contract.v1",
            "main_metrics_exactly": [
                "A_BEST_PARENT_RELATIVE_DEVELOPMENT_EFFECT_NDCG_AT_10",
                "B_MECHANISM_IDENTIFIABLE_EPISODE_COUNT",
                "C_WALL_TIME_COST_PER_INFORMATIVE_EPISODE",
                "D_FULL_EPISODE_COMPLETION_RATE_SELECTED_DENOMINATOR",
            ],
            "A": "MAX_CANDIDATE_MINUS_MATCHED_PARENT_NDCG_AT_10_OVER_COMPLETE_EPISODES_ONLY_MISSING_EXCLUDED",
            "B": "COMPLETE_EPISODE_AND_MECHANISM_STATE_IN_INACTIVE_ACTIVE_SUPPORTED_ACTIVE_CONTRADICTED",
            "C": {
                "primary": "ACTUAL_WALL_TIME_MS_PER_INFORMATIVE_EPISODE",
                "informative": "COMPLETE_EPISODE_WITH_EFFECT_OR_IDENTIFIABLE_MECHANISM",
                "components": ["FULL_SHARED_POOL_PROVIDER_COST_CHARGED_TO_EACH_POLICY", "IMPLEMENTER", "QUALIFICATION", "RESOURCE_PROBE", "MECHANISM_PROBE", "MATCHED_TRAINING"],
                "zero_denominator": "INF_UNDEFINED_NO_SMOOTHING",
                "call_token_training_counts": "AUDIT_COMPONENTS_NOT_EXTRA_MAIN_METRICS",
            },
            "D": {
                "main_denominator": "ONE_POLICY_SELECTED_CANDIDATE",
                "full_pool_denominator": len(PROPOSAL_SLOTS),
                "pool_coverage": "DIAGNOSTIC_ONLY_NOT_RANKING",
            },
            "ranking_uses_only_four_metrics": True,
            "development_only": True,
        }
    )
    prefreeze = canonical_value(
        {
            "schema": "recclaw.research-line.q4-prospective-prefreeze.v1",
            "campaign_id": CAMPAIGN_ID,
            "policy_order": POLICY_ORDER,
            "gpu_maximum_concurrency": 1,
            "shared_pool": {
                "generated_once": True,
                "byte_identical_input_to_all_policies": True,
                "origin_blind": True,
                "complete_denominator": len(PROPOSAL_SLOTS),
                "proposal_slots": [
                    {"slot": slot, "producer_role": role, "idea_mode": mode, "seed": seed}
                    for slot, role, mode, seed in PROPOSAL_SLOTS
                ],
                "replacement_or_retry_on_failure": False,
                "non_innovation_candidate_action": "HARD_BLOCK_WHOLE_POOL_NO_REPLACEMENT",
            },
            "common_execution": COMMON_EXECUTION,
            "physical_roots": {
                policy: f"arms/{index:02d}_{policy.lower()}"
                for index, policy in enumerate(POLICY_ORDER, 1)
            },
            "selected_chain": ["RESOLVER", "IMPLEMENTER", "MATERIALIZE", "MECHANICAL_QUALIFIER", "Q0R2_RESOURCE_ADMISSION", "Q2_CHEAP_MECHANISM_PROBE", "FRESH_MATCHED_DEVELOPMENT_IF_ADMITTED", "TYPED_EPISODE"],
            "cross_policy_reuse_after_selection": False,
            "single_policy_failure_isolated": True,
            "resource_failure_updates_effect_or_mechanism": False,
            "retries": 0,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
            "bindings_digest": sha256_digest(bindings),
            "authority_matrix_digest": sha256_digest(authority),
            "metric_contract_digest": sha256_digest(metrics),
        }
    )
    _write(output_root / "INPUT_BINDINGS_BEFORE_PROVIDER.json", bindings)
    _write(output_root / "FAIR_AUTHORITY_MATRIX_BEFORE_PROVIDER.json", authority)
    _write(output_root / "FOUR_METRICS_BEFORE_PROVIDER.json", metrics)
    _write(output_root / "PREFREEZE_CONTRACT_BEFORE_PROVIDER.json", prefreeze)
    print(json.dumps({"status": "PREFROZEN", "output_root": str(output_root), "prefreeze_digest": sha256_digest(prefreeze)}, sort_keys=True))


def _shared_contract_instruction(mode: str) -> str:
    diagnosis = (
        "For DIAGNOSIS_DRIVEN, cite one real gap in the supplied accepted context. "
        if mode == "DIAGNOSIS_DRIVEN"
        else "For FRONTIER_HYPOTHESIS, do not invent an observed failure. "
    )
    return (
        "Propose one high-change open recommender-method hypothesis outside the current executable profile. "
        "Set current_profile_expressibility_claim to NOT_EXPRESSIBLE, requested_current_semantics_digest to null, "
        "and provide non-empty capability_diff and high_change_dimensions. Do not propose a parameter or configuration change. "
        f"Use enriched OpenSpec fields and set idea_mode to {mode}. "
        + diagnosis
        + "State research_question, closest_parent, minimal_testable_wedge, causal_chain, competing explanation, "
        "discriminative prediction, executable mechanism_off_definition, resource_hypothesis, and realization_mode. "
        "The candidate must be implementable as a candidate-local RecBole GeneralRecommender on the frozen ML-1M protocol."
    )


def _render_shared_prompt(
    template: str,
    *,
    slot: str,
    role: str,
    mode: str,
    seed: int,
    context: Mapping[str, Any],
    catalog: list[dict[str, str]],
    active: Any,
    context_digest: str,
) -> str:
    replacements = {
        "{{LOGICAL_SLOT_ID}}": slot,
        "{{PROPOSAL_SEED}}": str(seed),
        "{{PRODUCER_ROLE}}": role,
        "{{CONTRACT_INSTRUCTION}}": _shared_contract_instruction(mode),
        "{{RESEARCH_CONTEXT_JSON}}": json.dumps(context, sort_keys=True, separators=(",", ":")),
        "{{PROFILE_CATALOG_JSON}}": json.dumps(catalog, sort_keys=True, separators=(",", ":")),
        "{{PROTOCOL_REF}}": active.protocol_ref,
        "{{PROTOCOL_DIGEST}}": active.protocol_digest,
        "{{CONTEXT_REF}}": Q1_CONTEXT_REF,
        "{{CONTEXT_DIGEST}}": context_digest,
        "{{PROFILE_REF}}": active.profile_ref,
        "{{PROFILE_DIGEST}}": active.profile_digest,
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    if re.search(r"\{\{[A-Z][A-Z0-9_]*\}\}", rendered):
        raise ProspectivePolicyComparisonError("shared Provider prompt has unresolved placeholders")
    lowered = rendered.lower()
    for forbidden in ("expected winner", "candidate_ndcg", "observed_ndcg", "qualification result"):
        if forbidden in lowered:
            raise ProspectivePolicyComparisonError("shared Provider prompt leaked an outcome")
    return rendered


def _provider_usage(attempts: list[list[Mapping[str, Any]]]) -> dict[str, int]:
    rows = [row for group in attempts for row in group]
    return {
        "physical_calls": len(rows),
        "retries": sum(max(0, len(group) - 1) for group in attempts),
        "provider_wall_time_ms": sum(int(row.get("latency_ms") or 0) for row in rows),
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
        "billed_tokens": sum(int(row.get("billed_tokens") or 0) for row in rows),
    }


def pool(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    pool_root = output_root / "shared_pool"
    if pool_root.exists():
        raise ProspectivePolicyComparisonError("shared pool already exists")
    prefreeze = read_object(output_root / "PREFREEZE_CONTRACT_BEFORE_PROVIDER.json")
    if prefreeze.get("campaign_id") != CAMPAIGN_ID:
        raise ProspectivePolicyComparisonError("prefreeze identity drift")
    os.environ["RECCLAW_PROJECTS_ROOT"] = str(args.projects_root.resolve())
    os.environ["RECCLAW_SEARCH_DATA_ROOT"] = str(args.search_data_root.resolve())
    os.environ["RECCLAW_RECBOLE_ROOT"] = str(args.recbole_root.resolve())
    os.environ["RECCLAW_PYTHON_EXECUTABLE"] = str(args.python_executable.resolve())
    os.environ["RECCLAW_API_CONFIG"] = str(args.api_config.resolve())
    verify_file(args.api_config.resolve(), COMMON_EXECUTION["provider_endpoint_digest"], label="Provider config")
    started_ns = time.monotonic_ns()
    context = build_research_context(repo_root)
    artifacts, _receipt = load_registered_r1_artifacts(repo_root)
    registry = build_r1_registry(artifacts)
    _current, _manifest, _next, _build, active = build_active_r2_profile(registry)
    catalog = public_active_profile_catalog(active, artifacts, seed=57000)
    bindings = canonical_value({**_r2_bindings(active), "context_ref": Q1_CONTEXT_REF, "context_digest": sha256_digest(context)})
    environment = _r2_environment(active)
    resources = repo_root / "src/recclaw_core/experiments/helix_abc_v1/resources"
    template_path = resources / "idea_quality_producer_prompt_v1.txt"
    schema = derive_enriched_proposal_schema()
    schema_path = pool_root / "contracts/shared.schema.json"
    pool_root.mkdir(parents=True)
    _write(schema_path, schema)
    _write(pool_root / "RESEARCH_CONTEXT.json", context)
    template = template_path.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    attempt_groups: list[list[Mapping[str, Any]]] = []
    hard_block_reasons: list[dict[str, Any]] = []
    for slot, role, mode, seed in PROPOSAL_SLOTS:
        if role not in ALLOWED_DIRECTIONS:
            raise ProspectivePolicyComparisonError("proposal direction is outside F1 authority")
        prompt = _render_shared_prompt(
            template,
            slot=slot,
            role=role,
            mode=mode,
            seed=seed,
            context=context,
            catalog=catalog,
            active=active,
            context_digest=sha256_digest(context),
        )
        call = bounded_provider_call(
            call_root=pool_root / "provider" / slot,
            schema_path=schema_path,
            logical_call_id=f"{CAMPAIGN_ID}:shared-pool:{slot}",
            session_id=f"{CAMPAIGN_ID}:shared-pool:{slot}:session",
            prompt=prompt,
            token_ceiling=Q1_PROPOSAL_TOKEN_CEILING,
            maximum_physical_attempts=1,
        )
        attempt_groups.append(list(call.attempts))
        if call.call is None:
            hard_block_reasons.append({"slot": slot, "stage": "PROPOSAL_PROVIDER_FAILURE", "failure": call.failure, "provider_attempts": call.attempts})
            continue
        validate_v4_response_contract(call.call.response, provider_schema=schema)
        draft = call.call.response["proposals"][0]
        if draft.get("producer_role") != role or draft.get("idea_mode") != mode:
            hard_block_reasons.append({"slot": slot, "stage": "PROVIDER_CONTRACT_DRIFT", "observed_role": draft.get("producer_role"), "observed_mode": draft.get("idea_mode")})
            continue
        spec, facts = project_open_producer_draft(draft, bindings=bindings, strict_resolution_contract=True)
        resolution = resolve_capability(spec, resolution_facts=facts, environment=environment)
        feasible = resolution.resolution in {CapabilityResolutionResultV1.SEARCH_READY, CapabilityResolutionResultV1.INNOVATION_REQUIRED}
        score = score_preoutcome_testability(spec, q0r2_resource_feasible=feasible)
        row = canonical_value(
            {
                "slot": slot,
                "producer_role": role,
                "proposal_seed": seed,
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
        rows.append(row)
        if resolution.resolution is not CapabilityResolutionResultV1.INNOVATION_REQUIRED:
            hard_block_reasons.append({"slot": slot, "stage": "NON_INNOVATION_RESOLUTION", "resolution": resolution.resolution.value, "candidate_id": spec.digest})
    usage = _provider_usage(attempt_groups)
    if usage["retries"] != 0 or usage["physical_calls"] != len(PROPOSAL_SLOTS):
        hard_block_reasons.append({"stage": "PROVIDER_DENOMINATOR_DRIFT", "usage": usage})
    pool_value = canonical_value(
        {
            "schema": "recclaw.research-line.q4-prospective-shared-pool.v1",
            "candidate_pools": {"shared": rows},
            "candidate_count": len(rows),
            "provider_denominator": len(PROPOSAL_SLOTS),
            "selection_rule": "NONE_POOL_ONLY_POLICIES_SELECT_AFTER_BYTE_FREEZE",
            "implementation_or_qualification_outcomes_present_when_written": 0,
            "outcome_fields_consumed": [],
            "held_out_reads": 0,
        }
    )
    pool_path = pool_root / "FROZEN_SHARED_POOL_BEFORE_SELECTION.json"
    pool_sha = _write(pool_path, pool_value)
    status = "POOL_FROZEN_BEFORE_SELECTION" if not hard_block_reasons and len(rows) == len(PROPOSAL_SLOTS) else "HARD_BLOCK_INCOMPLETE_OR_OUT_OF_SUPPORT_POOL"
    receipt = canonical_value(
        {
            "schema": "recclaw.research-line.q4-prospective-pool-receipt.v1",
            "status": status,
            "pool_file": str(pool_path),
            "pool_file_sha256": pool_sha,
            "pool_semantic_digest": sha256_digest(pool_value),
            "provider_model": MODEL,
            "provider_prompt_template_sha256": bytes_sha256(template_path.read_bytes()),
            "provider_schema_sha256": bytes_sha256(schema_path.read_bytes()),
            "proposal_token_ceiling": Q1_PROPOSAL_TOKEN_CEILING,
            "candidate_count_budget": len(PROPOSAL_SLOTS),
            "proposal_provider_usage": usage,
            "hard_block_reasons": hard_block_reasons,
            "manual_candidate_patches": 0,
            "implementation_provider_calls": 0,
            "wall_time_ms": (time.monotonic_ns() - started_ns) // 1_000_000,
            "retries": 0,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    _write(pool_root / "SHARED_POOL_RECEIPT.json", receipt)
    print(json.dumps({"status": status, "pool_file": str(pool_path), "pool_sha256": pool_sha, "usage": usage}, sort_keys=True))
    if status != "POOL_FROZEN_BEFORE_SELECTION":
        raise ProspectivePolicyComparisonError(status)


def select(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    pool_receipt = read_object(output_root / "shared_pool/SHARED_POOL_RECEIPT.json")
    if pool_receipt.get("status") != "POOL_FROZEN_BEFORE_SELECTION":
        raise ProspectivePolicyComparisonError("shared pool Gate did not pass")
    pool_path = Path(pool_receipt["pool_file"])
    verify_file(pool_path, pool_receipt["pool_file_sha256"], label="shared pool")
    pool_value = read_object(pool_path)
    pool_digest = str(pool_receipt["pool_semantic_digest"])
    f1_policy_path = output_root / "policy_bindings/current_f1/policy/versioned_policy.json"
    f1_activation_path = output_root / "policy_bindings/current_f1/activation/activation.json"
    outcome_policy_path = output_root / "policy_bindings/outcome_aware/versioned_policy.json"
    outcome_activation_path = output_root / "policy_bindings/outcome_aware/active_policy.json"
    verify_file(f1_policy_path, F1_FILE_SHA256["policy/versioned_policy.json"], label="F1 policy snapshot")
    verify_file(f1_activation_path, F1_FILE_SHA256["activation/activation.json"], label="F1 activation snapshot")
    verify_file(outcome_policy_path, OUTCOME_POLICY_FILE_SHA256, label="outcome policy snapshot")
    verify_file(outcome_activation_path, OUTCOME_ACTIVATION_FILE_SHA256, label="outcome activation snapshot")
    selections = {
        "STATIC": select_static(pool_value, pool_digest=pool_digest),
        "CURRENT_F1": select_current_f1(pool_value, pool_digest=pool_digest, policy=read_object(f1_policy_path)),
        "OUTCOME_AWARE": select_outcome_aware(
            pool_value,
            pool_digest=pool_digest,
            policy=read_object(outcome_policy_path),
            activation=read_object(outcome_activation_path),
            random_seed=OUTCOME_AWARE_SELECTION_SEED,
        ),
    }
    selection_paths: dict[str, str] = {}
    arm_paths: dict[str, str] = {}
    for index, policy_name in enumerate(POLICY_ORDER, 1):
        arm_root = output_root / "arms" / f"{index:02d}_{policy_name.lower()}"
        selection_path = arm_root / "FROZEN_SELECTION_BEFORE_PHYSICAL.json"
        selection_sha = _write(selection_path, selections[policy_name])
        manifest = build_arm_manifest(
            campaign_id=CAMPAIGN_ID,
            arm_index=index,
            policy_name=policy_name,
            selection=selections[policy_name],
            full_pool_file=pool_path,
            common_execution=COMMON_EXECUTION,
        )
        manifest_path = arm_root / "ARM_MANIFEST.json"
        _write(manifest_path, manifest)
        selection_paths[policy_name] = str(selection_path)
        arm_paths[policy_name] = str(manifest_path)
        if bytes_sha256(selection_path.read_bytes()) != selection_sha:
            raise ProspectivePolicyComparisonError("selection byte write drift")
    package = canonical_value(
        {
            "schema": "recclaw.research-line.q4-prospective-selections.v1",
            "campaign_id": CAMPAIGN_ID,
            "full_pool_file": str(pool_path),
            "full_pool_file_sha256": bytes_sha256(pool_path.read_bytes()),
            "full_pool_semantic_digest": pool_digest,
            "full_pool_denominator": len(PROPOSAL_SLOTS),
            "policy_order": POLICY_ORDER,
            "selection_files": selection_paths,
            "arm_manifests": arm_paths,
            "selected_candidate_ids": {name: selections[name]["selected_candidate_id"] for name in POLICY_ORDER},
            "all_selection_probabilities_sum_to_one": all(
                abs(sum(float(row["selection_probability"]) for row in selections[name]["candidates"]) - 1.0) < 1e-9
                for name in POLICY_ORDER
            ),
            "outcomes_present_when_written": 0,
            "retries": 0,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    _write(output_root / "FROZEN_SELECTIONS_BEFORE_PHYSICAL.json", package)
    print(json.dumps({"status": "SELECTIONS_FROZEN", "selected": package["selected_candidate_ids"], "policy_order": POLICY_ORDER}, sort_keys=True))


def correct_data_digest(args: argparse.Namespace) -> None:
    """Correct the pre-outcome 65-character data identity typo exactly once."""

    output_root = args.output_root.resolve()
    bad_digest = "99213591aa3344b023e2d07f99f9f970fcdf49e80ef0122db29e89660d8fdf98e"
    good_digest = str(COMMON_EXECUTION["data_digest"])
    for arm_root in (output_root / "arms").glob("*"):
        forbidden = (
            "IMPLEMENTER_RECEIPT.json",
            "MATERIALIZE_QUALIFIER_RECEIPT.json",
            "RESOURCE_ADMISSION_RECEIPT.json",
            "MATCHED_EXECUTION_RECEIPT.json",
            "EPISODE_RECEIPT.json",
        )
        if any((arm_root / name).exists() for name in forbidden):
            raise ProspectivePolicyComparisonError(
                "data identity correction is no longer pre-outcome"
            )
    correction_root = output_root / "preoutcome_data_digest_typo_failure_01"
    if correction_root.exists():
        raise ProspectivePolicyComparisonError("data identity correction already used")
    correction_root.mkdir(parents=True)
    prefreeze_path = output_root / "PREFREEZE_CONTRACT_BEFORE_PROVIDER.json"
    prefreeze = read_object(prefreeze_path)
    if prefreeze["common_execution"].get("data_digest") != bad_digest:
        raise ProspectivePolicyComparisonError("expected pre-outcome typo is absent")
    original_prefreeze_sha = bytes_sha256(prefreeze_path.read_bytes())
    original_prefreeze_path = correction_root / "PREFREEZE_CONTRACT_ORIGINAL_TYPO.json"
    prefreeze_path.rename(original_prefreeze_path)
    prefreeze["common_execution"]["data_digest"] = good_digest
    corrected_prefreeze_sha = _write(prefreeze_path, prefreeze)
    arm_corrections = {}
    for index, policy_name in enumerate(POLICY_ORDER, 1):
        arm_root = output_root / "arms" / f"{index:02d}_{policy_name.lower()}"
        manifest_path = arm_root / "ARM_MANIFEST.json"
        manifest = read_object(manifest_path)
        if manifest["frozen_execution"].get("data_digest") != bad_digest:
            raise ProspectivePolicyComparisonError("arm data typo identity drift")
        original_sha = bytes_sha256(manifest_path.read_bytes())
        diagnostic_root = arm_root / "preoutcome_data_digest_typo_failure_01"
        diagnostic_root.mkdir(parents=True)
        manifest_path.rename(diagnostic_root / "ARM_MANIFEST_ORIGINAL_TYPO.json")
        manifest.pop("manifest_digest", None)
        manifest["frozen_execution"]["data_digest"] = good_digest
        corrected = {**canonical_value(manifest), "manifest_digest": sha256_digest(manifest)}
        corrected_sha = _write(manifest_path, corrected)
        arm_corrections[policy_name] = {
            "original_manifest_sha256": original_sha,
            "corrected_manifest_sha256": corrected_sha,
            "selected_candidate_id": manifest["selected_candidate"]["candidate_id"],
        }
    correction = canonical_value(
        {
            "schema": "recclaw.research-line.q4-prospective-preoutcome-correction.v1",
            "status": "CORRECTED_ONCE_BEFORE_IMPLEMENTATION_QUALIFICATION_RESOURCE_OR_OUTCOME",
            "field": "common_execution.data_digest",
            "original_invalid_value": bad_digest,
            "corrected_existing_file_sha256": good_digest,
            "reason": "65_CHARACTER_TRANSCRIPTION_TYPO_ONLY",
            "scientific_equivalence": "SAME_EXISTING_FROZEN_SEARCH_PARTITION_FILE",
            "original_prefreeze_sha256": original_prefreeze_sha,
            "corrected_prefreeze_sha256": corrected_prefreeze_sha,
            "arm_corrections": arm_corrections,
            "candidate_changed": False,
            "selection_changed": False,
            "seed_batch_model_endpoint_budget_denominator_interpretation_changed": False,
            "pool_provider_calls_reexecuted": False,
            "implementation_provider_calls_before_correction": 0,
            "qualification_runs_before_correction": 0,
            "training_runs_before_correction": 0,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    _write(correction_root / "CORRECTION_RECEIPT.json", correction)
    print(
        json.dumps(
            {
                "status": correction["status"],
                "corrected_data_digest": good_digest,
                "arm_corrections": arm_corrections,
            },
            sort_keys=True,
        )
    )


def correct_resource_probe_seed(args: argparse.Namespace) -> None:
    """Create a v2 fairness binding without altering the sealed invalid v1 line."""

    output_root = args.output_root.resolve()
    bad_seed = 54302
    good_seed = int(COMMON_EXECUTION["resource_probe_seed"])
    v2_root = output_root / "fairness_v2"
    if v2_root.exists():
        raise ProspectivePolicyComparisonError("fairness v2 binding already exists")
    prefreeze_path = output_root / "PREFREEZE_CONTRACT_BEFORE_PROVIDER.json"
    prefreeze = read_object(prefreeze_path)
    if int(prefreeze["common_execution"].get("resource_probe_seed")) != bad_seed:
        raise ProspectivePolicyComparisonError("sealed v1 resource seed identity drift")

    invalid_launches: list[dict[str, Any]] = []
    upstream_bindings: dict[str, Any] = {}
    v2_root.mkdir(parents=True)
    for index, policy_name in enumerate(POLICY_ORDER, 1):
        v1_arm = output_root / "arms" / f"{index:02d}_{policy_name.lower()}"
        v1_manifest_path = v1_arm / "ARM_MANIFEST.json"
        v1_manifest = read_object(v1_manifest_path)
        if int(v1_manifest["frozen_execution"].get("resource_probe_seed")) != bad_seed:
            raise ProspectivePolicyComparisonError("sealed v1 arm seed identity drift")
        v2_arm = v2_root / "arms" / f"{index:02d}_{policy_name.lower()}"
        v2_arm.mkdir(parents=True)

        binding_artifacts: dict[str, Any] = {}
        for name in (
            "FROZEN_SELECTION_BEFORE_PHYSICAL.json",
            "RESOLVER_RECEIPT.json",
            "IMPLEMENTER_RECEIPT.json",
            "MATERIALIZE_QUALIFIER_RECEIPT.json",
        ):
            path = v1_arm / name
            if not path.is_file():
                raise ProspectivePolicyComparisonError(
                    f"sealed upstream artifact missing for {policy_name}: {name}"
                )
            binding_artifacts[name] = _artifact(path)
        for name in ("implementer.json", "materialize-qualifier.json"):
            path = v1_arm / "stage_costs" / name
            if not path.is_file():
                raise ProspectivePolicyComparisonError(
                    f"sealed upstream cost missing for {policy_name}: {name}"
                )
            binding_artifacts[f"stage_costs/{name}"] = _artifact(path)

        v2_manifest = dict(v1_manifest)
        v2_manifest.pop("manifest_digest", None)
        v2_manifest["campaign_id"] = f"{CAMPAIGN_ID}-fairness-v2"
        v2_manifest["fairness_contract_version"] = 2
        v2_manifest["frozen_execution"] = dict(v2_manifest["frozen_execution"])
        v2_manifest["frozen_execution"]["resource_probe_seed"] = good_seed
        v2_manifest["sealed_v1_manifest"] = _artifact(v1_manifest_path)
        v2_manifest["sealed_upstream_binding_digest"] = sha256_digest(binding_artifacts)
        v2_manifest = {
            **canonical_value(v2_manifest),
            "manifest_digest": sha256_digest(v2_manifest),
        }
        v2_manifest_path = v2_arm / "ARM_MANIFEST_V2.json"
        _write(v2_manifest_path, v2_manifest)
        upstream_binding = canonical_value(
            {
                "schema": "recclaw.research-line.q4-prospective-v2-upstream-binding.v1",
                "policy_name": policy_name,
                "v1_arm_root": str(v1_arm),
                "v2_arm_root": str(v2_arm),
                "sealed_artifacts": binding_artifacts,
                "provider_reexecuted": False,
                "implementer_reexecuted": False,
                "qualifier_reexecuted": False,
                "candidate_retry_count": 0,
                "held_out_reads": 0,
            }
        )
        _write(
            v2_arm / "SEALED_UPSTREAM_BINDING.json",
            {**upstream_binding, "binding_digest": sha256_digest(upstream_binding)},
        )
        upstream_bindings[policy_name] = {
            "manifest_v2": _artifact(v2_manifest_path),
            "upstream_binding": _artifact(v2_arm / "SEALED_UPSTREAM_BINDING.json"),
        }

        invalid_resource = v1_arm / "RESOURCE_ADMISSION_RECEIPT.json"
        if invalid_resource.is_file():
            confirmation = v1_arm / (
                "resource_probe/experiments/selected-candidate-prefix/worker/"
                "start_confirmation.json"
            )
            invalid = canonical_value(
                {
                    "schema": "recclaw.research-line.q4-invalid-protocol-launch.v1",
                    "policy_name": policy_name,
                    "classification": "INVALID_PROTOCOL_BINDING",
                    "reason": "RESOURCE_SEED_DID_NOT_MATCH_ACCEPTED_Q0R2_PREFIX_CONTRACT",
                    "sealed_v1_manifest": _artifact(v1_manifest_path),
                    "sealed_resource_receipt": _artifact(invalid_resource),
                    "start_confirmation": _artifact(confirmation),
                    "physical_launch_count": 1,
                    "training_batch_count": 0,
                    "candidate_retry_count": 0,
                    "effect_authority": 0,
                    "mechanism_authority": 0,
                    "held_out_reads": 0,
                    "development_only": True,
                }
            )
            invalid_path = v2_arm / "INVALID_PROTOCOL_BINDING_RECEIPT.json"
            _write(invalid_path, {**invalid, "receipt_digest": sha256_digest(invalid)})
            invalid_launches.append(_artifact(invalid_path))

    v2_contract = canonical_value(
        {
            "schema": "recclaw.research-line.q4-prospective-fairness-contract.v2",
            "campaign_id": f"{CAMPAIGN_ID}-fairness-v2",
            "status": "FROZEN_BEFORE_ANY_VALID_RESOURCE_OR_OUTCOME",
            "sealed_v1_prefreeze_contract": _artifact(prefreeze_path),
            "invalid_v1_resource_seed": bad_seed,
            "accepted_q0r2_resource_seed": good_seed,
            "accepted_q0r2_prefix_contract_sha256": str(
                COMMON_EXECUTION["resource_prefix_contract_sha256"]
            ),
            "shared_pool": _artifact(
                output_root / "shared_pool/FROZEN_SHARED_POOL_BEFORE_SELECTION.json"
            ),
            "frozen_selections": _artifact(
                output_root / "FROZEN_SELECTIONS_BEFORE_PHYSICAL.json"
            ),
            "policy_order": POLICY_ORDER,
            "upstream_bindings": upstream_bindings,
            "invalid_physical_launches": invalid_launches,
            "invalid_physical_launch_count": len(invalid_launches),
            "fresh_validation_failure_rule": "ANY_REPEAT_FAILURE_IS_HARD_BLOCK_NO_FURTHER_ATTEMPT",
            "candidate_retry_count": 0,
            "pool_changed": False,
            "selection_changed": False,
            "inputs_changed_other_than_protocol_identity_seed_correction": False,
            "held_out_reads": 0,
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    contract_path = v2_root / "PREFREEZE_FAIRNESS_CONTRACT_V2.json"
    _write(contract_path, {**v2_contract, "contract_digest": sha256_digest(v2_contract)})
    ledger = canonical_value(
        {
            "schema": "recclaw.research-line.q4-prospective-physical-ledger.v1",
            "invalid_protocol_launches": invalid_launches,
            "invalid_protocol_launch_count": len(invalid_launches),
            "valid_scientific_execution_count": 0,
            "candidate_retry_count": 0,
        }
    )
    _write(
        v2_root / "PHYSICAL_LEDGER_BEFORE_V2_EXECUTION.json",
        {**ledger, "ledger_digest": sha256_digest(ledger)},
    )
    print(json.dumps({"status": "FAIRNESS_V2_FROZEN", "contract": _artifact(contract_path)}, sort_keys=True))


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path), "sha256": bytes_sha256(path.read_bytes()), "size_bytes": path.stat().st_size}


def finalize(args: argparse.Namespace) -> None:
    output_root = args.output_root.resolve()
    repo_root = args.repo_root.resolve()
    v2_root = output_root / "fairness_v2"
    pool_receipt = read_object(output_root / "shared_pool/SHARED_POOL_RECEIPT.json")
    provider_usage = pool_receipt["proposal_provider_usage"]
    metrics: dict[str, Any] = {}
    failures: dict[str, list[dict[str, Any]]] = {}
    arm_summaries: dict[str, Any] = {}
    matched_training_runs = 0
    resource_training_runs = 0
    qualification_training_runs = 0
    total_implementation_calls = 0
    for index, policy_name in enumerate(POLICY_ORDER, 1):
        upstream_root = output_root / "arms" / f"{index:02d}_{policy_name.lower()}"
        arm_root = v2_root / "arms" / f"{index:02d}_{policy_name.lower()}"
        metric = compute_four_metrics(
            arm_root=arm_root,
            upstream_arm_root=upstream_root,
            full_pool_count=len(PROPOSAL_SLOTS),
            shared_pool_provider_wall_time_ms=int(provider_usage["provider_wall_time_ms"]),
        )
        _write(arm_root / "FOUR_MAIN_METRICS.json", metric)
        metrics[policy_name] = metric
        arm_failures = []
        for root, name in (
            (upstream_root, "RESOLVER_RECEIPT.json"),
            (upstream_root, "IMPLEMENTER_RECEIPT.json"),
            (upstream_root, "MATERIALIZE_QUALIFIER_RECEIPT.json"),
            (arm_root, "RESOURCE_ADMISSION_RECEIPT.json"),
            (arm_root, "MECHANISM_PROBE_RECEIPT.json"),
            (arm_root, "MATCHED_EXECUTION_RECEIPT.json"),
            (arm_root, "EPISODE_RECEIPT.json"),
        ):
            value = read_object(root / name)
            status = str(value.get("status"))
            if status not in {
                "RESOLUTION_CONFIRMED",
                "IMPLEMENTATION_PROVIDER_SUCCESS",
                "QUALIFICATION_PASS",
                "RESOURCE_ADMITTED",
                "PROBE_CLASSIFIED",
                "COMPLETED_MATCHED_PAIR",
                "EPISODE_CREATED",
            }:
                arm_failures.append({"stage": name, "status": status, "failure": value.get("failure"), "missingness": value.get("missingness")})
            if name == "IMPLEMENTER_RECEIPT.json":
                total_implementation_calls += len(value.get("attempts") or [])
            if name == "MATERIALIZE_QUALIFIER_RECEIPT.json":
                qualification_training_runs += int(
                    value.get("qualification", {}).get("smoke_executions", 0)
                )
            if name == "RESOURCE_ADMISSION_RECEIPT.json":
                resource_training_runs += int(value.get("resource_probe") is not None)
            if name == "MATCHED_EXECUTION_RECEIPT.json":
                matched_training_runs += int(value.get("baseline") is not None) + int(value.get("candidate") is not None)
        failures[policy_name] = arm_failures
        selection = read_object(upstream_root / "FROZEN_SELECTION_BEFORE_PHYSICAL.json")
        arm_summaries[policy_name] = {
            "selected_candidate_id": selection["selected_candidate_id"],
            "selection": _artifact(upstream_root / "FROZEN_SELECTION_BEFORE_PHYSICAL.json"),
            "sealed_v1_manifest": _artifact(upstream_root / "ARM_MANIFEST.json"),
            "fairness_v2_manifest": _artifact(arm_root / "ARM_MANIFEST_V2.json"),
            "sealed_upstream_binding": _artifact(arm_root / "SEALED_UPSTREAM_BINDING.json"),
            "resolver": _artifact(upstream_root / "RESOLVER_RECEIPT.json"),
            "implementer": _artifact(upstream_root / "IMPLEMENTER_RECEIPT.json"),
            "qualification": _artifact(upstream_root / "MATERIALIZE_QUALIFIER_RECEIPT.json"),
            "resource": _artifact(arm_root / "RESOURCE_ADMISSION_RECEIPT.json"),
            "mechanism": _artifact(arm_root / "MECHANISM_PROBE_RECEIPT.json"),
            "matched_execution": _artifact(arm_root / "MATCHED_EXECUTION_RECEIPT.json"),
            "episode": _artifact(arm_root / "EPISODE_RECEIPT.json"),
            "metrics": _artifact(arm_root / "FOUR_MAIN_METRICS.json"),
        }
    preoutcome_failure_evidence = [
        _artifact(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file()
        and any(part.startswith("preoutcome_") for part in path.relative_to(output_root).parts)
    ]
    invalid_ledger_path = v2_root / "PHYSICAL_LEDGER_BEFORE_V2_EXECUTION.json"
    invalid_ledger = read_object(invalid_ledger_path)
    invalid_launches = int(invalid_ledger["invalid_protocol_launch_count"])
    physical = canonical_value(
        {
            "schema": "recclaw.research-line.q4-prospective-physical-receipt.v1",
            "campaign_id": f"{CAMPAIGN_ID}-fairness-v2",
            "status": "DEVELOPMENT_ONLY_PROSPECTIVE_PILOT_CLOSED",
            "policy_order": POLICY_ORDER,
            "gpu_maximum_concurrency": 1,
            "shared_pool": _artifact(
                output_root / "shared_pool/FROZEN_SHARED_POOL_BEFORE_SELECTION.json"
            ),
            "shared_pool_provider_usage": provider_usage,
            "fairness_v2_contract": _artifact(v2_root / "PREFREEZE_FAIRNESS_CONTRACT_V2.json"),
            "invalid_protocol_ledger": _artifact(invalid_ledger_path),
            "invalid_protocol_launches": invalid_launches,
            "invalid_protocol_effect_authority": 0,
            "invalid_protocol_mechanism_authority": 0,
            "candidate_retry_count": 0,
            "arm_summaries": arm_summaries,
            "failures": failures,
            "preoutcome_failure_evidence": preoutcome_failure_evidence,
            "totals": {
                "proposal_provider_calls": provider_usage["physical_calls"],
                "implementation_provider_calls": total_implementation_calls,
                "qualification_training_runs": qualification_training_runs,
                "valid_resource_probe_training_runs": resource_training_runs,
                "matched_development_training_runs": matched_training_runs,
                "invalid_protocol_pretraining_launches": invalid_launches,
                "all_gpu_worker_launches_including_invalid": (
                    qualification_training_runs
                    + resource_training_runs
                    + matched_training_runs
                    + invalid_launches
                ),
                "complete_episodes": sum(value["episode_status"] == "EPISODE_CREATED" for value in metrics.values()),
                "missing_episodes": sum(value["episode_status"] != "EPISODE_CREATED" for value in metrics.values()),
                "candidate_retries": 0,
                "stage_retries": 0,
                "held_out_reads": 0,
            },
            "development_only": True,
            "scientific_effect_claim": False,
        }
    )
    physical_path = output_root / "Q4_PROSPECTIVE_PHYSICAL_RECEIPT.json"
    _write(physical_path, {**physical, "receipt_digest": sha256_digest(physical)})
    package = canonical_value(
        {
            "schema": "recclaw.research-line.q4-prospective-policy-comparison-package.v1",
            "campaign_id": f"{CAMPAIGN_ID}-fairness-v2",
            "status": "DEVELOPMENT_ONLY_PROSPECTIVE_PILOT_CLOSED",
            "full_pool_denominator": len(PROPOSAL_SLOTS),
            "policy_order": POLICY_ORDER,
            "metrics_exactly_four": metrics,
            "failures": failures,
            "preoutcome_failure_evidence": preoutcome_failure_evidence,
            "prefreeze": _artifact(output_root / "PREFREEZE_CONTRACT_BEFORE_PROVIDER.json"),
            "fairness_v2_contract": _artifact(v2_root / "PREFREEZE_FAIRNESS_CONTRACT_V2.json"),
            "authority_matrix": _artifact(output_root / "FAIR_AUTHORITY_MATRIX_BEFORE_PROVIDER.json"),
            "metric_contract": _artifact(output_root / "FOUR_METRICS_BEFORE_PROVIDER.json"),
            "selections": _artifact(output_root / "FROZEN_SELECTIONS_BEFORE_PHYSICAL.json"),
            "physical_receipt": _artifact(physical_path),
            "verification_audit": _artifact(output_root / "VERIFICATION_AUDIT_BEFORE_FINALIZE.json"),
            "held_out_reads": 0,
            "retries": 0,
            "development_only": True,
            "scientific_effect_claim": False,
            "outcome_aware_superiority_claim": False,
        }
    )
    package_path = output_root / "Q4_PROSPECTIVE_POLICY_COMPARISON_PACKAGE.json"
    _write(package_path, {**package, "package_digest": sha256_digest(package)})
    canonical_receipt = canonical_value(
        {
            "schema": "recclaw.research-line.q4-prospective-canonical-receipt.v1",
            "campaign_id": f"{CAMPAIGN_ID}-fairness-v2",
            "status": "DEVELOPMENT_ONLY_PROSPECTIVE_PILOT_CLOSED",
            "package_file": str(package_path),
            "package_file_sha256": bytes_sha256(package_path.read_bytes()),
            "physical_receipt_file": str(physical_path),
            "physical_receipt_file_sha256": bytes_sha256(physical_path.read_bytes()),
            "metrics_exactly_four": metrics,
            "failures": failures,
            "held_out_reads": 0,
            "retries": 0,
            "development_only": True,
            "scientific_effect_claim": False,
            "outcome_aware_superiority_claim": False,
        }
    )
    canonical_path = output_root / "Q4_PROSPECTIVE_CANONICAL_RECEIPT.json"
    _write(canonical_path, {**canonical_receipt, "receipt_digest": sha256_digest(canonical_receipt)})
    docs_canonical_path = (
        repo_root / "docs/research_line/vnext/Q4_PROSPECTIVE_POLICY_COMPARISON_CANONICAL_RECEIPT.json"
    )
    _write(docs_canonical_path, read_object(canonical_path))
    sums_path = output_root / "SHA256SUMS"
    files = sorted(path for path in output_root.rglob("*") if path.is_file() and path != sums_path)
    with sums_path.open("x", encoding="utf-8", newline="\n") as handle:
        for path in files:
            handle.write(f"{bytes_sha256(path.read_bytes())}  {path.relative_to(output_root).as_posix()}\n")
    print(json.dumps({"status": "FINALIZED", "canonical": _artifact(canonical_path), "docs_canonical": _artifact(docs_canonical_path), "physical": _artifact(physical_path), "package": _artifact(package_path), "metrics": metrics, "failures": failures}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--repo-root", type=Path, required=True)
    freeze_parser.add_argument("--output-root", type=Path, required=True)
    freeze_parser.add_argument("--f1-root", type=Path, required=True)
    pool_parser = subparsers.add_parser("pool")
    pool_parser.add_argument("--repo-root", type=Path, required=True)
    pool_parser.add_argument("--output-root", type=Path, required=True)
    pool_parser.add_argument("--projects-root", type=Path, required=True)
    pool_parser.add_argument("--search-data-root", type=Path, required=True)
    pool_parser.add_argument("--recbole-root", type=Path, required=True)
    pool_parser.add_argument("--python-executable", type=Path, required=True)
    pool_parser.add_argument("--api-config", type=Path, required=True)
    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--output-root", type=Path, required=True)
    correction_parser = subparsers.add_parser("correct-data-digest")
    correction_parser.add_argument("--output-root", type=Path, required=True)
    seed_correction_parser = subparsers.add_parser("correct-resource-probe-seed")
    seed_correction_parser.add_argument("--output-root", type=Path, required=True)
    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("--repo-root", type=Path, required=True)
    finalize_parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    {
        "freeze": freeze,
        "pool": pool,
        "select": select,
        "correct-data-digest": correct_data_digest,
        "correct-resource-probe-seed": correct_resource_probe_seed,
        "finalize": finalize,
    }[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
