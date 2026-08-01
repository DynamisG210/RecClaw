"""Fresh F1 open-Meta architecture-effect campaign.

This campaign fits the open Meta learner from all accepted R1/R2 evidence,
performs development replay and origin-blind shadow evaluation, promotes a
versioned DEVELOPMENT_ONLY policy under a pre-frozen engineering rule, then
activates it at a genuinely fresh campaign boundary.  The activated policy is
consumed by normal Producer, Resolver, registry, Implementer, Qualifier, and
development runtime paths.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

from .canonical import bytes_sha256, canonical_value, sha256_digest
from .fresh_r1 import (
    API_CONFIG,
    AVAILABLE_DEPENDENCIES,
    BACKOFF_MS,
    BUDGET_LIMITS,
    IMPLEMENTATION_TOKEN_CEILING,
    MAX_PHYSICAL_ATTEMPTS,
    MODEL,
    PROTOCOL_REQUIREMENTS,
    PYTHON_EXECUTABLE,
    RECBole_ROOT,
    SEARCH_DATASET_ROOT,
    EXPECTED_SEARCH_FILES,
    _credential_identity,
    _git,
    _materialize_and_qualify,
    _physical_usage,
    _read_json,
    _shared_policy,
    _write_new_json,
    bounded_provider_call,
    render_implementation_prompt,
    run_development_training,
)
from .fresh_r2 import (
    R1_EXTERNAL_ROOT,
    R2_ROLE_INSTRUCTIONS,
    build_r1_registry,
    derive_fresh_r2_proposal_schema,
    load_registered_r1_artifacts,
    public_active_profile_catalog,
)
from .innovation_spine import build_shared_implementer_request
from .innovation_recbole_adapter import snapshot_candidate_tree
from .next_fresh_profile import NextFreshProfileBuildManifest, build_next_fresh_profile
from .open_meta_f1 import (
    F1_POLICY_VERSION,
    STATIC_DIRECTION_ORDER,
    build_f1_replay_dataset,
    build_policy_activation,
    evaluate_development_promotion,
    fit_open_meta_policy,
    rank_search_ready_records,
    shadow_evaluate_open_meta,
)
from .open_spec import project_open_producer_draft, resolve_capability
from .search_adapter import (
    SearchProfileEntryOriginV1,
    activate_next_fresh_search_profile,
    adapt_current_search_profile,
    predecessor_executable_entries,
)
from .v4_response_contract import validate_v4_response_contract
from .vnext_contracts import (
    EpisodeEvidenceClassV1,
    QualificationStatusV1,
    ResearchFailureClassV1,
    TypedResearchEpisodeV1,
)


ACCEPTED_R2_COMMIT = "3505885c69737064ba5bd59a5aa8c96e5309d15d"
ACCEPTED_R2_PARENT = "11dae330dbbbbf3a8108b19e7f9b9020326f0a20"
ACCEPTED_R2_TREE = "dd2ff040814af2fc21d01655abbd4d35ca2e31e9"
R2_REPO_RECEIPT_SHA256 = (
    "5ed2ebddc5f0724fd63849cdac9b3155f039192f2002b895d73d0c7cee665393"
)
R2_EXTERNAL_RECEIPT_SHA256 = (
    "6de30cd5122084804e3c6b083b236dfbdc28677ace903cccf690f65fe61f9a54"
)
R2_EXTERNAL_ROOT = Path(
    "/root/projects/RecClaw_r1_r2_runs/fresh_r2_registry_effect_v2"
)
F1_ROOT = Path("/root/projects/RecClaw_f1_runs/fresh_f1_open_meta_v1")
F1_RUN_IDENTITY = "fresh-f1-open-meta-v1"
F1_CAMPAIGN_ID = "recclaw-fresh-f1-open-meta-v1"
F1_CONTEXT_REF = "recclaw-fresh-f1-open-meta-context-v1"
F1_CONTEXT_DIGEST = sha256_digest(
    {
        "campaign_id": F1_CAMPAIGN_ID,
        "purpose": "OPEN_META_ARCHITECTURE_EFFECT",
        "source_r2_commit": ACCEPTED_R2_COMMIT,
    }
)
F1_PROPOSAL_SEEDS = (53011, 53012, 53013, 53014)
F1_QUALIFICATION_SEED = 53101
F1_TRAINING_SEED = 53102
F1_PROPOSAL_TOKEN_CEILING = 12_000

F1_V1_EXTERNAL_RECEIPT_SHA256 = (
    "5bb7361179f13bca085884d6926e6002c9f7dfc9632c8471de9fcd0bd631cebb"
)
F1_V1_REPO_RECEIPT_SHA256 = (
    "f160436a9df1bb8d849849f6cb61e367dd54c62d4b5fcc68101a21c86a72b4b3"
)
F1_V1_POLICY_FILE_SHA256 = (
    "1bc607a8f0d4e2f266ae471b79c29c0251fdcf8ec37697c127875d6e38dfe88c"
)
F1_V1_ACTIVATION_FILE_SHA256 = (
    "7ad0d44da4d5357a36908a82b8edbe775ef347aedf0665e22d111be0da32969a"
)
F1_V1_SELECTION_FILE_SHA256 = (
    "c4fc92b5b53c1194cf9089ff09155ee578625324c9a625fbcfcc5cab750f44e6"
)
F1_V1_QUALIFICATION_FILE_SHA256 = (
    "1709c92bd91f6e332527aeb71a3ec7b501ed10390114aa5dbb3b87df2d04ee9c"
)
F1_RECOVERY_ROOT = Path(
    "/root/projects/RecClaw_f1_runs/fresh_f1_runtime_recovery_v2"
)
F1_RECOVERY_RUN_IDENTITY = "fresh-f1-runtime-recovery-v2"
F1_RECOVERY_CAMPAIGN_ID = "recclaw-fresh-f1-runtime-recovery-v2"
F1_RECOVERY_TIMEOUT_SECONDS = 2400

PREFROZEN_PROMOTION_RULE = canonical_value(
    {
        "schema": "recclaw.research-line.vnext.open-meta.f1-prefrozen-promotion-rule.v1",
        "rule": "F1_DEVELOPMENT_ARCHITECTURE_EFFECT_V1",
        "required_evidence_rows": 20,
        "required_typed_episodes": 8,
        "held_out_reads": 0,
        "requires_origin_blind_features": True,
        "requires_meaningful_static_controls": True,
        "requires_behavior_difference": True,
        "requires_information_coverage_not_worse": True,
        "requires_policy_superiority": False,
        "requires_scientific_effect": False,
        "activation_boundary": "NEXT_FRESH_CAMPAIGN",
    }
)


class FreshF1Error(RuntimeError):
    """The fresh F1 evidence or execution chain could not close."""


def _resource_root() -> Path:
    return Path(__file__).resolve().parent / "resources"


def verify_f1_source_identity(repo_root: Path, *, require_fresh_root: bool) -> dict[str, Any]:
    repo_receipt = (
        repo_root
        / "docs/research_line/vnext/R2_FRESH_REGISTRY_EFFECT_CANONICAL_RECEIPT.json"
    )
    external_receipt = R2_EXTERNAL_ROOT / "R2_CANONICAL_RECEIPT.json"
    observed = {
        "branch": _git(repo_root, "branch", "--show-current"),
        "head": _git(repo_root, "rev-parse", "HEAD"),
        "parent": _git(repo_root, "rev-parse", "HEAD^"),
        "head_tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "python_sha256": bytes_sha256(PYTHON_EXECUTABLE.read_bytes()),
        "r2_repo_receipt_sha256": bytes_sha256(repo_receipt.read_bytes()),
        "r2_external_receipt_sha256": bytes_sha256(external_receipt.read_bytes()),
        **_credential_identity(API_CONFIG),
    }
    expected = {
        "branch": "feat/research-line-f1-meta",
        "head": ACCEPTED_R2_COMMIT,
        "parent": ACCEPTED_R2_PARENT,
        "head_tree": ACCEPTED_R2_TREE,
        "r2_repo_receipt_sha256": R2_REPO_RECEIPT_SHA256,
        "r2_external_receipt_sha256": R2_EXTERNAL_RECEIPT_SHA256,
    }
    mismatches = {
        key: {"expected": value, "observed": observed[key]}
        for key, value in expected.items()
        if observed[key] != value
    }
    if mismatches:
        raise FreshF1Error(
            "fresh F1 source identity mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if _git(RECBole_ROOT, "rev-parse", "HEAD") != (
        "7b02be5ec80a88310f2d04a27a82adfcbb5dc211"
    ):
        raise FreshF1Error("RecBole commit identity mismatch")
    for name, digest in EXPECTED_SEARCH_FILES.items():
        if bytes_sha256((SEARCH_DATASET_ROOT / name).read_bytes()) != digest:
            raise FreshF1Error(f"search partition identity mismatch: {name}")
    if require_fresh_root and F1_ROOT.exists():
        raise FreshF1Error(f"fresh F1 root already exists: {F1_ROOT}")
    return canonical_value(
        {
            **observed,
            "campaign_id": F1_CAMPAIGN_ID,
            "model": MODEL,
            "proposal_seeds": F1_PROPOSAL_SEEDS,
            "qualification_seed": F1_QUALIFICATION_SEED,
            "training_seed": F1_TRAINING_SEED,
        }
    )


def _build_active_f1_profile(registry: Any) -> tuple[Any, Any, Any, Any, Any]:
    current = adapt_current_search_profile(
        campaign_id="recclaw-pre-f1-frozen-campaign-v1"
    )
    wave2_path = (
        Path(__file__).resolve().parents[4]
        / "docs/research_line/vnext/WAVE2_INTEGRATED_GATE_RECEIPT.json"
    )
    wave2 = json.loads(wave2_path.read_text(encoding="utf-8"))
    current_slate_digest = str(wave2["evidence"]["current_route"]["slate_digest"])
    manifest = NextFreshProfileBuildManifest(
        profile_version="fresh-f1-open-meta-registry-v1",
        predecessor_profile_ref=current.profile_ref,
        predecessor_profile_digest=current.profile_digest,
        current_campaign_profile_ref=current.profile_ref,
        current_campaign_profile_digest=current.profile_digest,
        current_campaign_slate_ref=(
            "recclaw-frozen-experiment-slate-e0-v1:" + current_slate_digest
        ),
        current_campaign_slate_digest=current_slate_digest,
        predecessor_executable_entries=predecessor_executable_entries(current),
        registry_ref=registry.registry_id,
        registry_digest=registry.digest,
        registry_version=registry.registry_version,
        protocol_ref=registry.protocol_ref,
        protocol_digest=registry.protocol_digest,
        compatibility_requirements=PROTOCOL_REQUIREMENTS,
    )
    next_profile, receipt = build_next_fresh_profile(manifest, registry)
    active = activate_next_fresh_search_profile(
        predecessor=current,
        next_profile=next_profile,
        registry=registry,
        fresh_campaign_id=F1_CAMPAIGN_ID,
    )
    return current, manifest, next_profile, receipt, active


def _bindings(active: Any) -> dict[str, Any]:
    return canonical_value(
        {
            "protocol_ref": active.protocol_ref,
            "protocol_digest": active.protocol_digest,
            "context_ref": F1_CONTEXT_REF,
            "context_digest": F1_CONTEXT_DIGEST,
            "current_profile_ref": active.profile_ref,
            "current_profile_digest": active.profile_digest,
            "implementation_requirements": (
                "RecBole general recommender interface",
                "candidate-local package",
            ),
            "compatibility_requirements": PROTOCOL_REQUIREMENTS,
        }
    )


def _environment(active: Any) -> dict[str, Any]:
    return canonical_value(
        {
            "current_profile_ref": active.profile_ref,
            "current_profile_digest": active.profile_digest,
            "current_capabilities": tuple(
                {
                    "capability_ref": entry.capability_ref,
                    "capability_digest": entry.capability_digest,
                    "semantics_digest": entry.semantic_identity_digest,
                }
                for entry in active.entries
            ),
            "protocol_ref": active.protocol_ref,
            "protocol_digest": active.protocol_digest,
            "protocol_requirements": PROTOCOL_REQUIREMENTS,
            "available_dependencies": AVAILABLE_DEPENDENCIES,
            "budget_limits": BUDGET_LIMITS,
        }
    )


def _render_proposal_prompt(
    template: str,
    *,
    slot_id: str,
    seed: int,
    direction: str,
    direction_rank: int,
    active: Any,
    catalog: Sequence[Mapping[str, str]],
    policy: Mapping[str, Any],
) -> str:
    instruction = R2_ROLE_INSTRUCTIONS[direction] + (
        " This direction and its slot order were allocated by the activated "
        "open Meta policy using only pre-outcome replay evidence."
        " Compatibility requirements must be exact tokens from: "
        + ", ".join(PROTOCOL_REQUIREMENTS)
        + ". Required dependencies must be exact tokens from: "
        + ", ".join(AVAILABLE_DEPENDENCIES)
        + ". Required budgets may not exceed "
        + json.dumps(BUDGET_LIMITS, sort_keys=True)
        + "."
    )
    replacements = {
        "{{CAMPAIGN_ID}}": F1_CAMPAIGN_ID,
        "{{LOGICAL_SLOT_ID}}": slot_id,
        "{{PROPOSAL_SEED}}": str(seed),
        "{{PRODUCER_ROLE}}": direction,
        "{{PRODUCER_ROLE_INSTRUCTION}}": instruction,
        "{{PROTOCOL_REF}}": active.protocol_ref,
        "{{PROTOCOL_DIGEST}}": active.protocol_digest,
        "{{CONTEXT_REF}}": F1_CONTEXT_REF,
        "{{CONTEXT_DIGEST}}": F1_CONTEXT_DIGEST,
        "{{PROFILE_REF}}": active.profile_ref,
        "{{PROFILE_DIGEST}}": active.profile_digest,
        "{{PROFILE_CATALOG_JSON}}": json.dumps(
            catalog, sort_keys=True, separators=(",", ":")
        ),
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    if "{{" in rendered or "}}" in rendered:
        raise FreshF1Error("fresh F1 proposal prompt has an unresolved placeholder")
    forbidden = ("side_a", "side_b", "outcome_digest", "ndcg_delta", "episode_digest")
    if any(token in rendered for token in forbidden):
        raise FreshF1Error("fresh F1 proposal prompt leaked origin or outcome")
    policy_binding = json.dumps(
        {
            "activated_policy_digest": policy["policy_digest"],
            "activated_policy_version": policy["policy_version"],
            "direction_rank": direction_rank,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return rendered + "\nActivated open Meta allocation: " + policy_binding + "\n"


def _episode(
    *,
    spec: Any,
    selected_capability: Any,
    materialized: Any,
    qualification: Any,
    baseline_run: Mapping[str, Any],
    candidate_run: Mapping[str, Any],
    active_profile: Any,
    activation: Mapping[str, Any],
) -> TypedResearchEpisodeV1:
    outcome = canonical_value(
        {
            "baseline_metrics": baseline_run["metrics"],
            "candidate_metrics": candidate_run["metrics"],
            "metric": "ndcg@10",
            "partition": "DEVELOPMENT_VALIDATION",
            "seed": candidate_run["seed"],
            "single_seed_interpretation": "INCONCLUSIVE",
        }
    )
    cost = canonical_value(
        {
            "baseline_wall_time_ms": baseline_run["wall_time_ms"],
            "candidate_wall_time_ms": candidate_run["wall_time_ms"],
            "physical_training_runs": 2,
        }
    )
    binding = canonical_value(
        {
            "active_profile_digest": active_profile.profile_digest,
            "activated_policy_digest": activation["policy_digest"],
            "activation_digest": activation["activation_digest"],
            "baseline_binding_digest": baseline_run["binding_digest"],
            "candidate_binding_digest": candidate_run["binding_digest"],
            "materialized_package_digest": materialized.package.digest,
            "matched_seed": candidate_run["seed"],
            "selected_registry_capability_digest": selected_capability.digest,
        }
    )
    return TypedResearchEpisodeV1(
        campaign_id=F1_CAMPAIGN_ID,
        context_ref=spec.context_ref,
        context_digest=spec.context_digest,
        hypothesis=spec.hypothesis,
        executable_capability_ref=selected_capability.capability_id,
        executable_capability_digest=selected_capability.digest,
        executable_profile_ref=active_profile.profile_ref,
        executable_profile_digest=active_profile.profile_digest,
        experiment_binding_ref=f"{F1_RUN_IDENTITY}-experiment-binding:{sha256_digest(binding)}",
        experiment_binding_digest=sha256_digest(binding),
        comparator_ref=f"{F1_RUN_IDENTITY}-bpr-comparator:{baseline_run['binding_digest']}",
        comparator_digest=sha256_digest(baseline_run),
        outcome_ref=f"{F1_RUN_IDENTITY}-development-outcome:{sha256_digest(outcome)}",
        outcome_digest=sha256_digest(outcome),
        cost_ref=f"{F1_RUN_IDENTITY}-development-cost:{sha256_digest(cost)}",
        cost_digest=sha256_digest(cost),
        protocol_ref=spec.protocol_ref,
        protocol_digest=spec.protocol_digest,
        evidence_class=EpisodeEvidenceClassV1.INCONCLUSIVE_EXPERIMENT,
        experiment_executed=True,
        mechanism_interpretation="NOT_ADJUDICATED",
        competing_explanation=spec.competing_explanation,
        failure_class=ResearchFailureClassV1.INCONCLUSIVE,
        mechanism_negative_evidence=False,
        next_discriminative_test=spec.falsifier,
        qualification_receipt_ref=qualification.receipt.receipt_id,
        qualification_receipt_digest=qualification.receipt.digest,
        qualification_evidence_used_as_scientific=False,
    )


def offline_f1_check(repo_root: Path) -> dict[str, Any]:
    identity = verify_f1_source_identity(repo_root, require_fresh_root=False)
    dataset = build_f1_replay_dataset(r1_root=R1_EXTERNAL_ROOT, r2_root=R2_EXTERNAL_ROOT)
    shadow_policy = fit_open_meta_policy(
        dataset,
        splits=("SEARCH_TRAIN",),
        policy_version=F1_POLICY_VERSION + "-shadow",
    )
    shadow = shadow_evaluate_open_meta(dataset, shadow_policy)
    final_policy = fit_open_meta_policy(
        dataset,
        splits=("SEARCH_TRAIN", "DEVELOPMENT_VALIDATION"),
        policy_version=F1_POLICY_VERSION,
    )
    promotion = evaluate_development_promotion(dataset, shadow, final_policy)
    activation = build_policy_activation(
        final_policy, promotion, campaign_id=F1_CAMPAIGN_ID
    )
    artifacts, _receipt = load_registered_r1_artifacts(repo_root)
    registry = build_r1_registry(artifacts)
    current, manifest, next_profile, build_receipt, active = _build_active_f1_profile(registry)
    catalog = public_active_profile_catalog(active, artifacts, seed=F1_PROPOSAL_SEEDS[0])
    return canonical_value(
        {
            "status": "F1_OFFLINE_ARCHITECTURE_READY",
            "identity": identity,
            "dataset_digest": dataset["dataset_digest"],
            "row_count": dataset["row_count"],
            "episode_count": dataset["episode_count"],
            "shadow_digest": shadow["shadow_digest"],
            "promotion_status": promotion["status"],
            "policy_digest": final_policy["policy_digest"],
            "activation_digest": activation["activation_digest"],
            "direction_order": final_policy["direction_order"],
            "current_entry_count": len(current.entries),
            "active_entry_count": len(active.entries),
            "registry_count": len(registry.capabilities),
            "manifest_digest": manifest.digest,
            "next_profile_digest": next_profile.digest,
            "profile_build_receipt_digest": build_receipt.digest,
            "catalog_count": len(catalog),
            "held_out_reads": 0,
        }
    )


def _verify_runtime_recovery_inputs(
    repo_root: Path,
    *,
    require_fresh_root: bool,
) -> tuple[dict[str, Any], Path, Any, dict[str, Any], dict[str, Any]]:
    """Bind the v1 pre-outcome chain without reusing its runtime outcome."""

    paths = {
        "v1_external_receipt": F1_ROOT / "F1_CANONICAL_RECEIPT.json",
        "v1_repo_receipt": (
            repo_root
            / "docs/research_line/vnext/F1_OPEN_META_CANONICAL_RECEIPT.json"
        ),
        "policy": F1_ROOT / "policy/versioned_policy.json",
        "activation": F1_ROOT / "activation/activation.json",
        "selection": F1_ROOT / "selection/selection.json",
        "qualification": F1_ROOT / "qualification/slot-01.json",
        "active_profile": F1_ROOT / "registry/active_search_profile.json",
        "selected_spec": F1_ROOT / "specs/slot-01.json",
    }
    expected_hashes = {
        "v1_external_receipt": F1_V1_EXTERNAL_RECEIPT_SHA256,
        "v1_repo_receipt": F1_V1_REPO_RECEIPT_SHA256,
        "policy": F1_V1_POLICY_FILE_SHA256,
        "activation": F1_V1_ACTIVATION_FILE_SHA256,
        "selection": F1_V1_SELECTION_FILE_SHA256,
        "qualification": F1_V1_QUALIFICATION_FILE_SHA256,
    }
    observed_hashes = {
        name: bytes_sha256(path.read_bytes()) for name, path in paths.items()
    }
    if any(
        observed_hashes[name] != expected_digest
        for name, expected_digest in expected_hashes.items()
    ):
        raise FreshF1Error(
            "v1 recovery input byte identity drift: "
            + json.dumps(
                {
                    name: {
                        "expected": expected_hashes[name],
                        "observed": observed_hashes[name],
                    }
                    for name in expected_hashes
                    if expected_hashes[name] != observed_hashes[name]
                },
                sort_keys=True,
            )
        )
    v1_receipt = _read_json(paths["v1_external_receipt"])
    selection = _read_json(paths["selection"])
    activation = _read_json(paths["activation"])
    policy = _read_json(paths["policy"])
    qualification = _read_json(paths["qualification"])
    active_profile = _read_json(paths["active_profile"])
    selected_spec_artifact = _read_json(paths["selected_spec"])
    if (
        v1_receipt.get("status") != "F1_ARCHITECTURE_EFFECT_FAIL"
        or v1_receipt.get("held_out_reads") != 0
        or v1_receipt.get("manual_candidate_patches") != 0
        or v1_receipt["matched_control"]["candidate"].get("launcher_return_code")
        != 124
        or v1_receipt["matched_control"]["candidate"].get("wall_time_ms")
        < 1_500_000
        or v1_receipt.get("episode") is not None
        or v1_receipt["architecture_effect_gates"].get(
            "serves_open_algorithm_research_target"
        )
        is not True
    ):
        raise FreshF1Error("v1 receipt is not the exact resource-timeout failure")
    if (
        activation.get("policy_digest") != policy.get("policy_digest")
        or selection.get("activated_policy_digest") != policy.get("policy_digest")
        or selection.get("activation_digest") != activation.get("activation_digest")
        or selection.get("selected_slot") != "slot-01"
        or v1_receipt["selection"] != selection
        or v1_receipt["activation"] != activation
    ):
        raise FreshF1Error("v1 policy/activation/selection binding drift")
    qualification_without_behavior = {
        key: value
        for key, value in qualification.items()
        if key != "behavioral_mechanism_evidence"
    }
    if (
        qualification_without_behavior != v1_receipt["qualification"]
        or qualification["receipt"].get("status") != "PASS"
        or qualification["behavioral_mechanism_evidence"].get("probe_status")
        != "PASS"
    ):
        raise FreshF1Error("v1 qualification binding drift")
    if (
        active_profile.get("profile_digest")
        != v1_receipt["profile"]["active_profile_digest"]
        or active_profile.get("profile_ref")
        != f"recclaw-executable-profile-vnext:{active_profile.get('profile_digest')}"
    ):
        raise FreshF1Error("v1 active profile binding drift")

    candidate_parent = F1_ROOT / "execution/candidates/slot-01"
    candidate_roots = tuple(path for path in candidate_parent.iterdir() if path.is_dir())
    if len(candidate_roots) != 1:
        raise FreshF1Error("v1 selected candidate root is not unique")
    candidate_root = candidate_roots[0]
    source_tree_digest = sha256_digest(
        {"files": snapshot_candidate_tree(candidate_root)}
    )
    if source_tree_digest != qualification["receipt"]["source_tree_digest"]:
        raise FreshF1Error("v1 selected candidate source tree byte drift")

    artifacts, _receipt = load_registered_r1_artifacts(repo_root)
    selected_capability = next(
        (
            item.capability
            for item in artifacts
            if item.capability.capability_id
            == selection["selected_capability_ref"]
        ),
        None,
    )
    if (
        selected_capability is None
        or selected_capability.digest != selection["selected_capability_digest"]
    ):
        raise FreshF1Error("v1 selected registry capability binding drift")
    spec_payload = selected_spec_artifact["research_spec"]
    selected_proposal = next(
        row
        for row in v1_receipt["proposal_records"]
        if row.get("logical_slot_id") == selection["selected_slot"]
    )
    if (
        sha256_digest(spec_payload) != selected_proposal.get("spec_digest")
        or selected_spec_artifact["resolution"].get("resolved_current_capability_ref")
        != selected_capability.capability_id
    ):
        raise FreshF1Error("v1 selected OpenSpec identity drift")
    branch = _git(repo_root, "branch", "--show-current")
    head = _git(repo_root, "rev-parse", "HEAD")
    source_tree = _git(repo_root, "rev-parse", "HEAD^{tree}")
    if (
        branch != "feat/research-line-f1-meta"
        or head != ACCEPTED_R2_COMMIT
        or source_tree != ACCEPTED_R2_TREE
    ):
        raise FreshF1Error("runtime recovery source identity drift")
    if _git(RECBole_ROOT, "rev-parse", "HEAD") != (
        "7b02be5ec80a88310f2d04a27a82adfcbb5dc211"
    ):
        raise FreshF1Error("RecBole commit identity mismatch")
    for name, digest in EXPECTED_SEARCH_FILES.items():
        if bytes_sha256((SEARCH_DATASET_ROOT / name).read_bytes()) != digest:
            raise FreshF1Error(f"search partition identity mismatch: {name}")
    if require_fresh_root and F1_RECOVERY_ROOT.exists():
        raise FreshF1Error(
            f"fresh F1 runtime recovery root already exists: {F1_RECOVERY_ROOT}"
        )
    identity = canonical_value(
        {
            "schema": "recclaw.research-line.fresh-f1-runtime-recovery-identity.v2",
            "run_identity": F1_RECOVERY_RUN_IDENTITY,
            "campaign_id": F1_RECOVERY_CAMPAIGN_ID,
            "branch": branch,
            "head": head,
            "source_tree": source_tree,
            "model": v1_receipt["attempt_identity"]["model"],
            "endpoint_digest": v1_receipt["attempt_identity"]["endpoint_digest"],
            "policy_digest": policy["policy_digest"],
            "activation_digest": activation["activation_digest"],
            "selection_file_sha256": observed_hashes["selection"],
            "qualification_file_sha256": observed_hashes["qualification"],
            "active_profile_file_sha256": observed_hashes["active_profile"],
            "selected_spec_file_sha256": observed_hashes["selected_spec"],
            "executable_profile_ref": active_profile["profile_ref"],
            "executable_profile_digest": active_profile["profile_digest"],
            "candidate_package_digest": qualification["receipt"][
                "candidate_package_digest"
            ],
            "candidate_source_tree_digest": source_tree_digest,
            "v1_external_receipt_sha256": observed_hashes["v1_external_receipt"],
            "v1_repo_receipt_sha256": observed_hashes["v1_repo_receipt"],
            "matched_seed": F1_TRAINING_SEED,
            "epochs_per_arm": 100,
            "timeout_seconds_per_arm": F1_RECOVERY_TIMEOUT_SECONDS,
            "timeout_is_symmetric": True,
            "held_out_reads": 0,
        }
    )
    return identity, candidate_root, selected_capability, spec_payload, qualification


def _runtime_recovery_episode(
    *,
    spec: Mapping[str, Any],
    selected_capability: Any,
    qualification: Mapping[str, Any],
    identity: Mapping[str, Any],
    baseline_run: Mapping[str, Any],
    candidate_run: Mapping[str, Any],
) -> TypedResearchEpisodeV1:
    outcome = canonical_value(
        {
            "baseline_metrics": baseline_run["metrics"],
            "candidate_metrics": candidate_run["metrics"],
            "metric": "ndcg@10",
            "partition": "DEVELOPMENT_VALIDATION",
            "seed": F1_TRAINING_SEED,
            "single_seed_interpretation": "INCONCLUSIVE",
        }
    )
    cost = canonical_value(
        {
            "baseline_wall_time_ms": baseline_run["wall_time_ms"],
            "candidate_wall_time_ms": candidate_run["wall_time_ms"],
            "physical_training_runs": 2,
            "symmetric_timeout_seconds_per_arm": F1_RECOVERY_TIMEOUT_SECONDS,
        }
    )
    binding = canonical_value(
        {
            "activation_digest": identity["activation_digest"],
            "baseline_binding_digest": baseline_run["binding_digest"],
            "candidate_binding_digest": candidate_run["binding_digest"],
            "candidate_package_digest": identity["candidate_package_digest"],
            "candidate_source_tree_digest": identity["candidate_source_tree_digest"],
            "matched_seed": F1_TRAINING_SEED,
            "policy_digest": identity["policy_digest"],
            "selection_file_sha256": identity["selection_file_sha256"],
        }
    )
    v1_receipt = _read_json(F1_ROOT / "F1_CANONICAL_RECEIPT.json")
    return TypedResearchEpisodeV1(
        campaign_id=F1_RECOVERY_CAMPAIGN_ID,
        context_ref=str(spec["context_ref"]),
        context_digest=str(spec["context_digest"]),
        hypothesis=str(spec["hypothesis"]),
        executable_capability_ref=selected_capability.capability_id,
        executable_capability_digest=selected_capability.digest,
        executable_profile_ref=str(identity["executable_profile_ref"]),
        executable_profile_digest=str(identity["executable_profile_digest"]),
        experiment_binding_ref=(
            f"{F1_RECOVERY_RUN_IDENTITY}-experiment-binding:{sha256_digest(binding)}"
        ),
        experiment_binding_digest=sha256_digest(binding),
        comparator_ref=(
            f"{F1_RECOVERY_RUN_IDENTITY}-bpr-comparator:{baseline_run['binding_digest']}"
        ),
        comparator_digest=sha256_digest(baseline_run),
        outcome_ref=(
            f"{F1_RECOVERY_RUN_IDENTITY}-development-outcome:{sha256_digest(outcome)}"
        ),
        outcome_digest=sha256_digest(outcome),
        cost_ref=(
            f"{F1_RECOVERY_RUN_IDENTITY}-development-cost:{sha256_digest(cost)}"
        ),
        cost_digest=sha256_digest(cost),
        protocol_ref=str(spec["protocol_ref"]),
        protocol_digest=str(spec["protocol_digest"]),
        evidence_class=EpisodeEvidenceClassV1.INCONCLUSIVE_EXPERIMENT,
        experiment_executed=True,
        mechanism_interpretation="NOT_ADJUDICATED",
        competing_explanation=str(spec["competing_explanation"]),
        failure_class=ResearchFailureClassV1.INCONCLUSIVE,
        mechanism_negative_evidence=False,
        next_discriminative_test=str(spec["falsifier"]),
        qualification_receipt_ref=(
            "recclaw-qualification-receipt-v1:"
            f"{sha256_digest(qualification['receipt'])}"
        ),
        qualification_receipt_digest=sha256_digest(qualification["receipt"]),
        qualification_evidence_used_as_scientific=False,
    )


def run_f1_runtime_recovery(
    repo_root: Path,
    *,
    canonical_receipt_path: Path,
) -> dict[str, Any]:
    """Fresh symmetric 2400-second recovery of the exact v1 matched arms."""

    repo_root = repo_root.resolve()
    started_ns = time.monotonic_ns()
    (
        identity,
        candidate_root,
        selected_capability,
        spec,
        qualification,
    ) = _verify_runtime_recovery_inputs(repo_root, require_fresh_root=True)
    if canonical_receipt_path.exists():
        raise FreshF1Error(
            f"runtime recovery repository receipt already exists: {canonical_receipt_path}"
        )
    F1_RECOVERY_ROOT.mkdir(parents=True)
    _write_new_json(F1_RECOVERY_ROOT / "RUN_IDENTITY.json", identity)
    frozen_runtime_rule = canonical_value(
        {
            "schema": "recclaw.research-line.fresh-f1-runtime-recovery-rule.v2",
            "arms": ("MATCHED_BPR_CONTROL", "LEARNED_POLICY_SELECTED_CANDIDATE"),
            "timeout_seconds_per_arm": F1_RECOVERY_TIMEOUT_SECONDS,
            "timeout_is_symmetric": True,
            "seed_per_arm": F1_TRAINING_SEED,
            "epochs_per_arm": 100,
            "dataset_partition": "SEARCH_TRAIN_PLUS_DEVELOPMENT_VALIDATION_ONLY",
            "policy_digest": identity["policy_digest"],
            "activation_digest": identity["activation_digest"],
            "selection_file_sha256": identity["selection_file_sha256"],
            "candidate_package_digest": identity["candidate_package_digest"],
            "candidate_source_tree_digest": identity["candidate_source_tree_digest"],
            "held_out_reads": 0,
        }
    )
    runtime_rule_digest = _write_new_json(
        F1_RECOVERY_ROOT / "PREFROZEN_RUNTIME_RULE.json", frozen_runtime_rule
    )

    baseline_source = RECBole_ROOT / "recbole/model/general_recommender/bpr.py"
    baseline_run = run_development_training(
        repo_root=repo_root,
        side_root=F1_RECOVERY_ROOT,
        run_id="matched-bpr-control",
        seed=F1_TRAINING_SEED,
        candidate_root=None,
        entrypoint="recbole.model.general_recommender.bpr:BPR",
        source_sha256=bytes_sha256(baseline_source.read_bytes()),
        run_identity=F1_RECOVERY_RUN_IDENTITY,
        authority="user-delegated-fresh-f1-runtime-recovery-v2",
        timeout_seconds=F1_RECOVERY_TIMEOUT_SECONDS,
    )
    candidate_source = candidate_root / "recclaw_ext/candidate.py"
    candidate_entrypoint = str(
        _read_json(
            F1_ROOT / "execution/experiments/learned-policy-selected-candidate/"
            "execution_recipe.json"
        )["execution_recipe"]["entrypoint"]
    )
    candidate_run = run_development_training(
        repo_root=repo_root,
        side_root=F1_RECOVERY_ROOT,
        run_id="learned-policy-selected-candidate",
        seed=F1_TRAINING_SEED,
        candidate_root=candidate_root,
        entrypoint=candidate_entrypoint,
        source_sha256=bytes_sha256(candidate_source.read_bytes()),
        run_identity=F1_RECOVERY_RUN_IDENTITY,
        authority="user-delegated-fresh-f1-runtime-recovery-v2",
        timeout_seconds=F1_RECOVERY_TIMEOUT_SECONDS,
    )
    training_closed = (
        baseline_run.get("exit_status") == "SUCCESS"
        and candidate_run.get("exit_status") == "SUCCESS"
        and "ndcg@10" in baseline_run.get("metrics", {})
        and "ndcg@10" in candidate_run.get("metrics", {})
    )
    episode = None
    if training_closed:
        episode = _runtime_recovery_episode(
            spec=spec,
            selected_capability=selected_capability,
            qualification=qualification,
            identity=identity,
            baseline_run=baseline_run,
            candidate_run=candidate_run,
        )
        _write_new_json(
            F1_RECOVERY_ROOT / "episodes/runtime_recovery_selected.json",
            episode.canonical_dict(),
        )
    v1_receipt = _read_json(F1_ROOT / "F1_CANONICAL_RECEIPT.json")
    architecture_gates = {
        "function_real_and_runnable": training_closed,
        "end_to_end_result_chain_real_and_valid": episode is not None,
        "serves_open_algorithm_research_target": (
            v1_receipt["architecture_effect_gates"][
                "serves_open_algorithm_research_target"
            ]
            is True
            and identity["policy_digest"]
            == v1_receipt["learner"]["policy_semantic_digest"]
            and identity["activation_digest"]
            == v1_receipt["activation"]["activation_digest"]
        ),
        "no_fixed_66_tuning_static_wrapper_fallback_mock_or_smoke_substitution": (
            training_closed
            and baseline_run.get("epochs_requested") == 100
            and candidate_run.get("epochs_requested") == 100
            and baseline_run.get("seed") == candidate_run.get("seed") == F1_TRAINING_SEED
            and identity["candidate_source_tree_digest"]
            == qualification["receipt"]["source_tree_digest"]
        ),
    }
    status = (
        "F1_ARCHITECTURE_EFFECT_PASS"
        if all(architecture_gates.values())
        else "F1_RESOURCE_HARD_BLOCKED"
    )
    receipt = canonical_value(
        {
            "schema": "recclaw.research-line.fresh-f1-runtime-recovery-canonical-receipt.v2",
            "status": status,
            "development_only": True,
            "architecture_effect_gates": architecture_gates,
            "attempt_identity": identity,
            "v1_failure_preserved": {
                "root": str(F1_ROOT),
                "external_receipt_sha256": F1_V1_EXTERNAL_RECEIPT_SHA256,
                "repo_receipt_sha256": F1_V1_REPO_RECEIPT_SHA256,
                "classification": "RESOURCE_TIMEOUT_MISSINGNESS_NOT_MECHANISM_EVIDENCE",
            },
            "runtime_recovery": {
                "prefrozen_rule_digest": runtime_rule_digest,
                "timeout_seconds_per_arm": F1_RECOVERY_TIMEOUT_SECONDS,
                "timeout_is_symmetric": True,
                "control": baseline_run,
                "candidate": candidate_run,
                "matched_seed": F1_TRAINING_SEED,
                "epochs_per_arm": 100,
                "new_physical_training_runs": 2,
                "reused_runtime_outcomes": 0,
            },
            "frozen_chain_binding": {
                "policy_digest": identity["policy_digest"],
                "activation_digest": identity["activation_digest"],
                "selection_file_sha256": identity["selection_file_sha256"],
                "qualification_file_sha256": identity["qualification_file_sha256"],
                "active_profile_file_sha256": identity[
                    "active_profile_file_sha256"
                ],
                "selected_spec_file_sha256": identity[
                    "selected_spec_file_sha256"
                ],
                "candidate_package_digest": identity["candidate_package_digest"],
                "candidate_source_tree_digest": identity["candidate_source_tree_digest"],
                "new_provider_calls": 0,
                "new_provider_retries": 0,
                "selection_rule_changed": False,
                "candidate_manual_patches": 0,
            },
            "episode": episode.canonical_dict() if episode else None,
            "scientific_interpretation": "INCONCLUSIVE_NOT_ADJUDICATED",
            "policy_superiority_claim": False,
            "scientific_effect_claim": False,
            "held_out_reads": 0,
            "wall_time_ms": max(1, (time.monotonic_ns() - started_ns) // 1_000_000),
        }
    )
    external_sha256 = _write_new_json(
        F1_RECOVERY_ROOT / "F1_RUNTIME_RECOVERY_CANONICAL_RECEIPT.json",
        receipt,
    )
    repository_receipt = canonical_value(
        {
            **receipt,
            "external_receipt_ref": str(
                F1_RECOVERY_ROOT / "F1_RUNTIME_RECOVERY_CANONICAL_RECEIPT.json"
            ),
            "external_receipt_sha256": external_sha256,
        }
    )
    _write_new_json(canonical_receipt_path, repository_receipt)
    return repository_receipt


def run_formal_fresh_f1(
    repo_root: Path,
    *,
    canonical_receipt_path: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    started_ns = time.monotonic_ns()
    identity = verify_f1_source_identity(repo_root, require_fresh_root=True)
    if canonical_receipt_path.exists():
        raise FreshF1Error(f"canonical repository receipt already exists: {canonical_receipt_path}")

    F1_ROOT.mkdir(parents=True)
    _write_new_json(F1_ROOT / "RUN_IDENTITY.json", identity)
    prefrozen_rule_digest = _write_new_json(
        F1_ROOT / "promotion/PREFROZEN_RULE.json", PREFROZEN_PROMOTION_RULE
    )

    dataset = build_f1_replay_dataset(r1_root=R1_EXTERNAL_ROOT, r2_root=R2_EXTERNAL_ROOT)
    replay_digest = _write_new_json(F1_ROOT / "learner/evidence_replay.json", dataset)
    shadow_policy = fit_open_meta_policy(
        dataset,
        splits=("SEARCH_TRAIN",),
        policy_version=F1_POLICY_VERSION + "-shadow",
    )
    shadow_policy_digest = _write_new_json(
        F1_ROOT / "learner/shadow_policy.json", shadow_policy
    )
    shadow = shadow_evaluate_open_meta(dataset, shadow_policy)
    shadow_digest = _write_new_json(F1_ROOT / "shadow/shadow_evaluation.json", shadow)
    final_policy = fit_open_meta_policy(
        dataset,
        splits=("SEARCH_TRAIN", "DEVELOPMENT_VALIDATION"),
        policy_version=F1_POLICY_VERSION,
    )
    policy_digest = _write_new_json(F1_ROOT / "policy/versioned_policy.json", final_policy)
    promotion = evaluate_development_promotion(dataset, shadow, final_policy)
    promotion = canonical_value(
        {**promotion, "prefrozen_rule_digest": prefrozen_rule_digest}
    )
    promotion_digest = _write_new_json(F1_ROOT / "promotion/promotion.json", promotion)
    activation = build_policy_activation(
        final_policy, promotion, campaign_id=F1_CAMPAIGN_ID
    )
    activation = canonical_value(
        {**activation, "promotion_artifact_digest": promotion_digest}
    )
    activation_digest = _write_new_json(F1_ROOT / "activation/activation.json", activation)

    artifacts, _r1_receipt = load_registered_r1_artifacts(repo_root)
    registry = build_r1_registry(artifacts)
    current, manifest, next_profile, profile_receipt, active = _build_active_f1_profile(registry)
    _write_new_json(F1_ROOT / "registry/versioned_capability_registry.json", registry.canonical_dict())
    _write_new_json(F1_ROOT / "registry/next_profile_build_manifest.json", manifest.canonical_dict())
    _write_new_json(F1_ROOT / "registry/next_fresh_profile.json", next_profile.canonical_dict())
    _write_new_json(F1_ROOT / "registry/profile_build_receipt.json", profile_receipt.canonical_dict())
    _write_new_json(F1_ROOT / "registry/active_search_profile.json", active.canonical_dict())

    catalog = public_active_profile_catalog(active, artifacts, seed=F1_PROPOSAL_SEEDS[0])
    catalog_digest = _write_new_json(
        F1_ROOT / "registry/origin_blind_active_catalog.json",
        {"schema": "recclaw.origin-blind-active-catalog.f1.v1", "entries": catalog},
    )
    proposal_schema = derive_fresh_r2_proposal_schema()
    schema_path = F1_ROOT / "contracts/fresh_f1_proposal_response.schema.json"
    proposal_schema_digest = _write_new_json(schema_path, proposal_schema)
    resources = _resource_root()
    proposal_template_path = resources / "fresh_r2_producer_prompt_v1.txt"
    implementation_template_path = resources / "fresh_r1_implementer_prompt_v1.txt"
    implementation_schema_path = resources / "fresh_r1_implementation_response_v1.schema.json"
    tool_policy_path = resources / "fresh_open_spec_tool_policy_v1.json"
    implementation_template = implementation_template_path.read_text(encoding="utf-8")
    implementation_policy = _shared_policy(
        bytes_sha256(implementation_template_path.read_bytes()),
        bytes_sha256(tool_policy_path.read_bytes()),
    )
    proposal_template = proposal_template_path.read_text(encoding="utf-8")
    bindings = _bindings(active)
    environment = _environment(active)
    registry_refs = {item.capability.capability_id for item in artifacts}

    direction_order = tuple(str(value) for value in final_policy["direction_order"])
    if set(direction_order) != set(STATIC_DIRECTION_ORDER):
        raise FreshF1Error("learned Idea policy did not retain the full direction domain")
    proposal_records: list[dict[str, Any]] = []
    live_specs: dict[str, tuple[Any, Mapping[str, Any], Any]] = {}
    for index, direction in enumerate(direction_order):
        slot_id = f"slot-{index + 1:02d}"
        seed = F1_PROPOSAL_SEEDS[index]
        prompt = _render_proposal_prompt(
            proposal_template,
            slot_id=slot_id,
            seed=seed,
            direction=direction,
            direction_rank=index + 1,
            active=active,
            catalog=catalog,
            policy=final_policy,
        )
        call_result = bounded_provider_call(
            call_root=F1_ROOT / "provider/proposals" / slot_id,
            schema_path=schema_path,
            logical_call_id=f"{F1_RUN_IDENTITY}:{slot_id}:proposal",
            session_id=f"{F1_RUN_IDENTITY}:proposal-session",
            prompt=prompt,
            token_ceiling=F1_PROPOSAL_TOKEN_CEILING,
        )
        record: dict[str, Any] = {
            "denominator_included": True,
            "logical_slot_id": slot_id,
            "producer_direction": direction,
            "learned_direction_rank": index + 1,
            "proposal_seed": seed,
            "activated_policy_digest": final_policy["policy_digest"],
            "activation_digest": activation["activation_digest"],
            "provider_attempts": call_result.attempts,
        }
        if call_result.call is None:
            record.update(
                {"failure": call_result.failure, "slot_status": "PROPOSAL_PROVIDER_FAILURE"}
            )
            proposal_records.append(record)
            _write_new_json(F1_ROOT / "slots" / f"{slot_id}.json", record)
            continue
        call = call_result.call
        record["proposal_response_digest"] = call.response_digest
        record["returned_model"] = call.returned_model
        try:
            validate_v4_response_contract(call.response, provider_schema=proposal_schema)
            draft = call.response["proposals"][0]
            if draft["producer_role"] != direction:
                raise FreshF1Error("Provider changed the learned direction allocation")
            spec, facts = project_open_producer_draft(draft, bindings=bindings)
            resolution = resolve_capability(spec, resolution_facts=facts, environment=environment)
            record.update(
                {
                    "resolution": resolution.resolution.value,
                    "resolution_digest": resolution.digest,
                    "resolution_reason_codes": resolution.reason_codes,
                    "resolved_capability_digest": resolution.resolved_current_capability_digest,
                    "resolved_capability_ref": resolution.resolved_current_capability_ref,
                    "selected_registry_capability": (
                        resolution.resolved_current_capability_ref in registry_refs
                    ),
                    "spec_digest": spec.digest,
                    "spec_ref": spec.spec_id,
                    "slot_status": "RESOLVED",
                }
            )
            live_specs[slot_id] = (spec, facts, resolution)
            _write_new_json(
                F1_ROOT / "specs" / f"{slot_id}.json",
                {
                    "research_spec": spec.canonical_dict(),
                    "resolution_facts": facts,
                    "resolution": resolution.canonical_dict(),
                },
            )
        except (FreshF1Error, jsonschema.ValidationError, ValueError) as error:
            record.update(
                {
                    "failure": {
                        "error_type": type(error).__name__,
                        "mechanism_negative_evidence": False,
                        "stage": "PROPOSAL_RESOLUTION",
                    },
                    "slot_status": "PROPOSAL_RESOLUTION_FAILURE",
                }
            )
        proposal_records.append(record)
        _write_new_json(F1_ROOT / "slots" / f"{slot_id}.json", record)

    ranked = rank_search_ready_records(final_policy, proposal_records)
    if not ranked:
        blocked = canonical_value(
            {
                "schema": "recclaw.research-line.fresh-f1-blocked-receipt.v1",
                "status": "F1_HARD_BLOCKED_NO_SEARCH_READY_FRESH_PROPOSAL",
                "activation_digest": activation_digest,
                "policy_digest": policy_digest,
                "proposal_records": proposal_records,
                "held_out_reads": 0,
                "manual_candidate_patches": 0,
            }
        )
        _write_new_json(F1_ROOT / "F1_BLOCKED_RECEIPT.json", blocked)
        raise FreshF1Error(
            "fresh blinded Producers produced no SEARCH_READY registry consumer; "
            "candidate-specific intervention is forbidden"
        )
    selected_record = ranked[0]
    selected_slot = str(selected_record["logical_slot_id"])
    selected_spec, _facts, selected_resolution = live_specs[selected_slot]
    selected_capability = next(
        item.capability
        for item in artifacts
        if item.capability.capability_id == selected_resolution.resolved_current_capability_ref
    )
    static_search_ready = next(
        row for row in proposal_records if row.get("resolution") == "SEARCH_READY"
    )
    selection = canonical_value(
        {
            "activated_policy_digest": final_policy["policy_digest"],
            "activation_digest": activation["activation_digest"],
            "active_profile_digest": active.profile_digest,
            "catalog_digest": catalog_digest,
            "learned_rule": "UNRESOLVED_ATTEMPT_GAP_THEN_MISSING_EPISODE_INFORMATION_VALUE_AND_RUNTIME_RELIABILITY",
            "learned_ranked_search_ready": tuple(
                {
                    "slot": row["logical_slot_id"],
                    "capability_ref": row["resolved_capability_ref"],
                    "score": row["learned_experiment_score"],
                }
                for row in ranked
            ),
            "selected_capability_ref": selected_capability.capability_id,
            "selected_capability_digest": selected_capability.digest,
            "selected_slot": selected_slot,
            "static_control_rule": "FIRST_SEARCH_READY_IN_STATIC_DIRECTION_SLOT_ORDER",
            "static_control_selected_slot_under_observed_calls": static_search_ready[
                "logical_slot_id"
            ],
            "idea_orchestration_changed_from_static": direction_order
            != STATIC_DIRECTION_ORDER,
            "experiment_selection_changed_within_observed_calls": selected_slot
            != static_search_ready["logical_slot_id"],
        }
    )
    selection_digest = _write_new_json(F1_ROOT / "selection/selection.json", selection)

    request_prompt = render_implementation_prompt(
        implementation_template,
        build_shared_implementer_request(selected_spec, policy=implementation_policy),
    )
    implementation_call = bounded_provider_call(
        call_root=F1_ROOT / "provider/implementation" / selected_slot,
        schema_path=implementation_schema_path,
        logical_call_id=f"{F1_RUN_IDENTITY}:{selected_slot}:implementation",
        session_id=f"{F1_RUN_IDENTITY}:shared-origin-blind-implementation-session",
        prompt=request_prompt,
        token_ceiling=IMPLEMENTATION_TOKEN_CEILING,
    )
    if implementation_call.call is None:
        raise FreshF1Error("learned-policy-selected implementation call failed")
    materialized, qualification, behavior = _materialize_and_qualify(
        repo_root=repo_root,
        side_root=F1_ROOT / "execution",
        slot_id=selected_slot,
        seed=F1_QUALIFICATION_SEED,
        spec=selected_spec,
        implementation=implementation_call.call.response["proposals"][0],
        implementation_prompt_digest=bytes_sha256(implementation_template_path.read_bytes()),
        tool_policy_digest=bytes_sha256(tool_policy_path.read_bytes()),
        run_identity=F1_RUN_IDENTITY,
    )
    _write_new_json(
        F1_ROOT / "qualification" / f"{selected_slot}.json",
        {**qualification.to_dict(), "behavioral_mechanism_evidence": behavior},
    )
    if qualification.receipt.status is not QualificationStatusV1.PASS:
        raise FreshF1Error("learned-policy-selected candidate failed Mechanical Qualifier")

    candidate_root = (
        F1_ROOT
        / "execution/candidates"
        / selected_slot
        / str(materialized.shared_request["blind_candidate_id"])
    )
    baseline_source = RECBole_ROOT / "recbole/model/general_recommender/bpr.py"
    baseline_run = run_development_training(
        repo_root=repo_root,
        side_root=F1_ROOT / "execution",
        run_id="matched-bpr-control",
        seed=F1_TRAINING_SEED,
        candidate_root=None,
        entrypoint="recbole.model.general_recommender.bpr:BPR",
        source_sha256=bytes_sha256(baseline_source.read_bytes()),
        run_identity=F1_RUN_IDENTITY,
        authority="user-delegated-fresh-f1-open-meta",
    )
    candidate_source = candidate_root / "recclaw_ext/candidate.py"
    candidate_run = run_development_training(
        repo_root=repo_root,
        side_root=F1_ROOT / "execution",
        run_id="learned-policy-selected-candidate",
        seed=F1_TRAINING_SEED,
        candidate_root=candidate_root,
        entrypoint=materialized.package.executable_entrypoint,
        source_sha256=bytes_sha256(candidate_source.read_bytes()),
        run_identity=F1_RUN_IDENTITY,
        authority="user-delegated-fresh-f1-open-meta",
    )
    training_closed = (
        baseline_run.get("exit_status") == "SUCCESS"
        and candidate_run.get("exit_status") == "SUCCESS"
        and "ndcg@10" in baseline_run.get("metrics", {})
        and "ndcg@10" in candidate_run.get("metrics", {})
    )
    episode = None
    if training_closed:
        episode = _episode(
            spec=selected_spec,
            selected_capability=selected_capability,
            materialized=materialized,
            qualification=qualification,
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            active_profile=active,
            activation=activation,
        )
        _write_new_json(F1_ROOT / "episodes/learned_policy_selected.json", episode.canonical_dict())

    proposal_usage = _physical_usage(proposal_records)
    implementation_usage = _physical_usage(
        [{"provider_attempts": implementation_call.attempts}]
    )
    architecture_gates = {
        "function_real_and_runnable": (
            qualification.receipt.status is QualificationStatusV1.PASS and training_closed
        ),
        "end_to_end_result_chain_real_and_valid": episode is not None,
        "serves_open_algorithm_research_target": (
            activation["replaces_static_for_campaign"] is True
            and selection["idea_orchestration_changed_from_static"] is True
            and selected_capability.capability_id in registry_refs
            and behavior.get("probe_status") == "PASS"
            and bool(behavior.get("extra_parameter_names"))
        ),
        "no_fixed_66_tuning_static_wrapper_fallback_mock_or_smoke_substitution": (
            active.entry(selected_capability.capability_id).origin
            is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
            and candidate_root.is_relative_to(F1_ROOT)
            and implementation_call.call is not None
            and baseline_run.get("epochs_requested") == 100
            and candidate_run.get("epochs_requested") == 100
            and training_closed
        ),
    }
    status = (
        "F1_ARCHITECTURE_EFFECT_PASS"
        if all(architecture_gates.values())
        else "F1_ARCHITECTURE_EFFECT_FAIL"
    )
    receipt = canonical_value(
        {
            "schema": "recclaw.research-line.fresh-f1-open-meta-canonical-receipt.v1",
            "status": status,
            "development_only": True,
            "architecture_effect_gates": architecture_gates,
            "attempt_identity": identity,
            "source_evidence": {
                "r1_rows": 16,
                "r1_typed_episodes": 7,
                "r2_rows": 4,
                "r2_typed_episodes": 1,
                "r2_repo_receipt_sha256": R2_REPO_RECEIPT_SHA256,
                "r2_external_receipt_sha256": R2_EXTERNAL_RECEIPT_SHA256,
            },
            "learner": {
                "replay_digest": replay_digest,
                "replay_semantic_digest": dataset["dataset_digest"],
                "shadow_policy_digest": shadow_policy_digest,
                "shadow_digest": shadow_digest,
                "policy_file_digest": policy_digest,
                "policy_semantic_digest": final_policy["policy_digest"],
                "policy_version": final_policy["policy_version"],
                "training_splits": final_policy["training_splits"],
                "scientific_outcome_usage": final_policy["scientific_audit"][
                    "interpretation"
                ],
            },
            "promotion": promotion,
            "promotion_artifact_digest": promotion_digest,
            "prefrozen_promotion_rule_digest": prefrozen_rule_digest,
            "activation": activation,
            "activation_artifact_digest": activation_digest,
            "activation_consumer": {
                "producer_direction_order": direction_order,
                "producer_records_bind_policy": all(
                    row["activated_policy_digest"] == final_policy["policy_digest"]
                    for row in proposal_records
                ),
                "resolver_distribution": dict(
                    sorted(
                        Counter(
                            str(row.get("resolution"))
                            for row in proposal_records
                            if row.get("resolution")
                        ).items()
                    )
                ),
                "registry_selected_capability_ref": selected_capability.capability_id,
                "runtime_selected_slot": selected_slot,
                "selection_digest": selection_digest,
                "traceable_behavior_change": selection[
                    "idea_orchestration_changed_from_static"
                ]
                or selection["experiment_selection_changed_within_observed_calls"],
            },
            "profile": {
                "current_entry_count": len(current.entries),
                "active_entry_count": len(active.entries),
                "registered_capability_count": len(registry.capabilities),
                "registry_digest": registry.digest,
                "next_profile_digest": next_profile.digest,
                "active_profile_digest": active.profile_digest,
                "profile_build_receipt_digest": profile_receipt.digest,
                "origin_blind_catalog_digest": catalog_digest,
            },
            "proposal_call_contract": {
                "model": MODEL,
                "origin_blind": True,
                "policy_activated_pre_call": True,
                "response_schema_digest": proposal_schema_digest,
                "slot_count": len(direction_order),
                "token_ceiling": F1_PROPOSAL_TOKEN_CEILING,
                "tools": [],
            },
            "proposal_records": proposal_records,
            "proposal_provider_usage": proposal_usage,
            "implementation_call_contract": {
                "model": MODEL,
                "origin_blind": True,
                "response_schema_digest": bytes_sha256(
                    implementation_schema_path.read_bytes()
                ),
                "token_ceiling": IMPLEMENTATION_TOKEN_CEILING,
                "tools": [],
            },
            "implementation_provider_attempts": implementation_call.attempts,
            "implementation_provider_usage": implementation_usage,
            "qualification": qualification.to_dict(),
            "qualification_evidence_class": "DEVELOPMENT_ONLY",
            "selection": selection,
            "matched_control": {
                "baseline": baseline_run,
                "candidate": candidate_run,
                "matched_seed": F1_TRAINING_SEED,
            },
            "episode": episode.canonical_dict() if episode else None,
            "scientific_interpretation": "INCONCLUSIVE_NOT_ADJUDICATED",
            "policy_superiority_claim": False,
            "scientific_effect_claim": False,
            "training": {
                "dataset_partition": "SEARCH_TRAIN_PLUS_DEVELOPMENT_VALIDATION_ONLY",
                "epochs_requested_per_run": 100,
                "physical_runs": 2,
                "held_out_exposed": False,
                "runtime_ceiling_seconds_per_run": 1500,
            },
            "retry_policy": {
                "backoff_ms": BACKOFF_MS,
                "maximum_physical_attempts": MAX_PHYSICAL_ATTEMPTS,
                "same_slot_same_payload_only": True,
            },
            "manual_candidate_patches": 0,
            "held_out_reads": 0,
            "wall_time_ms": max(1, (time.monotonic_ns() - started_ns) // 1_000_000),
        }
    )
    external_sha256 = _write_new_json(F1_ROOT / "F1_CANONICAL_RECEIPT.json", receipt)
    repository_receipt = canonical_value(
        {
            **receipt,
            "external_receipt_ref": str(F1_ROOT / "F1_CANONICAL_RECEIPT.json"),
            "external_receipt_sha256": external_sha256,
        }
    )
    _write_new_json(canonical_receipt_path, repository_receipt)
    return repository_receipt


__all__ = [
    "F1_ROOT",
    "F1_RECOVERY_ROOT",
    "FreshF1Error",
    "offline_f1_check",
    "run_formal_fresh_f1",
    "run_f1_runtime_recovery",
    "verify_f1_source_identity",
]
