"""Fresh R2 registry-consumer architecture-effect campaign.

R2 consumes the accepted R1 qualified-capability bytes through the existing
versioned registry, next-fresh profile, OpenSpec Resolver, shared blind
Implementer, mechanical RecBole Qualifier, and real development runtime.  It
does not add a catalog, fallback, mutable registry service, or outcome-based
selection path.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import jsonschema

from .canonical import bytes_sha256, canonical_value, sha256_digest
from .capability_admission import VersionedCapabilityRegistry
from .campaign_runtime import executable_mechanisms
from .fresh_r1 import (
    API_CONFIG,
    AVAILABLE_DEPENDENCIES,
    BACKOFF_MS,
    BUDGET_LIMITS,
    IMPLEMENTATION_TOKEN_CEILING,
    MAX_PHYSICAL_ATTEMPTS,
    MODEL,
    PROTOCOL_REQUIREMENTS,
    PROJECTS_ROOT,
    PYTHON_EXECUTABLE,
    RECBole_ROOT,
    SEARCH_DATA_ROOT,
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
    derive_fresh_r1_proposal_schema,
    render_implementation_prompt,
    run_development_training,
)
from .innovation_recbole_adapter import snapshot_candidate_tree
from .innovation_spine import build_shared_implementer_request
from .next_fresh_profile import NextFreshProfileBuildManifest, build_next_fresh_profile
from .open_spec import project_open_producer_draft, resolve_capability
from .search_adapter import (
    SearchProfileEntryOriginV1,
    activate_next_fresh_search_profile,
    adapt_current_search_profile,
    predecessor_executable_entries,
)
from .v4_response_contract import validate_v4_response_contract
from .vnext_contracts import (
    CapabilityKindV1,
    CapabilityResolutionResultV1,
    EpisodeEvidenceClassV1,
    OpenResearchSpecV1,
    QualificationStageV1,
    QualificationStatusV1,
    QualifiedCapabilityV1,
    ResearchFailureClassV1,
    TypedResearchEpisodeV1,
)


ACCEPTED_R1_COMMIT = "11dae330dbbbbf3a8108b19e7f9b9020326f0a20"
ACCEPTED_R1_PARENT = "e43bb78acbfbe1e616cc320cf9ff6b55b2144286"
ACCEPTED_R1_TREE = "3d9bf25a9dbab869ce16e9f6df213f01f139925b"
R1_REPO_RECEIPT_SHA256 = (
    "46c4ac25d068e84221913f5f3cd1fa7f5f7f3452e6ea44ac01f77bdbdd0a41de"
)
R1_EXTERNAL_RECEIPT_SHA256 = (
    "3c952acde2efec2d70285b19b3650799ee6d3eeb7826983767e46112872b6427"
)
R1_EXTERNAL_ROOT = (
    PROJECTS_ROOT
    / "RecClaw_r1_r2_runs/fresh_r1_training_filesystem_fix_v3"
)
R2_ROOT = PROJECTS_ROOT / "RecClaw_r1_r2_runs/fresh_r2_registry_effect_v2"
R2_RUN_IDENTITY = "fresh-r2-registry-effect-v2"
R2_CAMPAIGN_ID = "recclaw-fresh-r2-registry-effect-v2"
R2_LINEAGE = (
    "recclaw-fresh-r2-lineage-v2:source=" + ACCEPTED_R1_COMMIT
    + ":repair=uniform-provider-token-ceiling"
)
R2_CONTEXT_REF = "recclaw-fresh-r2-registry-effect-context-v2"
R2_CONTEXT_DIGEST = sha256_digest(
    {
        "campaign_id": R2_CAMPAIGN_ID,
        "lineage": R2_LINEAGE,
        "purpose": "REGISTRY_CONSUMER_ARCHITECTURE_EFFECT",
    }
)
R2_PROPOSAL_SEEDS = (52011, 52012, 52013, 52014)
R2_QUALIFICATION_SEED = 52101
R2_TRAINING_SEED = 52102
R2_PROPOSAL_TOKEN_CEILING = 12_000
R2_PREDECESSOR_FAILURE_RECEIPT_SHA256 = (
    "6c56a3e716acb4533d91487dbc2a261381f2d6d5b30928d9cc02563c2d05828e"
)
R2_ROLE_SCHEDULE = (
    "mechanism_composer",
    "lineage_refiner",
    "falsification_designer",
    "frontier_architect",
)
R2_ROLE_INSTRUCTIONS = {
    "mechanism_composer": (
        "Compose a coherent multi-part mechanism hypothesis, reusing an active "
        "mechanism only when it is an exact substantive match."
    ),
    "lineage_refiner": (
        "Identify a mechanistic weakness and decide whether an active mechanism "
        "already supplies the required descendant behavior."
    ),
    "falsification_designer": (
        "Choose a mechanism around a decisive matched-control falsifier, without "
        "treating prior outcomes as selection evidence."
    ),
    "frontier_architect": (
        "Seek a structural research frontier while refusing novelty claims when "
        "an active mechanism already implements it."
    ),
}


class FreshR2Error(RuntimeError):
    """A run-level R2 identity, artifact, or orchestration failure."""


@dataclass(frozen=True, slots=True)
class RegisteredR1Artifact:
    capability: QualifiedCapabilityV1
    spec_payload: Mapping[str, Any]
    qualification_payload: Mapping[str, Any]
    episode_payload: Mapping[str, Any] | None
    source_root: Path
    evidence_locator: str


def _resource_root() -> Path:
    return Path(__file__).resolve().parent / "resources"


def _qualified_capability(payload: Mapping[str, Any]) -> QualifiedCapabilityV1:
    value = dict(payload)
    if value.pop("schema", None) != QualifiedCapabilityV1.schema:
        raise FreshR2Error("R1 capability schema identity mismatch")
    value["capability_kind"] = CapabilityKindV1(value["capability_kind"])
    value["qualification_stage"] = QualificationStageV1(
        value["qualification_stage"]
    )
    value["qualification_status"] = QualificationStatusV1(
        value["qualification_status"]
    )
    value["compatibility_requirements"] = tuple(
        value["compatibility_requirements"]
    )
    return QualifiedCapabilityV1(**value)


def _r1_receipt(repo_root: Path) -> tuple[Path, dict[str, Any]]:
    path = (
        repo_root
        / "docs/research_line/vnext/"
        "R1_FRESH_TRAINING_FILESYSTEM_FIX_V3_CANONICAL_RECEIPT.json"
    )
    if bytes_sha256(path.read_bytes()) != R1_REPO_RECEIPT_SHA256:
        raise FreshR2Error("accepted R1 repository receipt byte identity drift")
    receipt = _read_json(path)
    if (
        receipt.get("status") != "R1_PASS"
        or receipt.get("end_to_end_result_chain_pass") is not True
        or receipt.get("gate", {}).get("pass") is not True
        or receipt.get("held_out_reads") != 0
        or receipt.get("external_receipt_sha256")
        != R1_EXTERNAL_RECEIPT_SHA256
    ):
        raise FreshR2Error("accepted R1 receipt no longer satisfies its result boundary")
    external = Path(str(receipt.get("external_receipt_ref")))
    expected_external = R1_EXTERNAL_ROOT / "R1_CANONICAL_RECEIPT.json"
    # The sealed repository receipt preserves its original absolute locator;
    # RuntimeBinding owns the relocated root.  Compare the accepted relative
    # identity, then read bytes only from the bound R1_EXTERNAL_ROOT.
    if tuple(external.parts[-3:]) != tuple(expected_external.parts[-3:]):
        raise FreshR2Error("accepted R1 external receipt reference drift")
    relocated_external = R1_EXTERNAL_ROOT / "R1_CANONICAL_RECEIPT.json"
    if bytes_sha256(relocated_external.read_bytes()) != R1_EXTERNAL_RECEIPT_SHA256:
        raise FreshR2Error("accepted R1 external receipt byte identity drift")
    return path, receipt


def load_registered_r1_artifacts(
    repo_root: Path,
) -> tuple[tuple[RegisteredR1Artifact, ...], dict[str, Any]]:
    """Load every and only accepted R1 qualification from real run artifacts."""

    _receipt_path, receipt = _r1_receipt(repo_root)
    expected_capabilities = {
        str(row["capability_ref"]): str(row["capability_digest"])
        for side_rows in receipt["side_records"].values()
        for row in side_rows
        if row.get("qualification_status") == "PASS"
    }
    expected_episode_digests = {
        str(value)
        for values in receipt["typed_research_episode_digests"].values()
        for value in values
    }
    artifacts: list[RegisteredR1Artifact] = []
    observed_episodes: set[str] = set()
    for side in ("side_a", "side_b"):
        side_root = R1_EXTERNAL_ROOT / side
        for capability_path in sorted((side_root / "capabilities").glob("slot-*.json")):
            slot_id = capability_path.stem
            capability_payload = _read_json(capability_path)
            capability = _qualified_capability(capability_payload)
            if expected_capabilities.get(capability.capability_id) != capability.digest:
                raise FreshR2Error("R1 capability is absent from the accepted receipt")
            if sha256_digest(capability_payload) != capability.digest:
                raise FreshR2Error("R1 capability canonical identity drift")

            qualification_path = side_root / "qualifications" / f"{slot_id}.json"
            qualification = _read_json(qualification_path)
            qualification_receipt = qualification.get("receipt")
            if not isinstance(qualification_receipt, Mapping):
                raise FreshR2Error("R1 qualification receipt is missing")
            if (
                qualification_receipt.get("status") != "PASS"
                or qualification_receipt.get("stage") != "ONE_EPOCH_SMOKE"
                or qualification_receipt.get("evidence_class") != "DEVELOPMENT_ONLY"
                or sha256_digest(qualification_receipt)
                != capability.qualification_receipt_digest
            ):
                raise FreshR2Error("R1 qualification evidence identity drift")

            spec_path = side_root / "specs" / f"{slot_id}.json"
            spec_artifact = _read_json(spec_path)
            spec_payload = spec_artifact.get("research_spec")
            if not isinstance(spec_payload, Mapping) or (
                sha256_digest(spec_payload)
                != qualification_receipt.get("research_spec_digest")
            ):
                raise FreshR2Error("R1 OpenSpec identity drift")

            candidate_parent = side_root / "candidates" / slot_id
            candidate_roots = tuple(
                item for item in candidate_parent.iterdir() if item.is_dir()
            )
            if len(candidate_roots) != 1:
                raise FreshR2Error("R1 qualified capability source root is ambiguous")
            source_root = candidate_roots[0]
            if sha256_digest({"files": snapshot_candidate_tree(source_root)}) != (
                capability.source_tree_digest
            ):
                raise FreshR2Error("R1 qualified source tree identity drift")

            episode_path = side_root / "episodes" / f"{slot_id}.json"
            episode: Mapping[str, Any] | None = None
            if episode_path.is_file():
                episode = _read_json(episode_path)
                episode_digest = sha256_digest(episode)
                if (
                    episode_digest not in expected_episode_digests
                    or episode.get("executable_capability_ref")
                    != capability.capability_id
                    or episode.get("experiment_executed") is not True
                    or episode.get("evidence_class") != "INCONCLUSIVE_EXPERIMENT"
                    or episode.get("mechanism_interpretation") != "NOT_ADJUDICATED"
                ):
                    raise FreshR2Error("R1 TypedResearchEpisode identity drift")
                observed_episodes.add(episode_digest)
            artifacts.append(
                RegisteredR1Artifact(
                    capability=capability,
                    spec_payload=canonical_value(spec_payload),
                    qualification_payload=canonical_value(qualification),
                    episode_payload=(canonical_value(episode) if episode else None),
                    source_root=source_root,
                    evidence_locator=f"{side}/{slot_id}",
                )
            )
    if (
        len(artifacts) != 11
        or len(expected_capabilities) != 11
        or observed_episodes != expected_episode_digests
        or len(observed_episodes) != 7
    ):
        raise FreshR2Error("R1 accepted capability or episode denominator drift")
    return tuple(artifacts), receipt


def build_r1_registry(
    artifacts: Sequence[RegisteredR1Artifact],
) -> VersionedCapabilityRegistry:
    capabilities = tuple(item.capability for item in artifacts)
    if not capabilities:
        raise FreshR2Error("R2 registry cannot be empty")
    first = capabilities[0]
    return VersionedCapabilityRegistry.build(
        registry_version="fresh-r2-r1-qualified-v1",
        predecessor_registry_ref=None,
        predecessor_registry_digest=None,
        protocol_ref=first.protocol_ref,
        protocol_digest=first.protocol_digest,
        capabilities=capabilities,
    )


def build_active_r2_profile(
    registry: VersionedCapabilityRegistry,
) -> tuple[Any, NextFreshProfileBuildManifest, Any, Any, Any]:
    current = adapt_current_search_profile(
        campaign_id="recclaw-pre-r2-frozen-campaign-v1"
    )
    wave2 = _read_json(
        Path(__file__).resolve().parents[4]
        / "docs/research_line/vnext/WAVE2_INTEGRATED_GATE_RECEIPT.json"
    )
    current_slate_digest = str(
        wave2["evidence"]["current_route"]["slate_digest"]
    )
    manifest = NextFreshProfileBuildManifest(
        profile_version="fresh-r2-registry-enabled-v1",
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
    next_profile, build_receipt = build_next_fresh_profile(manifest, registry)
    active = activate_next_fresh_search_profile(
        predecessor=current,
        next_profile=next_profile,
        registry=registry,
        fresh_campaign_id=R2_CAMPAIGN_ID,
    )
    return current, manifest, next_profile, build_receipt, active


def derive_fresh_r2_proposal_schema() -> dict[str, Any]:
    resources = _resource_root()
    base_path = resources / "fresh_open_spec_proposal_response_v4_provider.schema.json"
    r1_delta_path = resources / "fresh_r1_proposal_schema_delta_v1.json"
    r2_delta_path = resources / "fresh_r2_proposal_schema_delta_v1.json"
    base = _read_json(base_path)
    delta = _read_json(r2_delta_path)
    if (
        delta.get("schema")
        != "recclaw.research-line.fresh-r2-proposal-schema-delta.v1"
        or delta.get("base_schema_sha256") != bytes_sha256(base_path.read_bytes())
        or tuple(delta.get("expressibility_claims", ()))
        != ("EXPRESSIBLE", "NOT_EXPRESSIBLE")
    ):
        raise FreshR2Error("fresh R2 proposal schema delta identity drift")
    schema = derive_fresh_r1_proposal_schema(base, _read_json(r1_delta_path))
    proposal = schema["properties"]["proposals"]["items"]["properties"]
    proposal["current_profile_expressibility_claim"]["enum"] = [
        "EXPRESSIBLE",
        "NOT_EXPRESSIBLE",
    ]
    facts = proposal["resolution_facts"]["properties"]
    facts["requested_current_semantics_digest"] = {
        "maxLength": 64,
        "minLength": 64,
        "type": ["string", "null"],
    }
    facts["capability_diff"].pop("minItems", None)
    facts["high_change_dimensions"].pop("minItems", None)
    jsonschema.validators.validator_for(schema).check_schema(schema)
    return canonical_value(schema)


def public_active_profile_catalog(
    active: Any,
    artifacts: Sequence[RegisteredR1Artifact],
    *,
    seed: int,
) -> tuple[dict[str, str], ...]:
    artifact_by_semantics = {
        item.capability.semantic_identity_digest: item for item in artifacts
    }
    fixed_by_semantics = {
        item.mechanism_semantics_digest: item for item in executable_mechanisms()
    }
    rows: list[dict[str, str]] = []
    for entry in active.entries:
        if entry.origin is SearchProfileEntryOriginV1.FIXED_66:
            mechanism = fixed_by_semantics[entry.semantic_identity_digest]
            mechanism_id = mechanism.mechanism_id
            hypothesis = (
                f"{mechanism.mechanism_id}; axis={mechanism.mechanism_axis}; "
                f"operators={','.join(mechanism.operator_ids)}"
            )
        else:
            artifact = artifact_by_semantics[entry.semantic_identity_digest]
            mechanism_id = artifact.capability.capability_id
            hypothesis = str(artifact.spec_payload["hypothesis"])
        parent_id = str(entry.capability_ref)
        rows.append(
            {
                "mechanism_id": mechanism_id,
                "mechanism_summary": parent_id,
                "parent_id": parent_id,
                "profile_ref": parent_id,
                "semantics_digest": entry.semantic_identity_digest,
                "hypothesis_summary": hypothesis,
            }
        )
    random.Random(seed).shuffle(rows)
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    forbidden = ("side_a", "side_b", "fresh-r1", "R1_", "episode_digest")
    if any(token in payload for token in forbidden):
        raise FreshR2Error("public active profile catalog leaked R1 origin")
    return tuple(canonical_value(rows))


def _r2_bindings(active: Any) -> dict[str, Any]:
    return canonical_value(
        {
            "protocol_ref": active.protocol_ref,
            "protocol_digest": active.protocol_digest,
            "context_ref": R2_CONTEXT_REF,
            "context_digest": R2_CONTEXT_DIGEST,
            "current_profile_ref": active.profile_ref,
            "current_profile_digest": active.profile_digest,
            "implementation_requirements": (
                "RecBole general recommender interface",
                "candidate-local package",
            ),
            "compatibility_requirements": PROTOCOL_REQUIREMENTS,
        }
    )


def _r2_environment(active: Any) -> dict[str, Any]:
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


def render_r2_proposal_prompt(
    template: str,
    *,
    slot_id: str,
    seed: int,
    role: str,
    active: Any,
    catalog: Sequence[Mapping[str, str]],
) -> str:
    role_instruction = R2_ROLE_INSTRUCTIONS[role] + (
        " Compatibility requirements must be exact tokens from: "
        + ", ".join(PROTOCOL_REQUIREMENTS)
        + ". Required dependencies must be exact tokens from: "
        + ", ".join(AVAILABLE_DEPENDENCIES)
        + ". Required budgets may not exceed "
        + json.dumps(BUDGET_LIMITS, sort_keys=True)
        + "."
    )
    replacements = {
        "{{CAMPAIGN_ID}}": R2_CAMPAIGN_ID,
        "{{LOGICAL_SLOT_ID}}": slot_id,
        "{{PROPOSAL_SEED}}": str(seed),
        "{{PRODUCER_ROLE}}": role,
        "{{PRODUCER_ROLE_INSTRUCTION}}": role_instruction,
        "{{PROTOCOL_REF}}": active.protocol_ref,
        "{{PROTOCOL_DIGEST}}": active.protocol_digest,
        "{{CONTEXT_REF}}": R2_CONTEXT_REF,
        "{{CONTEXT_DIGEST}}": R2_CONTEXT_DIGEST,
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
        raise FreshR2Error("fresh R2 proposal prompt has an unresolved placeholder")
    forbidden = ("side_a", "side_b", "R1_FRESH", "outcome_digest")
    if any(token in rendered for token in forbidden):
        raise FreshR2Error("fresh R2 proposal prompt leaked origin or outcome")
    return rendered


def verify_r2_source_identity(repo_root: Path, *, require_fresh_root: bool) -> dict[str, Any]:
    receipt_path, _receipt = _r1_receipt(repo_root)
    predecessor_failure_path = (
        repo_root
        / "docs/research_line/vnext/"
        "R2_FRESH_ATTEMPT_V1_TOKEN_CEILING_BLOCKED_RECEIPT.json"
    )
    observed = {
        "branch": _git(repo_root, "branch", "--show-current"),
        "head": _git(repo_root, "rev-parse", "HEAD"),
        "parent": _git(repo_root, "rev-parse", "HEAD^"),
        "head_tree": _git(repo_root, "rev-parse", "HEAD^{tree}"),
        "python_sha256": bytes_sha256(PYTHON_EXECUTABLE.read_bytes()),
        "r1_repo_receipt_sha256": bytes_sha256(receipt_path.read_bytes()),
        "r1_external_receipt_sha256": bytes_sha256(
            (R1_EXTERNAL_ROOT / "R1_CANONICAL_RECEIPT.json").read_bytes()
        ),
        "r2_predecessor_failure_receipt_sha256": bytes_sha256(
            predecessor_failure_path.read_bytes()
        ),
        **_credential_identity(API_CONFIG),
    }
    expected = {
        "branch": "feat/research-line-fresh-r2",
        "head": ACCEPTED_R1_COMMIT,
        "parent": ACCEPTED_R1_PARENT,
        "head_tree": ACCEPTED_R1_TREE,
        "python_sha256": "d99cded726bcf8b1576305ef425915fc7009c40325ecc2659553ab1c94997938",
        "r1_repo_receipt_sha256": R1_REPO_RECEIPT_SHA256,
        "r1_external_receipt_sha256": R1_EXTERNAL_RECEIPT_SHA256,
        "r2_predecessor_failure_receipt_sha256": (
            R2_PREDECESSOR_FAILURE_RECEIPT_SHA256
        ),
    }
    mismatches = {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if observed[key] != expected[key]
    }
    if mismatches:
        raise FreshR2Error(
            "fresh R2 source identity mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )
    if _git(RECBole_ROOT, "rev-parse", "HEAD") != (
        "7b02be5ec80a88310f2d04a27a82adfcbb5dc211"
    ):
        raise FreshR2Error("RecBole commit identity mismatch")
    for name, digest in EXPECTED_SEARCH_FILES.items():
        if bytes_sha256((SEARCH_DATASET_ROOT / name).read_bytes()) != digest:
            raise FreshR2Error(f"search partition identity mismatch: {name}")
    if require_fresh_root and R2_ROOT.exists():
        raise FreshR2Error(f"fresh R2 root already exists: {R2_ROOT}")
    return canonical_value(
        {
            **observed,
            "campaign_id": R2_CAMPAIGN_ID,
            "lineage": R2_LINEAGE,
            "model": MODEL,
            "provider_db_ref": str(
                R2_ROOT
                / "provider/proposals/slot-01/physical_attempt_01/broker.sqlite3"
            ),
            "proposal_seeds": R2_PROPOSAL_SEEDS,
            "qualification_seed": R2_QUALIFICATION_SEED,
            "training_seed": R2_TRAINING_SEED,
        }
    )


def offline_registry_consumer_check(repo_root: Path) -> dict[str, Any]:
    identity = verify_r2_source_identity(repo_root, require_fresh_root=False)
    artifacts, receipt = load_registered_r1_artifacts(repo_root)
    registry = build_r1_registry(artifacts)
    current, manifest, next_profile, build_receipt, active = build_active_r2_profile(
        registry
    )
    catalog = public_active_profile_catalog(
        active, artifacts, seed=R2_PROPOSAL_SEEDS[0]
    )
    schema = derive_fresh_r2_proposal_schema()
    return canonical_value(
        {
            "active_entry_count": len(active.entries),
            "active_profile_digest": active.profile_digest,
            "build_manifest_digest": manifest.digest,
            "build_receipt_digest": build_receipt.digest,
            "catalog_digest": sha256_digest(catalog),
            "catalog_entry_count": len(catalog),
            "current_entry_count": len(current.entries),
            "held_out_reads": int(receipt["held_out_reads"]),
            "identity": identity,
            "next_profile_digest": next_profile.digest,
            "r1_episode_count": sum(
                item.episode_payload is not None for item in artifacts
            ),
            "registered_capability_count": len(registry.capabilities),
            "registry_digest": registry.digest,
            "schema_digest": sha256_digest(schema),
            "status": "OFFLINE_CONSUMER_PASS",
        }
    )


def _r2_episode(
    *,
    spec: OpenResearchSpecV1,
    selected_capability: QualifiedCapabilityV1,
    materialized: Any,
    qualification: Any,
    baseline_run: Mapping[str, Any],
    candidate_run: Mapping[str, Any],
    active_profile: Any,
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
            "baseline_binding_digest": baseline_run["binding_digest"],
            "candidate_binding_digest": candidate_run["binding_digest"],
            "materialized_package_digest": materialized.package.digest,
            "matched_seed": candidate_run["seed"],
            "selected_registry_capability_digest": selected_capability.digest,
        }
    )
    return TypedResearchEpisodeV1(
        campaign_id=R2_CAMPAIGN_ID,
        context_ref=spec.context_ref,
        context_digest=spec.context_digest,
        hypothesis=spec.hypothesis,
        executable_capability_ref=selected_capability.capability_id,
        executable_capability_digest=selected_capability.digest,
        executable_profile_ref=active_profile.profile_ref,
        executable_profile_digest=active_profile.profile_digest,
        experiment_binding_ref=(
            f"{R2_RUN_IDENTITY}-experiment-binding:{sha256_digest(binding)}"
        ),
        experiment_binding_digest=sha256_digest(binding),
        comparator_ref=(
            f"{R2_RUN_IDENTITY}-bpr-comparator:{baseline_run['binding_digest']}"
        ),
        comparator_digest=sha256_digest(baseline_run),
        outcome_ref=f"{R2_RUN_IDENTITY}-development-outcome:{sha256_digest(outcome)}",
        outcome_digest=sha256_digest(outcome),
        cost_ref=f"{R2_RUN_IDENTITY}-development-cost:{sha256_digest(cost)}",
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


def run_formal_fresh_r2(
    repo_root: Path,
    *,
    canonical_receipt_path: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    started_ns = time.monotonic_ns()
    identity = verify_r2_source_identity(repo_root, require_fresh_root=True)
    artifacts, r1_receipt = load_registered_r1_artifacts(repo_root)
    registry = build_r1_registry(artifacts)
    current, manifest, next_profile, build_receipt, active = build_active_r2_profile(
        registry
    )
    proposal_schema = derive_fresh_r2_proposal_schema()
    resources = _resource_root()
    proposal_template_path = resources / "fresh_r2_producer_prompt_v1.txt"
    implementation_template_path = resources / "fresh_r1_implementer_prompt_v1.txt"
    implementation_schema_path = resources / "fresh_r1_implementation_response_v1.schema.json"
    tool_policy_path = resources / "fresh_open_spec_tool_policy_v1.json"
    implementation_template = implementation_template_path.read_text(encoding="utf-8")
    policy = _shared_policy(
        bytes_sha256(implementation_template_path.read_bytes()),
        bytes_sha256(tool_policy_path.read_bytes()),
    )

    R2_ROOT.mkdir(parents=True)
    _write_new_json(R2_ROOT / "RUN_IDENTITY.json", identity)
    schema_path = R2_ROOT / "contracts/fresh_r2_proposal_response.schema.json"
    proposal_schema_digest = _write_new_json(schema_path, proposal_schema)
    _write_new_json(R2_ROOT / "registry/versioned_capability_registry.json", registry.canonical_dict())
    _write_new_json(R2_ROOT / "registry/next_profile_build_manifest.json", manifest.canonical_dict())
    _write_new_json(R2_ROOT / "registry/next_fresh_profile.json", next_profile.canonical_dict())
    _write_new_json(R2_ROOT / "registry/profile_build_receipt.json", build_receipt.canonical_dict())
    _write_new_json(R2_ROOT / "registry/active_search_profile.json", active.canonical_dict())
    episode_index = {
        "schema": "recclaw.research-line.r1-typed-episode-index.r2-consumer.v1",
        "evidence_policy": "INCONCLUSIVE_NOT_ADJUDICATED_NO_SELECTION_UTILITY",
        "entries": tuple(
            {
                "capability_ref": item.capability.capability_id,
                "capability_digest": item.capability.digest,
                "episode": item.episode_payload,
                "evidence_locator": item.evidence_locator,
            }
            for item in artifacts
        ),
    }
    _write_new_json(R2_ROOT / "registry/r1_typed_episode_index.json", episode_index)

    catalog = public_active_profile_catalog(
        active, artifacts, seed=R2_PROPOSAL_SEEDS[0]
    )
    catalog_digest = _write_new_json(
        R2_ROOT / "registry/origin_blind_active_catalog.json",
        {"entries": catalog, "schema": "recclaw.origin-blind-active-catalog.v1"},
    )
    bindings = _r2_bindings(active)
    environment = _r2_environment(active)
    proposal_template = proposal_template_path.read_text(encoding="utf-8")
    proposal_records: list[dict[str, Any]] = []
    live_specs: dict[str, tuple[OpenResearchSpecV1, Mapping[str, Any], Any]] = {}
    registry_refs = {item.capability.capability_id for item in artifacts}

    # Freeze every Producer response and Resolver result before implementation,
    # qualification, or training. Slot order is the pre-outcome selection rule.
    for index, role in enumerate(R2_ROLE_SCHEDULE):
        slot_id = f"slot-{index + 1:02d}"
        seed = R2_PROPOSAL_SEEDS[index]
        prompt = render_r2_proposal_prompt(
            proposal_template,
            slot_id=slot_id,
            seed=seed,
            role=role,
            active=active,
            catalog=catalog,
        )
        call_result = bounded_provider_call(
            call_root=R2_ROOT / "provider/proposals" / slot_id,
            schema_path=schema_path,
            logical_call_id=f"{R2_RUN_IDENTITY}:{slot_id}:proposal",
            session_id=f"{R2_RUN_IDENTITY}:proposal-session",
            prompt=prompt,
            token_ceiling=R2_PROPOSAL_TOKEN_CEILING,
        )
        record: dict[str, Any] = {
            "denominator_included": True,
            "logical_slot_id": slot_id,
            "producer_role": role,
            "proposal_seed": seed,
            "provider_attempts": call_result.attempts,
        }
        if call_result.call is None:
            record.update(
                {
                    "failure": call_result.failure,
                    "slot_status": "PROPOSAL_PROVIDER_FAILURE",
                }
            )
            proposal_records.append(record)
            _write_new_json(R2_ROOT / "slots" / f"{slot_id}.json", record)
            continue
        call = call_result.call
        record["proposal_response_digest"] = call.response_digest
        record["returned_model"] = call.returned_model
        try:
            validate_v4_response_contract(call.response, provider_schema=proposal_schema)
            draft = call.response["proposals"][0]
            if draft["producer_role"] != role:
                raise FreshR2Error("Provider changed the preassigned Producer role")
            spec, facts = project_open_producer_draft(draft, bindings=bindings)
            resolution = resolve_capability(
                spec,
                resolution_facts=facts,
                environment=environment,
            )
            selected_registry_ref = (
                resolution.resolved_current_capability_ref
                if resolution.resolved_current_capability_ref in registry_refs
                else None
            )
            record.update(
                {
                    "resolution": resolution.resolution.value,
                    "resolution_digest": resolution.digest,
                    "resolution_reason_codes": resolution.reason_codes,
                    "resolved_capability_digest": resolution.resolved_current_capability_digest,
                    "resolved_capability_ref": resolution.resolved_current_capability_ref,
                    "selected_registry_capability": selected_registry_ref is not None,
                    "spec_digest": spec.digest,
                    "spec_ref": spec.spec_id,
                    "slot_status": "RESOLVED",
                }
            )
            live_specs[slot_id] = (spec, facts, resolution)
            _write_new_json(
                R2_ROOT / "specs" / f"{slot_id}.json",
                {
                    "research_spec": spec.canonical_dict(),
                    "resolution_facts": facts,
                    "resolution": resolution.canonical_dict(),
                },
            )
        except (FreshR2Error, jsonschema.ValidationError, ValueError) as error:
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
        _write_new_json(R2_ROOT / "slots" / f"{slot_id}.json", record)

    selected_record = next(
        (
            row
            for row in proposal_records
            if row.get("selected_registry_capability") is True
        ),
        None,
    )
    if selected_record is None:
        raise FreshR2Error(
            "fresh preassigned Producers selected no registry capability; "
            "the unique R2 architecture-effect chain cannot continue without "
            "candidate-specific intervention"
        )
    selected_slot = str(selected_record["logical_slot_id"])
    selected_spec, _selected_facts, selected_resolution = live_specs[selected_slot]
    selected_capability = next(
        item.capability
        for item in artifacts
        if item.capability.capability_id
        == selected_resolution.resolved_current_capability_ref
    )
    selection = canonical_value(
        {
            "active_profile_digest": active.profile_digest,
            "catalog_digest": catalog_digest,
            "pre_outcome_rule": "FIRST_PREASSIGNED_REGISTRY_SEARCH_READY_IN_SLOT_ORDER",
            "resolution_digest": selected_resolution.digest,
            "selected_capability_digest": selected_capability.digest,
            "selected_capability_ref": selected_capability.capability_id,
            "selected_slot": selected_slot,
        }
    )
    selection_digest = _write_new_json(R2_ROOT / "selection/selection.json", selection)

    request_prompt = render_implementation_prompt(
        implementation_template,
        build_shared_implementer_request(selected_spec, policy=policy),
    )
    implementation_call = bounded_provider_call(
        call_root=R2_ROOT / "provider/implementation" / selected_slot,
        schema_path=implementation_schema_path,
        logical_call_id=f"{R2_RUN_IDENTITY}:{selected_slot}:implementation",
        session_id=f"{R2_RUN_IDENTITY}:shared-origin-blind-implementation-session",
        prompt=request_prompt,
        token_ceiling=IMPLEMENTATION_TOKEN_CEILING,
    )
    if implementation_call.call is None:
        raise FreshR2Error("selected registry capability implementation call failed")
    materialized, qualification, behavior = _materialize_and_qualify(
        repo_root=repo_root,
        side_root=R2_ROOT / "execution",
        slot_id=selected_slot,
        seed=R2_QUALIFICATION_SEED,
        spec=selected_spec,
        implementation=implementation_call.call.response["proposals"][0],
        implementation_prompt_digest=bytes_sha256(
            implementation_template_path.read_bytes()
        ),
        tool_policy_digest=bytes_sha256(tool_policy_path.read_bytes()),
        run_identity=R2_RUN_IDENTITY,
    )
    _write_new_json(
        R2_ROOT / "qualification" / f"{selected_slot}.json",
        {**qualification.to_dict(), "behavioral_mechanism_evidence": behavior},
    )
    if qualification.receipt.status is not QualificationStatusV1.PASS:
        raise FreshR2Error("selected registry capability failed Mechanical Qualifier")
    candidate_root = (
        R2_ROOT
        / "execution/candidates"
        / selected_slot
        / str(materialized.shared_request["blind_candidate_id"])
    )
    materialization_binding = canonical_value(
        {
            "fresh_candidate_root_ref": materialized.package.candidate_root_ref,
            "fresh_package_digest": materialized.package.digest,
            "fresh_source_tree_digest": materialized.package.source_tree_digest,
            "qualification_receipt_digest": qualification.receipt.digest,
            "selected_r1_capability_digest": selected_capability.digest,
            "selected_r1_capability_ref": selected_capability.capability_id,
            "semantic_identity_digest": selected_capability.semantic_identity_digest,
        }
    )
    materialization_binding_digest = _write_new_json(
        R2_ROOT / "selection/materialization_binding.json",
        materialization_binding,
    )

    baseline_source = RECBole_ROOT / "recbole/model/general_recommender/bpr.py"
    baseline_run = run_development_training(
        repo_root=repo_root,
        side_root=R2_ROOT / "execution",
        run_id="matched-bpr-control",
        seed=R2_TRAINING_SEED,
        candidate_root=None,
        entrypoint="recbole.model.general_recommender.bpr:BPR",
        source_sha256=bytes_sha256(baseline_source.read_bytes()),
        run_identity=R2_RUN_IDENTITY,
        authority="user-delegated-fresh-r2-registry-effect",
    )
    candidate_source = candidate_root / "recclaw_ext/candidate.py"
    candidate_run = run_development_training(
        repo_root=repo_root,
        side_root=R2_ROOT / "execution",
        run_id="registry-selected-candidate",
        seed=R2_TRAINING_SEED,
        candidate_root=candidate_root,
        entrypoint=materialized.package.executable_entrypoint,
        source_sha256=bytes_sha256(candidate_source.read_bytes()),
        run_identity=R2_RUN_IDENTITY,
        authority="user-delegated-fresh-r2-registry-effect",
    )
    training_closed = (
        baseline_run.get("exit_status") == "SUCCESS"
        and candidate_run.get("exit_status") == "SUCCESS"
        and "ndcg@10" in baseline_run.get("metrics", {})
        and "ndcg@10" in candidate_run.get("metrics", {})
    )
    episode = None
    if training_closed:
        episode = _r2_episode(
            spec=selected_spec,
            selected_capability=selected_capability,
            materialized=materialized,
            qualification=qualification,
            baseline_run=baseline_run,
            candidate_run=candidate_run,
            active_profile=active,
        )
        _write_new_json(R2_ROOT / "episodes/registry_selected.json", episode.canonical_dict())

    resolver_distribution = Counter(
        str(row.get("resolution"))
        for row in proposal_records
        if row.get("resolution")
    )
    proposal_usage = _physical_usage(proposal_records)
    implementation_usage = _physical_usage(
        [{"provider_attempts": implementation_call.attempts}]
    )
    architecture_gates = {
        "function_real_and_runnable": (
            qualification.receipt.status is QualificationStatusV1.PASS
            and training_closed
        ),
        "end_to_end_result_chain_real_and_valid": episode is not None,
        "serves_open_algorithm_research_target": (
            selected_resolution.resolution
            is CapabilityResolutionResultV1.SEARCH_READY
            and selected_capability.capability_id in registry_refs
            and behavior.get("probe_status") == "PASS"
            and bool(behavior.get("extra_parameter_names"))
        ),
        "no_fixed_66_tuning_static_wrapper_fallback_mock_or_smoke_substitution": (
            active.entry(
                selected_capability.capability_id
            ).origin is SearchProfileEntryOriginV1.QUALIFIED_REGISTRY
            and candidate_root.is_relative_to(R2_ROOT)
            and implementation_call.call is not None
            and training_closed
        ),
    }
    status = (
        "R2_ARCHITECTURE_EFFECT_PASS"
        if all(architecture_gates.values())
        else "R2_ARCHITECTURE_EFFECT_FAIL"
    )
    receipt = canonical_value(
        {
            "architecture_effect_gates": architecture_gates,
            "attempt_identity": identity,
            "development_only": True,
            "episode": episode.canonical_dict() if episode else None,
            "held_out_reads": 0,
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
            "manual_candidate_patches": 0,
            "materialization_binding_digest": materialization_binding_digest,
            "matched_control": {
                "baseline": baseline_run,
                "candidate": candidate_run,
                "matched_seed": R2_TRAINING_SEED,
            },
            "model": MODEL,
            "profile": {
                "active_entry_count": len(active.entries),
                "active_profile_digest": active.profile_digest,
                "build_receipt_digest": build_receipt.digest,
                "current_entry_count": len(current.entries),
                "next_profile_digest": next_profile.digest,
                "registry_digest": registry.digest,
                "registered_capability_count": len(registry.capabilities),
                "registered_typed_episode_count": sum(
                    item.episode_payload is not None for item in artifacts
                ),
            },
            "proposal_call_contract": {
                "model": MODEL,
                "origin_blind_catalog_digest": catalog_digest,
                "response_schema_digest": proposal_schema_digest,
                "slot_count": len(R2_ROLE_SCHEDULE),
                "token_ceiling": R2_PROPOSAL_TOKEN_CEILING,
                "tools": [],
            },
            "proposal_provider_usage": proposal_usage,
            "proposal_records": proposal_records,
            "qualification": qualification.to_dict(),
            "qualification_evidence_class": "DEVELOPMENT_ONLY",
            "r1_source_receipt_sha256": R1_REPO_RECEIPT_SHA256,
            "r2_predecessor_failure_receipt_sha256": (
                R2_PREDECESSOR_FAILURE_RECEIPT_SHA256
            ),
            "resolver_distribution": dict(sorted(resolver_distribution.items())),
            "retry_policy": {
                "backoff_ms": BACKOFF_MS,
                "maximum_physical_attempts": MAX_PHYSICAL_ATTEMPTS,
                "same_slot_same_payload_only": True,
            },
            "schema": "recclaw.research-line.fresh-r2-canonical-receipt.v1",
            "scientific_interpretation": "INCONCLUSIVE_NOT_ADJUDICATED",
            "selection": selection,
            "selection_digest": selection_digest,
            "status": status,
            "training": {
                "dataset_partition": "SEARCH_TRAIN_PLUS_DEVELOPMENT_VALIDATION_ONLY",
                "held_out_exposed": False,
                "physical_runs": 2,
                "runtime_ceiling_seconds_per_run": 1500,
            },
            "wall_time_ms": max(
                1, (time.monotonic_ns() - started_ns) // 1_000_000
            ),
        }
    )
    external_sha256 = _write_new_json(R2_ROOT / "R2_CANONICAL_RECEIPT.json", receipt)
    if canonical_receipt_path.exists():
        raise FreshR2Error(
            f"canonical repository receipt already exists: {canonical_receipt_path}"
        )
    repository_receipt = {
        **receipt,
        "external_receipt_ref": str(R2_ROOT / "R2_CANONICAL_RECEIPT.json"),
        "external_receipt_sha256": external_sha256,
        "r1_external_receipt_sha256": R1_EXTERNAL_RECEIPT_SHA256,
    }
    _write_new_json(canonical_receipt_path, repository_receipt)
    return repository_receipt


__all__ = [
    "FreshR2Error",
    "R2_ROOT",
    "build_active_r2_profile",
    "build_r1_registry",
    "derive_fresh_r2_proposal_schema",
    "load_registered_r1_artifacts",
    "offline_registry_consumer_check",
    "public_active_profile_catalog",
    "render_r2_proposal_prompt",
    "run_formal_fresh_r2",
    "verify_r2_source_identity",
]
