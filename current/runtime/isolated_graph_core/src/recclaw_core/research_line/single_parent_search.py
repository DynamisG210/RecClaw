"""Research-facing mechanism languages for one strong parent per search space.

The full registries stay available to compilers and implementers.  Research
Producers see a focused causal map and an open innovation lane while successful
measured descendants remain eligible to become the next construction parent.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
    validate_sha256,
)
from recclaw_core.mechanism_space import (
    CompileStatus,
    compile_program,
    prompt_projection,
)


CONTEXT_SCHEMA = "recclaw.research-baseline-context.v2"
LANGUAGE_ID = "recclaw.bl-icf.lightgcnpp-parent-language.v2"
RESEARCH_PROFILE_ID = "BL_ICF_LIGHTGCNPP_PARENT_V2"
COMPILER_SPACE_ID = "BL_ICF_MECHANISM_SPACE_V1"
PARENT_BASE_MODEL_CONFIG = "LightGCN"

_RESOURCE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "helix_abc_v1"
    / "resources"
)


@dataclass(frozen=True, slots=True)
class SingleParentSearchSpaceSpec:
    """Stable identity and parent ABI for one focused search profile."""

    profile_key: str
    research_profile_id: str
    mechanism_space_id: str
    mechanism_language_id: str
    parent_name: str
    parent_base_model_config: str
    adapter_id: str
    resource_name: str

    @property
    def resource_path(self) -> Path:
        return _RESOURCE_ROOT / self.resource_name


BL_ICF_SINGLE_PARENT_SPEC = SingleParentSearchSpaceSpec(
    profile_key="bl_icf",
    research_profile_id=RESEARCH_PROFILE_ID,
    mechanism_space_id=COMPILER_SPACE_ID,
    mechanism_language_id=LANGUAGE_ID,
    parent_name="LightGCN++",
    parent_base_model_config=PARENT_BASE_MODEL_CONFIG,
    adapter_id="recclaw.search-space-adapter.bl-icf.v1",
    resource_name="bl_icf_single_parent_mechanism_language_v2.json",
)
SEQUENTIAL_SCALING_SINGLE_PARENT_SPEC = SingleParentSearchSpaceSpec(
    profile_key="sequential_scaling",
    research_profile_id="SEQUENTIAL_SCALING_SSD4REC_PARENT_V2",
    mechanism_space_id="SEQUENTIAL_SCALING_MECHANISM_SPACE_V1",
    mechanism_language_id="recclaw.sequential-scaling.ssd4rec-parent-language.v2",
    parent_name="SSD4Rec",
    parent_base_model_config="BERT4Rec",
    adapter_id="recclaw.search-space-adapter.sequential-scaling.single-parent.v2",
    resource_name="sequential_scaling_single_parent_mechanism_language_v2.json",
)
SEMANTIC_ID_SINGLE_PARENT_SPEC = SingleParentSearchSpaceSpec(
    profile_key="semantic_id_generative",
    research_profile_id="SEMANTIC_ID_GENERATIVE_LIGER_PARENT_V2",
    mechanism_space_id="SEMANTIC_ID_GENERATIVE_MECHANISM_SPACE_V1",
    mechanism_language_id="recclaw.semantic-id-generative.liger-parent-language.v2",
    parent_name="LIGER",
    parent_base_model_config="BERT4Rec",
    adapter_id="recclaw.search-space-adapter.semantic-id-generative.single-parent.v2",
    resource_name="semantic_id_generative_single_parent_mechanism_language_v2.json",
)
DIFFUSION_FLOW_SINGLE_PARENT_SPEC = SingleParentSearchSpaceSpec(
    profile_key="diffusion_flow_cf",
    research_profile_id="DIFFUSION_FLOW_CF_DIFFREC_PARENT_V2",
    mechanism_space_id="DIFFUSION_FLOW_CF_MECHANISM_SPACE_V1",
    mechanism_language_id="recclaw.diffusion-flow-cf.diffrec-parent-language.v2",
    parent_name="DiffRec",
    parent_base_model_config="DiffRec",
    adapter_id="recclaw.search-space-adapter.diffusion-flow-cf.single-parent.v2",
    resource_name="diffusion_flow_cf_single_parent_mechanism_language_v2.json",
)

SINGLE_PARENT_SEARCH_SPACE_SPECS = (
    BL_ICF_SINGLE_PARENT_SPEC,
    SEQUENTIAL_SCALING_SINGLE_PARENT_SPEC,
    SEMANTIC_ID_SINGLE_PARENT_SPEC,
    DIFFUSION_FLOW_SINGLE_PARENT_SPEC,
)
_SPEC_BY_PROFILE_ID = {
    spec.research_profile_id: spec for spec in SINGLE_PARENT_SEARCH_SPACE_SPECS
}
_SPEC_BY_PROFILE_KEY = {
    spec.profile_key: spec for spec in SINGLE_PARENT_SEARCH_SPACE_SPECS
}


def single_parent_spec_for_profile_key(
    profile_key: str,
) -> SingleParentSearchSpaceSpec:
    try:
        return _SPEC_BY_PROFILE_KEY[profile_key]
    except KeyError as error:
        raise KeyError(f"unsupported single-parent profile: {profile_key}") from error


def single_parent_spec_for_context(
    value: Mapping[str, Any],
) -> SingleParentSearchSpaceSpec | None:
    """Resolve a registered profile only when all three identities agree."""

    if not is_single_parent_context(value):
        return None
    spec = _SPEC_BY_PROFILE_ID.get(str(value.get("research_profile_id")))
    if spec is None:
        return None
    if (
        value.get("mechanism_space") != spec.mechanism_space_id
        or value.get("mechanism_language_id") != spec.mechanism_language_id
    ):
        return None
    return spec


def single_parent_spec_for_objective(
    value: Mapping[str, Any],
) -> SingleParentSearchSpaceSpec | None:
    spec = _SPEC_BY_PROFILE_ID.get(str(value.get("research_profile_id")))
    if spec is None:
        return None
    if value.get("mechanism_space") not in (None, spec.mechanism_space_id):
        return None
    if value.get("mechanism_language_id") != spec.mechanism_language_id:
        return None
    return spec


@lru_cache(maxsize=None)
def focused_mechanism_language(
    spec: SingleParentSearchSpaceSpec = BL_ICF_SINGLE_PARENT_SPEC,
) -> dict[str, Any]:
    """Build the focused research projection from compiler-owned data."""

    resource = canonical_value(
        json.loads(spec.resource_path.read_text(encoding="utf-8"))
    )
    if (
        resource.get("language_id") != spec.mechanism_language_id
        or resource.get("research_profile_id") != spec.research_profile_id
        or resource.get("compiler_search_space_id") != spec.mechanism_space_id
        or resource.get("parent_contract", {}).get("name") != spec.parent_name
        or resource.get("parent_contract", {}).get("base_model_config")
        != spec.parent_base_model_config
    ):
        raise RuntimeError(
            f"focused mechanism language identity drift for {spec.profile_key}"
        )
    compiler = canonical_value(prompt_projection(spec.mechanism_space_id))
    primitive_by_id = {
        primitive["primitive_id"]: primitive
        for axis in compiler["axes"]
        for primitive in axis["primitives"]
    }
    parent_ids = tuple(
        resource["parent_contract"]["required_parent_foundation_primitives"]
    )
    affordance_ids = tuple(
        primitive_id
        for zone in resource["causal_zones"]
        for primitive_id in zone["known_affordance_ids"]
    )
    missing = sorted(set(parent_ids + affordance_ids) - set(primitive_by_id))
    if missing:
        raise RuntimeError(
            f"focused {spec.profile_key} language references unknown primitives: "
            + ", ".join(missing)
        )

    zones = [
        {
            **dict(zone),
            "known_affordances": [
                primitive_by_id[primitive_id]
                for primitive_id in zone["known_affordance_ids"]
            ],
        }
        for zone in resource["causal_zones"]
    ]
    innovation_extension = {
        **dict(resource["innovation_extension"]),
        "allowed_slots": [axis["slot_id"] for axis in compiler["axes"]],
        "allowed_data_roles": compiler["allowed_data_roles"],
        "architecture_operators": compiler["architecture_operators"],
        "output_contract": compiler["output_contract"],
    }
    for key in (
        "custom_mechanism_requirements",
        "proposal_obligations",
        "qualification_contract",
        "episode_contract",
    ):
        if key in compiler:
            innovation_extension[key] = compiler[key]

    projection = {
        "schema": resource["schema"],
        "language_id": resource["language_id"],
        "research_profile_id": resource["research_profile_id"],
        "compiler_space": compiler["space_identity"],
        # The exact parent program is already carried in Research Context.
        # Repeat only its stable primitive identities here; duplicating ten
        # full parent contracts spends attention without adding search freedom.
        "parent_contract": dict(resource["parent_contract"]),
        "language_semantics": resource["language_semantics"],
        "causal_zones": zones,
        "innovation_extension": innovation_extension,
        "projection_counts": {
            "full_internal_registry": sum(
                len(axis["primitives"]) for axis in compiler["axes"]
            ),
            "projected_parent_foundation": len(parent_ids),
            "projected_known_affordances": len(affordance_ids),
            "causal_zones": len(zones),
        },
    }
    return canonical_value(
        {**projection, "projection_digest": sha256_digest(projection)}
    )


def _declarative_profile_language(
    value: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    spec = single_parent_spec_for_context(value)
    if spec is None or spec == BL_ICF_SINGLE_PARENT_SPEC:
        return None
    return focused_mechanism_language(spec)


def single_parent_research_axes(
    value: Mapping[str, Any],
) -> tuple[str, ...]:
    """Return focused Provider slots followed by any non-focused tail."""

    language = _declarative_profile_language(value)
    if language is None:
        return ()
    allowed = tuple(language["innovation_extension"]["allowed_slots"])
    focused = tuple(
        dict.fromkeys(
            parent_slot
            for zone in language["causal_zones"]
            for parent_slot in zone["parent_slots"]
            if parent_slot in allowed
        )
    )
    return (*focused, *(slot for slot in allowed if slot not in focused))


def single_parent_research_axis_questions(
    value: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    """Project the Profile-owned research axes and their unresolved questions."""

    language = _declarative_profile_language(value)
    if language is None:
        return ()
    axes = set(single_parent_research_axes(value))
    return tuple(
        {
            "mechanism_axis": str(parent_slot),
            "question": str(zone["research_question"]),
        }
        for zone in language["causal_zones"]
        for parent_slot in zone["parent_slots"]
        if parent_slot in axes
    )


@lru_cache(maxsize=None)
def innovation_mechanism_language(
    spec: SingleParentSearchSpaceSpec = BL_ICF_SINGLE_PARENT_SPEC,
) -> dict[str, Any]:
    """Return the causal map without ordinary-lane implementation contracts."""

    focused = focused_mechanism_language(spec)
    parent = focused["parent_contract"]
    extension = focused["innovation_extension"]
    innovation_projection = {
        key: extension[key]
        for key in (
            "primary_role",
            "available_to_other_roles_when_scientifically_required",
            "construction_modes",
            "may_cross_multiple_support_zones",
            "may_define_new_typed_custom_components",
            "may_use_unlisted_registry_primitives",
            "known_affordances_limit_innovation",
            "source_ownership_compilation",
            "required_scientific_content",
            "forbidden_shortcuts",
            "allowed_slots",
            "allowed_data_roles",
            "output_contract",
        )
    }
    for key in (
        "custom_mechanism_requirements",
        "proposal_obligations",
        "qualification_contract",
        "episode_contract",
    ):
        if key in extension:
            innovation_projection[key] = extension[key]
    projection = {
        "schema": focused["schema"],
        "language_id": focused["language_id"],
        "research_profile_id": focused["research_profile_id"],
        "compiler_space": focused["compiler_space"],
        "parent": {
            key: parent[key]
            for key in (
                "name",
                "base_model_config",
                "role",
                "parent_is_construction_anchor",
                "successful_frontier_may_become_active_construction_parent",
                "frozen_root_remains_paired_comparator",
                "unchanged_parent_slots_are_inherited",
            )
        },
        "language_semantics": focused["language_semantics"],
        "causal_zones": [
            {
                key: zone[key]
                for key in (
                    "zone_id",
                    "parent_slots",
                    "research_question",
                    "known_affordance_ids",
                )
            }
            for zone in focused["causal_zones"]
        ],
        "innovation_extension": innovation_projection,
        "architecture_operator_ids": [
            item["operator_id"]
            for item in extension["architecture_operators"]
        ],
    }
    return canonical_value(
        {**projection, "projection_digest": sha256_digest(projection)}
    )


def is_single_parent_context(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("schema") == CONTEXT_SCHEMA


def is_bl_icf_single_parent_context(value: Any) -> bool:
    """Return whether a generic single-parent context selects BL-ICF v2."""

    return bool(
        isinstance(value, Mapping)
        and single_parent_spec_for_context(value) == BL_ICF_SINGLE_PARENT_SPEC
    )


def project_single_parent_objective(value: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the effect objective without recreating a baseline pack."""

    parent = value.get("parent_anchor")
    if not isinstance(parent, Mapping):
        return {}
    return canonical_value(
        {
            "research_profile_id": value.get("research_profile_id"),
            "mechanism_language_id": value.get("mechanism_language_id"),
            "mechanism_space": value.get("mechanism_space"),
            "parent_anchor": {
                key: parent[key]
                for key in (
                    "name",
                    "base_model_config",
                    "mechanism_summary",
                    "paired_metric",
                )
                if key in parent
            },
            "search_objective": value.get("search_objective"),
            "baseline_decision_semantics": value.get("decision_semantics", {}),
        }
    )


def bound_parent_from_context(value: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a supplied executable parent, or ``None`` while chain repair works."""

    parent = value.get("parent_anchor")
    if not isinstance(parent, Mapping):
        return None
    binding = parent.get("binding")
    program = parent.get("mechanism_program")
    if not isinstance(binding, Mapping) or not isinstance(program, Mapping):
        return None
    result = {"binding": binding, "mechanism_program": program}
    source_bundle = parent.get("source_bundle")
    if isinstance(source_bundle, Mapping):
        result["source_bundle"] = source_bundle
    if isinstance(parent.get("execution_contract"), Mapping):
        result["execution_contract"] = parent["execution_contract"]
    elif isinstance(source_bundle, Mapping) and isinstance(source_bundle.get("execution_contract"), Mapping):
        result["execution_contract"] = source_bundle["execution_contract"]
    elif isinstance(program.get("program_payload", {}).get("execution_contract"), Mapping):
        # Native profiles such as E1 carry the actual frozen recipe in their
        # program. Expose that same recipe to parent selection and source loading.
        result["execution_contract"] = program["program_payload"]["execution_contract"]
    return canonical_value(result)


def _require_actual_parent_contract(bundle: Mapping[str, Any]) -> None:
    contract = bundle.get("execution_contract")
    if (not isinstance(contract, Mapping)
            or not isinstance(contract.get("config"), Mapping)
            or any(not isinstance(contract.get(key), str) or not contract[key]
                   for key in ("base_model_config", "model", "capability_family"))):
        raise ValueError("frozen parent requires its actual execution_contract; do not reconstruct model defaults")


def exact_parent_bundle_from_context(
    value: Mapping[str, Any],
    *,
    allowed_files: tuple[str, ...] | None = None,
    require_executable_abi: bool = False,
) -> dict[str, Any] | None:
    """Return the exact source bundle used for parent-relative implementation."""

    bound = bound_parent_from_context(value)
    if bound is None or not isinstance(bound.get("source_bundle"), Mapping):
        return None
    binding = bound["binding"]
    bundle = canonical_value(dict(bound["source_bundle"]))
    if "execution_contract" not in bundle and "execution_contract" in bound:
        bundle["execution_contract"] = bound["execution_contract"]
    if require_executable_abi:
        _require_actual_parent_contract(bundle)
    if (
        bundle.get("candidate_id") != binding.get("candidate_id")
        or bundle.get("program_digest") != binding.get("program_digest")
        or bundle.get("instruction") != "CLONE_EXACT_PARENT_AND_LOCAL_PATCH"
        or not isinstance(bundle.get("capability_ref"), str)
        or not bundle.get("capability_ref")
    ):
        raise ValueError("frozen parent source bundle differs from its binding")
    files = bundle.get("files")
    if not isinstance(files, (tuple, list)) or not files:
        raise ValueError("frozen parent source bundle must contain source files")
    normalized_files: list[dict[str, Any]] = []
    source_by_path: dict[str, str] = {}
    for row in files:
        if not isinstance(row, Mapping):
            raise ValueError("frozen parent source bundle file is invalid")
        path = row.get("path")
        content = row.get("content")
        digest = row.get("sha256")
        if not all(isinstance(item, str) and item for item in (path, content, digest)):
            raise ValueError("frozen parent source bundle file is incomplete")
        payload = content.encode("utf-8")
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("frozen parent source bundle file digest drift")
        normalized_files.append(
            {
                "path": path,
                "sha256": digest,
                "size_bytes": len(payload),
            }
        )
        source_by_path[path] = content
    paths = tuple(sorted(row["path"] for row in normalized_files))
    if len(paths) != len(set(paths)):
        raise ValueError("frozen parent source bundle repeats a file path")
    if allowed_files is not None and paths != tuple(sorted(allowed_files)):
        raise ValueError("frozen parent source bundle differs from implementation policy")
    expected_tree_digest = sha256_digest(
        {"files": sorted(normalized_files, key=lambda row: row["path"])}
    )
    if bundle.get("source_tree_digest") != expected_tree_digest:
        raise ValueError("frozen parent source tree digest drift")
    if require_executable_abi:
        try:
            candidate_tree = ast.parse(source_by_path["recclaw_ext/candidate.py"])
            trainer_tree = ast.parse(source_by_path["recclaw_ext/trainer.py"])
            package_tree = ast.parse(source_by_path["recclaw_ext/__init__.py"])
        except (KeyError, SyntaxError) as error:
            raise ValueError(
                "frozen parent source bundle is not importable Python"
            ) from error
        candidate_class = next(
            (
                node
                for node in candidate_tree.body
                if isinstance(node, ast.ClassDef)
                and node.name == "FreshCandidateModel"
            ),
            None,
        )
        required_methods = {"calculate_loss", "predict", "full_sort_predict"}
        direct_methods = {
            node.name
            for node in (candidate_class.body if candidate_class is not None else ())
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if candidate_class is None or not required_methods <= direct_methods:
            raise ValueError(
                "frozen parent FreshCandidateModel lacks the executable RecBole ABI"
            )
        if not any(
            isinstance(node, ast.ClassDef)
            and node.name == "FreshCandidateTrainer"
            for node in trainer_tree.body
        ):
            raise ValueError(
                "frozen parent source bundle lacks FreshCandidateTrainer"
            )
        if not any(
            isinstance(node, ast.ImportFrom)
            and node.module == "candidate"
            and any(alias.name == "FreshCandidateModel" for alias in node.names)
            for node in package_tree.body
        ):
            raise ValueError(
                "frozen parent package does not export FreshCandidateModel"
            )
    normalized_bundle = {
        key: bundle[key]
        for key in (
            "candidate_id",
            "capability_ref",
            "files",
            "instruction",
            "program_digest",
            "source_tree_digest",
        )
    }
    if "execution_contract" in bundle:
        normalized_bundle["execution_contract"] = bundle["execution_contract"]
    return canonical_value(normalized_bundle)


def validate_single_parent_runtime_context(
    value: Mapping[str, Any],
    *,
    baseline_seed: int,
    baseline_value: float,
    allowed_files: tuple[str, ...] | None = None,
    active_profile_ref: Mapping[str, Any] | None = None,
    require_executable_parent_abi: bool = False,
) -> dict[str, Any]:
    """Validate only facts required to run a real one-parent search."""

    context = canonical_value(dict(value))
    if not is_single_parent_context(context):
        return context
    spec = single_parent_spec_for_context(context)
    if spec is None:
        raise ValueError(
            "single-parent context identity differs from every registered search space"
        )
    parent = context.get("parent_anchor")
    if not isinstance(parent, Mapping):
        raise ValueError("single-parent context lacks parent_anchor")
    if context.get("metric") != "NDCG@10" or context.get("split") != "data/dev":
        raise ValueError("single-parent context must target data/dev NDCG@10")
    if (
        parent.get("name") != spec.parent_name
        or parent.get("base_model_config") != spec.parent_base_model_config
    ):
        raise ValueError(
            f"{spec.profile_key} single-parent context must bind "
            f"{spec.parent_name} via {spec.parent_base_model_config}"
        )
    bound = bound_parent_from_context(context)
    if bound is None:
        raise ValueError("single-parent context lacks an executable parent program")
    binding = bound["binding"]
    candidate_id = binding.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise ValueError("single-parent binding lacks candidate_id")
    try:
        program_digest = validate_sha256(
            str(binding.get("program_digest")),
            field_name="single-parent binding program_digest",
        )
    except ValueError as error:
        raise ValueError(str(error)) from error
    report = compile_program(bound["mechanism_program"])
    if (
        report.status is not CompileStatus.VALID_NEEDS_IMPLEMENTATION
        or report.space_identity is None
        or report.space_identity.search_space_id != spec.mechanism_space_id
        or report.candidate_id != candidate_id
        or report.mechanism_program_digest != program_digest
    ):
        raise ValueError("single-parent mechanism program differs from its binding")
    if active_profile_ref is not None and canonical_value(
        bound["mechanism_program"].get("profile_ref")
    ) != canonical_value(dict(active_profile_ref)):
        raise ValueError("single-parent mechanism program differs from active profile")
    parent_primitives = {
        str(item["primitive_id"])
        for item in bound["mechanism_program"].get("program_payload", {}).get(
            "components", ()
        )
        if isinstance(item, Mapping) and isinstance(item.get("primitive_id"), str)
    }
    required_parent_primitives = set(
        focused_mechanism_language(spec)["parent_contract"][
            "required_parent_foundation_primitives"
        ]
    )
    if not required_parent_primitives <= parent_primitives:
        raise ValueError(
            "single-parent mechanism program lacks the registered parent foundation"
        )
    paired = parent.get("paired_metric")
    if not isinstance(paired, Mapping):
        raise ValueError("single-parent context lacks its paired parent metric")
    seed = paired.get("seed")
    metric_value = paired.get("value")
    if seed != baseline_seed:
        raise ValueError("single-parent paired metric seed differs from runtime baseline")
    if (
        isinstance(metric_value, bool)
        or not isinstance(metric_value, (int, float))
        or not math.isfinite(float(metric_value))
        or not math.isclose(
            float(metric_value),
            float(baseline_value),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("single-parent paired metric differs from runtime baseline")
    exact_bundle = exact_parent_bundle_from_context(
        context,
        allowed_files=allowed_files,
        require_executable_abi=require_executable_parent_abi,
    )
    if exact_bundle is None:
        raise ValueError("single-parent context lacks the exact parent source bundle")
    _require_actual_parent_contract(exact_bundle)
    return context


__all__ = [
    "BL_ICF_SINGLE_PARENT_SPEC",
    "COMPILER_SPACE_ID",
    "CONTEXT_SCHEMA",
    "DIFFUSION_FLOW_SINGLE_PARENT_SPEC",
    "LANGUAGE_ID",
    "PARENT_BASE_MODEL_CONFIG",
    "RESEARCH_PROFILE_ID",
    "SEMANTIC_ID_SINGLE_PARENT_SPEC",
    "SEQUENTIAL_SCALING_SINGLE_PARENT_SPEC",
    "SINGLE_PARENT_SEARCH_SPACE_SPECS",
    "SingleParentSearchSpaceSpec",
    "bound_parent_from_context",
    "exact_parent_bundle_from_context",
    "focused_mechanism_language",
    "innovation_mechanism_language",
    "is_bl_icf_single_parent_context",
    "is_single_parent_context",
    "project_single_parent_objective",
    "single_parent_spec_for_context",
    "single_parent_spec_for_objective",
    "single_parent_spec_for_profile_key",
    "single_parent_research_axes",
    "single_parent_research_axis_questions",
    "validate_single_parent_runtime_context",
]
