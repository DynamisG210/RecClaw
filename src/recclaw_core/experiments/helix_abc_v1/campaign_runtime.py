"""Single executable campaign projection shared by Pilot and Main.

The static BL-ICF grammar is intentionally broader than the package-owned
training runtime.  This module is the one projection boundary: every mechanism
exposed to a proposal prompt has an exact program and an exact execution
recipe, and every execution recipe is exposed to all three Arms.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
from itertools import combinations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from recclaw_core.mechanism_space import compile_program
from recclaw_core.mechanism_space.canonical import deep_thaw

from .canonical import bytes_sha256, canonical_value, sha256_digest


class CampaignRuntimeError(ValueError):
    pass


_RESOURCE_ROOT = Path(__file__).resolve().parent / "resources"
_CATALOG_PATH = _RESOURCE_ROOT / "executable_operator_catalog_v2.json"
_ANCHOR_PATH = _RESOURCE_ROOT / "campaign_anchor_programs_v1.json"
_TRAINING_PROFILE_PATH = _RESOURCE_ROOT / "campaign_training_profile_v1.json"
_PARTITION_PROFILE_PATH = _RESOURCE_ROOT / "campaign_partition_profile_v1.json"
_PROPOSAL_SCHEMA_PATH = _RESOURCE_ROOT / "campaign_proposal_response_v1.schema.json"
_PROTOCOL_PATH = _RESOURCE_ROOT / "development_protocol_v1.json"
_CAMPAIGN_PROTOCOL_PATH = (
    _RESOURCE_ROOT / "campaign_development_protocol_v1.json"
)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignRuntimeError(f"{path.name} must contain a JSON object")
    return value


def executable_catalog_spec() -> dict[str, Any]:
    return _json_object(_CATALOG_PATH)


def _composition_entries() -> tuple[dict[str, Any], ...]:
    catalog = executable_catalog_spec()
    rows: list[dict[str, Any]] = []
    for base in catalog["bases"]:
        root = str(base["root_mechanism_id"])
        rows.append(
            {
                "anchor_name": base["anchor_name"],
                "base_model_config": base["base_model_config"],
                "config": {},
                "entrypoint": base["root_entrypoint"],
                "mechanism_axis": (
                    "objective" if root == "BPR_MF" else "propagation"
                ),
                "mechanism_id": root,
                "model": base["base_model_config"],
                "operator_ids": [],
                "parent_mechanism_id": None,
                "summary": f"frozen {root} base control",
                "transforms": [],
            }
        )
        operators = [
            item
            for item in catalog["operators"]
            if str(item["base"]) == root
        ]
        selections = [(item,) for item in operators]
        selections.extend(
            pair
            for pair in combinations(operators, 2)
            if _operators_compatible(*pair)
        )
        for selection in selections:
            operator_ids = tuple(
                str(item["operator_id"]) for item in selection
            )
            config: dict[str, Any] = {
                "composition_operators": list(operator_ids)
            }
            transforms: list[dict[str, Any]] = []
            for item in selection:
                config.update(copy.deepcopy(dict(item["config"])))
                transforms.extend(
                    copy.deepcopy(list(item["transforms"]))
                )
            rows.append(
                {
                    "anchor_name": base["anchor_name"],
                    "base_model_config": base["base_model_config"],
                    "config": config,
                    "entrypoint": base["entrypoint"],
                    "mechanism_axis": selection[0]["axis"],
                    "mechanism_id": root + "__" + "__".join(operator_ids),
                    "model": base["model"],
                    "operator_ids": list(operator_ids),
                    "parent_mechanism_id": root,
                    "summary": " + ".join(
                        str(item["summary"]) for item in selection
                    ),
                    "transforms": transforms,
                }
            )
    return tuple(rows)


def _operators_compatible(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    if str(left["axis"]) == str(right["axis"]):
        return False
    left_targets = {
        str(item["component_id"])
        for item in left["transforms"]
        if item["op"] != "append_regularizer"
    }
    right_targets = {
        str(item["component_id"])
        for item in right["transforms"]
        if item["op"] != "append_regularizer"
    }
    return not bool(left_targets & right_targets)


def campaign_training_profile() -> dict[str, Any]:
    return _json_object(_TRAINING_PROFILE_PATH)


def campaign_proposal_schema_path() -> Path:
    return _PROPOSAL_SCHEMA_PATH


def campaign_proposal_schema() -> dict[str, Any]:
    return _json_object(_PROPOSAL_SCHEMA_PATH)


def _anchor_programs() -> dict[str, dict[str, Any]]:
    document = _json_object(_ANCHOR_PATH)
    return {
        str(item["anchor_name"]): copy.deepcopy(item["program"])
        for item in document["fixtures"]
    }


def _component(payload: Mapping[str, Any], component_id: str) -> dict[str, Any]:
    matches = [
        item
        for item in payload["components"]
        if str(item["component_id"]) == component_id
    ]
    if len(matches) != 1:
        raise CampaignRuntimeError(
            f"program does not contain one component named {component_id}"
        )
    return matches[0]


def _record_changed_slot(payload: dict[str, Any], slot_id: str) -> None:
    rows = {
        (str(item["slot_id"]), str(item["change_role"]))
        for item in payload["changed_slots"]
    }
    rows.add((slot_id, "CORE"))
    payload["changed_slots"] = [
        {"change_role": role, "slot_id": slot}
        for slot, role in sorted(rows)
    ]


def _finalize_changed_slots(
    payload: dict[str, Any], entry: Mapping[str, Any]
) -> None:
    transforms = tuple(entry["transforms"])
    if not transforms:
        return
    touched: list[str] = []
    for transform in transforms:
        if transform["op"] == "append_regularizer":
            slot = "GEOMETRY_REGULARIZATION"
        else:
            slot = str(
                _component(payload, str(transform["component_id"]))["slot_id"]
            )
        if slot not in touched:
            touched.append(slot)
    preferred = {
        "fusion": "FUSION_ROUTING",
        "negative_sampling": "NEGATIVE_SAMPLER",
        "objective": "PRIMARY_OBJECTIVE",
        "objective_sampling_composition": "PRIMARY_OBJECTIVE",
        "propagation": "PROPAGATION_AGGREGATION",
        "regularization": "GEOMETRY_REGULARIZATION",
    }.get(str(entry["mechanism_axis"]))
    core = preferred if preferred in touched else touched[0]
    payload["changed_slots"] = [
        {
            "change_role": "CORE" if slot == core else "SUPPORT",
            "slot_id": slot,
        }
        for slot in touched
    ]


def _apply_transform(payload: dict[str, Any], transform: Mapping[str, Any]) -> None:
    operation = str(transform["op"])
    if operation in {"replace_component", "set_component_parameters"}:
        item = _component(payload, str(transform["component_id"]))
        if operation == "replace_component":
            item["primitive_id"] = str(transform["primitive_id"])
        item["parameters"] = copy.deepcopy(dict(transform["parameters"]))
        _record_changed_slot(payload, str(item["slot_id"]))
        return
    if operation == "append_regularizer":
        component_id = str(transform["component_id"])
        if any(
            str(item["component_id"]) == component_id
            for item in payload["components"]
        ):
            raise CampaignRuntimeError(
                f"duplicate generated component id: {component_id}"
            )
        source = _component(payload, str(transform["source_component_id"]))
        payload["components"].append(
            {
                "component_id": component_id,
                "inputs": [
                    {
                        "port": "representation",
                        "source": {
                            "component_id": str(source["component_id"]),
                            "kind": "COMPONENT",
                            "output_port": (
                                "embedding"
                                if str(source["slot_id"]) == "EMBEDDING"
                                else "representation"
                            ),
                        },
                    }
                ],
                "parameters": copy.deepcopy(dict(transform["parameters"])),
                "primitive_id": str(transform["primitive_id"]),
                "slot_id": "GEOMETRY_REGULARIZATION",
            }
        )
        _record_changed_slot(payload, "GEOMETRY_REGULARIZATION")
        return
    raise CampaignRuntimeError(f"unknown catalog transform: {operation}")


def _catalog_program(entry: Mapping[str, Any]) -> dict[str, Any]:
    anchors = _anchor_programs()
    anchor_name = str(entry["anchor_name"])
    if anchor_name not in anchors:
        raise CampaignRuntimeError(f"unknown package anchor: {anchor_name}")
    program = anchors[anchor_name]
    catalog_digest = sha256_digest(executable_catalog_spec())
    program["profile_ref"] = {
        "profile_digest": catalog_digest,
        "profile_id": "BL_ICF_EXECUTABLE_PROFILE_V2",
        "profile_kind": "OFFLINE_TOPN",
    }
    payload = program["program_payload"]
    for transform in entry["transforms"]:
        _apply_transform(payload, transform)
    _finalize_changed_slots(payload, entry)
    mechanism_id = str(entry["mechanism_id"])
    parent = entry.get("parent_mechanism_id")
    operator_ids = tuple(
        str(item) for item in entry.get("operator_ids", ())
    )
    medium_cost = (
        len(operator_ids) == 2
        or bool(
            set(operator_ids)
            & {
                "LGCN_AUX_ALIGNMENT",
                "LGCN_DUAL_PATH",
                "LGCN_EDGE_DROPOUT",
            }
        )
    )
    payload.update(
        {
            "ablation_plan": [
                {
                    "ablation_id": "matched_parent_comparison",
                    "expected_observation": (
                        "The mechanism-specific signal should disappear or weaken "
                        "under the declared matched parent."
                    ),
                    "remove_component_ids": [
                        str(transform["component_id"])
                        for transform in entry["transforms"]
                        if transform["op"] != "set_component_parameters"
                    ]
                    or ["objective"],
                }
            ],
            "claim_ceiling": "DEVELOPMENT_ONLY_SINGLE_PROTOCOL_NO_GENERAL_CLAIM",
            "core_hypothesis": (
                f"{mechanism_id} is a package-owned executable mechanism whose "
                "declared intervention may change development NDCG@10."
            ),
            "estimated_cost": {
                "precompute_required": False,
                "relative_memory": "MEDIUM" if medium_cost else "LOW",
                "relative_training_compute": (
                    "MEDIUM" if medium_cost else "LOW"
                ),
            },
            "expected_effects": {
                "coverage": "frozen ML-1M development-validation users",
                "efficiency": "measured under the common campaign budget",
                "relevance": str(entry["summary"]),
                "robustness": "no claim beyond the exact executable recipe",
            },
            "failure_modes": [
                "The declared mechanism may be neutral or harmful under the "
                "frozen protocol and budget."
            ],
            "implementation_plan": [
                "Resolve the exact package-owned execution recipe from the "
                "compiled mechanism semantics digest."
            ],
            "matched_control": {
                "control_ref": str(parent or "FROZEN_COMPARATOR"),
                "rationale": (
                    "Use the declared parent mechanism as the controlled "
                    "comparison without changing protocol or budget."
                ),
            },
            "mechanism_explanation": (
                f"Package-owned executable catalog entry {mechanism_id}: "
                f"{entry['summary']}"
            ),
            "parent_refs": [],
            "protocol_impact": {"requested_changes": [], "status": "UNCHANGED"},
            "research_question": (
                f"Does {mechanism_id} improve or discriminate mechanism behavior "
                "under the frozen ML-1M development protocol?"
            ),
        }
    )
    return program


@dataclass(frozen=True, slots=True)
class ExecutableMechanismV1:
    mechanism_id: str
    mechanism_axis: str
    parent_mechanism_id: str | None
    summary: str
    entrypoint: str
    entrypoint_source_sha256: str
    model: str
    base_model_config: str
    config: Mapping[str, Any]
    mechanism_program: Mapping[str, Any]
    mechanism_program_digest: str
    mechanism_semantics_digest: str
    candidate_id: str
    base_mechanism_id: str
    operator_ids: tuple[str, ...]

    @property
    def execution_recipe_digest(self) -> str:
        return sha256_digest(self.execution_recipe())

    def execution_recipe(self) -> dict[str, Any]:
        return canonical_value(
            {
                "candidate_id": self.candidate_id,
                "config": dict(self.config),
                "base_model_config": self.base_model_config,
                "entrypoint": self.entrypoint,
                "entrypoint_source_sha256": self.entrypoint_source_sha256,
                "mechanism_id": self.mechanism_id,
                "mechanism_program_digest": self.mechanism_program_digest,
                "mechanism_semantics_digest": self.mechanism_semantics_digest,
                "model": self.model,
                "base_mechanism_id": self.base_mechanism_id,
                "operator_ids": self.operator_ids,
            }
        )

    def prompt_projection(self) -> dict[str, Any]:
        return canonical_value(
            {
                "mechanism_axis": self.mechanism_axis,
                "mechanism_id": self.mechanism_id,
                "parent_mechanism_id": self.parent_mechanism_id,
                "composition": {
                    "base_mechanism_id": self.base_mechanism_id,
                    "primary_operator_id": (
                        self.operator_ids[0] if self.operator_ids else None
                    ),
                    "secondary_operator_id": (
                        self.operator_ids[1]
                        if len(self.operator_ids) > 1
                        else None
                    ),
                },
                "summary": self.summary,
            }
        )


def _entrypoint_source_digest(entrypoint: str) -> str:
    module_name, separator, _attribute = entrypoint.partition(":")
    if not separator:
        raise CampaignRuntimeError(f"invalid entrypoint: {entrypoint}")
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise CampaignRuntimeError(f"entrypoint module is unavailable: {entrypoint}")
    path = Path(spec.origin)
    if not path.is_file():
        raise CampaignRuntimeError(f"entrypoint source is unavailable: {entrypoint}")
    return bytes_sha256(path.read_bytes())


def root_parent_mechanism_id_from_entries(
    entries: Sequence[Mapping[str, Any]], mechanism_id: str
) -> str:
    by_id = {str(item["mechanism_id"]): item for item in entries}
    current = mechanism_id
    seen: set[str] = set()
    while by_id[current].get("parent_mechanism_id") is not None:
        if current in seen:
            raise CampaignRuntimeError("catalog parent lineage contains a cycle")
        seen.add(current)
        current = str(by_id[current]["parent_mechanism_id"])
        if current not in by_id:
            raise CampaignRuntimeError("catalog parent lineage is incomplete")
    return current


@lru_cache(maxsize=1)
def executable_mechanisms() -> tuple[ExecutableMechanismV1, ...]:
    entries = _composition_entries()
    mechanisms: list[ExecutableMechanismV1] = []
    seen_semantics: set[str] = set()
    for entry in entries:
        program = _catalog_program(entry)
        report = compile_program(program)
        if not report.is_valid:
            raise CampaignRuntimeError(
                f"{entry['mechanism_id']} does not compile: {report.to_dict()}"
            )
        semantics = str(report.mechanism_semantics_digest)
        if semantics in seen_semantics:
            raise CampaignRuntimeError(
                f"catalog mechanisms collapse to one semantics digest: "
                f"{entry['mechanism_id']}"
            )
        seen_semantics.add(semantics)
        mechanisms.append(
            ExecutableMechanismV1(
                mechanism_id=str(entry["mechanism_id"]),
                mechanism_axis=str(entry["mechanism_axis"]),
                parent_mechanism_id=(
                    str(entry["parent_mechanism_id"])
                    if entry.get("parent_mechanism_id") is not None
                    else None
                ),
                summary=str(entry["summary"]),
                entrypoint=str(entry["entrypoint"]),
                entrypoint_source_sha256=_entrypoint_source_digest(
                    str(entry["entrypoint"])
                ),
                model=str(entry["model"]),
                base_model_config=(
                    str(entry["base_model_config"])
                ),
                config=canonical_value(dict(entry["config"])),
                mechanism_program=canonical_value(program),
                mechanism_program_digest=str(report.mechanism_program_digest),
                mechanism_semantics_digest=semantics,
                candidate_id=str(report.candidate_id),
                base_mechanism_id=root_parent_mechanism_id_from_entries(
                    entries, str(entry["mechanism_id"])
                ),
                operator_ids=tuple(
                    str(item) for item in entry["operator_ids"]
                ),
            )
        )
    return tuple(mechanisms)


def executable_mechanism(mechanism_id: str) -> ExecutableMechanismV1:
    aliases = {
        "BPR_MARGIN": "BPR_MF__BPR_MARGIN",
        "BPR_MIXED_NEGATIVE": "BPR_MF__BPR_MIXED_NEGATIVE",
        "BPR_POPULARITY_NEGATIVE": (
            "BPR_MF__BPR_POPULARITY_NEGATIVE"
        ),
        "BPR_NORM_CONSTRAINED": "BPR_MF__BPR_NORM_CONSTRAINT",
        "BPR_POPULARITY_REGULARIZED": "BPR_MF__BPR_POPULARITY_REG",
        "BPR_MIXED_NEGATIVE_MARGIN": (
            "BPR_MF__BPR_MIXED_NEGATIVE__BPR_MARGIN"
        ),
        "BPR_POPULARITY_MARGIN": (
            "BPR_MF__BPR_POPULARITY_NEGATIVE__BPR_MARGIN"
        ),
        "LIGHTGCN_SHALLOW": "LIGHTGCN__LGCN_SHALLOW",
        "LIGHTGCN_LAYER_WEIGHTED": (
            "LIGHTGCN__LGCN_LAYER_WEIGHTED"
        ),
        "LIGHTGCN_RESIDUAL": "LIGHTGCN__LGCN_RESIDUAL",
        "LIGHTGCN_RESIDUAL_NORM": (
            "LIGHTGCN__LGCN_RESIDUAL__LGCN_NORM_CONSTRAINT"
        ),
        "LIGHTGCN_EDGE_DROPOUT_RESIDUAL": (
            "LIGHTGCN__LGCN_EDGE_DROPOUT__LGCN_RESIDUAL"
        ),
        "LIGHTGCN_DEBIASED_NEGATIVE": (
            "LIGHTGCN__LGCN_DEBIASED_NEGATIVE"
        ),
        "LIGHTGCN_AUX_ALIGNMENT": (
            "LIGHTGCN__LGCN_AUX_ALIGNMENT"
        ),
        "LIGHTGCN_RANK_AWARE": "LIGHTGCN__LGCN_RANK_AWARE",
    }
    mechanism_id = aliases.get(mechanism_id, mechanism_id)
    matches = [
        item for item in executable_mechanisms() if item.mechanism_id == mechanism_id
    ]
    if len(matches) != 1:
        raise CampaignRuntimeError(f"unknown mechanism_id: {mechanism_id}")
    return matches[0]


def root_parent_mechanism_id(mechanism_id: str) -> str:
    current = executable_mechanism(mechanism_id)
    seen: set[str] = set()
    while current.parent_mechanism_id is not None:
        if current.mechanism_id in seen:
            raise CampaignRuntimeError("catalog parent lineage contains a cycle")
        seen.add(current.mechanism_id)
        current = executable_mechanism(current.parent_mechanism_id)
    return current.mechanism_id


def lineage_catalog_projection(
    *,
    root_mechanism_id: str,
    targeted_axes: Sequence[str] = (),
    executed_mechanism_ids: Sequence[str] = (),
    control_only: bool = False,
) -> dict[str, Any]:
    root = root_parent_mechanism_id(root_mechanism_id)
    executed = set(executed_mechanism_ids)
    axes = set(targeted_axes)
    rows = [
        item
        for item in executable_mechanisms()
        if root_parent_mechanism_id(item.mechanism_id) == root
        and (
            control_only
            and item.mechanism_id == root
            or not control_only
            and item.mechanism_id != root
            and (not axes or item.mechanism_axis in axes)
            and item.mechanism_id not in executed
        )
    ]
    if not rows and not control_only:
        rows = [
            item
            for item in executable_mechanisms()
            if root_parent_mechanism_id(item.mechanism_id) == root
            and item.mechanism_id != root
        ]
    payload = {
        "control_only": control_only,
        "mechanisms": [item.prompt_projection() for item in rows],
        "root_parent_mechanism_id": root,
    }
    return {**canonical_value(payload), "projection_digest": sha256_digest(payload)}


def execution_recipe_for_program(
    program: Mapping[str, Any],
) -> dict[str, Any]:
    report = compile_program(dict(program))
    if not report.is_valid:
        raise CampaignRuntimeError("mechanism program is not BL-ICF valid")
    matches = [
        item
        for item in executable_mechanisms()
        if item.mechanism_semantics_digest == report.mechanism_semantics_digest
    ]
    if len(matches) != 1:
        raise CampaignRuntimeError(
            "mechanism program is outside the exact executable campaign catalog"
        )
    recipe = matches[0].execution_recipe()
    recipe["candidate_id"] = str(report.candidate_id)
    recipe["mechanism_program_digest"] = str(report.mechanism_program_digest)
    recipe["execution_recipe_digest"] = sha256_digest(recipe)
    return canonical_value(recipe)


def program_from_proposal(proposal: Mapping[str, Any]) -> dict[str, Any]:
    composition = proposal.get("composition")
    if isinstance(composition, Mapping):
        operator_ids = tuple(
            item
            for item in (
                composition.get("primary_operator_id"),
                composition.get("secondary_operator_id"),
            )
            if item is not None
        )
        matches = [
            item
            for item in executable_mechanisms()
            if item.base_mechanism_id
            == str(composition["base_mechanism_id"])
            and item.operator_ids == operator_ids
        ]
        if len(matches) != 1:
            raise CampaignRuntimeError(
                "proposal composition is not one compatible executable program"
            )
        mechanism = matches[0]
        declared = proposal.get("mechanism_id")
        if declared is not None and str(declared) != mechanism.mechanism_id:
            raise CampaignRuntimeError(
                "declared mechanism_id does not match its composition"
            )
    else:
        mechanism = executable_mechanism(str(proposal["mechanism_id"]))
    program = deep_thaw(mechanism.mechanism_program)
    report = compile_program(program)
    if (
        not report.is_valid
        or report.mechanism_program_digest
        != mechanism.mechanism_program_digest
        or report.mechanism_semantics_digest
        != mechanism.mechanism_semantics_digest
    ):
        raise CampaignRuntimeError(
            "catalog mechanism program identity is inconsistent"
        )
    resolved = execution_recipe_for_program(program)
    if resolved["mechanism_id"] != mechanism.mechanism_id:
        raise CampaignRuntimeError("proposal materialized another mechanism")
    return program


def campaign_projection() -> dict[str, Any]:
    mechanisms = executable_mechanisms()
    catalog = executable_catalog_spec()
    payload = {
        "bases": copy.deepcopy(catalog["bases"]),
        "catalog_id": catalog["catalog_id"],
        "composition_contract": copy.deepcopy(
            catalog["composition_contract"]
        ),
        "mechanisms": [item.prompt_projection() for item in mechanisms],
        "operators": copy.deepcopy(catalog["operators"]),
        "projection_contract": "EXPOSURE_EQUALS_EXACT_EXECUTION_V1",
        "search_space_id": "BL_ICF_EXECUTABLE_PROFILE_V2",
    }
    return {**canonical_value(payload), "projection_digest": sha256_digest(payload)}


@lru_cache(maxsize=1)
def bl_icf_executable_profile_v2() -> dict[str, Any]:
    mechanisms = executable_mechanisms()
    handler_sources = {
        item.entrypoint: item.entrypoint_source_sha256
        for item in mechanisms
    }
    payload = {
        "profile_id": "BL_ICF_EXECUTABLE_PROFILE_V2",
        "catalog_digest": sha256_digest(executable_catalog_spec()),
        "composition_contract": executable_catalog_spec()[
            "composition_contract"
        ],
        "development_protocol_digest": sha256_digest(
            _json_object(_PROTOCOL_PATH)
        ),
        "executable_mechanism_count": len(mechanisms),
        "handler_source_manifest": [
            {"entrypoint": entrypoint, "sha256": digest}
            for entrypoint, digest in sorted(handler_sources.items())
        ],
        "mechanism_identity_manifest": [
            {
                "base_mechanism_id": item.base_mechanism_id,
                "candidate_id": item.candidate_id,
                "execution_recipe_digest": item.execution_recipe_digest,
                "mechanism_id": item.mechanism_id,
                "mechanism_program_digest": item.mechanism_program_digest,
                "mechanism_semantics_digest": (
                    item.mechanism_semantics_digest
                ),
                "operator_ids": item.operator_ids,
            }
            for item in mechanisms
        ],
        "proposal_schema_sha256": bytes_sha256(
            _PROPOSAL_SCHEMA_PATH.read_bytes()
        ),
        "training_profile_digest": sha256_digest(
            campaign_training_profile()
        ),
    }
    return {
        **canonical_value(payload),
        "profile_digest": sha256_digest(payload),
    }


@lru_cache(maxsize=1)
def campaign_runtime_profile() -> dict[str, Any]:
    source_files = (
        _ANCHOR_PATH,
        _CATALOG_PATH,
        _PROPOSAL_SCHEMA_PATH,
        _PROTOCOL_PATH,
        _CAMPAIGN_PROTOCOL_PATH,
        _PARTITION_PROFILE_PATH,
        _TRAINING_PROFILE_PATH,
    )
    project_root = Path(__file__).resolve().parents[4]
    runtime_sources = (
        Path(__file__).resolve(),
        Path(__file__).with_name("materialization.py"),
        Path(__file__).with_name("original_main.py"),
        Path(__file__).with_name("campaign_dataset.py"),
        Path(__file__).with_name("common_execution_guard.py"),
        Path(__file__).with_name("campaign_pilot.py"),
        Path(__file__).with_name("controllers.py"),
        Path(__file__).with_name("pilot_training.py"),
        Path(__file__).with_name("precanary_orchestration.py"),
        Path(__file__).with_name("real_canary.py"),
        Path(__file__).with_name("real_pilot.py"),
        Path(__file__).with_name("research_capability.py"),
        Path(__file__).with_name("research_contracts.py"),
        Path(__file__).with_name("research_science.py"),
        Path(__file__).with_name("training_execution_guard.py"),
        Path(__file__).with_name("training_materialization.py"),
        Path(__file__).with_name("training_runtime_release.py"),
        Path(__file__).with_name("meta_vnext_campaign.py"),
        Path(__file__).parents[2] / "helix" / "composition.py",
        Path(__file__).parents[2] / "helix" / "contracts.py",
        Path(__file__).parents[2] / "helix" / "fusion.py",
        Path(__file__).parents[2] / "helix" / "guard_adapter.py",
        Path(__file__).parents[2] / "helix" / "ledger.py",
        Path(__file__).parents[2] / "helix" / "ports.py",
        Path(__file__).parents[2]
        / "helix"
        / "scientific_attribution.py",
        project_root / "scripts" / "campaign_train_worker.py",
    )
    payload = {
        "profile_id": "BL_ICF_EXECUTABLE_PROFILE_V2",
        "executable_profile_digest": bl_icf_executable_profile_v2()[
            "profile_digest"
        ],
        "candidate_contract": "CandidateProposalV4",
        "catalog_digest": sha256_digest(executable_catalog_spec()),
        "development_protocol_digest": sha256_digest(_json_object(_PROTOCOL_PATH)),
        "executable_mechanism_count": len(executable_mechanisms()),
        "entrypoint_source_manifest": [
            {
                "entrypoint": item.entrypoint,
                "sha256": item.entrypoint_source_sha256,
            }
            for item in executable_mechanisms()
        ],
        "pilot_main_equivalence_rule": (
            "ONLY_SEARCH_SEED_ROUND_COUNT_ROOTS_AND_EVIDENCE_CLASS_MAY_DIFFER"
        ),
        "projection_digest": campaign_projection()["projection_digest"],
        "proposal_schema_sha256": bytes_sha256(_PROPOSAL_SCHEMA_PATH.read_bytes()),
        "source_manifest": [
            {
                "path": path.name,
                "sha256": bytes_sha256(path.read_bytes()),
            }
            for path in source_files
        ],
        "runtime_source_manifest": [
            {
                "path": path.relative_to(project_root).as_posix(),
                "sha256": bytes_sha256(path.read_bytes()),
            }
            for path in runtime_sources
        ],
        "training_profile_digest": sha256_digest(campaign_training_profile()),
    }
    return {**canonical_value(payload), "profile_digest": sha256_digest(payload)}


def normalized_campaign_overlay(
    overlay: Mapping[str, Any],
) -> dict[str, Any]:
    allowed = {
        "evidence_class",
        "output_roots",
        "scheduled_rounds",
        "search_seed",
    }
    unexpected = set(overlay) - allowed
    if unexpected:
        raise CampaignRuntimeError(
            f"Pilot/Main overlay changes frozen runtime content: {sorted(unexpected)}"
        )
    return canonical_value(dict(overlay))


def _import_entrypoint(entrypoint: str) -> None:
    module_name, separator, attribute = entrypoint.partition(":")
    if not separator or not module_name or not attribute:
        raise CampaignRuntimeError(f"invalid entrypoint: {entrypoint}")
    module = importlib.import_module(module_name)
    if not hasattr(module, attribute):
        raise CampaignRuntimeError(f"entrypoint is unavailable: {entrypoint}")


@lru_cache(maxsize=2)
def campaign_readiness_failures(
    *,
    import_entrypoints: bool = False,
) -> tuple[str, ...]:
    failures: list[str] = []
    mechanisms = executable_mechanisms()
    ids = tuple(item.mechanism_id for item in mechanisms)
    if len(ids) != len(set(ids)):
        failures.append("PROMPT_EXECUTION_CATALOG_MISMATCH")
    if not 50 <= len(mechanisms) <= 100:
        failures.append("MAIN_GRADE_POOL_SIZE_MISMATCH")
    if any(len(item.operator_ids) > 2 for item in mechanisms):
        failures.append("COMPOSITION_DEPTH_EXCEEDED")
    proposal_required = set(
        campaign_proposal_schema()["properties"]["proposals"]["items"][
            "required"
        ]
    )
    if "composition" not in proposal_required:
        failures.append("TYPED_COMPOSITION_CONTRACT_MISSING")
    parent_ids = {
        item.parent_mechanism_id
        for item in mechanisms
        if item.parent_mechanism_id is not None
    }
    if not parent_ids.issubset(set(ids)):
        failures.append("CATALOG_PARENT_MISSING")
    axes = {item.mechanism_axis for item in mechanisms}
    required_axes = {
        "architecture",
        "geometry",
        "message_transform",
        "objective",
        "propagation",
        "sampling",
        "self_supervision",
    }
    if not required_axes.issubset(axes):
        failures.append("SCIENTIFIC_COVERAGE_FLOOR_MISSING")
    training = campaign_training_profile()
    if (
        training.get("online_metric_source") != "BEST_VALID_RESULT"
        or training.get("online_partition_role") != "DEVELOPMENT_VALIDATION"
        or training.get("heldout_access") != "POST_SELECTION_ONLY"
    ):
        failures.append("ONLINE_HELDOUT_BOUNDARY_MISMATCH")
    for item in mechanisms:
        try:
            resolved = execution_recipe_for_program(item.mechanism_program)
        except (CampaignRuntimeError, ValueError):
            failures.append(f"EXACT_RENDER_FAILED:{item.mechanism_id}")
            continue
        if resolved["mechanism_id"] != item.mechanism_id:
            failures.append(f"EXACT_RENDER_SUBSTITUTION:{item.mechanism_id}")
        if import_entrypoints:
            try:
                _import_entrypoint(item.entrypoint)
            except (CampaignRuntimeError, ImportError):
                failures.append(f"ENTRYPOINT_UNAVAILABLE:{item.mechanism_id}")
    return tuple(sorted(set(failures)))


def role_catalog_projection(
    *,
    targeted_axes: Sequence[str],
    executed_mechanism_ids: Sequence[str],
) -> tuple[dict[str, Any], ...]:
    axes = tuple(str(item) for item in targeted_axes)
    executed = set(str(item) for item in executed_mechanism_ids)
    items = sorted(
        executable_mechanisms(),
        key=lambda item: (
            0 if item.mechanism_axis in axes else 1,
            1 if item.mechanism_id in executed else 0,
            item.mechanism_id,
        ),
    )
    return tuple(item.prompt_projection() for item in items)


__all__ = [
    "CampaignRuntimeError",
    "ExecutableMechanismV1",
    "bl_icf_executable_profile_v2",
    "campaign_projection",
    "campaign_proposal_schema",
    "campaign_proposal_schema_path",
    "campaign_readiness_failures",
    "campaign_runtime_profile",
    "campaign_training_profile",
    "executable_catalog_spec",
    "executable_mechanism",
    "executable_mechanisms",
    "execution_recipe_for_program",
    "normalized_campaign_overlay",
    "program_from_proposal",
    "role_catalog_projection",
]
