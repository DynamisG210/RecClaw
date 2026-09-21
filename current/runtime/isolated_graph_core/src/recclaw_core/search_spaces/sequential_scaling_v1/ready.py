"""Build a self-contained SSD4Rec single-parent launch descriptor."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
)
from recclaw_core.mechanism_space import CompileStatus
from recclaw_core.mechanism_space.canonical import deep_thaw
from recclaw_core.mechanism_space.catalog import resolve_provider
from recclaw_core.research_line.frozen_family_profile import (
    build_frozen_family_profile,
    verify_frozen_family_assets,
)
from recclaw_core.research_line.single_parent_profile import (
    ACTUAL_TRAINING_SEED,
    load_single_parent_launch_payload,
)
from recclaw_core.research_line.single_parent_search import (
    SEQUENTIAL_SCALING_SINGLE_PARENT_SPEC,
    focused_mechanism_language,
)


class SequentialScalingReadyError(ValueError):
    """Raised when real SSD4Rec launch inputs do not match the frozen profile."""


_PARENT_PRIMITIVES = {
    "representation.shared_item_embedding",
    "view.observed_history_masking",
    "packing.padded_prefix_batch",
    "backbone.bidirectional_prefix_reversal_state_space_duality",
    "prediction.tied_dot_product",
    "objective.next_item_softmax",
}
_EXPECTED_EVAL_ARGS = canonical_value(
    {
        "split": {"LS": "valid_only"},
        "order": "TO",
        "group_by": "user",
        "mode": "full",
    }
)


def _read_json(path: Path, *, field_name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SequentialScalingReadyError(
            f"{field_name} is not readable UTF-8 JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise SequentialScalingReadyError(f"{field_name} must contain an object")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256_bytes(payload)


def _copy_search_partition(
    source: Path,
    destination: Path,
    *,
    expected_suffix: str,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if not source.is_file() or not source.name.endswith(expected_suffix):
        raise SequentialScalingReadyError(
            f"search partition must be an existing {expected_suffix} file: {source}"
        )
    with source.open("rb") as handle:
        header = handle.readline().decode("utf-8", errors="strict").strip().split("\t")
        rows = sum(1 for line in handle if line.strip())
    required_fields = {
        "user_id:token",
        "item_id:token",
        "chrono_order:float",
        "timestamp:float",
        "item_id_list:token_seq",
        "chrono_order_list:float_seq",
        "timestamp_list:float_seq",
    }
    if not required_fields <= set(header):
        missing = ", ".join(sorted(required_fields - set(header)))
        raise SequentialScalingReadyError(
            f"sequential search partition lacks frozen prefix fields: {missing}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return {"sha256": _sha256_file(destination), "rows": rows}


def _ssd4rec_parent_program(
    *,
    provider: Any,
    profile_ref: Mapping[str, Any],
) -> dict[str, Any]:
    identity = provider.identity()
    components = [
        {
            "component_id": "parent_item_embedding",
            "slot_id": "ITEM_REPRESENTATION",
            "primitive_id": "representation.shared_item_embedding",
            "inputs": [
                {
                    "port": "items",
                    "source": {
                        "kind": "DATA",
                        "data_role": "TRAIN_PREFIX_ITEM_SEQUENCE",
                    },
                }
            ],
            "parameters": {"dimension": 256, "tie_output": True},
        },
        {
            "component_id": "parent_mask_view",
            "slot_id": "TRAIN_INPUT_VIEW",
            "primitive_id": "view.observed_history_masking",
            "inputs": [
                {
                    "port": "representation",
                    "source": {
                        "kind": "COMPONENT",
                        "component_id": "parent_item_embedding",
                        "output_port": "representation",
                    },
                }
            ],
            "parameters": {
                "mask_probability": 0.1,
                "replacement": "LEARNED_MASK_TOKEN",
            },
        },
        {
            "component_id": "parent_padded_prefix",
            "slot_id": "SEQUENCE_REGISTER_GEOMETRY",
            "primitive_id": "packing.padded_prefix_batch",
            "inputs": [
                {
                    "port": "representation",
                    "source": {
                        "kind": "COMPONENT",
                        "component_id": "parent_mask_view",
                        "output_port": "representation",
                    },
                }
            ],
            "parameters": {"maximum_length": 50},
        },
        {
            "component_id": "parent_bidirectional_ssd",
            "slot_id": "SEQUENCE_BACKBONE",
            "primitive_id": (
                "backbone.bidirectional_prefix_reversal_state_space_duality"
            ),
            "inputs": [
                {
                    "port": "representation",
                    "source": {
                        "kind": "COMPONENT",
                        "component_id": "parent_padded_prefix",
                        "output_port": "representation",
                    },
                }
            ],
            "parameters": {
                "layers": 2,
                "state_dimension": 64,
                "head_dimension": 16,
                "expansion": 2,
                "local_convolution": 4,
                "backward_weight": 0.1,
                "direction_parameter_sharing": "SHARED",
                "reverse_output_alignment": "AS_EMITTED_REVERSE_ORDER",
                "post_mixer": "RESIDUAL_FFN",
                "ffn_multiplier": 4.0,
            },
        },
        {
            "component_id": "parent_tied_score",
            "slot_id": "PREDICTION_HEAD",
            "primitive_id": "prediction.tied_dot_product",
            "inputs": [
                {
                    "port": "state",
                    "source": {
                        "kind": "COMPONENT",
                        "component_id": "parent_bidirectional_ssd",
                        "output_port": "state",
                    },
                },
                {
                    "port": "candidate_items",
                    "source": {"kind": "DATA", "data_role": "ITEM_ID"},
                },
            ],
            "parameters": {"temperature": 1.0},
        },
        {
            "component_id": "parent_next_item_objective",
            "slot_id": "SEQUENTIAL_OBJECTIVE",
            "primitive_id": "objective.next_item_softmax",
            "inputs": [
                {
                    "port": "score",
                    "source": {
                        "kind": "COMPONENT",
                        "component_id": "parent_tied_score",
                        "output_port": "score",
                    },
                },
                {
                    "port": "targets",
                    "source": {
                        "kind": "DATA",
                        "data_role": "TRAIN_NEXT_ITEM_TARGETS",
                    },
                },
            ],
            "parameters": {"label_smoothing": 0.0},
        },
    ]
    return canonical_value(
        {
            "record_type": "MECHANISM_PROGRAM_ENVELOPE",
            "kernel_schema_version": "recclaw.mechanism-space.kernel.v1",
            "search_space_id": identity.search_space_id,
            "search_space_digest": identity.search_space_digest,
            "family_id": identity.family_id,
            "family_version": identity.family_version,
            "profile_ref": dict(profile_ref),
            "program_payload": {
                "schema_version": "recclaw.sequential-scaling-v1.mechanism-program.v1",
                "family_contract_id": identity.family_id,
                "construction_mode": "COMPOSITION",
                "parent_refs": [],
                "research_question": (
                    "Can an exact SSD4Rec parent improve chronological development ranking?"
                ),
                "core_hypothesis": (
                    "Valid-prefix reversal exposes complementary prefix context while "
                    "shared state-space duality keeps sequence mixing efficient."
                ),
                "declared_data_roles": [
                    "ITEM_ID",
                    "TRAIN_NEXT_ITEM_TARGETS",
                    "TRAIN_PREFIX_ITEM_SEQUENCE",
                ],
                "components": components,
                "architecture_operators": [
                    {
                        "operator_id": "add_component",
                        "targets": [],
                        "replacements": [],
                        "parameters": {},
                        "rationale": "Bind the exact executable SSD4Rec parent graph.",
                    }
                ],
                "changed_slots": [
                    {"slot_id": "SEQUENCE_BACKBONE", "change_role": "CORE"}
                ],
                "removed_slots": [],
                "custom_components": [],
                "mechanism_explanation": (
                    "Observed prefixes are mask-augmented only during training, kept "
                    "isolated in padded rows, scanned forward and in valid-prefix "
                    "reverse order by shared Mamba2 weights, and scored through the "
                    "tied item table."
                ),
                "expected_effects": {
                    "relevance": "Bidirectional prefix context improves dev ranking.",
                    "efficiency": "Two linear SSD scans remain practical at length 50.",
                    "robustness": "Padding and reverse alignment do not alter valid-prefix identity.",
                    "coverage": "Full-catalog tied scoring remains unchanged.",
                },
                "matched_control": {
                    "control_ref": "frozen_ssd4rec_parent",
                    "rationale": "Keep data, evaluator, update budget, and source fixed.",
                },
                "ablation_plan": [
                    {
                        "ablation_id": "backward_ssd_off",
                        "remove_component_ids": ["parent_bidirectional_ssd"],
                        "expected_observation": (
                            "Removing the reversed-prefix contribution reduces its effect."
                        ),
                    }
                ],
                "discriminating_predictions": [
                    {
                        "metric_or_probe": "same-protocol development NDCG at ten",
                        "if_supported": "The exact parent remains a strong construction anchor.",
                        "if_refuted": "A matched simpler sequential state-space model is at least as strong.",
                    }
                ],
                "failure_interpretation": {
                    "mechanism_failure": "The directional contribution is inert or harmful.",
                    "optimization_failure": "The finite training objective does not converge.",
                    "protocol_failure": "Chronology, prefix, or evaluator identity differs.",
                    "resource_failure": "The two SSD passes cannot fit the frozen budget.",
                },
                "implementation_plan": [
                    "Clone the exact SSD4Rec adapter and make only parent-relative local changes."
                ],
                "resource_contract": {
                    "relative_training_compute": "MEDIUM",
                    "relative_memory": "MEDIUM",
                    "precompute_required": False,
                    "separate_budget_stages": [],
                },
                "claim_ceiling": "DEVELOPMENT_ONLY_SINGLE_PROTOCOL_NO_GENERAL_CLAIM",
                "protocol_impact": {"status": "UNCHANGED", "requested_changes": []},
            },
        }
    )


def _parent_source_bundle(
    *,
    adapter_path: Path,
    candidate_id: str,
    program_digest: str,
) -> tuple[dict[str, Any], str]:
    adapter_bytes = adapter_path.read_bytes()
    try:
        adapter_source = adapter_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SequentialScalingReadyError(
            "SSD4Rec parent adapter must be UTF-8 Python"
        ) from error
    declaration = "class SSD4Rec(SequentialRecommender):"
    if adapter_source.count(declaration) != 1:
        raise SequentialScalingReadyError(
            "SSD4Rec parent adapter lacks one canonical model declaration"
        )
    candidate_source = adapter_source.replace(
        declaration,
        "class FreshCandidateModel(SequentialRecommender):",
        1,
    )
    files = []
    for path, content in (
        ("recclaw_ext/__init__.py", "from .candidate import FreshCandidateModel\n"),
        ("recclaw_ext/candidate.py", candidate_source),
        (
            "recclaw_ext/trainer.py",
            "from recbole.trainer import Trainer\n\n\n"
            "class FreshCandidateTrainer(Trainer):\n"
            "    pass\n",
        ),
    ):
        files.append(
            {
                "path": path,
                "content": content,
                "sha256": _sha256_bytes(content.encode("utf-8")),
            }
        )
    rows = [
        {
            "path": row["path"],
            "sha256": row["sha256"],
            "size_bytes": len(row["content"].encode("utf-8")),
        }
        for row in files
    ]
    source_tree_digest = sha256_digest(
        {"files": sorted(rows, key=lambda row: row["path"])}
    )
    bundle = canonical_value(
        {
            "candidate_id": candidate_id,
            "program_digest": program_digest,
            "capability_ref": "parent:sequential-scaling:ssd4rec:seed54201",
            "instruction": "CLONE_EXACT_PARENT_AND_LOCAL_PATCH",
            "source_tree_digest": source_tree_digest,
            "files": files,
        }
    )
    return bundle, _sha256_bytes(adapter_bytes)


def _normalized_parent_result(
    *,
    source: Mapping[str, Any],
    source_path: Path,
    expected_dataset: str,
    expected_dataset_sha256: str,
    expected_adapter_sha256: str,
    expected_partition_files: Mapping[str, Any],
    expected_protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    protocol = source.get("protocol")
    best_valid = source.get("best_valid_result")
    external_source = source.get("external_model_source")
    partition_access = source.get("partition_access", {})
    partition_files = source.get("partition_files")
    if not all(
        isinstance(value, Mapping)
        for value in (protocol, best_valid, external_source, partition_files)
    ):
        raise SequentialScalingReadyError(
            "SSD4Rec parent result lacks protocol, metric, or source identity"
        )
    ndcg_key = next(
        (key for key in best_valid if str(key).casefold() == "ndcg@10"), None
    )
    if ndcg_key is None:
        raise SequentialScalingReadyError(
            "SSD4Rec parent result lacks development NDCG@10"
        )
    value = best_valid[ndcg_key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise SequentialScalingReadyError(
            "SSD4Rec parent development NDCG@10 is not a finite metric"
        )
    expected_access = {
        "partitions_opened": ["train", "development"],
        "heldout_opened": False,
    }
    split_identity = source.get("split_identity", {})
    recorded_test_identity = (
        split_identity.get("test")
        if isinstance(split_identity, Mapping)
        else None
    )
    observed_protocol = {
        key: protocol.get(key)
        for key in expected_protocol
    }
    if (
        source.get("model") != "SSD4Rec"
        or canonical_value(observed_protocol) != canonical_value(expected_protocol)
        or source.get("test_result") is not None
        or canonical_value(partition_access) != canonical_value(expected_access)
        or recorded_test_identity not in (None, {})
        or external_source.get("sha256") != expected_adapter_sha256
        or float(source.get("best_valid_score", float("nan"))) != float(value)
    ):
        raise SequentialScalingReadyError(
            "SSD4Rec parent must be the exact seed-54201 validation-only run "
            "on the frozen dataset, with explicit heldout_opened=false and no "
            "test identity or result"
        )
    if canonical_value(partition_files) != canonical_value(expected_partition_files):
        raise SequentialScalingReadyError(
            "SSD4Rec parent result is not bound to the supplied physical "
            "train and development files"
        )
    normalized = canonical_value(
        {
            "best_valid_result": {"ndcg@10": float(value)},
            "best_valid_score": float(value),
            "exit_status": "SUCCESS",
            "metric_source": "BEST_VALID_RESULT",
            "online_partition_role": "DEVELOPMENT_VALIDATION",
            "seed": ACTUAL_TRAINING_SEED,
            "split": "data/dev",
            "test_result": None,
            "source_result_ref": str(source_path.resolve()),
            "source_result_sha256": _sha256_file(source_path),
            "source_partition_files": expected_partition_files,
        }
    )
    return normalized, float(value)


def build_sequential_scaling_ready_launch(
    *,
    template_path: Path,
    parent_adapter_path: Path,
    parent_result_path: Path,
    train_path: Path,
    dev_path: Path,
    output_root: Path,
) -> Path:
    """Build and validate a no-run SSD4Rec launch from explicit real inputs."""

    template_path = template_path.expanduser().resolve()
    parent_adapter_path = parent_adapter_path.expanduser().resolve()
    parent_result_path = parent_result_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not parent_adapter_path.is_file():
        raise SequentialScalingReadyError(
            f"SSD4Rec parent adapter does not exist: {parent_adapter_path}"
        )
    payload = _read_json(template_path, field_name="sequential launch template")
    if payload.get("profile_key") != "sequential_scaling":
        raise SequentialScalingReadyError(
            "launch template is not the sequential-scaling profile"
        )
    config_overrides = payload["execution_contract"]["config_overrides"]
    if canonical_value(config_overrides.get("eval_args")) != _EXPECTED_EVAL_ARGS:
        raise SequentialScalingReadyError(
            "SSD4Rec launch template differs from the strict valid-only protocol"
        )
    spec = SEQUENTIAL_SCALING_SINGLE_PARENT_SPEC
    provider = resolve_provider(spec.mechanism_space_id)
    required = set(
        focused_mechanism_language(spec)["parent_contract"][
            "required_parent_foundation_primitives"
        ]
    )
    if required != _PARENT_PRIMITIVES:
        raise SequentialScalingReadyError(
            "focused mechanism language differs from the executable SSD4Rec parent"
        )

    frozen = payload["frozen_profile"]["frozen_fields"]
    expected_frozen_split = canonical_value(
        {
            "strategy": "TO_LS",
            "valid": "LAST_EVENT",
            "group_by": "user",
        }
    )
    if canonical_value(frozen.get("chronological_split")) != expected_frozen_split:
        raise SequentialScalingReadyError(
            "SSD4Rec frozen split differs from the strict valid-only protocol"
        )
    dataset = str(frozen["dataset"])
    dataset_sha256 = str(frozen["dataset_snapshot"]["interaction_sha256"])
    dataset_root = output_root / "search_data" / dataset
    train_destination = dataset_root / f"{dataset}.train.inter"
    dev_destination = dataset_root / f"{dataset}.dev.inter"
    train_identity = _copy_search_partition(
        train_path,
        train_destination,
        expected_suffix=".train.inter",
    )
    dev_identity = _copy_search_partition(
        dev_path,
        dev_destination,
        expected_suffix=".dev.inter",
    )
    manifest = canonical_value(
        {
            "schema": "recclaw.search-only-dataset.v1",
            "dataset": dataset,
            "parent_interaction_sha256": dataset_sha256,
            "parent_split": frozen["chronological_split"],
            "partition_files": {
                "train": {
                    "path": f"{dataset}/{dataset}.train.inter",
                    "sha256": train_identity["sha256"],
                },
                "development": {
                    "path": f"{dataset}/{dataset}.dev.inter",
                    "sha256": dev_identity["sha256"],
                },
            },
            "abi_partition_roles": {
                "train": "TRAIN",
                "valid": "DEVELOPMENT_VALIDATION",
                "test": "DEVELOPMENT_VALIDATION",
            },
            "heldout_partition_present": False,
        }
    )
    manifest_path = output_root / "search_data" / "search-data-manifest.json"
    manifest_sha256 = _write_json(manifest_path, manifest)
    payload["execution_contract"]["search_data"] = {
        "root": str((output_root / "search_data").resolve()),
        "manifest_ref": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
    }

    profile_ref, _ = build_frozen_family_profile(
        provider,
        profile_id=payload["frozen_profile"]["profile_id"],
        frozen_fields=frozen,
    )
    program = _ssd4rec_parent_program(provider=provider, profile_ref=profile_ref)
    report = provider.compile(deep_thaw(program))
    if report.status is not CompileStatus.VALID_NEEDS_IMPLEMENTATION:
        raise SequentialScalingReadyError(
            "executable SSD4Rec parent program does not compile: "
            + json.dumps(report.to_dict(), sort_keys=True)
        )
    binding = {
        "candidate_id": report.candidate_id,
        "program_digest": report.mechanism_program_digest,
    }
    source_bundle, adapter_sha256 = _parent_source_bundle(
        adapter_path=parent_adapter_path,
        candidate_id=report.candidate_id,
        program_digest=report.mechanism_program_digest,
    )
    payload["baseline_context"]["parent_anchor"].update(
        {
            "binding": binding,
            "mechanism_program": program,
            "source_bundle": source_bundle,
        }
    )

    source_result = _read_json(
        parent_result_path,
        field_name="SSD4Rec parent result",
    )
    normalized_result, metric_value = _normalized_parent_result(
        source=source_result,
        source_path=parent_result_path,
        expected_dataset=dataset,
        expected_dataset_sha256=dataset_sha256,
        expected_adapter_sha256=adapter_sha256,
        expected_partition_files={
            "train": train_identity,
            "development": dev_identity,
        },
        expected_protocol={
            "dataset": dataset,
            "raw_dataset_sha256": dataset_sha256,
            "seed": ACTUAL_TRAINING_SEED,
            "selection_partition": "validation",
            "split": config_overrides["eval_args"]["split"],
            "group_by": config_overrides["eval_args"]["group_by"],
            "order": config_overrides["eval_args"]["order"],
            "mode": config_overrides["eval_args"]["mode"],
        },
    )
    worker_result_path = output_root / "observations" / "parent-worker-result.json"
    worker_result_sha256 = _write_json(worker_result_path, normalized_result)
    payload["baseline_context"]["parent_anchor"]["paired_metric"][
        "value"
    ] = metric_value

    verified_assets = verify_frozen_family_assets(
        frozen_fields=frozen,
        config_overrides=payload["execution_contract"]["config_overrides"],
        search_data=payload["execution_contract"]["search_data"],
    )
    expected_parent_binding = {
        **binding,
        "source_tree_digest": source_bundle["source_tree_digest"],
    }
    comparator_digest = sha256_digest(
        {
            "schema": "recclaw.single-parent-comparator-executable.v1",
            "parent_binding": expected_parent_binding,
            "protocol_digest": profile_ref["profile_digest"],
            "asset_manifest_digest": verified_assets["digest"],
        }
    )
    receipt = canonical_value(
        {
            "schema": "recclaw.single-parent-parent-observation.v1",
            "profile_key": "sequential_scaling",
            "parent_binding": expected_parent_binding,
            "protocol_digest": profile_ref["profile_digest"],
            "asset_manifest_digest": verified_assets["digest"],
            "seed": ACTUAL_TRAINING_SEED,
            "split": "data/dev",
            "metric": {
                "name": "NDCG@10",
                "source": "BEST_VALID_RESULT",
                "partition_role": "DEVELOPMENT_VALIDATION",
                "value": metric_value,
            },
            "comparator_ref": source_bundle["capability_ref"],
            "comparator_digest": comparator_digest,
            "worker_result_ref": str(worker_result_path.resolve()),
            "worker_result_digest": worker_result_sha256,
        }
    )
    receipt_path = output_root / "observations" / "parent-observation.json"
    receipt_sha256 = _write_json(receipt_path, receipt)
    payload["parent_source"] = {
        "source_ref": str(receipt_path.resolve()),
        "source_sha256": receipt_sha256,
        "comparator_ref": source_bundle["capability_ref"],
        "comparator_digest": comparator_digest,
        "frozen_ndcg_at_10": metric_value,
    }
    payload["evidence"] = {
        "known_parent_result_seed": ACTUAL_TRAINING_SEED,
        "known_parent_development_ndcg_at_10": metric_value,
        "known_result_ref": str(parent_result_path),
        "launch_asset_status": "READY_NO_RUN",
        "formal_launch_rule": (
            "Use only this frozen seed-54201 development comparator and the "
            "physical train-plus-development-only search root."
        ),
    }

    load_single_parent_launch_payload(payload)
    launch_path = output_root / "sequential_scaling_ssd4rec_single_parent.ready.json"
    _write_json(launch_path, payload)
    return launch_path


__all__ = [
    "SequentialScalingReadyError",
    "build_sequential_scaling_ready_launch",
]
