"""Build a self-contained DiffRec single-parent launch descriptor."""

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
    DIFFUSION_FLOW_SINGLE_PARENT_SPEC,
    focused_mechanism_language,
)


class DiffusionFlowReadyError(ValueError):
    """Raised when real DiffRec launch inputs do not match the frozen profile."""


_PARENT_PRIMITIVES = {
    "state.full_interaction_vector",
    "forward.gaussian_variance_preserving",
    "dynamics.time_conditioned_mlp",
    "objective.predict_clean_state",
    "schedule.linear",
    "solver.deterministic_reduced_step",
    "score.recovered_logits",
}
_PARENT_CONFIG_SHA256 = (
    "151b1e443628e9e496428cc1f133392eae24674d4087d6eba703d3e88cc819fc"
)
_DIFFREC_MODEL_CONFIG_SHA256 = (
    "c4c70c548004c9eec17944128426b505bfbac1a884ede70c8ebcce2490f6ca08"
)
_EXPECTED_EXECUTION_CONFIG = canonical_value(
    {
        "eval_args": {
            "split": None,
            "order": "TO",
            "group_by": "user",
            "mode": "full",
        },
        "metrics": ["Recall", "NDCG", "Hit"],
        "topk": [10, 20],
        "valid_metric": "NDCG@10",
        "train_batch_size": 4096,
        "eval_batch_size": 131072,
        "epochs": 300,
        "eval_step": 1,
        "stopping_step": 10,
        "learning_rate": 0.001,
        "reproducibility": True,
        "repeatable": False,
        "metric_decimal_place": 4,
        "train_neg_sample_args": None,
    }
)
_EXPECTED_TRAINING = canonical_value(
    {
        "epochs_cap": 300,
        "stopping_step": 10,
        "train_batch_size": 4096,
        "eval_batch_size": 4096,
        "learning_rate": 0.001,
    }
)


def _read_json(path: Path, *, field_name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DiffusionFlowReadyError(
            f"{field_name} is not readable UTF-8 JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise DiffusionFlowReadyError(f"{field_name} must contain an object")
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
        raise DiffusionFlowReadyError(
            f"search partition must be an existing {expected_suffix} file: {source}"
        )
    with source.open("rb") as handle:
        header = handle.readline().decode("utf-8", errors="strict").strip().split("\t")
        rows = sum(1 for line in handle if line.strip())
    required_fields = {"user_id:token", "item_id:token"}
    if not required_fields <= set(header):
        missing = ", ".join(sorted(required_fields - set(header)))
        raise DiffusionFlowReadyError(
            f"DiffRec search partition lacks frozen interaction fields: {missing}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return {"sha256": _sha256_file(destination), "rows": rows}


def _copy_catalog_items(
    source: Path,
    destination: Path,
    *,
    expected_items: int,
) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if not source.is_file() or not source.name.endswith(".item"):
        raise DiffusionFlowReadyError(
            f"catalog mapping must be an existing .item file: {source}"
        )
    with source.open("rb") as handle:
        header = handle.readline().decode("utf-8", errors="strict").strip().split("\t")
        if "item_id:token" not in header:
            raise DiffusionFlowReadyError("DiffRec catalog mapping lacks item_id:token")
        item_column = header.index("item_id:token")
        items: set[str] = set()
        rows = 0
        for raw_line in handle:
            if not raw_line.strip():
                continue
            values = raw_line.decode("utf-8", errors="strict").rstrip("\r\n").split("\t")
            if item_column >= len(values) or not values[item_column]:
                raise DiffusionFlowReadyError("DiffRec catalog mapping has an empty item id")
            rows += 1
            items.add(values[item_column])
    if rows != expected_items or len(items) != expected_items:
        raise DiffusionFlowReadyError(
            f"DiffRec catalog must contain exactly {expected_items} unique rated items"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return {"sha256": _sha256_file(destination), "items": expected_items}


def _diffrec_parent_program(
    *,
    provider: Any,
    profile_ref: Mapping[str, Any],
) -> dict[str, Any]:
    identity = provider.identity()
    def component_source(component_id: str, output_port: str) -> dict[str, str]:
        return {
            "kind": "COMPONENT",
            "component_id": component_id,
            "output_port": output_port,
        }

    components = [
        {
            "component_id": "parent_interaction_state",
            "slot_id": "STATE_REPRESENTATION",
            "primitive_id": "state.full_interaction_vector",
            "inputs": [
                {
                    "port": "history",
                    "source": {
                        "kind": "DATA",
                        "data_role": "TRAIN_USER_INTERACTION_SIGNAL",
                    },
                }
            ],
            "parameters": {"value_encoding": "BINARY"},
        },
        {
            "component_id": "parent_gaussian_forward",
            "slot_id": "FORWARD_PATH",
            "primitive_id": "forward.gaussian_variance_preserving",
            "inputs": [
                {
                    "port": "state",
                    "source": component_source("parent_interaction_state", "state"),
                }
            ],
            "parameters": {
                "steps": 5,
                "noise_scale": 0.001,
                "beta_fixed": True,
                "fixed_first_beta": 0.00001,
            },
        },
        {
            "component_id": "parent_linear_schedule",
            "slot_id": "TIME_SCHEDULE",
            "primitive_id": "schedule.linear",
            "inputs": [
                {
                    "port": "time",
                    "source": component_source("parent_gaussian_forward", "time"),
                }
            ],
            "parameters": {"start": 0.0005, "end": 0.005},
        },
        {
            "component_id": "parent_time_mlp",
            "slot_id": "GENERATIVE_DYNAMICS",
            "primitive_id": "dynamics.time_conditioned_mlp",
            "inputs": [
                {
                    "port": "state",
                    "source": component_source("parent_gaussian_forward", "state"),
                },
                {
                    "port": "time",
                    "source": component_source("parent_linear_schedule", "time"),
                },
            ],
            "parameters": {
                "layers": 1,
                "hidden_dimension": 300,
                "time_embedding_dimension": 10,
                "time_fusion": "CONCAT",
                "activation": "TANH",
                "dropout": 0.5,
                "normalize_input": False,
            },
        },
        {
            "component_id": "parent_clean_objective",
            "slot_id": "DENOISING_FLOW_OBJECTIVE",
            "primitive_id": "objective.predict_clean_state",
            "inputs": [
                {
                    "port": "field",
                    "source": component_source("parent_time_mlp", "field"),
                },
                {
                    "port": "clean",
                    "source": component_source("parent_interaction_state", "state"),
                },
                {
                    "port": "time",
                    "source": component_source("parent_linear_schedule", "time"),
                },
            ],
            "parameters": {
                "loss": "WEIGHTED_MSE",
                "time_weighting": "SNR",
                "timestep_sampling": "LOSS_SECOND_MOMENT_AFTER_WARMUP",
                "history_num_per_term": 10,
                "uniform_mixture_probability": 0.001,
            },
        },
        {
            "component_id": "parent_deterministic_solver",
            "slot_id": "RECOVERY_SOLVER",
            "primitive_id": "solver.deterministic_reduced_step",
            "inputs": [
                {
                    "port": "field",
                    "source": component_source("parent_time_mlp", "field"),
                },
                {
                    "port": "state",
                    "source": component_source("parent_gaussian_forward", "state"),
                },
                {
                    "port": "time",
                    "source": component_source("parent_linear_schedule", "time"),
                },
            ],
            "parameters": {
                "steps": 5,
                "spacing": "UNIFORM",
                "sampling_steps": 0,
                "sampling_noise": False,
                "start_state": "OBSERVED_INTERACTION_STATE",
            },
        },
        {
            "component_id": "parent_recovered_score",
            "slot_id": "SCORE_HEAD",
            "primitive_id": "score.recovered_logits",
            "inputs": [
                {
                    "port": "recovered",
                    "source": component_source(
                        "parent_deterministic_solver", "recovered"
                    ),
                }
            ],
            "parameters": {"temperature": 1.0},
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
                "schema_version": "recclaw.diffusion-flow-cf-v1.mechanism-program.v1",
                "family_contract_id": identity.family_id,
                "construction_mode": "COMPOSITION",
                "parent_refs": [],
                "research_question": (
                    "Can a faithful DiffRec intervention improve development ranking?"
                ),
                "core_hypothesis": (
                    "A learned time-conditioned reverse process can recover full-catalog "
                    "preference signal from a bounded Gaussian corruption path."
                ),
                "declared_data_roles": ["TRAIN_USER_INTERACTION_SIGNAL"],
                "components": components,
                "architecture_operators": [
                    {
                        "operator_id": "add_component",
                        "targets": [],
                        "replacements": [],
                        "parameters": {},
                        "rationale": "Bind the exact executable DiffRec parent graph.",
                    }
                ],
                "changed_slots": [
                    {"slot_id": "GENERATIVE_DYNAMICS", "change_role": "CORE"}
                ],
                "removed_slots": [],
                "custom_components": [],
                "mechanism_explanation": (
                    "DiffRec corrupts the full binary interaction vector over five "
                    "variance-preserving steps, predicts the clean state with a "
                    "time-conditioned MLP, and deterministically recovers catalog scores."
                ),
                "expected_effects": {
                    "relevance": "Denoising preserves personalized ranking signal.",
                    "efficiency": "Five learned diffusion steps bound field evaluations.",
                    "robustness": "Seeded corruption and deterministic recovery remain finite.",
                    "coverage": "The full observed catalog is scored directly.",
                },
                "matched_control": {
                    "control_ref": "frozen_diffrec_parent",
                    "rationale": "Keep data, evaluator, update budget, and source fixed.",
                },
                "ablation_plan": [
                    {
                        "ablation_id": "learned_reverse_path_off",
                        "remove_component_ids": ["parent_time_mlp"],
                        "expected_observation": (
                            "Removing the learned denoiser eliminates its ranking effect."
                        ),
                    }
                ],
                "discriminating_predictions": [
                    {
                        "metric_or_probe": "same-protocol development NDCG at ten",
                        "if_supported": "The candidate exceeds frozen DiffRec.",
                        "if_refuted": "Frozen DiffRec remains at least as strong.",
                    }
                ],
                "failure_interpretation": {
                    "mechanism_failure": "The learned recovery path is inert or harmful.",
                    "optimization_failure": "The denoising objective does not converge.",
                    "protocol_failure": "The random split or evaluator identity differs.",
                    "resource_failure": "The diffusion field cannot fit the frozen budget.",
                },
                "implementation_plan": [
                    "Clone the exact DiffRec source and make only parent-relative local changes."
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
    model_path: Path,
    candidate_id: str,
    program_digest: str,
) -> tuple[dict[str, Any], str]:
    model_bytes = model_path.read_bytes()
    try:
        model_source = model_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DiffusionFlowReadyError(
            "DiffRec parent model must be UTF-8 Python"
        ) from error
    declaration = "class DiffRec(GeneralRecommender, AutoEncoderMixin):"
    if model_source.count(declaration) != 1:
        raise DiffusionFlowReadyError(
            "DiffRec parent source lacks one canonical model declaration"
        )
    candidate_source = (
        model_source.rstrip()
        + "\n\n\nclass FreshCandidateModel(DiffRec):\n"
        + "    \"\"\"Exact frozen DiffRec parent exposed through the RecClaw ABI.\"\"\"\n"
        + "\n"
        + "    def calculate_loss(self, interaction):\n"
        + "        return super().calculate_loss(interaction)\n"
        + "\n"
        + "    def predict(self, interaction):\n"
        + "        return super().predict(interaction)\n"
        + "\n"
        + "    def full_sort_predict(self, interaction):\n"
        + "        return super().full_sort_predict(interaction)\n"
    )
    compile(candidate_source, "recclaw_ext/candidate.py", "exec")
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
            "capability_ref": "parent:diffusion-flow:diffrec:seed54201",
            "instruction": "CLONE_EXACT_PARENT_AND_LOCAL_PATCH",
            "source_tree_digest": source_tree_digest,
            "files": files,
        }
    )
    return bundle, _sha256_bytes(model_bytes)


def _mapping_contains(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _mapping_contains(actual[key], value)
            for key, value in expected.items()
        )
    return canonical_value(actual) == canonical_value(expected)


def _normalized_parent_result(
    *,
    source: Mapping[str, Any],
    source_path: Path,
    expected_dataset: str,
    expected_dataset_sha256: str,
    expected_model_sha256: str,
    expected_partition_files: Mapping[str, Any],
    expected_split_identity: Mapping[str, Any],
    expected_catalog_items: Mapping[str, Any],
    expected_protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    protocol = source.get("protocol")
    best_valid = source.get("best_valid_result")
    external_source = source.get("external_model_source")
    partition_access = source.get("partition_access", {})
    training = source.get("training")
    split_identity = source.get("split_identity")
    partition_files = source.get("partition_files")
    catalog_items = source.get("catalog_items")
    if not all(
        isinstance(value, Mapping)
        for value in (
            protocol,
            best_valid,
            external_source,
            training,
            split_identity,
            partition_files,
            catalog_items,
        )
    ):
        raise DiffusionFlowReadyError(
            "DiffRec parent result lacks protocol, metric, source, or data identity"
        )
    ndcg_key = next(
        (key for key in best_valid if str(key).casefold() == "ndcg@10"), None
    )
    if ndcg_key is None:
        raise DiffusionFlowReadyError(
            "DiffRec parent result lacks development NDCG@10"
        )
    value = best_valid[ndcg_key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        raise DiffusionFlowReadyError(
            "DiffRec parent development NDCG@10 is not a finite metric"
        )
    upstream_files = external_source.get("upstream_files")
    diffrec_configs = [
        row
        for row in (upstream_files if isinstance(upstream_files, list) else ())
        if isinstance(row, Mapping)
        and Path(str(row.get("path", ""))).name == "DiffRec.yaml"
        and row.get("sha256") == _DIFFREC_MODEL_CONFIG_SHA256
    ]
    if (
        source.get("config_file_sha256") != _PARENT_CONFIG_SHA256
        or not _mapping_contains(training, _EXPECTED_TRAINING)
        or not _mapping_contains(protocol, expected_protocol)
        or external_source.get("sha256") != expected_model_sha256
        or len(diffrec_configs) != 1
    ):
        raise DiffusionFlowReadyError(
            "DiffRec parent config, training, protocol, or source identity drift"
        )
    expected_access = {
        "partitions_opened": ["train", "development"],
        "heldout_opened": False,
    }
    if (
        source.get("model") != "DiffRec"
        or source.get("test_result") is not None
        or canonical_value(partition_access) != canonical_value(expected_access)
        or set(split_identity) != {"train", "validation"}
        or float(source.get("best_valid_score", float("nan"))) != float(value)
    ):
        raise DiffusionFlowReadyError(
            "DiffRec parent must be the exact seed-54201 validation-only run "
            "on the frozen dataset, with explicit heldout_opened=false and no "
            "test identity or result"
        )
    if (
        canonical_value(partition_files)
        != canonical_value(expected_partition_files)
        or canonical_value(catalog_items)
        != canonical_value(expected_catalog_items)
        or not _mapping_contains(
            split_identity.get("train"), expected_split_identity.get("train")
        )
        or not _mapping_contains(
            split_identity.get("validation"),
            expected_split_identity.get("development"),
        )
    ):
        raise DiffusionFlowReadyError(
            "DiffRec parent result is not bound to the supplied physical "
            "train, development, and catalog assets"
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
            "source_catalog_items": expected_catalog_items,
        }
    )
    return normalized, float(value)


def build_diffusion_flow_ready_launch(
    *,
    template_path: Path,
    parent_model_path: Path,
    parent_result_path: Path,
    train_path: Path,
    dev_path: Path,
    catalog_items_path: Path,
    output_root: Path,
) -> Path:
    """Build and validate a no-run DiffRec launch from explicit real inputs."""

    template_path = template_path.expanduser().resolve()
    parent_model_path = parent_model_path.expanduser().resolve()
    parent_result_path = parent_result_path.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not parent_model_path.is_file():
        raise DiffusionFlowReadyError(
            f"DiffRec parent model does not exist: {parent_model_path}"
        )
    payload = _read_json(template_path, field_name="diffusion-flow launch template")
    if payload.get("profile_key") != "diffusion_flow_cf":
        raise DiffusionFlowReadyError(
            "launch template is not the diffusion-flow profile"
        )
    config_overrides = payload["execution_contract"]["config_overrides"]
    if not _mapping_contains(config_overrides, _EXPECTED_EXECUTION_CONFIG):
        raise DiffusionFlowReadyError(
            "DiffRec launch template differs from the frozen parent config"
        )
    spec = DIFFUSION_FLOW_SINGLE_PARENT_SPEC
    provider = resolve_provider(spec.mechanism_space_id)
    required = set(
        focused_mechanism_language(spec)["parent_contract"][
            "required_parent_foundation_primitives"
        ]
    )
    if required != _PARENT_PRIMITIVES:
        raise DiffusionFlowReadyError(
            "focused mechanism language differs from the executable DiffRec parent"
        )

    frozen = payload["frozen_profile"]["frozen_fields"]
    expected_frozen_split = canonical_value(
        {
            "strategy": "BENCHMARK",
            "benchmark_filename": ["train", "dev", "dev"],
            "split": config_overrides["eval_args"]["split"],
            "group_by": config_overrides["eval_args"]["group_by"],
            "order": config_overrides["eval_args"]["order"],
        }
    )
    if canonical_value(frozen.get("split")) != expected_frozen_split:
        raise DiffusionFlowReadyError(
            "DiffRec frozen split differs from the strict benchmark protocol"
        )
    dataset = str(frozen["dataset"])
    dataset_sha256 = str(frozen["dataset_snapshot"]["interaction_sha256"])
    expected_items = int(frozen["dataset_snapshot"]["items"])
    dataset_root = output_root / "search_data" / dataset
    train_destination = dataset_root / f"{dataset}.train.inter"
    dev_destination = dataset_root / f"{dataset}.dev.inter"
    catalog_destination = dataset_root / f"{dataset}.item"
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
    catalog_identity = _copy_catalog_items(
        catalog_items_path,
        catalog_destination,
        expected_items=expected_items,
    )
    frozen_catalog = frozen.get("item_id_mapping_identity")
    if (
        not isinstance(frozen_catalog, dict)
        or frozen_catalog.get("items") != expected_items
    ):
        raise DiffusionFlowReadyError("frozen DiffRec catalog identity is incomplete")
    frozen_catalog["catalog_items_sha256"] = catalog_identity["sha256"]
    config_overrides["recclaw_catalog_mapping"] = str(catalog_destination.resolve())
    physical_partitions = canonical_value(
        {"train": train_identity, "development": dev_identity}
    )
    manifest = canonical_value(
        {
            "schema": "recclaw.search-only-dataset.v1",
            "dataset": dataset,
            "parent_interaction_sha256": dataset_sha256,
            "parent_split": frozen["split"],
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
            "item_feature": {
                "path": f"{dataset}/{dataset}.item",
                "sha256": catalog_identity["sha256"],
                "items": expected_items,
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
    verified_assets = verify_frozen_family_assets(
        frozen_fields=frozen,
        config_overrides=config_overrides,
        search_data=payload["execution_contract"]["search_data"],
    )

    profile_ref, _ = build_frozen_family_profile(
        provider,
        profile_id=payload["frozen_profile"]["profile_id"],
        frozen_fields=frozen,
    )
    program = _diffrec_parent_program(provider=provider, profile_ref=profile_ref)
    report = provider.compile(deep_thaw(program))
    if report.status is not CompileStatus.VALID_NEEDS_IMPLEMENTATION:
        raise DiffusionFlowReadyError(
            "executable DiffRec parent program does not compile: "
            + json.dumps(report.to_dict(), sort_keys=True)
        )
    binding = {
        "candidate_id": report.candidate_id,
        "program_digest": report.mechanism_program_digest,
    }
    source_bundle, model_sha256 = _parent_source_bundle(
        model_path=parent_model_path,
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
        field_name="DiffRec parent result",
    )
    normalized_result, metric_value = _normalized_parent_result(
        source=source_result,
        source_path=parent_result_path,
        expected_dataset=dataset,
        expected_dataset_sha256=dataset_sha256,
        expected_model_sha256=model_sha256,
        expected_partition_files=physical_partitions,
        expected_split_identity=frozen["dataset_snapshot"][
            "search_partition_semantic_identity"
        ],
        expected_catalog_items=catalog_identity,
        expected_protocol={
            "dataset": dataset,
            "raw_dataset_sha256": dataset_sha256,
            "seed": ACTUAL_TRAINING_SEED,
            "selection_partition": "validation",
            "benchmark_filename": verified_assets["search_data"][
                "benchmark_filename"
            ],
            "split": config_overrides["eval_args"]["split"],
            "group_by": config_overrides["eval_args"]["group_by"],
            "order": config_overrides["eval_args"]["order"],
            "mode": config_overrides["eval_args"]["mode"],
            "exclude_seen": True,
            "topk": config_overrides["topk"],
            "test_read_after_selection": False,
        },
    )
    worker_result_path = output_root / "observations" / "parent-worker-result.json"
    worker_result_sha256 = _write_json(worker_result_path, normalized_result)
    payload["baseline_context"]["parent_anchor"]["paired_metric"][
        "value"
    ] = metric_value

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
            "profile_key": "diffusion_flow_cf",
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
    launch_path = output_root / "diffusion_flow_diffrec_single_parent.ready.json"
    _write_json(launch_path, payload)
    return launch_path


__all__ = [
    "DiffusionFlowReadyError",
    "build_diffusion_flow_ready_launch",
]
