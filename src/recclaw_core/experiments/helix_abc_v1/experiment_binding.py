"""Content-addressed execution bindings for the common campaign worker."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from .canonical import canonical_value, sha256_digest


EXPERIMENT_BINDING_SCHEMA = "recclaw.research-line.experiment-binding.v1"
COMMON_DATASET = "ml-1m"
COMMON_SPLIT = "train/dev/heldout"
COMMON_EVALUATOR = canonical_value(
    {
        "candidate_universe": "FULL_SORT",
        "heldout_access": "AFTER_BEST_CHECKPOINT_SELECTION",
        "metric": "NDCG@10",
        "nonfinite_policy": "REJECT",
        "partition_role": "ROUND_TEST_FEEDBACK",
        "source": "BEST_CHECKPOINT_TEST_RESULT",
    }
)
_EXECUTION_ROLES = frozenset({"CANDIDATE", "COMPARATOR"})
_DIGEST_FIELDS = (
    "capability_digest",
    "profile_digest",
    "entrypoint_source_sha256",
)


class ExperimentBindingError(ValueError):
    """The selected execution recipe is incomplete or inconsistent."""


def _require_string(recipe: Mapping[str, Any], field_name: str) -> str:
    value = recipe.get(field_name)
    if not isinstance(value, str) or not value or value != value.strip():
        raise ExperimentBindingError(
            f"execution_recipe requires non-empty {field_name}"
        )
    return value


def _require_digest(recipe: Mapping[str, Any], field_name: str) -> str:
    value = _require_string(recipe, field_name)
    if len(value) != 64:
        raise ExperimentBindingError(
            f"execution_recipe {field_name} must be a 64-character digest"
        )
    return value


def validate_execution_recipe(recipe: Mapping[str, Any]) -> None:
    """Validate recipe fields that are independent of a physical run."""

    if not isinstance(recipe, Mapping):
        raise ExperimentBindingError(
            "explicit execution_recipe is required; generic BPR execution is disabled"
        )
    for field_name in (
        "capability_family",
        "capability_ref",
        "profile_ref",
        "entrypoint",
        "model",
        "base_model_config",
        "dataset",
        "split",
        "execution_role",
        "mechanism_id",
    ):
        _require_string(recipe, field_name)
    for field_name in _DIGEST_FIELDS + ("capability_digest", "profile_digest"):
        _require_digest(recipe, field_name)
    if recipe["dataset"] != COMMON_DATASET:
        raise ExperimentBindingError(
            f"execution_recipe dataset must remain {COMMON_DATASET!r}"
        )
    if recipe["split"] != COMMON_SPLIT:
        raise ExperimentBindingError(
            f"execution_recipe split must remain {COMMON_SPLIT!r}"
        )
    if recipe["execution_role"] not in _EXECUTION_ROLES:
        raise ExperimentBindingError(
            "execution_recipe execution_role must be CANDIDATE or COMPARATOR"
        )
    if not isinstance(recipe.get("config"), Mapping):
        raise ExperimentBindingError("execution_recipe config must be an object")
    evaluator = recipe.get("evaluator")
    if not isinstance(evaluator, Mapping):
        raise ExperimentBindingError(
            "execution_recipe requires the common evaluator contract"
        )
    if canonical_value(dict(evaluator)) != COMMON_EVALUATOR:
        raise ExperimentBindingError(
            "execution_recipe evaluator must be the common FULL_SORT NDCG@10 contract"
        )
    evaluator_digest = recipe.get("evaluator_digest")
    if evaluator_digest is not None:
        if not isinstance(evaluator_digest, str) or evaluator_digest != sha256_digest(
            evaluator
        ):
            raise ExperimentBindingError("execution_recipe evaluator digest drift")
    if recipe["execution_role"] == "COMPARATOR":
        comparator_ref = _require_string(recipe, "comparator_ref")
        comparator_digest = _require_digest(recipe, "comparator_digest")
        if not comparator_ref or not comparator_digest:
            raise ExperimentBindingError("comparator identity is incomplete")
    for ref_field, digest_field in (
        ("candidate_package_ref", "candidate_package_digest"),
        ("candidate_root_ref", "candidate_root_digest"),
    ):
        ref_value = recipe.get(ref_field)
        digest_value = recipe.get(digest_field)
        if (ref_value is None) != (digest_value is None):
            raise ExperimentBindingError(
                f"{ref_field} and {digest_field} must be supplied together"
            )
        if ref_value is not None:
            if not isinstance(ref_value, str) or not ref_value:
                raise ExperimentBindingError(f"{ref_field} is invalid")
            if not isinstance(digest_value, str) or len(digest_value) != 64:
                raise ExperimentBindingError(f"{digest_field} is invalid")
    source_tree_digest = recipe.get("candidate_source_tree_digest")
    if source_tree_digest is not None and (
        not isinstance(source_tree_digest, str) or len(source_tree_digest) != 64
    ):
        raise ExperimentBindingError("candidate_source_tree_digest is invalid")


@dataclass(frozen=True, slots=True)
class ExperimentBindingV1:
    """One immutable execution identity consumed by the package worker."""

    capability_family: str
    capability_ref: str
    capability_digest: str
    profile_ref: str
    profile_digest: str
    mechanism_id: str
    candidate_package_ref: str | None
    candidate_package_digest: str | None
    candidate_root_ref: str | None
    candidate_root_digest: str | None
    candidate_root_path: str | None
    candidate_source_tree_digest: str | None
    entrypoint: str
    entrypoint_source_sha256: str
    model: str
    base_model_config: str
    config: Mapping[str, Any]
    dataset: str
    dataset_manifest_digest: str
    split: str
    evaluator: Mapping[str, Any]
    seed: int
    epochs: int
    timeout_seconds: int
    execution_purpose: str
    execution_role: str
    resource_telemetry: bool
    watchdog_seconds: int | None
    prefix_contract_digest: str | None
    run_id: str
    round_id: str
    claim_id: str
    permit_digest: str
    runtime_binding_digest: str
    runtime_release_digest: str
    runner_abi: str
    filesystem_capability_digest: str
    execution_recipe_digest: str
    comparator_ref: str | None
    comparator_digest: str | None

    schema = EXPERIMENT_BINDING_SCHEMA

    @classmethod
    def from_execution_recipe(
        cls,
        recipe: Mapping[str, Any],
        *,
        candidate_root: Path | None,
        dataset_manifest_digest: str,
        seed: int,
        epochs: int,
        timeout_seconds: int,
        execution_purpose: str,
        resource_telemetry: bool,
        watchdog_seconds: int | None,
        prefix_contract_digest: str | None,
        run_id: str,
        round_id: str,
        claim_id: str,
        permit_digest: str,
        runtime_binding_digest: str,
        runtime_release_digest: str,
        runner_abi: str,
        filesystem_capability_digest: str,
    ) -> "ExperimentBindingV1":
        validate_execution_recipe(recipe)
        if not isinstance(dataset_manifest_digest, str) or len(dataset_manifest_digest) != 64:
            raise ExperimentBindingError("dataset_manifest_digest is invalid")
        if candidate_root is not None and recipe["execution_role"] != "CANDIDATE":
            raise ExperimentBindingError(
                "candidate_root is only valid for a CANDIDATE binding"
            )
        if recipe.get("execution_purpose") not in (None, execution_purpose):
            raise ExperimentBindingError("execution purpose drift")
        if recipe.get("run_id") not in (None, run_id):
            raise ExperimentBindingError("run identity drift")
        if recipe.get("seed") not in (None, seed):
            raise ExperimentBindingError("execution seed drift")
        evaluator = canonical_value(dict(recipe["evaluator"]))
        candidate_root_path = (
            str(candidate_root.resolve()) if candidate_root is not None else None
        )
        return cls(
            capability_family=str(recipe["capability_family"]),
            capability_ref=str(recipe["capability_ref"]),
            capability_digest=str(recipe["capability_digest"]),
            profile_ref=str(recipe["profile_ref"]),
            profile_digest=str(recipe["profile_digest"]),
            mechanism_id=str(recipe["mechanism_id"]),
            candidate_package_ref=recipe.get("candidate_package_ref"),
            candidate_package_digest=recipe.get("candidate_package_digest"),
            candidate_root_ref=recipe.get("candidate_root_ref"),
            candidate_root_digest=recipe.get("candidate_root_digest"),
            candidate_root_path=candidate_root_path,
            candidate_source_tree_digest=recipe.get("candidate_source_tree_digest"),
            entrypoint=str(recipe["entrypoint"]),
            entrypoint_source_sha256=str(recipe["entrypoint_source_sha256"]),
            model=str(recipe["model"]),
            base_model_config=str(recipe["base_model_config"]),
            config=canonical_value(dict(recipe["config"])),
            dataset=str(recipe["dataset"]),
            dataset_manifest_digest=dataset_manifest_digest,
            split=str(recipe["split"]),
            evaluator=evaluator,
            seed=int(seed),
            epochs=int(epochs),
            timeout_seconds=int(timeout_seconds),
            execution_purpose=str(execution_purpose),
            execution_role=str(recipe["execution_role"]),
            resource_telemetry=bool(resource_telemetry),
            watchdog_seconds=(
                int(watchdog_seconds) if watchdog_seconds is not None else None
            ),
            prefix_contract_digest=prefix_contract_digest,
            run_id=str(run_id),
            round_id=str(round_id),
            claim_id=str(claim_id),
            permit_digest=str(permit_digest),
            runtime_binding_digest=str(runtime_binding_digest),
            runtime_release_digest=str(runtime_release_digest),
            runner_abi=str(runner_abi),
            filesystem_capability_digest=str(filesystem_capability_digest),
            execution_recipe_digest=sha256_digest(recipe),
            comparator_ref=recipe.get("comparator_ref"),
            comparator_digest=recipe.get("comparator_digest"),
        )

    @classmethod
    def from_canonical_dict(
        cls, payload: Mapping[str, Any]
    ) -> "ExperimentBindingV1":
        """Rehydrate the complete binding emitted by a physical runner."""

        if not isinstance(payload, Mapping):
            raise ExperimentBindingError("runner did not return a complete ExperimentBindingV1")
        normalized = canonical_value(dict(payload))
        if normalized.get("schema") != cls.schema:
            raise ExperimentBindingError("runner Experiment Binding schema drift")
        field_names = {field.name for field in fields(cls)}
        if set(normalized) != field_names | {"schema"}:
            raise ExperimentBindingError("runner Experiment Binding fields are incomplete")
        binding = cls(**{name: normalized[name] for name in field_names})
        if binding.canonical_dict() != normalized:
            raise ExperimentBindingError("runner Experiment Binding is not canonical")
        return binding

    @property
    def digest(self) -> str:
        return sha256_digest(self.canonical_dict())

    @property
    def ref(self) -> str:
        return f"{self.schema}:{self.digest}"

    def canonical_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "schema": self.schema,
                "capability_family": self.capability_family,
                "capability_ref": self.capability_ref,
                "capability_digest": self.capability_digest,
                "profile_ref": self.profile_ref,
                "profile_digest": self.profile_digest,
                "mechanism_id": self.mechanism_id,
                "candidate_package_ref": self.candidate_package_ref,
                "candidate_package_digest": self.candidate_package_digest,
                "candidate_root_ref": self.candidate_root_ref,
                "candidate_root_digest": self.candidate_root_digest,
                "candidate_root_path": self.candidate_root_path,
                "candidate_source_tree_digest": self.candidate_source_tree_digest,
                "entrypoint": self.entrypoint,
                "entrypoint_source_sha256": self.entrypoint_source_sha256,
                "model": self.model,
                "base_model_config": self.base_model_config,
                "config": dict(self.config),
                "dataset": self.dataset,
                "dataset_manifest_digest": self.dataset_manifest_digest,
                "split": self.split,
                "evaluator": dict(self.evaluator),
                "seed": self.seed,
                "epochs": self.epochs,
                "timeout_seconds": self.timeout_seconds,
                "execution_purpose": self.execution_purpose,
                "execution_role": self.execution_role,
                "resource_telemetry": self.resource_telemetry,
                "watchdog_seconds": self.watchdog_seconds,
                "prefix_contract_digest": self.prefix_contract_digest,
                "run_id": self.run_id,
                "round_id": self.round_id,
                "claim_id": self.claim_id,
                "permit_digest": self.permit_digest,
                "runtime_binding_digest": self.runtime_binding_digest,
                "runtime_release_digest": self.runtime_release_digest,
                "runner_abi": self.runner_abi,
                "filesystem_capability_digest": self.filesystem_capability_digest,
                "execution_recipe_digest": self.execution_recipe_digest,
                "comparator_ref": self.comparator_ref,
                "comparator_digest": self.comparator_digest,
            }
        )

    def worker_recipe(self) -> dict[str, Any]:
        """Project only the recipe fields consumed by campaign_train_worker."""

        return canonical_value(
            {
                "base_model_config": self.base_model_config,
                "config": dict(self.config),
                "entrypoint": self.entrypoint,
                "entrypoint_source_sha256": self.entrypoint_source_sha256,
                "execution_binding_digest": self.digest,
                "execution_binding_ref": self.ref,
                "execution_role": self.execution_role,
                "mechanism_id": self.mechanism_id,
                "model": self.model,
            }
        )


def render_campaign_worker_command(
    binding: ExperimentBindingV1,
    *,
    python_executable: Path,
    worker_path: Path,
    checkpoint_dir: Path,
    data_path: Path,
    execution_recipe_path: Path,
    filesystem_capability_path: Path,
    log_path: Path,
    output_path: Path,
    project_root: Path,
    recbole_root: Path,
    start_confirmation_path: Path,
    start_gate_path: Path,
    resource_telemetry_path: Path | None = None,
    prefix_contract_path: Path | None = None,
) -> list[str]:
    """Purely render the worker argv from one binding and run paths."""

    if binding.resource_telemetry and resource_telemetry_path is None:
        raise ExperimentBindingError(
            "resource telemetry binding requires a telemetry output path"
        )
    if binding.prefix_contract_digest is not None and prefix_contract_path is None:
        raise ExperimentBindingError(
            "prefix contract binding requires a prefix contract path"
        )
    command = [
        str(python_executable),
        str(worker_path),
        "--binding-digest",
        binding.digest,
        "--claim-id",
        binding.claim_id,
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--data-path",
        str(data_path),
        "--dataset",
        binding.dataset,
        "--epochs",
        str(binding.epochs),
        "--execution-purpose",
        binding.execution_purpose,
        "--execution-recipe-path",
        str(execution_recipe_path),
        "--filesystem-capability-path",
        str(filesystem_capability_path),
        "--filesystem-mode",
        "HASH_AUDITED_PRIVATE_ROOT_V1",
        "--log-path",
        str(log_path),
        "--model",
        binding.model,
        "--output-path",
        str(output_path),
        "--permit-digest",
        binding.permit_digest,
        "--project-root",
        str(project_root),
        "--recbole-root",
        str(recbole_root),
        "--round-id",
        binding.round_id,
        "--run-id",
        binding.run_id,
        "--runner-abi",
        binding.runner_abi,
        "--runtime-binding-digest",
        binding.runtime_binding_digest,
        "--runtime-release-digest",
        binding.runtime_release_digest,
        "--seed",
        str(binding.seed),
        "--start-confirmation-path",
        str(start_confirmation_path),
        "--start-gate-path",
        str(start_gate_path),
    ]
    if binding.resource_telemetry:
        command.extend(
            [
                "--resource-telemetry",
                "--resource-telemetry-path",
                str(resource_telemetry_path),
            ]
        )
    if binding.prefix_contract_digest is not None:
        command.extend(
            ["--prefix-contract-path", str(prefix_contract_path.resolve())]
        )
    return command


__all__ = [
    "COMMON_DATASET",
    "COMMON_EVALUATOR",
    "COMMON_SPLIT",
    "EXPERIMENT_BINDING_SCHEMA",
    "ExperimentBindingError",
    "ExperimentBindingV1",
    "render_campaign_worker_command",
    "validate_execution_recipe",
]
