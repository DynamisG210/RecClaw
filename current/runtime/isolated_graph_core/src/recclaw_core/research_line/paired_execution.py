"""Physical Provider and common-evaluator bindings for the Original arm."""

from __future__ import annotations

import importlib.util
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jsonschema
import yaml

from recclaw_core.experiments.helix_abc_v1 import fresh_r1
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    COMMON_DATASET,
    COMMON_EVALUATOR,
    COMMON_SPLIT,
    validate_execution_recipe,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (
    load_lab_api_credential_pairs,
    validate_provider_strict_schema,
)

from .original_arm import PinnedOriginalArmV1
from .paired import ORIGINAL_RECCLAW_ARM, ArmRequest


class PairedExecutionError(ValueError):
    """A physical paired-arm binding is incomplete or inconsistent."""


def _write_stable_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise PairedExecutionError(f"persisted identity drift: {path}")
        return
    path.write_text(payload, encoding="utf-8")


class OriginalProviderResponder:
    """Use the existing bounded gpt-5.4 transport for pinned Original schemas."""

    def __init__(
        self,
        *,
        api_config_path: Path,
        call_root: Path,
        campaign_id: str,
        total_token_ceiling: int = 200_000,
    ) -> None:
        self.api_config_path = Path(api_config_path).resolve()
        self.call_root = Path(call_root).resolve()
        self.campaign_id = str(campaign_id)
        self.total_token_ceiling = int(total_token_ceiling)
        if self.total_token_ceiling < fresh_r1.PROPOSAL_TOKEN_CEILING:
            raise PairedExecutionError("Original token ceiling is below its output ceiling")
        self.round_index: int | None = None
        self.call_index = 0

    def bind_request(self, request: ArmRequest) -> None:
        if request.arm != ORIGINAL_RECCLAW_ARM:
            raise PairedExecutionError("Original Provider received a non-Original request")
        self.round_index = int(request.round_index)
        self.call_index = 0

    def __call__(
        self,
        messages: list[dict[str, str]],
        *,
        schema_name: str,
        response_schema: dict[str, Any],
    ) -> Mapping[str, Any]:
        if self.round_index is None:
            raise PairedExecutionError("Original Provider request is not bound to a round")
        if not isinstance(messages, list) or not messages:
            raise PairedExecutionError("Original Provider messages are empty")
        jsonschema.validators.validator_for(response_schema).check_schema(response_schema)
        wrapper = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "proposals": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 1,
                    "items": response_schema,
                }
            },
            "required": ["proposals"],
        }
        validate_provider_strict_schema(wrapper)
        schema_digest = sha256_digest(wrapper)
        schema_path = self.call_root / "schemas" / f"{schema_digest}.json"
        _write_stable_json(schema_path, wrapper)
        self.call_index += 1
        logical_call_id = (
            f"{self.campaign_id}:original:r{self.round_index:04d}:"
            f"c{self.call_index:02d}:{schema_name}"
        )
        prompt = (
            "Return exactly one JSON object inside the outer proposals array. "
            "That inner object must satisfy the supplied Original RecClaw response schema.\n\n"
            + json.dumps(messages, ensure_ascii=True, sort_keys=True)
        )
        result = fresh_r1.bounded_provider_call(
            call_root=self.call_root / "rounds" / f"round-{self.round_index:04d}" / f"call-{self.call_index:02d}",
            schema_path=schema_path,
            logical_call_id=logical_call_id,
            session_id=f"{self.campaign_id}:original",
            prompt=prompt,
            token_ceiling=self.total_token_ceiling,
            output_token_ceiling=fresh_r1.PROPOSAL_TOKEN_CEILING,
            credential_config_path=self.api_config_path,
        )
        if result.call is None:
            raise fresh_r1.FreshR1Error("Original Provider call failed")
        proposals = result.call.response.get("proposals")
        if not isinstance(proposals, list) or len(proposals) != 1 or not isinstance(proposals[0], Mapping):
            raise fresh_r1.FreshR1Error("Original Provider wrapper response is invalid")
        return dict(proposals[0])


def original_llm_runtime(api_config_path: Path) -> dict[str, str]:
    """Bind the pinned implementation subprocess to the primary configured API."""

    pairs = load_lab_api_credential_pairs(Path(api_config_path).resolve())
    base_url, api_key = pairs[0]
    key_name = "RECCLAW_ORIGINAL_GPT54_API_KEY"
    os.environ[key_name] = api_key
    return {
        "provider": "openai",
        "model": fresh_r1.MODEL,
        "base_url": base_url,
        "api_key_env": key_name,
    }


def _entrypoint_source(root: Path, entrypoint: str) -> tuple[Path, str]:
    module, separator, _name = entrypoint.partition(":")
    if not separator or not module.startswith("recclaw_ext."):
        raise PairedExecutionError("generated Original candidate lacks a local entrypoint")
    source = root / (module.replace(".", "/") + ".py")
    if not source.is_file():
        raise PairedExecutionError(f"generated Original source is unavailable: {source}")
    return source, fresh_r1.bytes_sha256(source.read_bytes())


def _native_entrypoint_source(
    arm: PinnedOriginalArmV1, entrypoint: str
) -> tuple[Path, str, Path | None]:
    module, separator, _name = entrypoint.partition(":")
    if not separator:
        raise PairedExecutionError("native Original candidate has an invalid entrypoint")
    if module.startswith("recclaw_ext."):
        source, digest = _entrypoint_source(arm.source_root, entrypoint)
        return source, digest, arm.source_root
    if not module.startswith("recbole.model.general_recommender."):
        raise PairedExecutionError(
            f"native Original entrypoint is outside its frozen method space: {entrypoint}"
        )
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        raise PairedExecutionError(f"native Original entrypoint is unavailable: {entrypoint}")
    source = Path(spec.origin).resolve()
    if not source.is_file():
        raise PairedExecutionError(f"native Original source is unavailable: {source}")
    return source, fresh_r1.bytes_sha256(source.read_bytes()), None


def _original_recipe(
    arm: PinnedOriginalArmV1,
    candidate: Mapping[str, Any],
    params: Mapping[str, Any],
) -> tuple[dict[str, Any], Path | None]:
    stored = candidate.get("execution_recipe")
    if isinstance(stored, Mapping):
        recipe = dict(stored)
        initial = arm.initial_executable_space if isinstance(arm.initial_executable_space, Mapping) else {}
        recipe.setdefault("capability_family", str(candidate.get("mechanism_axis") or "bl_icf_fixed"))
        recipe.setdefault("capability_ref", f"fixed-66:{candidate['candidate_id']}")
        recipe.setdefault("capability_digest", str(candidate.get("mechanism_semantics_digest") or sha256_digest(candidate)))
        recipe.setdefault("profile_ref", str(initial.get("profile_ref") or "BL_ICF_EXECUTABLE_PROFILE_V2"))
        recipe.setdefault("profile_digest", str(initial.get("profile_digest") or sha256_digest(initial)))
        recipe.setdefault("dataset", COMMON_DATASET)
        recipe.setdefault("split", COMMON_SPLIT)
        recipe.setdefault("evaluator", COMMON_EVALUATOR)
        recipe.setdefault("evaluator_digest", sha256_digest(COMMON_EVALUATOR))
        recipe.setdefault("execution_role", "CANDIDATE")
        config = dict(recipe.get("config") or {})
        config.update({str(key): value for key, value in params.items() if value is not None})
        recipe["config"] = config
        validate_execution_recipe(recipe)
        return recipe, None

    entrypoint = str(candidate.get("entrypoint") or "")
    source, source_digest, candidate_root = _native_entrypoint_source(arm, entrypoint)
    candidate_id = str(candidate.get("candidate_id") or source.stem)
    model_name = entrypoint.rsplit(":", 1)[-1]
    config = dict(candidate.get("parameter_overrides") or candidate.get("config") or {})
    config.update({str(key): value for key, value in params.items() if value is not None})
    registry_path = Path(arm.mutable_paths["registry"])
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, Mapping):
        raise PairedExecutionError(f"Original registry is not a mapping: {registry_path}")
    profile_digest = sha256_digest(registry)
    capability_identity = {
        "arm_id": arm.arm_id,
        "candidate_id": candidate_id,
        "entrypoint": entrypoint,
        "source_sha256": source_digest,
        "config": config,
    }
    recipe = {
        "capability_family": str(candidate.get("mechanism") or candidate.get("category") or "original_generated_model"),
        "capability_ref": f"original:{candidate_id}",
        "capability_digest": sha256_digest(capability_identity),
        "profile_ref": f"{arm.arm_id}:registry",
        "profile_digest": profile_digest,
        "entrypoint": entrypoint,
        "entrypoint_source_sha256": source_digest,
        "model": model_name,
        "base_model_config": str(candidate.get("base_model") or "LightGCN"),
        "config": config,
        "dataset": COMMON_DATASET,
        "split": COMMON_SPLIT,
        "evaluator": COMMON_EVALUATOR,
        "evaluator_digest": sha256_digest(COMMON_EVALUATOR),
        "execution_role": "CANDIDATE",
        "mechanism_id": str(candidate.get("mechanism") or candidate_id),
    }
    validate_execution_recipe(recipe)
    return recipe, candidate_root


def make_original_runner_factory(
    *,
    arm: PinnedOriginalArmV1,
    repo_root: Path,
    root: Path,
    epochs: int,
    timeout_seconds: int,
    recbole_commit_identity: str,
    expected_recbole_source_tree_digest: str,
    watchdog_seconds: int | None,
    launch: Any = None,
) -> Any:
    """Adapt Original's selected candidate to the same family-neutral worker."""

    launcher = fresh_r1.run_development_training if launch is None else launch

    def factory(request: ArmRequest) -> Any:
        def run(candidate: Mapping[str, Any], params: Mapping[str, Any], round_index: int, _run_root: Path) -> Mapping[str, Any]:
            if round_index != request.round_index:
                raise PairedExecutionError("Original runner round identity mismatch")
            recipe, candidate_root = _original_recipe(arm, candidate, params)
            run_id = f"{request.campaign_id}-original-round-{round_index:04d}"
            physical = launcher(
                repo_root=Path(repo_root).resolve(),
                side_root=Path(root).resolve() / "original_recclaw" / "execution" / f"round-{round_index:04d}",
                run_id=run_id,
                seed=request.seed,
                candidate_root=candidate_root,
                entrypoint=str(recipe["entrypoint"]),
                source_sha256=str(recipe["entrypoint_source_sha256"]),
                timeout_seconds=timeout_seconds,
                epochs=epochs,
                execution_purpose="ORIGINAL_RECCLAW_PAIRED_OFFLINE_TOPN",
                run_identity="recclaw-original-arm-v1",
                authority="user-delegated-paired-physical-execution",
                recbole_commit_identity=recbole_commit_identity,
                expected_recbole_source_tree_digest=expected_recbole_source_tree_digest,
                resource_telemetry=True,
                watchdog_seconds=watchdog_seconds,
                execution_recipe=recipe,
                cuda_visible_devices=request.device,
            )
            metrics = dict(physical.get("metrics") or {})
            exit_status = str(physical.get("exit_status") or "RUNTIME_FAILURE")
            return {
                "run_id": run_id,
                "model": str(recipe["model"]),
                "dataset": COMMON_DATASET,
                "status": "success" if exit_status == "SUCCESS" else "censored" if exit_status == "RESOURCE_CENSORED" else "crash",
                "exit_code": 0 if exit_status == "SUCCESS" else 1,
                "ndcg@10": metrics.get("ndcg@10"),
                "metrics": metrics,
                "wall_time_ms": physical.get("wall_time_ms"),
                "experiment_binding": physical.get("experiment_binding"),
                "physical_result": physical,
            }

        return run

    return factory


__all__ = [
    "OriginalProviderResponder",
    "PairedExecutionError",
    "make_original_runner_factory",
    "original_llm_runtime",
]
