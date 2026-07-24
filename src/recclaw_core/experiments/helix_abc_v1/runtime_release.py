"""M1 runtime release, executable profile, and common identity projection."""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

from recclaw_core.mechanism_space import catalog_digest, prompt_projection, space_identity

from .canonical import bytes_sha256, canonical_json_bytes, sha256_digest
from .runtime_contracts import DevelopmentRecSysProtocolV1


RESOURCE_PACKAGE = "recclaw_core.experiments.helix_abc_v1.resources"
SPACE_ID = "BL_ICF_MECHANISM_SPACE_V1"
SOURCE_FILES = (
    "runtime_contracts.py",
    "runtime_release.py",
    "runtime_handlers.py",
    "materialization.py",
    "common_execution_guard.py",
    "fake_runner.py",
    "state_store.py",
)


def _resource_bytes(name: str) -> bytes:
    return resources.files(RESOURCE_PACKAGE).joinpath(name).read_bytes()


def _resource_json(name: str) -> dict[str, Any]:
    value = json.loads(_resource_bytes(name))
    if not isinstance(value, dict):
        raise ValueError(f"package resource {name} must contain an object")
    return value


def executable_profile() -> dict[str, Any]:
    return _resource_json("bl_icf_executable_profile_v1.json")


def executable_profile_digest() -> str:
    return bytes_sha256(_resource_bytes("bl_icf_executable_profile_v1.json"))


def common_guard_policy() -> dict[str, Any]:
    return _resource_json("common_execution_guard_policy_v1.json")


def common_guard_policy_digest() -> str:
    return bytes_sha256(_resource_bytes("common_execution_guard_policy_v1.json"))


def development_protocol() -> DevelopmentRecSysProtocolV1:
    return DevelopmentRecSysProtocolV1(_resource_json("development_protocol_v1.json"))


def runtime_release_contract() -> dict[str, Any]:
    return _resource_json("runtime_release_contract_v1.json")


def runtime_release_digest() -> str:
    return bytes_sha256(_resource_bytes("runtime_release_contract_v1.json"))


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def source_manifest() -> tuple[dict[str, Any], ...]:
    package_dir = _package_dir()
    rows: list[dict[str, Any]] = []
    for name in SOURCE_FILES:
        payload = package_dir.joinpath(name).read_bytes()
        rows.append({"path": name, "sha256": bytes_sha256(payload), "size_bytes": len(payload)})
    return tuple(rows)


def source_manifest_digest() -> str:
    return sha256_digest(source_manifest())


def validate_frozen_environment() -> tuple[str, ...]:
    """Check the cheap, identity-bearing release inputs used by the fake runner."""

    release = runtime_release_contract()
    project_root = _package_dir().parents[3]
    failures: list[str] = []
    lock_path = project_root / str(release["environment_lock"])
    if not lock_path.is_file():
        failures.append("ENVIRONMENT_LOCK_MISSING")
    elif bytes_sha256(lock_path.read_bytes()) != release["environment_lock_sha256"]:
        failures.append("ENVIRONMENT_LOCK_DIGEST_MISMATCH")
    authorization = release["task_authorization"]
    authorization_path = project_root / str(authorization["path"])
    if not authorization_path.is_file():
        failures.append("TASK_AUTHORIZATION_INPUT_MISSING")
    elif bytes_sha256(authorization_path.read_bytes()) != authorization["sha256"]:
        failures.append("TASK_AUTHORIZATION_INPUT_DIGEST_MISMATCH")

    recbole_root = project_root.parent / "RecBole"
    if not recbole_root.joinpath(".git").exists():
        failures.append("RECBOLE_IDENTITY_UNAVAILABLE")
    else:
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(recbole_root), "rev-parse", "HEAD"],
                text=True,
                timeout=5,
            ).strip()
            tree = subprocess.check_output(
                ["git", "-C", str(recbole_root), "rev-parse", "HEAD^{tree}"],
                text=True,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            failures.append("RECBOLE_IDENTITY_UNAVAILABLE")
        else:
            if commit != release["recbole_commit"] or tree != release["recbole_tree"]:
                failures.append("RECBOLE_IDENTITY_MISMATCH")
    return tuple(sorted(failures))


def coverage_manifest() -> dict[str, Any]:
    profile = executable_profile()
    return {
        "campaign_capability_policy_digest": executable_profile_digest(),
        "materializer_and_runner_digest": source_manifest_digest(),
        "profile_id": profile["profile_id"],
        "prompt_projection_digest": campaign_projection()["effective_projection_digest"],
        "search_space_digest": space_identity(SPACE_ID).search_space_digest,
        "supported_operators": profile["supported_operators"],
        "supported_parameter_domains": profile["supported_parameter_domains"],
        "supported_primitives": profile["supported_primitives"],
        "unsupported_policy": profile["unsupported_policy"],
    }


def campaign_projection() -> dict[str, Any]:
    base = prompt_projection(SPACE_ID)
    profile = executable_profile()
    body = {
        "architecture_templates": profile["architecture_templates"],
        "profile_id": profile["profile_id"],
        "profile_version": profile["profile_version"],
        "supported_operators": profile["supported_operators"],
        "supported_primitives": profile["supported_primitives"],
    }
    return {
        "base_projection_digest": sha256_digest(base),
        "coverage_digest": executable_profile_digest(),
        "effective_projection": body,
        "effective_projection_digest": sha256_digest(body),
    }


def common_release_projection() -> dict[str, Any]:
    policy = common_guard_policy()
    release = runtime_release_contract()
    identity = space_identity(SPACE_ID)
    return {
        "catalog_digest": catalog_digest(),
        "common_guard_policy_digest": common_guard_policy_digest(),
        "executable_profile_digest": executable_profile_digest(),
        "phase_schedule": policy["phase_schedule"],
        "protocol_digest": development_protocol().digest,
        "reason_registry": policy["reason_codes"],
        "runner_abi": release["runner_abi"],
        "runtime_release_digest": runtime_release_digest(),
        "search_space_digest": identity.search_space_digest,
        "search_space_id": identity.search_space_id,
        "source_manifest": source_manifest(),
        "source_manifest_digest": source_manifest_digest(),
        "subcheck_order": policy["subcheck_order"],
    }


def common_release_projection_digest() -> str:
    return sha256_digest(common_release_projection())


def instance_binding(
    *, opaque_arm_instance_id: str, arm_private_root: str | Path
) -> dict[str, Any]:
    return {
        "arm_private_root": Path(arm_private_root).resolve().as_posix(),
        "common_release_projection_digest": common_release_projection_digest(),
        "opaque_arm_instance_id": opaque_arm_instance_id,
    }


def profile_supports(program: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
    payload = dict(program).get("program_payload")
    if not isinstance(payload, Mapping):
        return False, ("PROGRAM_PAYLOAD_MISSING",)
    primitives = {
        str(component.get("primitive_id"))
        for component in payload.get("components", ())
        if isinstance(component, Mapping) and component.get("primitive_id") is not None
    }
    operators = {
        str(operator.get("operator_id"))
        for operator in payload.get("architecture_operators", ())
        if isinstance(operator, Mapping) and operator.get("operator_id") is not None
    }
    profile = executable_profile()
    construction_mode = str(payload.get("construction_mode") or "")
    custom_components = payload.get("custom_components", ())
    missing_primitive_components = [
        component
        for component in payload.get("components", ())
        if isinstance(component, Mapping) and "primitive_id" not in component
    ]
    unsupported_primitives = sorted(primitives - set(profile["supported_primitives"]))
    unsupported_operators = sorted(operators - set(profile["supported_operators"]))
    matching_templates = [
        item["template_id"]
        for item in profile["architecture_templates"]
        if set(item["trigger_primitives"]).issubset(primitives)
    ]
    reasons = tuple(
        (
            ["CANDIDATE_CONTROLLED_CUSTOM_MODEL"]
            if construction_mode == "CUSTOM_MODEL" or custom_components
            else []
        )
        + (["COMPONENT_WITHOUT_PACKAGE_PRIMITIVE"] if missing_primitive_components else [])
        + [f"UNSUPPORTED_PRIMITIVE:{item}" for item in unsupported_primitives]
        + [f"UNSUPPORTED_OPERATOR:{item}" for item in unsupported_operators]
        + (["NO_PACKAGE_TEMPLATE"] if not matching_templates else [])
    )
    return not reasons, reasons


def implementation_identity_digest(files: Mapping[str, bytes]) -> str:
    rows = [
        {"path": path, "sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
        for path, payload in sorted(files.items())
    ]
    return sha256_digest(
        {
            "files": rows,
            "runtime_release_digest": runtime_release_digest(),
            "source_manifest_digest": source_manifest_digest(),
        }
    )


def canonical_resource_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value)


__all__ = [
    "SPACE_ID",
    "campaign_projection",
    "canonical_resource_bytes",
    "common_guard_policy",
    "common_guard_policy_digest",
    "common_release_projection",
    "common_release_projection_digest",
    "coverage_manifest",
    "development_protocol",
    "executable_profile",
    "executable_profile_digest",
    "implementation_identity_digest",
    "instance_binding",
    "profile_supports",
    "runtime_release_contract",
    "runtime_release_digest",
    "source_manifest",
    "source_manifest_digest",
    "validate_frozen_environment",
]
