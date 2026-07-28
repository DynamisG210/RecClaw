"""Deterministic BL-ICF materialization, trust classification, and V2 binding."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .compilation_cache import compile_campaign_program as compile_program

from .campaign_runtime import (
    CampaignRuntimeError,
    campaign_runtime_profile,
    execution_recipe_for_program,
)
from .canonical import (
    bytes_sha256,
    canonical_json_bytes,
    content_id,
    sha256_digest,
    validate_sha256,
)
from .runtime_contracts import (
    CandidateExecutionBindingV2,
    CommonEligibleActionV1,
    DevelopmentExecutionGateDecisionV1,
    ExecutionTrustClassificationV1,
    GateStatus,
    MaterializationReportV1,
    TrustClass,
)
from .runtime_release import (
    common_release_projection_digest,
    executable_profile,
    executable_profile_digest,
    implementation_identity_digest,
    runtime_release_contract,
    runtime_release_digest,
    source_manifest,
    source_manifest_digest,
)
from .state_store import (
    RegisterArtifactCommand,
    SingleWriterExperimentStoreV1,
)


_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_.:-]{2,127}$")
_MATERIALIZER_ID = "recclaw.bl-icf.deterministic-materializer.v1"
_CLASSIFIER_ID = "recclaw.package-template-trust-classifier.v1"
_GATE_ID = "recclaw.development-execution-gate.v1"


def _source_digest() -> str:
    return bytes_sha256(Path(__file__).read_bytes())


def _ensure_private_root(root: Path) -> Path:
    if not root.is_absolute():
        raise ValueError("Arm runtime root must be absolute")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or root.resolve() != root:
        raise ValueError("Arm runtime root must be a real, canonical directory")
    return root


def _safe_target(root: Path, relative: str) -> Path:
    target = root.joinpath(*relative.split("/"))
    resolved_parent = target.parent.resolve()
    try:
        resolved_parent.relative_to(root)
    except ValueError as exc:
        raise ValueError("materialization path escapes the Arm-private root") from exc
    for parent in (target.parent, *target.parent.parents):
        if parent == root:
            break
        if parent.is_symlink():
            raise ValueError("materialization path traverses a symlink")
        if parent.exists() and parent.is_mount():
            raise ValueError("materialization path traverses a nested mount")
    return target


def _write_exact(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.is_symlink() or target.stat().st_nlink != 1:
            raise ValueError("existing materialization target is linked")
        if target.read_bytes() != payload:
            raise ValueError("deterministic materialization target has different bytes")
        return
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _program_parts(program: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    payload = program["program_payload"]
    primitives = tuple(
        sorted(
            str(component["primitive_id"])
            for component in payload["components"]
            if "primitive_id" in component
        )
    )
    operators = tuple(
        sorted(str(operator["operator_id"]) for operator in payload["architecture_operators"])
    )
    return primitives, operators


def _select_template(primitives: tuple[str, ...]) -> str:
    profile = executable_profile()
    primitive_set = set(primitives)
    for template in profile["architecture_templates"]:
        if set(template["trigger_primitives"]).issubset(primitive_set):
            return str(template["template_id"])
    raise ValueError("no package-owned template covers this mechanism program")


def _materialization_projection(
    program: Mapping[str, Any],
    *,
    campaign: bool,
) -> tuple[str, str, str, Mapping[str, Any] | None]:
    if campaign:
        recipe = execution_recipe_for_program(program)
        return (
            str(campaign_runtime_profile()["profile_digest"]),
            str(recipe["mechanism_id"]),
            str(recipe["entrypoint"]),
            recipe,
        )
    primitives, _operators = _program_parts(program)
    return (
        executable_profile_digest(),
        _select_template(primitives),
        runtime_release_contract()["allowed_entrypoint"],
        None,
    )


class DeterministicMaterializerV1:
    materializer_id = _MATERIALIZER_ID

    def materialize(
        self,
        eligible: CommonEligibleActionV1,
        *,
        program: Mapping[str, Any],
        arm_runtime_root: str | Path,
    ) -> MaterializationReportV1:
        root = _ensure_private_root(Path(arm_runtime_root))
        compile_report = eligible.compile_projection
        observed_compile = compile_program(program)
        if observed_compile.mechanism_program_digest != eligible.program_digest:
            raise ValueError("materializer received substituted mechanism-program bytes")
        observed_projection = {
            "candidate_id": observed_compile.candidate_id,
            "mechanism_program_digest": observed_compile.mechanism_program_digest,
            "mechanism_semantics_digest": observed_compile.mechanism_semantics_digest,
            "required_capabilities": list(observed_compile.required_capabilities),
            "space_identity": (
                observed_compile.space_identity.to_dict()
                if observed_compile.space_identity
                else None
            ),
            "status": observed_compile.status.value,
        }
        if sha256_digest(observed_projection) != sha256_digest(compile_report):
            raise ValueError("materializer received a substituted compile projection")
        candidate_id = str(eligible.candidate_id)
        if not _SAFE_ID.fullmatch(candidate_id):
            raise ValueError("compiler-derived candidate ID is not path safe")
        primitives, operators = _program_parts(program)
        (
            campaign_profile_digest,
            template_id,
            entrypoint,
            execution_recipe,
        ) = _materialization_projection(
            program,
            campaign=(
                eligible.release_projection_digest
                == campaign_runtime_profile()["profile_digest"]
            ),
        )
        candidate_prefix = f"recclaw_ext/generated/{candidate_id}"
        program_bytes = canonical_json_bytes(program)
        handler_config = {
            "candidate_id": candidate_id,
            "mechanism_program_digest": compile_report["mechanism_program_digest"],
            "mechanism_semantics_digest": compile_report["mechanism_semantics_digest"],
            "operators": list(operators),
            "primitives": list(primitives),
            "template_id": template_id,
        }
        if execution_recipe is not None:
            handler_config["execution_recipe"] = execution_recipe
        config_bytes = canonical_json_bytes(handler_config)
        manifest = {
            "candidate_id": candidate_id,
            "entrypoint": entrypoint,
            "generated_files": [
                {
                    "path": f"{candidate_prefix}/handler_config.json",
                    "sha256": bytes_sha256(config_bytes),
                    "size_bytes": len(config_bytes),
                },
                {
                    "path": f"{candidate_prefix}/program.json",
                    "sha256": bytes_sha256(program_bytes),
                    "size_bytes": len(program_bytes),
                },
            ],
            "package_source_manifest_digest": source_manifest_digest(),
            "template_id": template_id,
        }
        manifest_bytes = canonical_json_bytes(manifest)
        files = {
            f"{candidate_prefix}/handler_config.json": config_bytes,
            f"{candidate_prefix}/implementation_manifest.json": manifest_bytes,
            f"{candidate_prefix}/program.json": program_bytes,
        }
        for relative, payload in files.items():
            _write_exact(_safe_target(root, relative), payload)
        file_rows = tuple(
            {
                "path": relative,
                "sha256": bytes_sha256(payload),
                "size_bytes": len(payload),
            }
            for relative, payload in sorted(files.items())
        )
        return MaterializationReportV1(
            {
                "candidate_id": candidate_id,
                "campaign_profile_digest": campaign_profile_digest,
                "compile_report_digest": eligible.compile_report_digest,
                "dependencies": [
                    {
                        "identity": runtime_release_contract()["recbole_commit"],
                        "name": "RecBole",
                        "usage": runtime_release_contract()["recbole_usage"],
                    },
                    {
                        "identity": campaign_profile_digest,
                        "name": (
                            "recclaw-campaign-executable-catalog"
                            if execution_recipe is not None
                            else "recclaw-m1-package-handlers"
                        ),
                        "usage": "EXECUTED",
                    },
                ],
                "diagnostics": [],
                "entrypoint": entrypoint,
                "files": file_rows,
                "implementation_digest": implementation_identity_digest(files),
                "materializer_digest": sha256_digest(
                    {"id": self.materializer_id, "source_sha256": _source_digest()}
                ),
                "mechanism_program_digest": compile_report["mechanism_program_digest"],
                "mechanism_semantics_digest": compile_report["mechanism_semantics_digest"],
                "required_capabilities": compile_report["required_capabilities"],
                "runner_abi": runtime_release_contract()["runner_abi"],
                "runtime_release_digest": runtime_release_digest(),
                "status": "MATERIALIZED",
                "template_id": template_id,
            }
        )


def classify_execution_trust(
    report: MaterializationReportV1,
    *,
    arm_runtime_root: str | Path,
) -> ExecutionTrustClassificationV1:
    root = _ensure_private_root(Path(arm_runtime_root))
    reasons: list[str] = []
    if report.status != "MATERIALIZED":
        reasons.append("MATERIALIZATION_NOT_COMPLETE")
    config_path = (
        root
        / "recclaw_ext"
        / "generated"
        / str(report.candidate_id)
        / "handler_config.json"
    )
    expected_profile = executable_profile_digest()
    expected_entrypoint = runtime_release_contract()["allowed_entrypoint"]
    try:
        handler_config = json.loads(config_path.read_text(encoding="utf-8"))
        recipe = handler_config.get("execution_recipe")
        if recipe is not None:
            expected_profile = str(campaign_runtime_profile()["profile_digest"])
            expected_entrypoint = str(recipe["entrypoint"])
    except (OSError, ValueError, json.JSONDecodeError, TypeError, KeyError):
        reasons.append("HANDLER_CONFIG_UNREADABLE")
    if report.entrypoint != expected_entrypoint:
        reasons.append("CALLABLE_OR_ENTRYPOINT_SUBSTITUTION")
    if report.runner_abi != runtime_release_contract()["runner_abi"]:
        reasons.append("RUNNER_ABI_SUBSTITUTION")
    if report.campaign_profile_digest != expected_profile:
        reasons.append("TEMPLATE_PROFILE_SUBSTITUTION")
    expected_sources = {item["path"]: item for item in source_manifest()}
    if "runtime_handlers.py" not in expected_sources:
        reasons.append("PACKAGE_HANDLER_SOURCE_MISSING")

    actual_files: dict[str, bytes] = {}
    reported_paths = {str(row["path"]) for row in report.files}
    for row in report.files:
        relative = str(row["path"])
        if relative.endswith(".py"):
            reasons.append("CANDIDATE_CONTROLLED_SOURCE_PRESENT")
            continue
        try:
            target = _safe_target(root, relative)
        except ValueError:
            reasons.append("PATH_OUT_OF_SCOPE")
            continue
        if not target.is_file() or target.is_symlink():
            reasons.append("MATERIALIZED_FILE_MISSING_OR_LINKED")
            continue
        if target.stat().st_nlink != 1:
            reasons.append("HARDLINK_DETECTED")
            continue
        payload = target.read_bytes()
        actual_files[relative] = payload
        if len(payload) != row["size_bytes"] or bytes_sha256(payload) != row["sha256"]:
            reasons.append("MATERIALIZED_FILE_DIGEST_MISMATCH")
    candidate_root = root / "recclaw_ext" / "generated" / str(report.candidate_id)
    if candidate_root.is_dir():
        observed_paths = {
            path.relative_to(root).as_posix()
            for path in candidate_root.rglob("*")
            if path.is_file()
        }
        if observed_paths != reported_paths:
            reasons.append("UNMANIFESTED_OR_MISSING_FILE")
    else:
        reasons.append("CANDIDATE_RUNTIME_ROOT_MISSING")
    if implementation_identity_digest(actual_files) != report.implementation_digest:
        reasons.append("IMPLEMENTATION_DIGEST_MISMATCH")

    classification = (
        TrustClass.PACKAGE_OWNED_TYPED_TEMPLATE
        if not reasons
        else TrustClass.CANDIDATE_CONTROLLED_EXECUTABLE
    )
    return ExecutionTrustClassificationV1(
        {
            "classification": classification.value,
            "classifier_digest": sha256_digest(
                {
                    "classifier_id": _CLASSIFIER_ID,
                    "materializer_source": _source_digest(),
                    "package_sources": source_manifest_digest(),
                }
            ),
            "implementation_digest": report.implementation_digest,
            "materialization_digest": report.digest,
            "reason_codes": sorted(set(reasons)),
        }
    )


def verify_materialization(
    report: MaterializationReportV1,
    *,
    eligible: CommonEligibleActionV1,
    arm_runtime_root: str | Path,
) -> tuple[bool, tuple[str, ...]]:
    """Re-derive the materializer output from actual bytes and the eligible action."""

    root = _ensure_private_root(Path(arm_runtime_root))
    reasons: list[str] = []
    prefix = f"recclaw_ext/generated/{eligible.candidate_id}"
    expected_paths = {
        f"{prefix}/handler_config.json",
        f"{prefix}/implementation_manifest.json",
        f"{prefix}/program.json",
    }
    rows = {str(row["path"]): row for row in report.files}
    if set(rows) != expected_paths:
        reasons.append("MATERIALIZATION_FILE_SET_MISMATCH")
        return False, tuple(reasons)
    try:
        program_bytes = _safe_target(root, f"{prefix}/program.json").read_bytes()
        config_bytes = _safe_target(root, f"{prefix}/handler_config.json").read_bytes()
        manifest_bytes = _safe_target(
            root, f"{prefix}/implementation_manifest.json"
        ).read_bytes()
        program = json.loads(program_bytes)
        config = json.loads(config_bytes)
        manifest = json.loads(manifest_bytes)
    except (OSError, ValueError, json.JSONDecodeError):
        return False, ("MATERIALIZATION_BYTES_UNREADABLE",)

    compiled = compile_program(program)
    if compiled.mechanism_program_digest != eligible.program_digest:
        reasons.append("MATERIALIZED_PROGRAM_SUBSTITUTION")
    projection = {
        "candidate_id": compiled.candidate_id,
        "mechanism_program_digest": compiled.mechanism_program_digest,
        "mechanism_semantics_digest": compiled.mechanism_semantics_digest,
        "required_capabilities": list(compiled.required_capabilities),
        "space_identity": compiled.space_identity.to_dict() if compiled.space_identity else None,
        "status": compiled.status.value,
    }
    if sha256_digest(projection) != sha256_digest(eligible.compile_projection):
        reasons.append("MATERIALIZED_COMPILE_PROJECTION_MISMATCH")
    try:
        primitives, operators = _program_parts(program)
        (
            campaign_profile_digest,
            template_id,
            entrypoint,
            execution_recipe,
        ) = _materialization_projection(
            program,
            campaign=(
                eligible.release_projection_digest
                == campaign_runtime_profile()["profile_digest"]
            ),
        )
    except (KeyError, TypeError, ValueError):
        reasons.append("MATERIALIZED_TEMPLATE_UNSUPPORTED")
        primitives, operators, template_id = (), (), ""
        campaign_profile_digest = executable_profile_digest()
        entrypoint = runtime_release_contract()["allowed_entrypoint"]
        execution_recipe = None
    expected_config = {
        "candidate_id": eligible.candidate_id,
        "mechanism_program_digest": compiled.mechanism_program_digest,
        "mechanism_semantics_digest": compiled.mechanism_semantics_digest,
        "operators": list(operators),
        "primitives": list(primitives),
        "template_id": template_id,
    }
    if execution_recipe is not None:
        expected_config["execution_recipe"] = execution_recipe
    if config != expected_config or config_bytes != canonical_json_bytes(expected_config):
        reasons.append("HANDLER_CONFIG_SUBSTITUTION")
    expected_manifest = {
        "candidate_id": eligible.candidate_id,
        "entrypoint": entrypoint,
        "generated_files": [
            {
                "path": f"{prefix}/handler_config.json",
                "sha256": bytes_sha256(canonical_json_bytes(expected_config)),
                "size_bytes": len(canonical_json_bytes(expected_config)),
            },
            {
                "path": f"{prefix}/program.json",
                "sha256": bytes_sha256(program_bytes),
                "size_bytes": len(program_bytes),
            },
        ],
        "package_source_manifest_digest": source_manifest_digest(),
        "template_id": template_id,
    }
    if manifest != expected_manifest or manifest_bytes != canonical_json_bytes(
        expected_manifest
    ):
        reasons.append("IMPLEMENTATION_MANIFEST_SUBSTITUTION")
    files = {
        f"{prefix}/handler_config.json": config_bytes,
        f"{prefix}/implementation_manifest.json": manifest_bytes,
        f"{prefix}/program.json": program_bytes,
    }
    if implementation_identity_digest(files) != report.implementation_digest:
        reasons.append("IMPLEMENTATION_DIGEST_MISMATCH")
    for path, payload in files.items():
        row = rows[path]
        if row["sha256"] != bytes_sha256(payload) or row["size_bytes"] != len(payload):
            reasons.append("MATERIALIZATION_REPORT_FILE_MISMATCH")
    expected_report = {
        "campaign_profile_digest": campaign_profile_digest,
        "candidate_id": eligible.candidate_id,
        "compile_report_digest": eligible.compile_report_digest,
        "dependencies": [
            {
                "identity": runtime_release_contract()["recbole_commit"],
                "name": "RecBole",
                "usage": runtime_release_contract()["recbole_usage"],
            },
            {
                "identity": campaign_profile_digest,
                "name": (
                    "recclaw-campaign-executable-catalog"
                    if execution_recipe is not None
                    else "recclaw-m1-package-handlers"
                ),
                "usage": "EXECUTED",
            },
        ],
        "diagnostics": [],
        "entrypoint": entrypoint,
        "materializer_digest": sha256_digest(
            {"id": _MATERIALIZER_ID, "source_sha256": _source_digest()}
        ),
        "mechanism_program_digest": compiled.mechanism_program_digest,
        "mechanism_semantics_digest": compiled.mechanism_semantics_digest,
        "required_capabilities": list(compiled.required_capabilities),
        "runner_abi": runtime_release_contract()["runner_abi"],
        "runtime_release_digest": runtime_release_digest(),
        "status": "MATERIALIZED",
        "template_id": template_id,
    }
    for field, expected in expected_report.items():
        if sha256_digest(getattr(report, field)) != sha256_digest(expected):
            reasons.append(f"MATERIALIZATION_REPORT_FIELD_MISMATCH:{field}")
    return not reasons, tuple(sorted(set(reasons)))


def build_binding_v2(
    *,
    eligible: CommonEligibleActionV1,
    report: MaterializationReportV1,
    trust: ExecutionTrustClassificationV1,
    opaque_arm_instance_id: str,
    arm_private_root: str | Path,
    round_id: str,
    search_seed: int,
) -> CandidateExecutionBindingV2:
    root = _ensure_private_root(Path(arm_private_root))
    run_id = content_id(
        "m1-run",
        {
            "candidate_id": eligible.candidate_id,
            "implementation_digest": report.implementation_digest,
            "round_id": round_id,
            "search_seed": search_seed,
        },
    )
    return CandidateExecutionBindingV2(
        {
            "arm_private_root": root.as_posix(),
            "budget_digest": eligible.budget_digest,
            "candidate_id": eligible.candidate_id,
            "implementation_digest": report.implementation_digest,
            "materialization_digest": report.digest,
            "mechanism_program_digest": report.mechanism_program_digest,
            "mechanism_semantics_digest": report.mechanism_semantics_digest,
            "opaque_arm_instance_id": opaque_arm_instance_id,
            "profile_digest": report.campaign_profile_digest,
            "round_id": round_id,
            "run_id": run_id,
            "runner_abi": runtime_release_contract()["runner_abi"],
            "runtime_release_digest": runtime_release_digest(),
            "search_seed": search_seed,
            "trust_classification_digest": trust.digest,
        }
    )


def verify_binding_v2(
    binding: CandidateExecutionBindingV2,
    *,
    eligible: CommonEligibleActionV1,
    report: MaterializationReportV1,
    trust: ExecutionTrustClassificationV1,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    expected = {
        "budget_digest": eligible.budget_digest,
        "candidate_id": eligible.candidate_id,
        "implementation_digest": report.implementation_digest,
        "materialization_digest": report.digest,
        "mechanism_program_digest": report.mechanism_program_digest,
        "mechanism_semantics_digest": report.mechanism_semantics_digest,
        "profile_digest": report.campaign_profile_digest,
        "runner_abi": runtime_release_contract()["runner_abi"],
        "runtime_release_digest": runtime_release_digest(),
        "trust_classification_digest": trust.digest,
    }
    for field, value in expected.items():
        if getattr(binding, field) != value:
            reasons.append(f"BINDING_FIELD_MISMATCH:{field}")
    for name in (
        "budget_digest",
        "implementation_digest",
        "materialization_digest",
        "mechanism_program_digest",
        "mechanism_semantics_digest",
        "profile_digest",
        "runtime_release_digest",
        "trust_classification_digest",
    ):
        try:
            validate_sha256(str(getattr(binding, name)), field_name=name)
        except ValueError:
            reasons.append(f"BINDING_DIGEST_INVALID:{name}")
    root = Path(str(binding.arm_private_root))
    try:
        _ensure_private_root(root)
    except ValueError:
        reasons.append("ARM_PRIVATE_ROOT_INVALID")
    observed_trust = classify_execution_trust(report, arm_runtime_root=root)
    if observed_trust.to_dict() != trust.to_dict():
        reasons.append("TRUST_CLASSIFICATION_SUBSTITUTION")
    return not reasons, tuple(sorted(set(reasons)))


def development_execution_gate(
    *,
    binding: CandidateExecutionBindingV2,
    eligible: CommonEligibleActionV1,
    report: MaterializationReportV1,
    trust: ExecutionTrustClassificationV1,
    task_authorization_ref: str,
) -> DevelopmentExecutionGateDecisionV1:
    valid_binding, binding_reasons = verify_binding_v2(
        binding, eligible=eligible, report=report, trust=trust
    )
    reasons = list(binding_reasons)
    if trust.classification != TrustClass.PACKAGE_OWNED_TYPED_TEMPLATE.value:
        reasons.append("TRUST_CLASSIFICATION_DENIED")
    if task_authorization_ref != "RecClaw_Codex_Autonomous_M1_M8_Master_Goal.md#M1":
        reasons.append("TASK_AUTHORIZATION_SCOPE_MISMATCH")
    decision = GateStatus.ALLOW if not reasons else GateStatus.DENY
    return DevelopmentExecutionGateDecisionV1(
        {
            "binding_digest": binding.digest,
            "decision": decision.value,
            "gate_source_digest": sha256_digest(
                {
                    "gate_id": _GATE_ID,
                    "release_projection": common_release_projection_digest(),
                    "source_sha256": _source_digest(),
                }
            ),
            "materialization_digest": report.digest,
            "reason_codes": sorted(set(reasons)),
            "runtime_release_digest": runtime_release_digest(),
            "task_authorization_ref": task_authorization_ref,
            "trust_classification_digest": trust.digest,
        }
    )


def register_materialization_artifacts(
    store: SingleWriterExperimentStoreV1,
    *,
    binding: CandidateExecutionBindingV2,
    report: MaterializationReportV1,
) -> tuple[dict[str, Any], ...]:
    """Index the exact implementation files and materialization report."""

    root = Path(str(binding.arm_private_root))
    rows: list[dict[str, Any]] = []
    type_by_suffix = {
        "handler_config.json": "M1_HANDLER_CONFIG",
        "implementation_manifest.json": "M1_IMPLEMENTATION_MANIFEST",
        "program.json": "M1_MECHANISM_PROGRAM",
    }
    for item in report.files:
        relative = str(item["path"])
        artifact_type = next(
            value for suffix, value in type_by_suffix.items() if relative.endswith(suffix)
        )
        rows.append(
            store.register_artifact(
                RegisterArtifactCommand(
                    round_id=str(binding.round_id),
                    artifact_type=artifact_type,
                    relative_path=relative,
                    producer="DeterministicMaterializerV1",
                    idempotency_key=f"materialized:{binding.round_id}:{artifact_type}",
                ),
                root.joinpath(*relative.split("/")).read_bytes(),
            )
        )
    report_path = f"artifacts/{binding.run_id}/materialization_report.v1.json"
    rows.append(
        store.register_artifact(
            RegisterArtifactCommand(
                round_id=str(binding.round_id),
                artifact_type="MATERIALIZATION_REPORT_V1",
                relative_path=report_path,
                producer="DeterministicMaterializerV1",
                idempotency_key=f"materialization-report:{binding.round_id}",
            ),
            canonical_json_bytes(report.to_dict()),
        )
    )
    return tuple(rows)


__all__ = [
    "DeterministicMaterializerV1",
    "build_binding_v2",
    "classify_execution_trust",
    "development_execution_gate",
    "register_materialization_artifacts",
    "verify_binding_v2",
    "verify_materialization",
]
