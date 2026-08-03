"""Deterministic OpenSpec-to-CandidatePackage boundary for Research Line vNext."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .canonical import (
    canonical_value,
    content_id,
    sha256_digest,
    validate_relative_artifact_path,
    validate_sha256,
)
from .innovation_recbole_adapter import (
    candidate_tree_identity,
    snapshot_candidate_tree,
)
from .vnext_contracts import CandidatePackageV1, OpenResearchSpecV1


_BLIND_SPEC_FIELDS = (
    "causal_chain",
    "closest_parent",
    "compatibility_requirements",
    "competing_explanation",
    "current_profile_digest",
    "current_profile_expressibility_claim",
    "current_profile_ref",
    "expected_evidence",
    "falsifier",
    "high_change_justification",
    "hypothesis",
    "idea_mode",
    "implementation_requirements",
    "matched_control_requirement",
    "mechanism_off_definition",
    "mechanism_change",
    "minimal_testable_wedge",
    "protocol_digest",
    "protocol_ref",
    "realization_mode",
    "research_question",
    "resource_hypothesis",
    "discriminative_predictions",
)
_FORBIDDEN_REQUEST_KEYS = frozenset(
    {
        "arm",
        "arm_code",
        "candidate_origin",
        "context_digest",
        "context_ref",
        "controller",
        "controller_id",
        "metric_observation",
        "metric_values",
        "origin",
        "outcome",
        "outcomes",
        "producer",
        "producer_id",
        "producer_role",
        "result",
        "results",
        "research_spec_digest",
        "research_spec_ref",
        "spec_id",
    }
)
_CANDIDATE_PREFIXES = ("recclaw_ext/", "tests/")


class InnovationSpineError(RuntimeError):
    """Typed implementation/package failure at the blind materialization boundary."""

    def __init__(
        self,
        *,
        failure_class: str,
        reason_code: str,
        message: str,
    ) -> None:
        if failure_class not in {"IMPLEMENTATION", "PACKAGE"}:
            raise ValueError("InnovationSpineError failure_class is invalid")
        super().__init__(message)
        self.failure_class = failure_class
        self.reason_code = reason_code


@dataclass(frozen=True, slots=True)
class SharedImplementerPolicy:
    """Common non-outcome service inputs shared by every implementation source."""

    allowed_files: tuple[str, ...]
    dependency_identity_ref: str
    dependency_identity_digest: str
    runtime_identity_ref: str
    runtime_identity_digest: str
    prompt_digest: str
    tool_policy_digest: str
    implementation_token_ceiling: int
    execution_contract: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        try:
            paths = tuple(
                validate_relative_artifact_path(str(path))
                for path in self.allowed_files
            )
        except Exception as error:
            raise InnovationSpineError(
                failure_class="IMPLEMENTATION",
                reason_code="INVALID_WRITE_ALLOWLIST",
                message="allowed_files contains an invalid relative path",
            ) from error
        if (
            not paths
            or len(set(paths)) != len(paths)
            or any(not path.startswith(_CANDIDATE_PREFIXES) for path in paths)
            or not any(path.startswith("recclaw_ext/") for path in paths)
        ):
            raise InnovationSpineError(
                failure_class="IMPLEMENTATION",
                reason_code="INVALID_WRITE_ALLOWLIST",
                message=(
                    "allowed_files must be unique candidate-local paths and "
                    "include an entrypoint source under recclaw_ext/"
                ),
            )
        object.__setattr__(self, "allowed_files", tuple(sorted(paths)))
        for field_name in (
            "dependency_identity_ref",
            "runtime_identity_ref",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or value != value.strip():
                raise InnovationSpineError(
                    failure_class="PACKAGE",
                    reason_code="INVALID_IDENTITY_REFERENCE",
                    message=f"{field_name} must be a normalized non-empty string",
                )
        for field_name in (
            "dependency_identity_digest",
            "runtime_identity_digest",
            "prompt_digest",
            "tool_policy_digest",
        ):
            try:
                validate_sha256(getattr(self, field_name), field_name=field_name)
            except Exception as error:
                raise InnovationSpineError(
                    failure_class="PACKAGE",
                    reason_code="INVALID_IDENTITY_DIGEST",
                    message=f"{field_name} must be a SHA-256 digest",
                ) from error
        if (
            not isinstance(self.implementation_token_ceiling, int)
            or isinstance(self.implementation_token_ceiling, bool)
            or self.implementation_token_ceiling < 1
        ):
            raise InnovationSpineError(
                failure_class="IMPLEMENTATION",
                reason_code="INVALID_TOKEN_CEILING",
                message="implementation_token_ceiling must be positive",
            )
        if self.execution_contract is not None and not isinstance(
            self.execution_contract, Mapping
        ):
            raise InnovationSpineError(
                failure_class="IMPLEMENTATION",
                reason_code="INVALID_EXECUTION_CONTRACT",
                message="execution_contract must be a mapping when supplied",
            )

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(self)


@dataclass(frozen=True, slots=True)
class MaterializedCandidate:
    """Successful local materialization and its RC0 CandidatePackageV1."""

    package: CandidatePackageV1
    blind_projection: Mapping[str, Any]
    shared_request: Mapping[str, Any]
    implementation_receipt: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return canonical_value(
            {
                "blind_projection": self.blind_projection,
                "implementation_receipt": self.implementation_receipt,
                "package": self.package.canonical_dict(),
                "shared_request": self.shared_request,
            }
        )


def _implementation_failure(reason_code: str, message: str) -> InnovationSpineError:
    return InnovationSpineError(
        failure_class="IMPLEMENTATION",
        reason_code=reason_code,
        message=message,
    )


def _package_failure(reason_code: str, message: str) -> InnovationSpineError:
    return InnovationSpineError(
        failure_class="PACKAGE",
        reason_code=reason_code,
        message=message,
    )


def origin_blind_projection(spec: OpenResearchSpecV1) -> dict[str, Any]:
    """Project semantic implementation inputs without source/context identity."""

    if not isinstance(spec, OpenResearchSpecV1):
        raise _implementation_failure(
            "INVALID_OPEN_SPEC",
            "origin-blind projection requires OpenResearchSpecV1",
        )
    source = spec.to_dict()
    projection = canonical_value(
        {
            field_name: source[field_name]
            for field_name in _BLIND_SPEC_FIELDS
            if field_name in source
        }
    )
    _reject_forbidden_request_keys(projection)
    return projection


def _reject_forbidden_request_keys(
    value: Any,
    *,
    path: tuple[str, ...] = (),
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            field_name = str(key)
            if field_name.lower() in _FORBIDDEN_REQUEST_KEYS:
                raise _implementation_failure(
                    "BLIND_REQUEST_IDENTITY_OR_OUTCOME_LEAK",
                    "blind request contains forbidden field "
                    + ".".join((*path, field_name)),
                )
            _reject_forbidden_request_keys(
                item,
                path=(*path, field_name),
            )
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _reject_forbidden_request_keys(
                item,
                path=(*path, str(index)),
            )


def build_shared_implementer_request(
    spec: OpenResearchSpecV1,
    *,
    policy: SharedImplementerPolicy,
) -> dict[str, Any]:
    """Build the only payload visible to a shared implementation service."""

    if not isinstance(policy, SharedImplementerPolicy):
        raise _implementation_failure(
            "INVALID_SHARED_POLICY",
            "shared implementer policy has the wrong type",
        )
    projection = origin_blind_projection(spec)
    projection_digest = sha256_digest(projection)
    service_policy: dict[str, Any] = {
        "candidate_local_write_only": True,
        "dependency_identity_digest": policy.dependency_identity_digest,
        "dependency_identity_ref": policy.dependency_identity_ref,
        "implementation_token_ceiling": policy.implementation_token_ceiling,
        "prompt_digest": policy.prompt_digest,
        "response_mode": "STRICT_JSON_FULL_FILE_CONTENTS",
        "runtime_identity_digest": policy.runtime_identity_digest,
        "runtime_identity_ref": policy.runtime_identity_ref,
        "tool_policy_digest": policy.tool_policy_digest,
    }
    if policy.execution_contract is not None:
        service_policy["execution_contract"] = canonical_value(
            policy.execution_contract
        )
    request = canonical_value(
        {
            "blind_candidate_id": (
                "innovation-candidate-" + projection_digest[:24]
            ),
            "blind_research_spec": projection,
            "candidate_local_write_allowlist": policy.allowed_files,
            "schema": "recclaw.shared-implementer-request.v1",
            "service_policy": service_policy,
        }
    )
    _reject_forbidden_request_keys(request)
    return request


def _validated_entrypoint(value: Any, *, allowed_files: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value.count(":") != 1:
        raise _implementation_failure(
            "INVALID_ENTRYPOINT",
            "entrypoint must use module.path:ClassName",
        )
    module_name, class_name = value.split(":", 1)
    if (
        not module_name.startswith("recclaw_ext.")
        or not class_name.isidentifier()
        or any(not part.isidentifier() for part in module_name.split("."))
    ):
        raise _implementation_failure(
            "INVALID_ENTRYPOINT",
            "entrypoint must identify a candidate-local recclaw_ext class",
        )
    source_path = module_name.replace(".", "/") + ".py"
    if source_path not in allowed_files:
        raise _implementation_failure(
            "ENTRYPOINT_OUTSIDE_ALLOWLIST",
            "entrypoint source is outside the exact write allowlist",
        )
    return value


def _validate_implementation_response(
    response: Mapping[str, Any],
    *,
    policy: SharedImplementerPolicy,
) -> dict[str, Any]:
    if not isinstance(response, Mapping) or set(response) != {
        "entrypoint",
        "files",
        "implementation_summary",
    }:
        raise _implementation_failure(
            "IMPLEMENTATION_RESPONSE_FIELDS_INVALID",
            "implementation response fields do not match the shared boundary",
        )
    summary = response["implementation_summary"]
    if not isinstance(summary, str) or not summary.strip():
        raise _implementation_failure(
            "IMPLEMENTATION_SUMMARY_MISSING",
            "implementation_summary must be non-empty",
        )
    entrypoint = _validated_entrypoint(
        response["entrypoint"],
        allowed_files=policy.allowed_files,
    )
    raw_files = response["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise _implementation_failure(
            "IMPLEMENTATION_FILES_MISSING",
            "implementation response must provide complete files",
        )
    normalized_files: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw_files:
        if not isinstance(item, Mapping) or set(item) != {"content", "path"}:
            raise _implementation_failure(
                "IMPLEMENTATION_FILE_FIELDS_INVALID",
                "implementation file fields must be path and content",
            )
        try:
            path = validate_relative_artifact_path(str(item["path"]))
        except Exception as error:
            raise _implementation_failure(
                "IMPLEMENTATION_FILE_INVALID",
                "implementation file path is not a safe relative artifact path",
            ) from error
        content = item["content"]
        if (
            path in seen
            or path not in policy.allowed_files
            or not isinstance(content, str)
            or not content
            or "\x00" in content
        ):
            raise _implementation_failure(
                "IMPLEMENTATION_FILE_INVALID",
                "implementation file is duplicated, forbidden, or invalid UTF-8 text",
            )
        seen.add(path)
        normalized_files.append({"content": content, "path": path})
    if not seen or not seen.issubset(set(policy.allowed_files)):
        raise _implementation_failure(
            "IMPLEMENTATION_FILE_SET_MISMATCH",
            "implementation response must materialize candidate-local files only",
        )
    entrypoint_path = response["entrypoint"].split(":", 1)[0].replace(".", "/") + ".py"
    if entrypoint_path not in seen:
        raise _implementation_failure(
            "ENTRYPOINT_SOURCE_MISSING",
            "implementation response must include its entrypoint source",
        )
    if "recclaw_ext/candidate.py" in policy.allowed_files:
        if "recclaw_ext/candidate.py" not in seen or not str(
            response["entrypoint"]
        ).startswith("recclaw_ext.candidate:"):
            raise _implementation_failure(
                "CANDIDATE_ENTRYPOINT_REQUIRED",
                "conversion packages must include recclaw_ext/candidate.py as the entrypoint",
            )
    return canonical_value(
        {
            "entrypoint": entrypoint,
            "files": sorted(normalized_files, key=lambda item: item["path"]),
            "implementation_summary": summary.strip(),
        }
    )


def _ensure_fresh_root(root: Path, *, blind_candidate_id: str) -> None:
    if root.name != blind_candidate_id:
        raise _package_failure(
            "CANDIDATE_ROOT_IDENTITY_MISMATCH",
            "candidate root name must equal the blind candidate identity",
        )
    if os.path.lexists(root):
        raise _package_failure(
            (
                "CANDIDATE_ROOT_SYMLINK"
                if root.is_symlink()
                else "CANDIDATE_ROOT_NOT_FRESH"
            ),
            "candidate root must not exist before materialization",
        )
    parent = root.parent
    if not parent.is_dir() or parent.is_symlink():
        raise _package_failure(
            "CANDIDATE_ROOT_PARENT_INVALID",
            "candidate root parent must be an existing non-symlink directory",
        )


def _exclusive_write(root: Path, *, relative: str, content: str) -> None:
    parts = PurePosixPath(relative).parts
    current = root
    for part in parts[:-1]:
        current = current / part
        if os.path.lexists(current):
            if current.is_symlink() or not current.is_dir():
                raise _package_failure(
                    "CANDIDATE_PACKAGE_PATH_UNSAFE",
                    "candidate package directory is not a real directory",
                )
            continue
        current.mkdir(mode=0o700)
    target = current / parts[-1]
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, 0o600)
    try:
        payload = content.encode("utf-8")
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def materialize_candidate_package(
    spec: OpenResearchSpecV1,
    *,
    policy: SharedImplementerPolicy,
    implementation_response: Mapping[str, Any],
    candidate_root: Path,
    candidate_root_ref: str,
) -> MaterializedCandidate:
    """Materialize one local response into a fresh root and RC0 package."""

    request = build_shared_implementer_request(spec, policy=policy)
    response = _validate_implementation_response(
        implementation_response,
        policy=policy,
    )
    root = candidate_root.absolute()
    _ensure_fresh_root(
        root,
        blind_candidate_id=str(request["blind_candidate_id"]),
    )
    if (
        not isinstance(candidate_root_ref, str)
        or not candidate_root_ref
        or candidate_root_ref != candidate_root_ref.strip()
    ):
        raise _package_failure(
            "CANDIDATE_ROOT_REF_INVALID",
            "candidate_root_ref must be a normalized non-empty string",
        )

    created_root = False
    try:
        root.mkdir(mode=0o700)
        created_root = True
        for item in response["files"]:
            _exclusive_write(
                root,
                relative=str(item["path"]),
                content=str(item["content"]),
            )
        manifest = snapshot_candidate_tree(root)
        package_allowed_files = tuple(
            str(item["path"]) for item in response["files"]
        )
        if (
            any(row["path"] not in package_allowed_files for row in manifest)
            or tuple(row["path"] for row in manifest) != package_allowed_files
            or any((root / row["path"]).is_symlink() for row in manifest)
        ):
            raise _package_failure(
                "MATERIALIZED_TREE_MISMATCH",
                "materialized tree differs from the exact allowlist",
            )
        source_tree_digest, candidate_root_digest = candidate_tree_identity(
            root,
            candidate_root_ref=candidate_root_ref,
        )
        projection = origin_blind_projection(spec)
        projection_digest = sha256_digest(projection)
        implementation_receipt = canonical_value(
            {
                "blind_candidate_id": request["blind_candidate_id"],
                "entrypoint": response["entrypoint"],
                "request_digest": sha256_digest(request),
                "response_digest": sha256_digest(response),
                "schema": (
                    "recclaw.shared-implementer-materialization-receipt.v1"
                ),
                "source_tree_digest": source_tree_digest,
                "written_files": tuple(
                    {
                        "path": row["path"],
                        "sha256": row["sha256"],
                        "size_bytes": row["size_bytes"],
                    }
                    for row in manifest
                ),
            }
        )
        receipt_digest = sha256_digest(implementation_receipt)
        receipt_ref = content_id(
            "recclaw-implementation-receipt-v1",
            implementation_receipt,
        )
        package = CandidatePackageV1(
            research_spec_ref=spec.spec_id,
            research_spec_digest=spec.digest,
            protocol_ref=spec.protocol_ref,
            protocol_digest=spec.protocol_digest,
            source_tree_digest=source_tree_digest,
            candidate_root_ref=candidate_root_ref,
            candidate_root_digest=candidate_root_digest,
            executable_entrypoint=str(response["entrypoint"]),
            allowed_files=package_allowed_files,
            dependency_identity_ref=policy.dependency_identity_ref,
            dependency_identity_digest=policy.dependency_identity_digest,
            runtime_identity_ref=policy.runtime_identity_ref,
            runtime_identity_digest=policy.runtime_identity_digest,
            implementation_receipt_ref=receipt_ref,
            implementation_receipt_digest=receipt_digest,
            origin_blind_projection_digest=projection_digest,
        )
        return MaterializedCandidate(
            package=package,
            blind_projection=projection,
            shared_request=request,
            implementation_receipt=implementation_receipt,
        )
    except InnovationSpineError:
        if created_root:
            shutil.rmtree(root)
        raise
    except OSError as error:
        if created_root:
            shutil.rmtree(root)
        raise _package_failure(
            "CANDIDATE_PACKAGE_FILESYSTEM_FAILURE",
            type(error).__name__,
        ) from error
    except Exception as error:
        if created_root:
            shutil.rmtree(root)
        raise _package_failure(
            "CANDIDATE_PACKAGE_IDENTITY_FAILURE",
            type(error).__name__,
        ) from error


__all__ = [
    "InnovationSpineError",
    "MaterializedCandidate",
    "SharedImplementerPolicy",
    "build_shared_implementer_request",
    "materialize_candidate_package",
    "origin_blind_projection",
]
