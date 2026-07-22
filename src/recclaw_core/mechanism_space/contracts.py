"""Stable public contracts for mechanism-space compilation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


class CompileStatus(str, Enum):
    VALID_WIRED = "VALID_WIRED"
    VALID_NEEDS_IMPLEMENTATION = "VALID_NEEDS_IMPLEMENTATION"
    PROTOCOL_BRANCH_REQUIRED = "PROTOCOL_BRANCH_REQUIRED"
    CAPABILITY_DENIED = "CAPABILITY_DENIED"
    INVALID = "INVALID"
    UNSUPPORTED = "UNSUPPORTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


@dataclass(frozen=True, slots=True)
class CompileDiagnostic:
    code: str
    message: str
    path: str = ""
    expected: Any = None
    actual: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "actual": self.actual,
            "code": self.code,
            "expected": self.expected,
            "message": self.message,
            "path": self.path,
        }


@dataclass(frozen=True, slots=True)
class SpaceIdentity:
    search_space_id: str
    search_space_version: str
    search_space_digest: str
    family_id: str
    family_version: str
    provider_id: str

    def to_dict(self) -> dict[str, str]:
        return {
            "family_id": self.family_id,
            "family_version": self.family_version,
            "provider_id": self.provider_id,
            "search_space_digest": self.search_space_digest,
            "search_space_id": self.search_space_id,
            "search_space_version": self.search_space_version,
        }


@dataclass(frozen=True, slots=True)
class CompileReportV1:
    status: CompileStatus
    operation: str = "compile_mechanism_program"
    diagnostics: Sequence[CompileDiagnostic] = field(default_factory=tuple)
    space_identity: SpaceIdentity | None = None
    candidate_id: str | None = None
    mechanism_semantics_digest: str | None = None
    mechanism_program_digest: str | None = None
    required_capabilities: Sequence[str] = field(default_factory=tuple)
    resolved_ir: Mapping[str, Any] | None = None

    @property
    def is_valid(self) -> bool:
        return self.status in {
            CompileStatus.VALID_WIRED,
            CompileStatus.VALID_NEEDS_IMPLEMENTATION,
        }

    def to_dict(self) -> dict[str, Any]:
        diagnostics = sorted(
            (item.to_dict() for item in self.diagnostics),
            key=lambda item: (item["code"], item["path"], item["message"]),
        )
        return {
            "authority": "NONE",
            "candidate_id": self.candidate_id,
            "diagnostics": diagnostics,
            "evidence_class": "DEVELOPMENT_ONLY",
            "formal_acceptance": False,
            "mechanism_program_digest": self.mechanism_program_digest,
            "mechanism_semantics_digest": self.mechanism_semantics_digest,
            "operation": self.operation,
            "report_version": "recclaw.mechanism-space.compile-report.v1",
            "required_capabilities": sorted(set(self.required_capabilities)),
            "resolved_ir": dict(self.resolved_ir) if self.resolved_ir is not None else None,
            "space_identity": self.space_identity.to_dict() if self.space_identity else None,
            "status": self.status.value,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
        )


@runtime_checkable
class MechanismSpaceProviderV1(Protocol):
    provider_id: str

    def identity(self) -> SpaceIdentity: ...

    def compile(self, envelope: Mapping[str, Any]) -> CompileReportV1: ...

    def prompt_projection(self) -> Mapping[str, Any]: ...

    def program_schema(self) -> Mapping[str, Any]: ...


__all__ = [
    "CompileDiagnostic",
    "CompileReportV1",
    "CompileStatus",
    "MechanismSpaceProviderV1",
    "SpaceIdentity",
]
