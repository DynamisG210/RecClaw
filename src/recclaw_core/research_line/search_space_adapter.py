"""Generic search-space boundary for feedback-driven confirmation work."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable


class ConfirmationResolutionKindV1(str, Enum):
    EXACT_BINDING = "EXACT_BINDING"
    NEEDS_PROPOSAL = "NEEDS_PROPOSAL"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True, slots=True)
class ConfirmationResolutionV1:
    kind: ConfirmationResolutionKindV1
    binding: Mapping[str, Any] | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ConfirmationResolutionKindV1):
            raise TypeError("kind must be ConfirmationResolutionKindV1")
        if self.kind is ConfirmationResolutionKindV1.EXACT_BINDING:
            if not isinstance(self.binding, Mapping):
                raise ValueError("EXACT_BINDING requires an opaque binding")
        elif self.binding is not None:
            raise ValueError("non-exact confirmation cannot carry a binding")

        if self.reason is not None and (
            not isinstance(self.reason, str) or not self.reason.strip()
        ):
            raise ValueError("confirmation reason must be non-empty when supplied")


@dataclass(frozen=True, slots=True)
class SearchSpaceExecutionBindingV1:
    """Opaque adapter-owned binding plus generic identity coordinates."""

    adapter_id: str
    binding_ref: str
    binding_digest: str
    candidate_id: str
    semantic_identity_digest: str
    native_binding: Any = field(repr=False, compare=False)
    execution_context: Mapping[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        for field_name in (
            "adapter_id",
            "binding_ref",
            "binding_digest",
            "candidate_id",
            "semantic_identity_digest",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a non-empty string")


@runtime_checkable
class SearchSpaceAdapter(Protocol):
    """Narrow port consumed by the generic feedback execution path."""

    adapter_id: str
    supported_frozen_profile_kinds: tuple[str, ...]

    def resolve_confirmation(
        self,
        kind: str,
        primary_binding: Any,
        context: Mapping[str, Any],
    ) -> ConfirmationResolutionV1: ...

    def validate_execution_binding(
        self,
        binding: SearchSpaceExecutionBindingV1,
    ) -> None: ...

    def effective_identity(
        self,
        binding: SearchSpaceExecutionBindingV1,
    ) -> Mapping[str, Any]: ...

    def execution_recipe(
        self,
        binding: SearchSpaceExecutionBindingV1,
    ) -> Mapping[str, Any]: ...


__all__ = [
    "ConfirmationResolutionKindV1",
    "ConfirmationResolutionV1",
    "SearchSpaceAdapter",
    "SearchSpaceExecutionBindingV1",
]
