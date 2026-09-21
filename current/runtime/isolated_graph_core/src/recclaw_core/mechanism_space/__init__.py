"""Versioned, development-only mechanism-space contracts.

The public entry points deliberately resolve package-owned providers from a
closed catalog.  Callers may submit programs and bindings, but may not inject
schemas, registries, compilers, or capability policies.
"""

from __future__ import annotations

from importlib import import_module

from .contracts import (
    CompileDiagnostic,
    CompileReportV1,
    CompileStatus,
    MechanismSpaceProviderV1,
    SpaceIdentity,
)

_KERNEL_EXPORTS = frozenset(
    {
        "available_space_ids",
        "capability_envelope_digest",
        "catalog_digest",
        "compile_program",
        "compile_program_bytes",
        "program_schema",
        "prompt_projection",
        "space_identity",
        "validate_development_binding",
        "validate_development_binding_bytes",
    }
)


def __getattr__(name: str):
    if name not in _KERNEL_EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(".kernel", __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _KERNEL_EXPORTS)

__all__ = [
    "CompileDiagnostic",
    "CompileReportV1",
    "CompileStatus",
    "MechanismSpaceProviderV1",
    "SpaceIdentity",
    "available_space_ids",
    "capability_envelope_digest",
    "catalog_digest",
    "compile_program",
    "compile_program_bytes",
    "program_schema",
    "prompt_projection",
    "space_identity",
    "validate_development_binding",
    "validate_development_binding_bytes",
]
