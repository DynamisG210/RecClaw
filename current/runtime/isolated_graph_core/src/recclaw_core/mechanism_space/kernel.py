"""Closed public kernel for compiling mechanism programs and bindings."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import resources
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from .canonical import (
    StrictJsonError,
    canonical_bytes,
    deep_thaw,
    domain_sha256,
    load_json_bytes,
    snapshot_json,
)
from .catalog import (
    available_space_ids,
    catalog_digest,
    resolve_provider,
)
from .contracts import (
    CompileDiagnostic,
    CompileReportV1,
    CompileStatus,
    MechanismSpaceProviderV1,
    SpaceIdentity,
)

_RESOURCE_PACKAGE = "recclaw_core.mechanism_space.resources"
_ENVELOPE_SCHEMA = "mechanism_program_envelope_v1.schema.json"
_BINDING_SCHEMA = "candidate_execution_binding_v1.schema.json"


def capability_envelope_digest(capability_envelope: Mapping[str, Any]) -> str:
    """Return the identity of a development capability envelope.

    The caller-supplied digest is deliberately excluded from its own preimage.
    This binds the exact capability set and write roots without treating an
    author-provided fingerprint as a trust root.
    """

    snapshot = snapshot_json(dict(capability_envelope))
    snapshot.pop("envelope_digest", None)
    return domain_sha256(
        "recclaw.candidate-capability-envelope.v1",
        snapshot,
    )


def _resource_json(name: str) -> Any:
    raw = resources.files(_RESOURCE_PACKAGE).joinpath(name).read_bytes()
    return load_json_bytes(raw)


def _schema_diagnostics(instance: Any, schema: Any) -> tuple[CompileDiagnostic, ...]:
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:  # package-owned defect
        return (
            CompileDiagnostic(
                "PACKAGE_SCHEMA_INVALID",
                "package-owned JSON Schema is invalid",
                actual=type(exc).__name__,
            ),
        )
    validator = Draft202012Validator(schema)
    diagnostics: list[CompileDiagnostic] = []
    for error in sorted(validator.iter_errors(instance), key=lambda item: list(item.absolute_path)):
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        diagnostics.append(
            CompileDiagnostic(
                "PROGRAM_SCHEMA_INVALID",
                error.message,
                path=path if path != "/" else "",
            )
        )
    return tuple(diagnostics)


def compile_program_bytes(raw: bytes | bytearray | memoryview) -> CompileReportV1:
    try:
        envelope = load_json_bytes(raw)
    except StrictJsonError as exc:
        return CompileReportV1(
            CompileStatus.INVALID,
            diagnostics=(CompileDiagnostic(exc.code, str(exc)),),
        )
    if not isinstance(envelope, dict):
        return CompileReportV1(
            CompileStatus.INVALID,
            diagnostics=(
                CompileDiagnostic("PROGRAM_NOT_OBJECT", "mechanism program envelope must be an object"),
            ),
        )
    diagnostics = _schema_diagnostics(envelope, _resource_json(_ENVELOPE_SCHEMA))
    if diagnostics:
        return CompileReportV1(CompileStatus.INVALID, diagnostics=diagnostics)
    space_id = str(envelope["search_space_id"])
    try:
        provider = resolve_provider(space_id)
    except KeyError:
        return CompileReportV1(
            CompileStatus.UNSUPPORTED,
            diagnostics=(
                CompileDiagnostic(
                    "UNSUPPORTED_SEARCH_SPACE",
                    "search space is not registered in the closed catalog",
                    path="/search_space_id",
                    actual=space_id,
                ),
            ),
        )
    return _compile_with_provider(envelope, provider)


def compile_program(program: Mapping[str, Any]) -> CompileReportV1:
    try:
        snapshot = snapshot_json(dict(program))
        raw = canonical_bytes(snapshot)
    except (StrictJsonError, TypeError, ValueError) as exc:
        code = exc.code if isinstance(exc, StrictJsonError) else "PROGRAM_NOT_JSON"
        return CompileReportV1(
            CompileStatus.INVALID,
            diagnostics=(CompileDiagnostic(code, str(exc)),),
        )
    return compile_program_bytes(raw)


def _compile_with_provider(
    envelope: Mapping[str, Any], provider: MechanismSpaceProviderV1
) -> CompileReportV1:
    """Compile with an explicit provider for internal tests and catalog dispatch.

    Public entry points never accept a provider from the caller.
    """

    try:
        snapshot = snapshot_json(dict(envelope))
        identity = provider.identity()
        identity_diagnostics: list[CompileDiagnostic] = []
        expected_fields = {
            "search_space_id": identity.search_space_id,
            "search_space_digest": identity.search_space_digest,
            "family_id": identity.family_id,
            "family_version": identity.family_version,
        }
        for field, expected in expected_fields.items():
            actual = snapshot.get(field)
            if actual != expected:
                identity_diagnostics.append(
                    CompileDiagnostic(
                        "SPACE_IDENTITY_MISMATCH",
                        "program is not bound to the selected package-owned search space",
                        path=f"/{field}",
                        expected=expected,
                        actual=actual,
                    )
                )
        if identity_diagnostics:
            return CompileReportV1(
                CompileStatus.INVALID,
                diagnostics=tuple(identity_diagnostics),
                space_identity=identity,
            )
        return provider.compile(snapshot)
    except StrictJsonError as exc:
        return CompileReportV1(
            CompileStatus.INVALID,
            diagnostics=(CompileDiagnostic(exc.code, str(exc)),),
        )
    except Exception as exc:  # fail closed on package/provider defects
        return CompileReportV1(
            CompileStatus.INTERNAL_ERROR,
            diagnostics=(
                CompileDiagnostic(
                    "PROVIDER_INTERNAL_ERROR",
                    "package-owned mechanism-space provider failed",
                    actual=type(exc).__name__,
                ),
            ),
        )


def space_identity(search_space_id: str) -> SpaceIdentity:
    return resolve_provider(search_space_id).identity()


def prompt_projection(search_space_id: str) -> dict[str, Any]:
    provider = resolve_provider(search_space_id)
    return deep_thaw(snapshot_json(deep_thaw(provider.prompt_projection())))


def program_schema(search_space_id: str) -> dict[str, Any]:
    provider = resolve_provider(search_space_id)
    return deep_thaw(snapshot_json(deep_thaw(provider.program_schema())))


def validate_development_binding(
    program: Mapping[str, Any], binding: Mapping[str, Any]
) -> CompileReportV1:
    compiled = compile_program(program)
    if not compiled.is_valid:
        return CompileReportV1(
            compiled.status,
            operation="validate_development_binding",
            diagnostics=compiled.diagnostics,
            space_identity=compiled.space_identity,
            candidate_id=compiled.candidate_id,
            mechanism_semantics_digest=compiled.mechanism_semantics_digest,
            mechanism_program_digest=compiled.mechanism_program_digest,
            required_capabilities=compiled.required_capabilities,
            resolved_ir=compiled.resolved_ir,
        )
    try:
        binding_snapshot = snapshot_json(dict(binding))
    except (StrictJsonError, TypeError, ValueError) as exc:
        code = exc.code if isinstance(exc, StrictJsonError) else "BINDING_NOT_JSON"
        return _binding_failure(compiled, CompileStatus.INVALID, CompileDiagnostic(code, str(exc)))
    diagnostics = list(_schema_diagnostics(binding_snapshot, _resource_json(_BINDING_SCHEMA)))
    if diagnostics:
        return _binding_failure(compiled, CompileStatus.INVALID, *diagnostics)

    expected = {
        "candidate_id": compiled.candidate_id,
        "search_space_id": compiled.space_identity.search_space_id if compiled.space_identity else None,
        "search_space_digest": compiled.space_identity.search_space_digest if compiled.space_identity else None,
        "mechanism_program_digest": compiled.mechanism_program_digest,
        "mechanism_semantics_digest": compiled.mechanism_semantics_digest,
        "profile_ref": snapshot_json(dict(program))["profile_ref"],
    }
    for field, expected_value in expected.items():
        if binding_snapshot.get(field) != expected_value:
            diagnostics.append(
                CompileDiagnostic(
                    "BINDING_IDENTITY_MISMATCH",
                    "development binding identity does not match the compiled program",
                    path=f"/{field}",
                    expected=expected_value,
                    actual=binding_snapshot.get(field),
                )
            )
    capability_envelope = binding_snapshot["capability_envelope"]
    granted = set(capability_envelope["granted_capabilities"])
    required = set(compiled.required_capabilities)
    if granted != required:
        diagnostics.append(
            CompileDiagnostic(
                "CAPABILITY_SET_MISMATCH",
                "development binding must grant exactly the compiler-derived capabilities",
                path="/capability_envelope/granted_capabilities",
                expected=sorted(required),
                actual=sorted(granted),
            )
        )
    observed_envelope_digest = capability_envelope_digest(capability_envelope)
    if capability_envelope["envelope_digest"] != observed_envelope_digest:
        diagnostics.append(
            CompileDiagnostic(
                "CAPABILITY_ENVELOPE_DIGEST_MISMATCH",
                "capability envelope digest does not bind its exact content",
                path="/capability_envelope/envelope_digest",
                expected=observed_envelope_digest,
                actual=capability_envelope["envelope_digest"],
            )
        )
    candidate_root = f"recclaw_ext/generated/{compiled.candidate_id}/"
    run_id = str(binding_snapshot["run_id"])
    artifact_root = f"artifacts/{run_id}/"
    roots = binding_snapshot["capability_envelope"]["write_roots"]
    if roots != [candidate_root, artifact_root]:
        diagnostics.append(
            CompileDiagnostic(
                "WRITE_ROOT_WIDENING",
                "development writes must be limited to the candidate and run roots",
                path="/capability_envelope/write_roots",
                expected=[candidate_root, artifact_root],
                actual=roots,
            )
        )
    if diagnostics:
        status = (
            CompileStatus.CAPABILITY_DENIED
            if any(
                item.code
                in {
                    "CAPABILITY_SET_MISMATCH",
                    "CAPABILITY_ENVELOPE_DIGEST_MISMATCH",
                    "WRITE_ROOT_WIDENING",
                }
                for item in diagnostics
            )
            else CompileStatus.INVALID
        )
        return _binding_failure(compiled, status, *diagnostics)
    return CompileReportV1(
        compiled.status,
        operation="validate_development_binding",
        space_identity=compiled.space_identity,
        candidate_id=compiled.candidate_id,
        mechanism_semantics_digest=compiled.mechanism_semantics_digest,
        mechanism_program_digest=compiled.mechanism_program_digest,
        required_capabilities=compiled.required_capabilities,
        resolved_ir=compiled.resolved_ir,
    )


def validate_development_binding_bytes(
    program_raw: bytes | bytearray | memoryview,
    binding_raw: bytes | bytearray | memoryview,
) -> CompileReportV1:
    """Strict-byte entry point for file, API, and runner boundaries."""

    try:
        program = load_json_bytes(program_raw)
        binding = load_json_bytes(binding_raw)
    except StrictJsonError as exc:
        return CompileReportV1(
            CompileStatus.INVALID,
            operation="validate_development_binding",
            diagnostics=(CompileDiagnostic(exc.code, str(exc)),),
        )
    if not isinstance(program, dict) or not isinstance(binding, dict):
        return CompileReportV1(
            CompileStatus.INVALID,
            operation="validate_development_binding",
            diagnostics=(
                CompileDiagnostic(
                    "BINDING_INPUT_NOT_OBJECT",
                    "program and development binding must both be JSON objects",
                ),
            ),
        )
    return validate_development_binding(program, binding)


def _binding_failure(
    compiled: CompileReportV1,
    status: CompileStatus,
    *diagnostics: CompileDiagnostic,
) -> CompileReportV1:
    return CompileReportV1(
        status,
        operation="validate_development_binding",
        diagnostics=diagnostics,
        space_identity=compiled.space_identity,
        candidate_id=compiled.candidate_id,
        mechanism_semantics_digest=compiled.mechanism_semantics_digest,
        mechanism_program_digest=compiled.mechanism_program_digest,
        required_capabilities=compiled.required_capabilities,
        resolved_ir=compiled.resolved_ir,
    )


__all__ = [
    "available_space_ids",
    "capability_envelope_digest",
    "catalog_digest",
    "compile_program",
    "compile_program_bytes",
    "prompt_projection",
    "program_schema",
    "space_identity",
    "validate_development_binding",
    "validate_development_binding_bytes",
]
