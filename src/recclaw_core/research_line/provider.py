"""Provider-backed Research Line callables.

The adapters keep the external Provider edge injectable.  They reuse the
existing fresh-R1 prompt, schema, and bounded call contract without creating a
second retry, service, or receipt store.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeAlias

import jsonschema

from recclaw_core.experiments.helix_abc_v1 import fresh_r1
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
)


ConfigSource: TypeAlias = Mapping[str, Any] | str | Path
ProviderCall: TypeAlias = Callable[..., fresh_r1.ProviderAttemptResult]

_RESOURCE_ROOT = Path(fresh_r1.__file__).resolve().parent / "resources"
_DEFAULT_PROPOSAL_TEMPLATE = (
    _RESOURCE_ROOT / "research_line_open_spec_proposal_prompt_v1.txt"
)
_DEFAULT_PROPOSAL_SCHEMA = (
    _RESOURCE_ROOT / "research_line_open_spec_proposal_response_v1.schema.json"
)
_DEFAULT_IMPLEMENTATION_TEMPLATE = (
    _RESOURCE_ROOT / "research_line_implementer_prompt_v1.txt"
)
_DEFAULT_IMPLEMENTATION_SCHEMA = (
    _RESOURCE_ROOT / "fresh_r1_implementation_response_v1.schema.json"
)

_SAFE_CONFIG_KEYS = (
    "source_ref",
    "source_digest",
    "config_ref",
    "config_digest",
    "selected_ref",
    "selected_digest",
    "release_ref",
    "release_digest",
    "endpoint_digest",
    "model",
    "transport",
    "request_mode",
    "selection",
)


def _read_text(source: str | Path) -> str:
    if isinstance(source, Path):
        return source.read_text(encoding="utf-8")
    return source


def _read_json(source: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(source, Mapping):
        return canonical_value(dict(source))
    value = json.loads(Path(source).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Provider schema source must contain a JSON object")
    return canonical_value(value)


def _config_identity(source: ConfigSource) -> dict[str, Any]:
    if isinstance(source, Mapping):
        identity = {
            str(key): canonical_value(source[key])
            for key in _SAFE_CONFIG_KEYS
            if key in source
            and (
                source[key] is None
                or isinstance(source[key], (bool, int, float, str))
            )
        }
        identity.setdefault("source_kind", "mapping")
        return canonical_value(identity)
    return canonical_value(
        {
            "source_kind": "path",
            "source_ref": str(Path(source)),
        }
    )


def _call_trace(
    *,
    kind: str,
    config_identity: Mapping[str, Any],
    logical_call_id: str,
    session_id: str,
    prompt: str,
    result: fresh_r1.ProviderAttemptResult,
) -> dict[str, Any]:
    call = result.call
    receipt = {
        "logical_call_id": logical_call_id,
        "proposal_generation_session_id": session_id,
        "request_digest": (
            getattr(call, "request_digest", None)
            or next(
                (
                    attempt.get("request_digest")
                    for attempt in result.attempts
                    if attempt.get("request_digest")
                ),
                None,
            )
        ),
        "response_digest": getattr(call, "response_digest", None),
        "release_digest": config_identity.get("release_digest"),
        "status": "SUCCESS" if call is not None else "FAILED",
        "latency_ms": getattr(call, "latency_ms", None),
        "returned_model": getattr(call, "returned_model", None),
    }
    usage = {
        "input_tokens": getattr(call, "input_tokens", 0),
        "cached_input_tokens": getattr(call, "cached_input_tokens", 0),
        "output_tokens": getattr(call, "output_tokens", 0),
        "billed_tokens": getattr(call, "total_tokens", 0),
    }
    return canonical_value(
        {
            "kind": kind,
            "config_identity": config_identity,
            "logical_call_id": logical_call_id,
            "session_id": session_id,
            "prompt_digest": sha256_digest(prompt),
            "receipt": receipt,
            "attempts": result.attempts,
            "usage": usage,
            "failure": result.failure,
        }
    )


def _record_trace(
    owner: Any,
    *,
    kind: str,
    logical_call_id: str,
    prompt: str,
    result: fresh_r1.ProviderAttemptResult,
) -> None:
    trace = _call_trace(
        kind=kind,
        config_identity=owner.config_identity,
        logical_call_id=logical_call_id,
        session_id=owner.session_id,
        prompt=prompt,
        result=result,
    )
    owner.last_call_trace = trace
    owner.call_traces = (*owner.call_traces, trace)


def _require_success(
    result: fresh_r1.ProviderAttemptResult,
    *,
    kind: str,
) -> Any:
    if result.call is None:
        raise fresh_r1.FreshR1Error(f"{kind} Provider call failed")
    return result.call.response


def _one_proposal(response: Mapping[str, Any], *, kind: str) -> Mapping[str, Any]:
    proposals = response.get("proposals")
    if not isinstance(proposals, list) or len(proposals) != 1:
        raise fresh_r1.FreshR1Error(
            f"{kind} Provider response must contain exactly one proposal"
        )
    proposal = proposals[0]
    if not isinstance(proposal, Mapping):
        raise fresh_r1.FreshR1Error(f"{kind} Provider proposal must be an object")
    return proposal


class ProviderResearchProducer:
    """A ResearchProducer backed by the existing bounded fresh-R1 call."""

    def __init__(
        self,
        *,
        config_source: ConfigSource,
        call_root: Path,
        session_id: str,
        total_token_ceiling: int | None = None,
        provider_call: ProviderCall | None = None,
        proposal_template_source: str | Path = _DEFAULT_PROPOSAL_TEMPLATE,
        proposal_schema_source: Mapping[str, Any] | str | Path = _DEFAULT_PROPOSAL_SCHEMA,
        proposal_schema_delta_source: Mapping[str, Any] | str | Path | None = None,
        proposal_schema_path: Path = _DEFAULT_PROPOSAL_SCHEMA,
        proposal_seed: int = 0,
        maximum_physical_attempts: int = fresh_r1.MAX_PHYSICAL_ATTEMPTS,
    ) -> None:
        self.config_identity = _config_identity(config_source)
        self.credential_config_path = (
            None
            if isinstance(config_source, Mapping)
            else Path(config_source).resolve()
        )
        self.call_root = Path(call_root)
        self.session_id = str(session_id)
        self.total_token_ceiling = int(
            fresh_r1.PROPOSAL_TOKEN_CEILING
            if total_token_ceiling is None
            else total_token_ceiling
        )
        if self.total_token_ceiling < fresh_r1.PROPOSAL_TOKEN_CEILING:
            raise ValueError("total_token_ceiling is below the proposal output ceiling")
        self.provider_call = provider_call or fresh_r1.bounded_provider_call
        self.proposal_template = _read_text(proposal_template_source)
        base_schema = _read_json(proposal_schema_source)
        self.proposal_schema = (
            base_schema
            if proposal_schema_delta_source is None
            else fresh_r1.derive_fresh_r1_proposal_schema(
                base_schema,
                _read_json(proposal_schema_delta_source),
            )
        )
        self.proposal_schema_path = Path(proposal_schema_path)
        self.proposal_seed = int(proposal_seed)
        self.maximum_physical_attempts = int(maximum_physical_attempts)
        self.last_call_trace: Mapping[str, Any] | None = None
        self.call_traces: tuple[Mapping[str, Any], ...] = ()

    def __call__(
        self,
        producer_role: str,
        context_view: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        return self._call(
            producer_role,
            context_view,
            logical_namespace=None,
        )

    def call_with_namespace(
        self,
        producer_role: str,
        context_view: Mapping[str, Any],
        *,
        logical_namespace: str,
    ) -> Mapping[str, Any]:
        """Run a policy-shadow call without colliding with the live call identity."""

        return self._call(
            producer_role,
            context_view,
            logical_namespace=logical_namespace,
        )

    def _call(
        self,
        producer_role: str,
        context_view: Mapping[str, Any],
        *,
        logical_namespace: str | None,
    ) -> Mapping[str, Any]:
        if producer_role not in DISCOVERY_PRODUCERS:
            raise ValueError("producer_role is outside the four-role portfolio")
        if not isinstance(context_view, Mapping):
            raise TypeError("context_view must be a mapping")
        context_ref = context_view.get("context_ref")
        if not isinstance(context_ref, str) or not context_ref:
            raise ValueError("context_view must contain context_ref")

        prompt = fresh_r1.render_proposal_prompt(
            self.proposal_template,
            side_identity=context_ref,
            logical_slot_id=f"{context_ref}:{producer_role}",
            proposal_seed=self.proposal_seed,
            producer_role=producer_role,
        )
        prompt += (
            "\n\nUse this complete role-scoped Research Context JSON as the "
            "authoritative input for this call:\n"
            + canonical_json_bytes(context_view).decode("utf-8")
        )
        namespace = f":{logical_namespace}" if logical_namespace else ""
        logical_call_id = (
            f"{self.session_id}{namespace}:{context_ref}:"
            f"{producer_role}:proposal"
        )
        result = self.provider_call(
            call_root=self.call_root / "proposals" / producer_role,
            schema_path=self.proposal_schema_path,
            logical_call_id=logical_call_id,
            session_id=self.session_id,
            prompt=prompt,
            token_ceiling=self.total_token_ceiling,
            output_token_ceiling=fresh_r1.PROPOSAL_TOKEN_CEILING,
            credential_config_path=self.credential_config_path,
            expected_transport_release_digest=self.config_identity.get(
                "release_digest"
            ),
            maximum_physical_attempts=self.maximum_physical_attempts,
        )
        _record_trace(
            self,
            kind="research_producer",
            logical_call_id=logical_call_id,
            prompt=prompt,
            result=result,
        )
        response = _require_success(result, kind="research proposal")
        if not isinstance(response, Mapping):
            raise fresh_r1.FreshR1Error("research proposal response must be an object")
        fresh_r1.validate_v4_response_contract(
            response,
            provider_schema=self.proposal_schema,
        )
        proposal = _one_proposal(response, kind="research proposal")
        if proposal.get("producer_role") != producer_role:
            raise fresh_r1.FreshR1Error(
                "Provider changed the preassigned Producer role"
            )
        normalized = dict(proposal)
        contract = normalized.get("execution_contract")
        if not isinstance(contract, Mapping):
            raise fresh_r1.FreshR1Error(
                "research proposal execution_contract must be an object"
            )
        config_json = contract.get("config_json")
        if not isinstance(config_json, str):
            raise fresh_r1.FreshR1Error(
                "research proposal execution_contract.config_json must be a string"
            )
        try:
            config = json.loads(config_json)
        except json.JSONDecodeError as error:
            raise fresh_r1.FreshR1Error(
                "research proposal execution_contract.config_json is not JSON"
            ) from error
        if not isinstance(config, Mapping):
            raise fresh_r1.FreshR1Error(
                "research proposal execution_contract.config_json must encode an object"
            )
        base_model_config = contract.get("base_model_config")
        if not isinstance(base_model_config, str) or not base_model_config:
            raise fresh_r1.FreshR1Error(
                "research proposal execution_contract.base_model_config must be a string"
            )
        for suffix in (".yaml", ".yml"):
            if base_model_config.endswith(suffix):
                base_model_config = base_model_config[: -len(suffix)]
                break
        if not base_model_config.isidentifier():
            raise fresh_r1.FreshR1Error(
                "research proposal execution_contract.base_model_config must be a bare model-config identifier"
            )
        normalized_contract = dict(contract)
        normalized_contract.pop("config_json")
        normalized_contract["base_model_config"] = base_model_config
        normalized_contract["config"] = canonical_value(config)
        normalized["execution_contract"] = normalized_contract
        return normalized


class ProviderImplementerGateway:
    """An ImplementerGateway backed by the existing bounded fresh-R1 call."""

    def __init__(
        self,
        *,
        config_source: ConfigSource,
        call_root: Path,
        session_id: str,
        total_token_ceiling: int | None = None,
        provider_call: ProviderCall | None = None,
        implementation_template_source: str | Path = _DEFAULT_IMPLEMENTATION_TEMPLATE,
        implementation_schema_source: Mapping[str, Any] | str | Path = _DEFAULT_IMPLEMENTATION_SCHEMA,
        implementation_schema_path: Path = _DEFAULT_IMPLEMENTATION_SCHEMA,
        maximum_physical_attempts: int = fresh_r1.MAX_PHYSICAL_ATTEMPTS,
    ) -> None:
        self.config_identity = _config_identity(config_source)
        self.credential_config_path = (
            None
            if isinstance(config_source, Mapping)
            else Path(config_source).resolve()
        )
        self.call_root = Path(call_root)
        self.session_id = str(session_id)
        self.total_token_ceiling = int(
            fresh_r1.IMPLEMENTATION_TOKEN_CEILING
            if total_token_ceiling is None
            else total_token_ceiling
        )
        if self.total_token_ceiling < fresh_r1.IMPLEMENTATION_TOKEN_CEILING:
            raise ValueError(
                "total_token_ceiling is below the implementation output ceiling"
            )
        self.provider_call = provider_call or fresh_r1.bounded_provider_call
        self.implementation_template = _read_text(implementation_template_source)
        self.implementation_schema = _read_json(implementation_schema_source)
        jsonschema.validators.validator_for(
            self.implementation_schema
        ).check_schema(self.implementation_schema)
        self.implementation_schema_path = Path(implementation_schema_path)
        self.maximum_physical_attempts = int(maximum_physical_attempts)
        self.last_call_trace: Mapping[str, Any] | None = None
        self.call_traces: tuple[Mapping[str, Any], ...] = ()

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(request, Mapping):
            raise TypeError("implementation request must be a mapping")
        candidate_id = str(request.get("blind_candidate_id", "request"))
        repair_attempt = int(request.get("repair_attempt", 0))
        logical_call_id = (
            f"{self.session_id}:{candidate_id}:implementation:{repair_attempt}"
        )
        prompt = fresh_r1.render_implementation_prompt(
            self.implementation_template,
            request,
        )
        result = self.provider_call(
            call_root=(
                self.call_root
                / "implementations"
                / candidate_id
                / f"revision_{repair_attempt:02d}"
            ),
            schema_path=self.implementation_schema_path,
            logical_call_id=logical_call_id,
            session_id=self.session_id,
            prompt=prompt,
            token_ceiling=self.total_token_ceiling,
            output_token_ceiling=fresh_r1.IMPLEMENTATION_TOKEN_CEILING,
            credential_config_path=self.credential_config_path,
            expected_transport_release_digest=self.config_identity.get(
                "release_digest"
            ),
            maximum_physical_attempts=self.maximum_physical_attempts,
        )
        _record_trace(
            self,
            kind="implementer_gateway",
            logical_call_id=logical_call_id,
            prompt=prompt,
            result=result,
        )
        response = _require_success(result, kind="implementation")
        if not isinstance(response, Mapping):
            raise fresh_r1.FreshR1Error("implementation response must be an object")
        jsonschema.validate(
            canonical_value(response),
            canonical_value(self.implementation_schema),
        )
        return dict(_one_proposal(response, kind="implementation"))


__all__ = [
    "ProviderImplementerGateway",
    "ProviderResearchProducer",
]
