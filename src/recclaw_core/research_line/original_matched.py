"""Original policy adapter for the shared strict BL-ICF execution substrate."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping, Sequence

import jsonschema

from recclaw_core.mechanism_space import (
    compile_program,
    program_schema,
    prompt_projection,
)
from recclaw_core.experiments.helix_abc_v1 import fresh_r1
from recclaw_core.experiments.helix_abc_v1.canary_broker import (
    CanaryBrokerCallV1,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    bytes_sha256,
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.controllers import (
    OriginalRuntimeAdapterV1,
)
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    DEVELOPMENT_EVALUATOR,
    DEVELOPMENT_SPLIT,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    StrongStaticRouterV1,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
    CandidateProposalV2,
    CandidateProposalV3,
    CandidateProposalV4,
    RouteTraceV1,
    SearchUtilityFeaturesV1,
)
from recclaw_core.experiments.helix_abc_v1.provider_model_routing import (
    select_provider_model,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (
    LabApiCallReceiptV1,
)
from recclaw_core.research_line.interfaces import project_provider_context_view

from .campaign import CampaignState, ResearchCampaign
from .provider import (
    ProviderMechanismProgramProposalV1,
    ProviderResearchProducer,
    _FROZEN_QUALIFICATION_BASE_MODEL_CONFIGS,
    _bl_icf_program_proposal_v1,
    _expected_parent_binding_for_proposal,
    _json_type_for_provider_wire,
    _parent_program_for_proposal,
    _provider_wire_program_schema,
    _provider_wire_with_explicit_json_types,
    _read_json,
    _record_trace,
    _require_success,
    _write_role_bound_schema,
)
from .single_parent_search import (
    PARENT_BASE_MODEL_CONFIG,
    bound_parent_from_context,
    focused_mechanism_language,
    is_bl_icf_single_parent_context,
)
from .standalone import (
    StandaloneControllerInterface,
    StandaloneResearchComposition,
    StandaloneResearchConfig,
    compose_standalone_campaign,
)


_RESOURCE_ROOT = Path(fresh_r1.__file__).resolve().parent / "resources"
_DEFAULT_BATCH_SCHEMA = (
    _RESOURCE_ROOT
    / "original_matched_bl_icf_program_proposal_response_v1.schema.json"
)
_ORIGINAL_STATE_KEY = "original_matched_controller_state"
_ORIGINAL_TRANSITION_KEY = "original_matched_last_transition"
# Original is one full GPT-5.4 physical response containing four proposals, not
# four compact 6k producer calls.  It shares the evidence-bounded 32k/80k full
# model envelope with Frontier while preserving its four-proposal schema; the
# independent mini roles retain their compact 6k contract.
ORIGINAL_MATCHED_BATCH_OUTPUT_TOKEN_CEILING = 32_000
ORIGINAL_MATCHED_TOTAL_TOKEN_CEILING = 80_000
_HISTORICAL_SUCCESS_REPLAY_SCHEMA = (
    "recclaw.original-matched.historical-success-replay.v1"
)
_ORIGINAL_RESPONSES_TRANSPORT_NAMESPACE = "original-matched-responses-v1"
_ORIGINAL_RESPONSES_WIRE_API = "responses"
_IDENTITY_COLLISION_MESSAGE = (
    "logical API call identity differs from stored bytes"
)


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _sealed_response_schema(sealed_request: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = sealed_request.get("request_payload")
    if not isinstance(payload, Mapping):
        raise fresh_r1.FreshR1Error(
            "Original historical replay lacks the sealed request payload"
        )
    response_format = payload.get("response_format")
    if isinstance(response_format, Mapping):
        json_schema = response_format.get("json_schema")
        schema = json_schema.get("schema") if isinstance(json_schema, Mapping) else None
    else:
        text = payload.get("text")
        text_format = text.get("format") if isinstance(text, Mapping) else None
        schema = text_format.get("schema") if isinstance(text_format, Mapping) else None
    if not isinstance(schema, Mapping):
        raise fresh_r1.FreshR1Error(
            "Original historical replay lacks the sealed response schema"
        )
    return canonical_value(dict(schema))


def _immutable_historical_original_success(
    *,
    call_root: Path,
    logical_call_id: str,
    session_id: str,
    request_digest: str,
    prompt: str,
    expected_prompt_digest: str,
    response_schema: Mapping[str, Any],
    expected_schema_digest: str,
) -> tuple[fresh_r1.ProviderAttemptResult, Mapping[str, Any]]:
    """Read one sealed Original SUCCESS without constructing a new broker."""

    if sha256_digest(prompt) != expected_prompt_digest:
        raise fresh_r1.FreshR1Error(
            "Original historical replay prompt identity differs from the sealed trace"
        )
    canonical_schema = canonical_value(dict(response_schema))
    if sha256_digest(canonical_schema) != expected_schema_digest:
        raise fresh_r1.FreshR1Error(
            "Original historical replay schema identity differs from the sealed trace"
        )
    canonical_root = call_root.resolve()
    databases = tuple(
        sorted(canonical_root.glob("physical_attempt_*/broker.sqlite3"))
    )
    if len(databases) != 1:
        raise fresh_r1.FreshR1Error(
            "Original historical replay requires exactly one physical broker source"
        )
    database = databases[0].resolve()
    if database.parent.parent != canonical_root or not database.parent.name.startswith(
        "physical_attempt_"
    ):
        raise fresh_r1.FreshR1Error(
            "Original historical replay broker source escaped its call root"
        )
    before_bytes = database.read_bytes()
    connection = sqlite3.connect(
        f"{database.as_uri()}?mode=ro&immutable=1",
        uri=True,
    )
    try:
        rows = connection.execute(
            "SELECT request_digest, response_digest, response_json, "
            "input_tokens, cached_input_tokens, output_tokens, total_tokens, "
            "latency_ms, returned_model, status, error_type, "
            "proposal_generation_session_id, receipt_digest, receipt_json, "
            "broker_release_digest FROM calls WHERE logical_call_id=?",
            (logical_call_id,),
        ).fetchall()
    except sqlite3.Error as error:
        raise fresh_r1.FreshR1Error(
            "Original historical replay broker evidence is unreadable"
        ) from error
    finally:
        connection.close()
    if len(rows) != 1:
        raise fresh_r1.FreshR1Error(
            "Original historical replay requires exactly one logical SUCCESS source"
        )
    (
        stored_request_digest,
        response_digest,
        response_json,
        input_tokens,
        cached_input_tokens,
        output_tokens,
        total_tokens,
        latency_ms,
        returned_model,
        status,
        error_type,
        stored_session_id,
        receipt_digest,
        receipt_json,
        broker_release_digest,
    ) = rows[0]
    if (
        stored_request_digest != request_digest
        or stored_session_id != session_id
        or status != "SUCCESS"
        or error_type is not None
    ):
        raise fresh_r1.FreshR1Error(
            "Original historical replay call identity or status differs"
        )
    if not all(
        _is_sha256(value)
        for value in (
            stored_request_digest,
            response_digest,
            receipt_digest,
            broker_release_digest,
        )
    ):
        raise fresh_r1.FreshR1Error(
            "Original historical replay contains an invalid digest"
        )
    try:
        response = json.loads(str(response_json))
        receipt_payload = json.loads(str(receipt_json))
    except json.JSONDecodeError as error:
        raise fresh_r1.FreshR1Error(
            "Original historical replay contains invalid JSON evidence"
        ) from error
    if not isinstance(response, Mapping) or not isinstance(receipt_payload, Mapping):
        raise fresh_r1.FreshR1Error(
            "Original historical replay response or receipt is not an object"
        )
    if sha256_digest(response) != response_digest:
        raise fresh_r1.FreshR1Error(
            "Original historical replay response digest differs"
        )
    try:
        stored_receipt = LabApiCallReceiptV1(**dict(receipt_payload))
        expected_receipt = LabApiCallReceiptV1.create(
            logical_call_id=logical_call_id,
            proposal_generation_session_id=session_id,
            request_digest=request_digest,
            response_digest=str(response_digest),
            release_digest=str(broker_release_digest),
            status="SUCCESS",
            error_type=None,
            error_detail_digest=None,
            http_status=200,
            latency_ms=int(latency_ms),
        )
    except (TypeError, ValueError) as error:
        raise fresh_r1.FreshR1Error(
            "Original historical replay receipt is invalid"
        ) from error
    if (
        stored_receipt.to_dict() != expected_receipt.to_dict()
        or stored_receipt.receipt_digest != receipt_digest
        or stored_receipt.release_digest != broker_release_digest
    ):
        raise fresh_r1.FreshR1Error(
            "Original historical replay receipt identity differs"
        )
    try:
        historical_input = int(input_tokens)
        historical_cached = int(cached_input_tokens)
        historical_output = int(output_tokens)
        historical_total = int(total_tokens)
        historical_latency = int(latency_ms)
    except (TypeError, ValueError, OverflowError) as error:
        raise fresh_r1.FreshR1Error(
            "Original historical replay usage is invalid"
        ) from error
    if (
        min(
            historical_input,
            historical_cached,
            historical_output,
            historical_total,
            historical_latency,
        )
        < 0
        or historical_cached > historical_input
        or historical_total != historical_input + historical_output
    ):
        raise fresh_r1.FreshR1Error(
            "Original historical replay usage accounting differs"
        )
    requested_model = select_provider_model(
        "original_matched_proposal"
    ).requested_model
    try:
        jsonschema.validate(canonical_value(response), canonical_schema)
    except jsonschema.ValidationError as error:
        raise fresh_r1.FreshR1Error(
            "Original historical replay failed the current strict schema"
        ) from error
    proposals = response.get("proposals")
    if not isinstance(proposals, list) or len(proposals) != len(DISCOVERY_PRODUCERS):
        raise fresh_r1.FreshR1Error(
            "Original historical replay must contain exactly four proposals"
        )
    after_bytes = database.read_bytes()
    if after_bytes != before_bytes:
        raise fresh_r1.FreshR1Error(
            "Original historical replay mutated its broker source"
        )
    historical_usage = canonical_value(
        {
            "cached_input_tokens": historical_cached,
            "input_tokens": historical_input,
            "latency_ms": historical_latency,
            "output_tokens": historical_output,
            "total_tokens": historical_total,
        }
    )
    provenance = canonical_value(
        {
            "broker_release_digest": str(broker_release_digest),
            "current_prompt_digest": expected_prompt_digest,
            "current_schema_digest": expected_schema_digest,
            "historical_usage": historical_usage,
            "logical_call_id": logical_call_id,
            "proposal_generation_session_id": session_id,
            "receipt": stored_receipt.to_dict(),
            "receipt_digest": str(receipt_digest),
            "request_digest": request_digest,
            "response_digest": str(response_digest),
            "returned_model": str(returned_model),
            "schema": _HISTORICAL_SUCCESS_REPLAY_SCHEMA,
            "source_broker_path": str(database),
            "source_broker_sha256": bytes_sha256(before_bytes),
            "source_physical_attempt_count": 1,
        }
    )
    call = CanaryBrokerCallV1(
        logical_call_id=logical_call_id,
        request_digest=request_digest,
        response_digest=str(response_digest),
        response=canonical_value(response),
        input_tokens=0,
        cached_input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        latency_ms=0,
        returned_model=str(returned_model),
    )
    attempt = canonical_value(
        {
            "billed_tokens": 0,
            "cached_input_tokens": 0,
            "historical_provenance": provenance,
            "input_tokens": 0,
            "latency_ms": 0,
            "ordinal": 1,
            "output_tokens": 0,
            "physical_call_count": 0,
            "provider_role": "original_matched_proposal",
            "release_digest": str(broker_release_digest),
            "request_digest": request_digest,
            "requested_model": requested_model,
            "returned_model": str(returned_model),
            "sealed_request_replay": True,
            "status": "SUCCESS",
            "usage_charge_basis": "HISTORICAL_SUCCESS_ALREADY_BILLED",
        }
    )
    sealed_request = canonical_value(
        {
            "historical_provenance": provenance,
            "logical_call_id": logical_call_id,
            "proposal_generation_session_id": session_id,
            "request_digest": request_digest,
            "schema": "recclaw.provider.sealed-request-replay.v1",
        }
    )
    return (
        fresh_r1.ProviderAttemptResult(
            call=call,
            attempts=(attempt,),
            failure=None,
            sealed_request=sealed_request,
        ),
        provenance,
    )


def render_original_matched_prompt(
    *,
    round_index: int,
    search_seed: int,
    baseline_context: Mapping[str, Any],
    original_state: Mapping[str, Any],
    discovery_generation: int = 0,
    compiler_failure_summaries: Mapping[str, Any] | None = None,
    provider_context_view: Mapping[str, Any] | None = None,
) -> str:
    """Render Original's policy over the active BL-ICF mechanism language."""

    single_parent_mode = is_bl_icf_single_parent_context(baseline_context)
    if single_parent_mode:
        language = focused_mechanism_language()
        counts = language["projection_counts"]
        language_attestation = {
            "search_space_id": language["compiler_space"]["search_space_id"],
            "known_affordance_count": counts["projected_known_affordances"],
            "parent_foundation_count": counts["projected_parent_foundation"],
            "projection_digest": language["projection_digest"],
        }
        provider_state = (
            provider_context_view.get("state")
            if isinstance(provider_context_view, Mapping)
            else None
        )
        provider_state = provider_state if isinstance(provider_state, Mapping) else {}
        lineage_binding = provider_state.get("lineage_parent_binding")
        lineage_program = provider_state.get("lineage_parent_mechanism_program")
        active_parent_projection = (
            "\nActive construction parent from Research Context:\n"
            + canonical_json_bytes(
                {
                    "binding": lineage_binding,
                    "mechanism_program": lineage_program,
                }
            ).decode("utf-8")
            if isinstance(lineage_binding, Mapping)
            and isinstance(lineage_program, Mapping)
            else ""
        )
        parent_direction = (
            "\nLightGCN++ remains the frozen root and paired comparator. Every "
            "proposal must be one faithful causal delta from the active construction "
            "parent shown below when present, otherwise from the exact frozen "
            "LightGCN++ root. Inherit every unchanged mechanism from that active "
            "parent. A genuinely improved measured descendant may become the next "
            "construction parent without replacing the frozen root comparator. Use "
            "the focused affordances as implementation material rather than a closed "
            "candidate catalog. Architecture rewrites and typed custom models remain "
            "available when their expected effect justifies the extra implementation "
            "and training cost."
            + active_parent_projection
        )
        language_heading = "\nFocused single-parent BL-ICF mechanism language:\n"
    else:
        language = prompt_projection("BL_ICF_MECHANISM_SPACE_V1")
        language_attestation = {
            "search_space_id": "BL_ICF_MECHANISM_SPACE_V1",
            "search_space_size": sum(
                len(axis["primitives"]) for axis in language["axes"]
            ),
            "architecture_operator_count": len(
                language["architecture_operators"]
            ),
            "projection_digest": language["space_identity"][
                "search_space_digest"
            ],
        }
        parent_direction = ""
        language_heading = "\nComplete search language BL_ICF_MECHANISM_SPACE_V1:\n"
    generation_line = (
        ""
        if discovery_generation == 0
        else f"Discovery generation: {int(discovery_generation)}.\n"
    )
    compiler_feedback = (
        ""
        if discovery_generation == 0 or not compiler_failure_summaries
        else (
            "Prior discovery generation compiler failures by transport role; "
            "correct these exact diagnostics:\n"
            + canonical_json_bytes(dict(compiler_failure_summaries)).decode("utf-8")
            + "\n"
        )
    )
    return (
        "You are the Original RecClaw proposal controller for a validation-only "
        "recommender search. Do not use tools or inspect files. Return exactly "
        "four diverse canonical mechanism_program proposals through the supplied "
        "strict JSON schema. Preserve Original's preference for novel runnable "
        "families, avoid recently executed semantics, and use Original's prior "
        "outcomes. Do not use multi-role Research policy, Research Router utility "
        "scores, Meta policy, or Evidence authority. This is proposal generation, "
        "not adjudication.\n"
        f"Search seed: {int(search_seed)}. Round: {int(round_index)}.\n"
        + generation_line
        + compiler_feedback
        + "Frozen baseline context:\n"
        + canonical_json_bytes(dict(baseline_context)).decode("utf-8")
        + "\nExact available dependencies:\n"
        + canonical_json_bytes(list(fresh_r1.AVAILABLE_DEPENDENCIES)).decode(
            "utf-8"
        )
        + "\nExact protocol requirements:\n"
        + canonical_json_bytes(list(fresh_r1.PROTOCOL_REQUIREMENTS)).decode(
            "utf-8"
        )
        + "\nExact resolver budget limits:\n"
        + canonical_json_bytes(fresh_r1.BUDGET_LIMITS).decode("utf-8")
        + (
            "\nEvery compatibility_requirements item MUST be copied from the exact "
            "protocol requirements, every required_dependencies item MUST be "
            "copied from the exact available dependencies, and required_budget "
            "MUST NOT exceed any corresponding limit."
        )
        + (
            "\nBefore returning, construct every program to satisfy the package "
            "compiler's existing validity rules. Every construction mode requires "
            "exactly one CORE entry in changed_slots, exactly one ENCODER component, "
            "exactly one SCORE_HEAD component, an acyclic typed component graph, "
            "and only final component IDs in operator targets/replacements and "
            "ablations. COMPOSITION permits at most one SUPPORT changed slot. "
            "CUSTOM_MODEL requires construction_mode CUSTOM_MODEL, at least one "
            "declared and instantiated custom component, and a synthesize_custom_model "
            "operator; custom component instances must use an empty parameters object "
            "because all custom parameters belong in the mathematical definition, "
            "allowed_read_roles must include every DATA role read directly by that "
            "custom component, and minimal_implementation.entrypoint_role must also "
            "appear in minimal_implementation.file_roles. A slot: target must also be "
            "declared in removed_slots. Every primitive instance must satisfy its "
            "primitive parameter_schema exactly. Do not return a program with missing "
            "required inputs, undeclared data reads, unknown component references, "
            "frozen-protocol changes, or more changed slots than the selected "
            "construction mode permits. These are compiler-validity constraints over "
            "the complete language, not a fixed-catalog restriction."
        )
        + parent_direction
        + "\nOriginal planner state:\n"
        + canonical_json_bytes(dict(original_state)).decode("utf-8")
        + "\nSearch language attestation:\n"
        + canonical_json_bytes(language_attestation).decode("utf-8")
        + language_heading
        + canonical_json_bytes(language).decode("utf-8")
    )


class OriginalMatchedProviderProducer(ProviderResearchProducer):
    """One Original refresh projected onto the shared four-slot transport."""

    def __init__(
        self,
        *,
        controller: OriginalRuntimeAdapterV1,
        baseline_context: Mapping[str, Any],
        batch_schema_path: Path = _DEFAULT_BATCH_SCHEMA,
        allow_historical_success_replay: bool = False,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault(
            "proposal_output_token_ceiling_per_physical_call",
            ORIGINAL_MATCHED_BATCH_OUTPUT_TOKEN_CEILING,
        )
        super().__init__(**kwargs)
        self.controller = controller
        self.baseline_context = canonical_value(dict(baseline_context))
        self.single_parent_mode = is_bl_icf_single_parent_context(
            self.baseline_context
        )
        self.batch_schema_template_path = Path(batch_schema_path)
        self.batch_schema_path = self.batch_schema_template_path
        self.batch_schema = _read_json(self.batch_schema_template_path)
        jsonschema.validators.validator_for(self.batch_schema).check_schema(
            self.batch_schema
        )
        if not isinstance(allow_historical_success_replay, bool):
            raise TypeError("allow_historical_success_replay must be a boolean")
        self.allow_historical_success_replay = allow_historical_success_replay
        self._historical_replay_by_logical_call: dict[
            str, Mapping[str, Any]
        ] = {}
        self._portfolio_context_key: tuple[str, int, str | None] | None = None
        self._portfolio_results: dict[
            str, ProviderMechanismProgramProposalV1
        ] = {}
        self._portfolio_failures: dict[str, fresh_r1.FreshR1Error] = {}
        self.priority_by_program_digest: dict[str, str] = {}

    def configure_resume_provider_replay(
        self,
        traces: Sequence[Mapping[str, Any]],
    ) -> None:
        """Bind only the observed zero-physical Original identity collision."""

        super().configure_resume_provider_replay(traces)
        historical: dict[str, Mapping[str, Any]] = {}
        for trace in traces:
            if not isinstance(trace, Mapping) or trace.get("kind") != (
                "original_matched_producer"
            ):
                continue
            logical_call_id = trace.get("logical_call_id")
            session_id = trace.get("session_id")
            prompt_digest = trace.get("prompt_digest")
            receipt = trace.get("receipt")
            sealed_request = trace.get("sealed_request")
            attempts = trace.get("attempts")
            if not (
                isinstance(logical_call_id, str)
                and logical_call_id
                and session_id == self.session_id
                and _is_sha256(prompt_digest)
                and isinstance(receipt, Mapping)
                and isinstance(sealed_request, Mapping)
                and isinstance(attempts, (tuple, list))
                and len(attempts) == 1
                and isinstance(attempts[0], Mapping)
            ):
                continue
            attempt = attempts[0]
            historical_request_digest = receipt.get("request_digest")
            current_request_digest = sealed_request.get("request_digest")
            exact_collision = (
                receipt.get("logical_call_id") == logical_call_id
                and receipt.get("proposal_generation_session_id") == session_id
                and receipt.get("status") == "FAILED"
                and sealed_request.get("logical_call_id") == logical_call_id
                and sealed_request.get("proposal_generation_session_id") == session_id
                and _is_sha256(historical_request_digest)
                and _is_sha256(current_request_digest)
                and historical_request_digest != current_request_digest
                and attempt.get("status") == "FAILED"
                and attempt.get("failure_class") == "LOCAL_BROKER_FAILURE"
                and attempt.get("physical_call_count") == 0
                and attempt.get("message") == _IDENTITY_COLLISION_MESSAGE
                and attempt.get("request_digest") == historical_request_digest
            )
            if not exact_collision:
                continue
            sealed_schema = _sealed_response_schema(sealed_request)
            replay = canonical_value(
                {
                    "current_request_digest": current_request_digest,
                    "expected_prompt_digest": prompt_digest,
                    "expected_schema_digest": sha256_digest(sealed_schema),
                    "historical_request_digest": historical_request_digest,
                    "logical_call_id": logical_call_id,
                    "session_id": session_id,
                    "source_trace_digest": sha256_digest(trace),
                }
            )
            prior = historical.get(logical_call_id)
            if prior is not None and prior != replay:
                raise fresh_r1.FreshR1Error(
                    "prepared checkpoint contains conflicting Original replay identities"
                )
            historical[logical_call_id] = replay
        self._historical_replay_by_logical_call = historical

    def _materialize_batch_schema(self) -> Path:
        if self.batch_schema_path != self.batch_schema_template_path:
            return self.batch_schema_path
        schema = canonical_value(_read_json(self.batch_schema_template_path))
        proposal_properties = schema["properties"]["proposals"]["items"][
            "properties"
        ]
        proposal_properties["mechanism_program"] = _provider_wire_program_schema(
            program_schema("BL_ICF_MECHANISM_SPACE_V1")
        )
        proposal_properties["implementation_research"]["properties"][
            "base_model_config"
        ] = {
            "enum": (
                [PARENT_BASE_MODEL_CONFIG]
                if self.single_parent_mode
                else list(_FROZEN_QUALIFICATION_BASE_MODEL_CONFIGS)
            ),
            "type": "string",
        }
        profile = canonical_value(dict(self.frozen_profile_ref))
        proposal_properties["mechanism_program"]["properties"]["profile_ref"] = {
            "additionalProperties": False,
            "properties": {
                key: {
                    "const": value,
                    "type": _json_type_for_provider_wire(value),
                }
                for key, value in sorted(profile.items())
            },
            "required": sorted(profile),
            "type": "object",
        }
        self.batch_schema = _provider_wire_with_explicit_json_types(schema)
        self.batch_schema_path = _write_role_bound_schema(
            self.batch_schema,
            self.call_root
            / "proposal-schemas"
            / "original-matched"
            / "batch.schema.json",
        )
        return self.batch_schema_path

    def _cached_portfolio(
        self, *, context_view: Mapping[str, Any]
    ) -> dict[str, ProviderMechanismProgramProposalV1]:
        provider_context_view = project_provider_context_view(context_view)
        expected_parent = _expected_parent_binding_for_proposal(
            context_view=context_view,
            provider_context_view=provider_context_view,
        )
        if self.single_parent_mode:
            frozen_parent = bound_parent_from_context(self.baseline_context)
            if expected_parent is None and frozen_parent is not None:
                expected_parent = frozen_parent["binding"]
            parent_program: Mapping[str, Any] | None = _parent_program_for_proposal(
                provider_context_view=provider_context_view,
                expected_parent_binding=(
                    expected_parent if isinstance(expected_parent, Mapping) else None
                ),
            )
            if (
                parent_program is None
                and frozen_parent is not None
                and expected_parent == frozen_parent["binding"]
            ):
                parent_program = frozen_parent["mechanism_program"]
            if not isinstance(expected_parent, Mapping) or parent_program is None:
                raise fresh_r1.FreshR1Error(
                    "Original single-parent search requires the active construction "
                    "parent binding and mechanism program"
                )
        else:
            parent_program = None
        results: dict[str, ProviderMechanismProgramProposalV1] = {}
        seen_programs: set[str] = set()
        self._portfolio_failures.clear()
        self.priority_by_program_digest.clear()
        for role, cached in zip(
            DISCOVERY_PRODUCERS,
            self.controller.cached_proposals,
            strict=True,
        ):
            wire = dict(cached["wire_proposal"])
            wire["producer_role"] = role
            try:
                proposal = _bl_icf_program_proposal_v1(
                    wire,
                    producer_role=role,
                    expected_parent_binding=expected_parent,
                    parent_program=parent_program,
                    force_parent_binding=self.single_parent_mode,
                )
                report = compile_program(proposal.mechanism_program)
                program_digest = str(report.mechanism_program_digest)
                if program_digest in seen_programs:
                    raise fresh_r1.FreshR1Error(
                        "Original-matched refresh contains duplicate canonical programs"
                    )
            except fresh_r1.FreshR1Error as error:
                self._portfolio_failures[role] = error
                continue
            seen_programs.add(program_digest)
            results[role] = proposal
            self.priority_by_program_digest[program_digest] = str(
                cached["original_priority"]
            )
        return results

    def _refresh_portfolio(
        self,
        *,
        context_view: Mapping[str, Any],
        transport_round_index: int,
        planner_round_index: int,
        discovery_generation: int,
        logical_namespace: str | None,
    ) -> dict[str, ProviderMechanismProgramProposalV1]:
        self._materialize_batch_schema()
        context_ref = str(context_view["context_ref"])
        provider_context_view = project_provider_context_view(context_view)
        compiler_failure_summaries = {
            role: report
            for role, error in self._portfolio_failures.items()
            if isinstance(
                report := getattr(error, "compiler_report", None),
                Mapping,
            )
        }
        prompt = render_original_matched_prompt(
            round_index=planner_round_index,
            search_seed=self.proposal_seed,
            baseline_context=self.baseline_context,
            original_state=self.controller.state_projection(),
            discovery_generation=discovery_generation,
            compiler_failure_summaries=compiler_failure_summaries,
            provider_context_view=provider_context_view,
        )
        generation_suffix = (
            ""
            if discovery_generation == 0
            else f":generation-{discovery_generation:04d}"
        )
        namespace_suffix = f":{logical_namespace}" if logical_namespace else ""
        logical_call_id = (
            f"{self.session_id}{namespace_suffix}:original-matched:"
            f"round-{transport_round_index:04d}{generation_suffix}:proposal"
        )
        historical_call_root = (
            self.call_root
            / "proposals"
            / "original-matched"
            / f"round_{transport_round_index:04d}"
        )
        live_call_root = (
            self.call_root
            / "proposals"
            / _ORIGINAL_RESPONSES_TRANSPORT_NAMESPACE
            / f"round_{transport_round_index:04d}"
        )
        if discovery_generation > 0:
            generation_root = f"generation_{discovery_generation:04d}"
            historical_call_root /= generation_root
            live_call_root /= generation_root
        if logical_namespace is not None:
            historical_call_root /= logical_namespace
            live_call_root /= logical_namespace
        replay = self._historical_replay_by_logical_call.pop(
            logical_call_id, None
        )
        replay_provenance: Mapping[str, Any] | None = None
        if replay is not None:
            if not self.allow_historical_success_replay:
                raise fresh_r1.FreshR1Error(
                    "Original historical success replay was not authorized"
                )
            self._resume_request_digest_by_logical_call.pop(
                logical_call_id, None
            )
            result, replay_provenance = _immutable_historical_original_success(
                call_root=historical_call_root,
                logical_call_id=logical_call_id,
                session_id=self.session_id,
                request_digest=str(replay["historical_request_digest"]),
                prompt=prompt,
                expected_prompt_digest=str(replay["expected_prompt_digest"]),
                response_schema=self.batch_schema,
                expected_schema_digest=str(replay["expected_schema_digest"]),
            )
        else:
            result = self.provider_call(
                call_root=live_call_root,
                schema_path=self.batch_schema_path,
                logical_call_id=logical_call_id,
                session_id=self.session_id,
                prompt=prompt,
                provider_role="original_matched_proposal",
                requested_model=select_provider_model(
                    "original_matched_proposal"
                ).requested_model,
                token_ceiling=ORIGINAL_MATCHED_TOTAL_TOKEN_CEILING,
                output_token_ceiling=self._budgeted_output_ceiling(context_ref),
                expected_proposal_count=4,
                credential_config_path=self.credential_config_path,
                expected_transport_release_digest=self.config_identity.get(
                    "release_digest"
                ),
                maximum_physical_attempts=self.maximum_physical_attempts,
                wire_api=_ORIGINAL_RESPONSES_WIRE_API,
            )
        self._charge_output_tokens(context_ref, result)
        _record_trace(
            self,
            kind="original_matched_producer",
            logical_call_id=logical_call_id,
            prompt=prompt,
            result=result,
        )
        if replay_provenance is not None:
            trace = dict(self.last_call_trace or {})
            receipt = dict(trace.get("receipt", {}))
            receipt.update(
                {
                    "historical_success_replay": True,
                    "receipt_digest": replay_provenance["receipt_digest"],
                    "release_digest": replay_provenance[
                        "broker_release_digest"
                    ],
                }
            )
            trace["receipt"] = receipt
            trace["historical_success_replay"] = replay_provenance
            canonical_trace = canonical_value(trace)
            self.last_call_trace = canonical_trace
            self.call_traces = (*self.call_traces[:-1], canonical_trace)
        response = _require_success(result, kind="Original-matched proposal")
        jsonschema.validate(canonical_value(response), self.batch_schema)
        raw_proposals = response.get("proposals")
        if not isinstance(raw_proposals, list) or len(raw_proposals) != 4:
            raise fresh_r1.FreshR1Error(
                "Original-matched response must contain exactly four proposals"
            )
        cached: list[Mapping[str, Any]] = []
        for raw in raw_proposals:
            wire = dict(raw)
            priority = str(wire.pop("original_priority"))
            cached.append(
                {
                    "wire_proposal": canonical_value(wire),
                    "original_priority": priority,
                }
            )
        self.controller.install_proposals(
            round_index=planner_round_index,
            proposals=tuple(cached),
        )
        return self._cached_portfolio(context_view=context_view)

    def _call_original(
        self,
        producer_role: str,
        context_view: Mapping[str, Any],
        *,
        logical_namespace: str | None,
    ) -> ProviderMechanismProgramProposalV1:
        if producer_role not in DISCOVERY_PRODUCERS:
            raise ValueError("producer_role is outside the four transport slots")
        context_ref = str(context_view.get("context_ref", ""))
        context_round_index = int(context_view["round_index"])
        scientific_memory = context_view.get("scientific_memory")
        discovery_generation = int(
            scientific_memory.get("discovery_generation", 0)
            if isinstance(scientific_memory, Mapping)
            else 0
        )
        bootstrap = str(context_view.get("campaign_id", "")).endswith(
            ":bootstrap-source"
        )
        transport_round_index = 0 if bootstrap else context_round_index
        planner_round_index = 1 if bootstrap else context_round_index
        portfolio_context_key = (
            context_ref,
            discovery_generation,
            logical_namespace,
        )
        if portfolio_context_key != self._portfolio_context_key:
            try:
                refreshed = (
                    self._refresh_portfolio(
                        context_view=context_view,
                        transport_round_index=transport_round_index,
                        planner_round_index=planner_round_index,
                        discovery_generation=discovery_generation,
                        logical_namespace=logical_namespace,
                    )
                    if discovery_generation > 0
                    or logical_namespace is not None
                    or self.controller.refresh_required(planner_round_index)
                    else self._cached_portfolio(context_view=context_view)
                )
            except fresh_r1.FreshR1Error as error:
                self._portfolio_results = {}
                self._portfolio_failures = {
                    role: error for role in DISCOVERY_PRODUCERS
                }
            else:
                self._portfolio_results = refreshed
            self._portfolio_context_key = portfolio_context_key
        if producer_role in self._portfolio_failures:
            raise self._portfolio_failures[producer_role]
        return self._portfolio_results[producer_role]

    def __call__(
        self, producer_role: str, context_view: Mapping[str, Any]
    ) -> ProviderMechanismProgramProposalV1:
        return self._call_original(
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
    ) -> ProviderMechanismProgramProposalV1:
        """Preserve Original's one-batch portfolio under a retry identity."""

        return self._call_original(
            producer_role,
            context_view,
            logical_namespace=logical_namespace,
        )


@dataclass(slots=True)
class OriginalMatchedRouterV1:
    """Apply common hard gates, then only Original's ranking policy."""

    controller: OriginalRuntimeAdapterV1
    priority_by_program_digest: Mapping[str, str]
    common_router: StrongStaticRouterV1

    @property
    def runnable_floor(self) -> float:
        return self.common_router.runnable_floor

    @property
    def utility_floor(self) -> float:
        return self.common_router.utility_floor

    @property
    def blocker_ceiling(self) -> float:
        return self.common_router.blocker_ceiling

    @property
    def cost_ceiling(self) -> float:
        return self.common_router.cost_ceiling

    @property
    def slate_ceiling(self) -> int:
        return self.common_router.slate_ceiling

    @property
    def policy_digest(self) -> str:
        return sha256_digest(
            {
                "controller_identity": self.controller.identity_digest,
                "common_hard_gate": self.common_router.policy_digest,
            }
        )

    def score(
        self,
        features: SearchUtilityFeaturesV1,
        policy_projection: Mapping[str, Any] | None = None,
    ) -> float:
        del policy_projection
        return self.common_router.score(features, policy_projection=None)

    def select_innovation_outcome(
        self,
        candidates: Sequence[tuple[Any, Any]],
    ) -> str | None:
        """Preserve Original's own selection before shared implementation."""

        actions: list[Mapping[str, Any]] = []
        outcome_digest_by_candidate: dict[str, str] = {}
        for outcome, _resolution in candidates:
            program = getattr(outcome, "source_mechanism_program", None)
            if not isinstance(program, Mapping):
                continue
            report = compile_program(program)
            if (
                report.candidate_id is None
                or report.mechanism_program_digest is None
                or report.mechanism_semantics_digest is None
            ):
                continue
            candidate_id = str(report.candidate_id)
            family_id = str(report.mechanism_semantics_digest)
            actions.append(
                {
                    "candidate_id": candidate_id,
                    "family_id": family_id,
                    "mechanism_id": family_id,
                    "mechanism_semantics_digest": family_id,
                    "priority": self.priority_by_program_digest.get(
                        str(report.mechanism_program_digest),
                        "high",
                    ),
                    "status": "implemented",
                }
            )
            outcome_digest_by_candidate[candidate_id] = str(outcome.digest)
        ranked = self.controller.rank(actions)
        if not ranked:
            return None
        return outcome_digest_by_candidate.get(str(ranked[0]["candidate_id"]))

    def route(
        self,
        proposals: Sequence[
            CandidateProposalV2 | CandidateProposalV3 | CandidateProposalV4
        ],
        policy_projection: Mapping[str, Any] | None = None,
    ) -> RouteTraceV1:
        del policy_projection
        common = self.common_router.route(proposals, policy_projection=None)
        allowed = {
            decision.candidate_id
            for decision in common.decisions
            if decision.allowed
        }
        actions: list[Mapping[str, Any]] = []
        for proposal in proposals:
            if proposal.candidate_id not in allowed:
                continue
            report = compile_program(proposal.mechanism_program)
            program_digest = str(report.mechanism_program_digest)
            actions.append(
                {
                    "candidate_id": proposal.candidate_id,
                    "family_id": proposal.mechanism_id,
                    "mechanism_id": proposal.mechanism_id,
                    "mechanism_semantics_digest": (
                        report.mechanism_semantics_digest
                    ),
                    "priority": self.priority_by_program_digest.get(
                        program_digest,
                        "high",
                    ),
                    "status": "implemented",
                }
            )
        ranked = self.controller.rank(actions)
        ranked_ids = tuple(str(item["candidate_id"]) for item in ranked)
        selected = ranked_ids[0] if ranked_ids else None
        return RouteTraceV1(
            pool_digest=common.pool_digest,
            ordered_candidate_ids=common.ordered_candidate_ids,
            ranked_candidate_ids=ranked_ids,
            decisions=common.decisions,
            selected_candidate_id=selected,
            selection_score=(
                self.controller._score(ranked[0]) if ranked else None
            ),
            policy_digest=self.policy_digest,
        )


@dataclass(frozen=True, slots=True)
class OriginalMatched264Config:
    """The strict 100-or-200-slot A contract over ``StandaloneResearchConfig``."""

    shared: StandaloneResearchConfig
    batch_schema_path: Path = _DEFAULT_BATCH_SCHEMA

    def __post_init__(self) -> None:
        config = self.shared
        expected = {
            "seed": 54201,
            "epochs": 100,
            "timeout_seconds": 3600,
            "watchdog_seconds": 3600,
            "final_worker_ceiling_seconds": 3600,
            "attempt_scheduler": True,
            "max_attempts_per_round": 4,
            "close_exhausted_no_metric_slot": True,
            "arm_code": "A",
        }
        for name, value in expected.items():
            if getattr(config, name) != value:
                raise ValueError(f"Original-Matched-264 contract drift: {name}")
        if config.round_count not in {100, 200}:
            raise ValueError("Original-Matched-264 contract drift: round_count")
        if config.search_seed not in {54201, 54202, 54203}:
            raise ValueError("Original-Matched-264 search seed is outside preregistration")
        if config.shared_implementation_root is None:
            raise ValueError(
                "Original-Matched-264 requires the paired shared implementation root"
            )
        if config.evaluator != DEVELOPMENT_EVALUATOR or config.split != DEVELOPMENT_SPLIT:
            raise ValueError(
                "Original-Matched-264 requires the shared development evaluator"
            )
        if config.observation_seed_schedule != (54201,) * config.round_count:
            raise ValueError(
                "Original-Matched-264 worker schedule must remain seed 54201"
            )
        schema_path = Path(self.batch_schema_path).resolve()
        if not schema_path.is_file():
            raise ValueError("Original-Matched-264 proposal schema is missing")
        object.__setattr__(self, "batch_schema_path", schema_path)


@dataclass(slots=True)
class OriginalMatched264ControllerInterface:
    """Bind Original policy state to the shared standalone execution hooks."""

    controller: OriginalRuntimeAdapterV1
    provider: OriginalMatchedProviderProducer
    router: OriginalMatchedRouterV1

    @property
    def identity(self) -> Mapping[str, Any]:
        return canonical_value(
            {
                "kind": "ORIGINAL_MATCHED_264",
                "controller_identity_digest": self.controller.identity_digest,
                "router_policy_digest": self.router.policy_digest,
                "meta_research_enabled": False,
                "search_space_id": "BL_ICF_MECHANISM_SPACE_V1",
                "search_space_size": 264,
            }
        )

    def _persist_controller_state(
        self,
        state: CampaignState,
        transition: Mapping[str, Any] | None = None,
    ) -> CampaignState:
        memory = dict(state.context.scientific_memory)
        global_memory = dict(memory.get("global_memory", {}))
        global_memory[_ORIGINAL_STATE_KEY] = self.controller.to_state()
        if transition is not None:
            global_memory[_ORIGINAL_TRANSITION_KEY] = canonical_value(
                dict(transition)
            )
        memory["global_memory"] = canonical_value(global_memory)
        return replace(
            state,
            context=replace(
                state.context,
                scientific_memory=canonical_value(memory),
            ),
        )

    def initial_state_transition(self, state: CampaignState) -> CampaignState:
        return self._persist_controller_state(state)

    def post_round_state_transition(
        self,
        state: CampaignState,
        result: Any,
        round_index: int,
        status: str,
    ) -> CampaignState:
        attempts = tuple(getattr(result, "attempts", ()) or ())
        if not attempts or state.next_round_index <= round_index:
            return self._persist_controller_state(state)
        attempt = attempts[-1]
        binding = attempt.binding
        candidate_run = getattr(result, "candidate_run", None)
        if not isinstance(candidate_run, Mapping):
            candidate_run = getattr(attempt, "candidate_run", None)
        outcome = dict(candidate_run) if isinstance(candidate_run, Mapping) else {}
        feedback = canonical_value(
            {
                "round_index": round_index,
                "candidate_id": attempt.candidate_id,
                "mechanism_id": binding.proposal.mechanism_id,
                "mechanism_semantics_digest": (
                    binding.mechanism_semantics_digest
                ),
                "slot_status": status,
                "search_outcome": {
                    "run_status": outcome.get(
                        "exit_status",
                        outcome.get("status", "FAILED"),
                    ),
                    "normalized_metrics": outcome.get("metrics", {}),
                },
            }
        )
        transition = self.controller.close_round(feedback)
        return self._persist_controller_state(state, transition)

    def as_standalone_interface(self) -> StandaloneControllerInterface:
        return StandaloneControllerInterface(
            identity=self.identity,
            provider=self.provider,
            router=self.router,
            enable_meta_research=False,
            initial_state_transition=self.initial_state_transition,
            post_round_state_transition=self.post_round_state_transition,
        )


def _controller_from_checkpoint(config: StandaloneResearchConfig) -> OriginalRuntimeAdapterV1:
    state = ResearchCampaign.read_checkpoint_state(config.run_root)
    global_memory = state.context.scientific_memory.get("global_memory")
    controller_state = (
        global_memory.get(_ORIGINAL_STATE_KEY)
        if isinstance(global_memory, Mapping)
        else None
    )
    if not isinstance(controller_state, Mapping):
        raise ValueError("resume checkpoint lacks Original controller state")
    return OriginalRuntimeAdapterV1.from_state(controller_state)


def build_original_matched_264_controller_interface(
    config: OriginalMatched264Config,
    *,
    resume: bool = False,
    provider_call: Any | None = None,
) -> OriginalMatched264ControllerInterface:
    """Build only Original's policy hooks; all execution objects stay shared."""

    if not isinstance(config, OriginalMatched264Config):
        raise TypeError("config must be OriginalMatched264Config")
    shared = config.shared
    controller = (
        _controller_from_checkpoint(shared)
        if resume
        else OriginalRuntimeAdapterV1()
    )
    provider = OriginalMatchedProviderProducer(
        controller=controller,
        baseline_context=shared.baseline_context,
        batch_schema_path=config.batch_schema_path,
        config_source=shared.api_config_source,
        call_root=shared.run_root / "provider_calls",
        session_id=shared.campaign_id,
        provider_call=provider_call,
        proposal_seed=shared.search_seed,
        frozen_profile_ref=shared.frozen_profile_ref,
        maximum_physical_attempts=shared.provider_maximum_physical_attempts,
        proposal_output_token_ceiling_total_per_context=(
            shared.proposal_output_token_ceiling_total_per_slot
        ),
        allow_historical_success_replay=(
            resume
            and shared.allow_resume_endpoint1_to_endpoint2_and_reasoning_drift
        ),
    )
    common_router = StrongStaticRouterV1(
        runnable_floor=0.0,
        utility_floor=0.0,
        blocker_ceiling=1.0,
        cost_ceiling=1.0,
        slate_ceiling=4,
    )
    router = OriginalMatchedRouterV1(
        controller=controller,
        priority_by_program_digest=provider.priority_by_program_digest,
        common_router=common_router,
    )
    controller_interface = OriginalMatched264ControllerInterface(
        controller=controller,
        provider=provider,
        router=router,
    )
    return controller_interface


def compose_original_matched_264_campaign(
    config: OriginalMatched264Config,
    *,
    resume: bool = False,
    provider_call: Any | None = None,
    launch: Any | None = None,
) -> StandaloneResearchComposition:
    """Compose A while reusing B/C implementation through evaluator objects."""

    controller_interface = build_original_matched_264_controller_interface(
        config,
        resume=resume,
        provider_call=provider_call,
    )
    return compose_standalone_campaign(
        # A owns its Original controller, not B's fixed-policy intervention.
        replace(config.shared, search_policy_mode="adaptive"),
        resume=resume,
        provider_call=provider_call,
        launch=launch,
        controller_interface=controller_interface.as_standalone_interface(),
    )


__all__ = [
    "OriginalMatched264Config",
    "OriginalMatched264ControllerInterface",
    "OriginalMatchedProviderProducer",
    "OriginalMatchedRouterV1",
    "ORIGINAL_MATCHED_BATCH_OUTPUT_TOKEN_CEILING",
    "ORIGINAL_MATCHED_TOTAL_TOKEN_CEILING",
    "build_original_matched_264_controller_interface",
    "compose_original_matched_264_campaign",
    "render_original_matched_prompt",
]
