"""Provider proposal lane for package-owned declarative search spaces.

This module adds one narrow program-producing edge to the existing Research
Line.  It does not select candidates or alter the campaign loop: the Provider
still authors one strict mechanism program, the family compiler validates it,
and the normal implementation/qualification/runtime path consumes the result.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
import json

import jsonschema

from recclaw_core.experiments.helix_abc_v1 import fresh_r1
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.provider_model_routing import (
    select_provider_model,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
)
from recclaw_core.mechanism_space import CompileStatus
from recclaw_core.mechanism_space.canonical import deep_thaw
from recclaw_core.mechanism_space.declarative_provider import (
    DeclarativeMechanismSpaceProvider,
    train_spectral_basis_rank_max,
)

from .bl_icf_realization import ProviderMechanismProgramProposalV1
from .interfaces import project_provider_context_view
from .provider import (
    ProviderResearchProducer,
    SealedProviderRequestUnavailableError,
    _compiler_owned_custom_marker_normalization,
    _decode_provider_wire_program,
    _expected_parent_binding_for_proposal,
    _inherit_unchanged_parent_slots,
    _one_proposal,
    _parent_program_for_proposal,
    _selected_construction_parent,
    _provider_wire_program_schema,
    _provider_wire_with_explicit_json_types,
    _record_replay_trace,
    _record_trace,
    _resume_transport_kwargs,
    _require_effective_single_parent_delta,
    _require_success,
    _stored_provider_request_replay,
    _write_role_bound_schema,
)


_ROLE_INSTRUCTIONS = {
    "mechanism_composer": (
        "Compose one causally coherent mechanism from the exposed typed axes; "
        "change one core slot and at most one support slot. With a frozen "
        "construction parent, normally use COMPOSITION so the implementation "
        "is a local parent-relative change rather than a full-source rewrite."
    ),
    "lineage_refiner": (
        "Refine the measured lineage parent with one explicit mechanism-level "
        "correction; preserve unrelated parent behavior. With a frozen parent, "
        "normally use COMPOSITION and keep the implementation local to that "
        "correction."
    ),
    "falsification_designer": (
        "Design one executable mechanism whose matched control and mechanism-off "
        "ablation distinguish the stated competing explanation. With a frozen "
        "parent, normally use COMPOSITION and change only what the falsifier "
        "needs."
    ),
    "frontier_architect": (
        "Propose one genuinely missing executable architecture, using the custom "
        "escape only when no declared primitive expresses the causal change. "
        "ARCHITECTURE_REWRITE and CUSTOM_MODEL remain available here when a "
        "local parent composition cannot express the innovation."
    ),
}


def _response_schema(
    provider: DeclarativeMechanismSpaceProvider,
    *,
    producer_role: str,
    frozen_profile_ref: Mapping[str, Any],
) -> dict[str, Any]:
    program = _provider_wire_program_schema(
        deep_thaw(provider.program_schema())
    )
    profile = program["properties"]["profile_ref"]
    profile["properties"] = {
        key: {"const": value} for key, value in sorted(frozen_profile_ref.items())
    }
    profile["required"] = sorted(frozen_profile_ref)
    return _provider_wire_with_explicit_json_types({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema": {
                "const": "recclaw.declarative-program-proposal-response.v1"
            },
            "proposals": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "producer_role": {"const": producer_role},
                        "mechanism_program": program,
                    },
                    "required": ["producer_role", "mechanism_program"],
                },
            },
        },
        "required": ["schema", "proposals"],
    })


def _compile_failure(report: Any) -> dict[str, Any]:
    value = report.to_dict()
    return canonical_value(
        {"status": value["status"], "diagnostics": value["diagnostics"]}
    )


class DeclarativeResearchProducer(ProviderResearchProducer):
    """Existing bounded Provider transport with one family-owned strict schema."""

    def __init__(
        self,
        *,
        declarative_provider: DeclarativeMechanismSpaceProvider,
        execution_contract: Mapping[str, Any],
        focused_language: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(declarative_provider, DeclarativeMechanismSpaceProvider):
            raise TypeError("declarative_provider must be a family Provider")
        if not isinstance(execution_contract, Mapping):
            raise TypeError("execution_contract must be a mapping")
        super().__init__(
            allowed_frozen_profile_kinds=(
                declarative_provider.spec.supported_profile_kinds
            ),
            **kwargs,
        )
        self.declarative_provider = declarative_provider
        self.declarative_execution_contract = canonical_value(
            dict(execution_contract)
        )
        self.focused_language = (
            None
            if focused_language is None
            else canonical_value(dict(focused_language))
        )

    def _program_prompt(
        self,
        *,
        producer_role: str,
        provider_context_view: Mapping[str, Any],
        active_task: Mapping[str, Any] | None,
    ) -> str:
        task_text = "none"
        if active_task is not None and active_task.get(
            "execution_eligible_this_round"
        ) is True:
            task_text = canonical_json_bytes(active_task).decode("utf-8")
        spectral_rank_max = (
            train_spectral_basis_rank_max(self.declarative_execution_contract)
            if "TRAIN_SPECTRAL_BASIS"
            in self.declarative_provider.spec.allowed_data_roles
            else None
        )
        spectral_constraint = (
            ""
            if spectral_rank_max is None
            else (
                "\nExecution-bound spectral constraint: every component that "
                "consumes TRAIN_SPECTRAL_BASIS must use one shared rank no "
                f"greater than {spectral_rank_max}. This maximum is derived "
                "from the frozen train-user and non-padding-item cardinalities. "
                "For spectral_index_ranges, the maximum interval end (required "
                "top-K solve depth) must also respect this bound. All consumers "
                "of this one shared DATA basis must select the same columns; "
                "consumers without a selection use [0,rank)."
            )
        )
        return (
            "You are one RecClaw Research Producer. Return exactly one JSON "
            "object matching the supplied schema and no prose. Author one strict "
            "typed mechanism_program; do not return model names, a static candidate "
            "choice, a configuration-only change, a hidden fallback, or a claimed "
            "metric. On the Provider response wire, encode every component and "
            "architecture-operator parameters map as a canonical JSON object "
            "string, never as prose or semicolon-separated key/value text. The "
            "Use exact final component_id values, never descriptive aliases, for "
            "architecture-operator targets and replacements; only an explicit "
            "slot:<removed_slot_id> target is not a component. Express data flow "
            "only through component inputs, "
            "and preserve parent component_id values exactly for inherited and "
            "in-place replacement components. CUSTOM_MODEL construction_mode "
            "requires a custom_components declaration, a components instance "
            "referring to its exact custom_component_id, and a "
            "synthesize_custom_model architecture operator. The profile_ref is "
            "frozen. Every changed core component must "
            "lie on the output path and have an explicit matched control, "
            "mechanism-off ablation, discriminating prediction, implementation "
            "plan, failure interpretation, and honest resource contract. Use a "
            "declared primitive when it expresses the mechanism; use one custom "
            "component only for a genuinely missing causal operation.\n\n"
            f"Assigned role: {producer_role}. {_ROLE_INSTRUCTIONS[producer_role]}\n\n"
            "The execution envelope below is fixed metadata, not a search axis. "
            "Do not change dataset, split, evaluator, candidate universe, seed, "
            "budget, or heldout policy:\n"
            + canonical_json_bytes(self.declarative_execution_contract).decode(
                "utf-8"
            )
            + spectral_constraint
            + (
                "\n\nFocused parent-relative mechanism language. Known "
                "affordances are implementation material, not a whitelist or "
                "candidate menu; the package compiler retains the complete "
                "language and the custom/rewrite innovation lane:\n"
                if self.focused_language is not None
                else "\n\nComplete family search language:\n"
            )
            + canonical_json_bytes(
                self.focused_language
                if self.focused_language is not None
                else deep_thaw(self.declarative_provider.prompt_projection())
            ).decode("utf-8")
            + "\n\nActive feedback task (binding only when marked executable):\n"
            + task_text
            + "\n\nComplete role-scoped Research Context. Use measured positive and "
            "negative evidence to avoid repeated effective families and to choose "
            "the next discriminative mechanism. Treat pre_metric_failure_feedback "
            "as executable-path evidence, not mechanism-effect evidence: never "
            "resubmit the same failing realization unchanged. Either preserve the "
            "hypothesis with a materially different, concretely efficient "
            "implementation that reuses vectorized or parent-native operations, "
            "or choose a different mechanism family; do not infer scientific "
            "rejection from a pre-metric failure:\n"
            + canonical_json_bytes(provider_context_view).decode("utf-8")
        )

    def _decode(
        self,
        response: Mapping[str, Any],
        *,
        producer_role: str,
        expected_parent_binding: Mapping[str, Any]
        | tuple[Mapping[str, Any], ...]
        | None,
        parent_program: Mapping[str, Any] | None,
        construction_parent_options: tuple[Mapping[str, Any], ...] = (),
    ) -> ProviderMechanismProgramProposalV1:
        proposal = _one_proposal(
            response, kind="declarative mechanism-program proposal"
        )
        if proposal.get("producer_role") != producer_role:
            raise fresh_r1.FreshR1Error(
                "Provider changed the preassigned Producer role"
            )
        wire_program = proposal.get("mechanism_program")
        if not isinstance(wire_program, Mapping):
            raise fresh_r1.FreshR1Error(
                "declarative proposal lacks mechanism_program"
            )
        program = _decode_provider_wire_program(wire_program)
        if construction_parent_options:
            expected_parent_binding, parent_program = _selected_construction_parent(
                program, construction_parent_options,
            )
        parent_refs = (
            []
            if expected_parent_binding is None
            else [dict(expected_parent_binding)]
            if isinstance(expected_parent_binding, Mapping)
            else [dict(item) for item in expected_parent_binding]
        )
        program = canonical_value(
            {
                **dict(program),
                "program_payload": {
                    **dict(program["program_payload"]),
                    "parent_refs": parent_refs,
                },
            }
        )
        program = _inherit_unchanged_parent_slots(program, parent_program)
        program = _compiler_owned_custom_marker_normalization(program)
        compile_kwargs = (
            {"execution_contract": self.declarative_execution_contract}
            if "TRAIN_SPECTRAL_BASIS"
            in self.declarative_provider.spec.allowed_data_roles
            else {}
        )
        report = self.declarative_provider.compile(
            deep_thaw(program),
            **compile_kwargs,
        )
        if report.status is not CompileStatus.VALID_NEEDS_IMPLEMENTATION:
            raise _DeclarativeCompileError(report)
        if parent_program is not None:
            _require_effective_single_parent_delta(program, parent_program)
        required_dependencies = self.declarative_execution_contract.get(
            "required_dependencies",
            ("numpy", "python-stdlib", "pytorch", "recbole-runtime", "scipy", "torch"),
        )
        capability_diff = tuple(report.required_capabilities) or (
            "CUSTOM_MODEL_IMPLEMENTATION",
        )
        facts = {
            "requested_current_semantics_digest": None,
            "capability_diff": capability_diff,
            "high_change_dimensions": ("CUSTOM_EXECUTABLE_CAPABILITY",),
            "required_dependencies": tuple(required_dependencies),
            "required_budget": dict(fresh_r1.BUDGET_LIMITS),
        }
        contract = dict(self.declarative_execution_contract)
        contract.pop("required_dependencies", None)
        return ProviderMechanismProgramProposalV1(
            producer_role=producer_role,
            mechanism_program=program,
            implementation_research={
                "base_model_config": contract["base_model_config"],
                "mechanism_config": {},
            },
            resolution_facts=facts,
            parent_binding=expected_parent_binding,
        )

    def _call(  # type: ignore[override]
        self,
        producer_role: str,
        context_view: Mapping[str, Any],
        *,
        logical_namespace: str | None,
        resume_trace: Mapping[str, Any] | None = None,
    ) -> ProviderMechanismProgramProposalV1:
        if producer_role not in DISCOVERY_PRODUCERS:
            raise ValueError("producer_role is outside the four-role portfolio")
        if not isinstance(context_view, Mapping):
            raise TypeError("context_view must be a mapping")
        context_ref = context_view.get("context_ref")
        if not isinstance(context_ref, str) or not context_ref:
            raise ValueError("context_view must contain context_ref")
        provider_context_view = project_provider_context_view(context_view)
        context_identity = (
            f"{context_ref}:context-{sha256_digest(provider_context_view)[:16]}"
        )
        provider_state = provider_context_view.get("state")
        provider_state = (
            provider_state if isinstance(provider_state, Mapping) else {}
        )
        active_task = provider_state.get("task")
        active_task = active_task if isinstance(active_task, Mapping) else None
        prompt = self._program_prompt(
            producer_role=producer_role,
            provider_context_view=provider_context_view,
            active_task=active_task,
        )
        schema = _response_schema(
            self.declarative_provider,
            producer_role=producer_role,
            frozen_profile_ref=self.frozen_profile_ref,
        )
        schema_path = _write_role_bound_schema(
            schema,
            self.call_root
            / "proposal-schemas"
            / "declarative-program"
            / self.declarative_provider.identity().search_space_id
            / f"{producer_role}.schema.json",
        )
        namespace = f":{logical_namespace}" if logical_namespace else ""
        logical_call_id = (
            f"{self.session_id}{namespace}:{context_identity}:{producer_role}:"
            "declarative-program-proposal"
        )
        resume_request_digest: str | None = None
        if logical_namespace is None:
            sealed_identity = self._resume_request_by_producer_role.pop(
                producer_role, None
            )
            if sealed_identity is not None:
                logical_call_id, resume_request_digest = sealed_identity
        provider_call_root = (
            self.call_root
            / "proposals"
            / "declarative-program"
            / self.declarative_provider.identity().search_space_id
            / producer_role
        )
        if resume_trace is not None and resume_trace.get("kind") == "declarative_research_producer_semantic_repair":
            # The first proposal already returned. Continue its saved compiler
            # correction directly instead of generating or charging it again.
            repair_id = logical_call_id + ":semantic-repair-01"
            repair_role = "mechanism_semantic_repair"
            repair_kwargs = {
                "call_root": provider_call_root / "semantic_repair_01",
                "schema_path": schema_path, "logical_call_id": repair_id,
                "session_id": self.session_id, "prompt": prompt,
                "provider_role": repair_role,
                "requested_model": select_provider_model(repair_role).requested_model,
                "token_ceiling": max(self.total_token_ceiling, 64_000),
                "output_token_ceiling": self._budgeted_output_ceiling(context_identity),
                "credential_config_path": self.credential_config_path,
                "expected_transport_release_digest": self.config_identity.get("release_digest"),
                "maximum_physical_attempts": self.maximum_physical_attempts,
            }
            _resume_transport_kwargs(repair_kwargs, resume_trace)
            result = self.provider_call(**repair_kwargs)
            self._charge_output_tokens(context_identity, result)
            _record_trace(self, kind="declarative_research_producer_semantic_repair",
                logical_call_id=repair_id, prompt=repair_kwargs["prompt"], result=result,
                transport_ceiling={"total_tokens": repair_kwargs["token_ceiling"],
                                   "output_tokens": repair_kwargs["output_token_ceiling"]})
            response = _require_success(result, kind="research proposal semantic repair")
            jsonschema.validate(canonical_value(response), json.loads(Path(repair_kwargs["schema_path"]).read_text()))
            parent_binding = _expected_parent_binding_for_proposal(
                context_view=context_view, provider_context_view=provider_context_view)
            return self._decode(response, producer_role=producer_role, expected_parent_binding=parent_binding,
                parent_program=_parent_program_for_proposal(
                    provider_context_view=provider_context_view, expected_parent_binding=parent_binding),
                construction_parent_options=tuple(context_view.get("construction_parent_options", ())),
            )
        replay = self._proposal_replay_remaining.pop(producer_role, None)
        if replay is None:
            model_selection = select_provider_model(producer_role)
            call_kwargs: dict[str, Any] = {
                "call_root": provider_call_root,
                "schema_path": schema_path,
                "logical_call_id": logical_call_id,
                "session_id": self.session_id,
                "prompt": prompt,
                "provider_role": producer_role,
                "requested_model": model_selection.requested_model,
                "token_ceiling": max(self.total_token_ceiling, 64_000),
                "output_token_ceiling": self._budgeted_output_ceiling(
                    context_identity
                ),
                "credential_config_path": self.credential_config_path,
                "expected_transport_release_digest": self.config_identity.get(
                    "release_digest"
                ),
                "maximum_physical_attempts": self.maximum_physical_attempts,
            }
            if resume_request_digest is None:
                resume_request_digest = (
                    self._resume_request_digest_by_logical_call.pop(
                        logical_call_id, None
                    )
                )
            else:
                self._resume_request_digest_by_logical_call.pop(
                    logical_call_id, None
                )
            if (
                resume_request_digest is None
                and logical_namespace is None
                and self._started_round_replay_context_ref
                == context_view.get("context_ref")
            ):
                (
                    request_started,
                    resume_request_digest,
                ) = _stored_provider_request_replay(
                    provider_call_root,
                    logical_call_id=logical_call_id,
                    session_id=self.session_id,
                )
                if request_started and resume_request_digest is None:
                    raise SealedProviderRequestUnavailableError(
                        "started Provider request is missing its exact sealed "
                        f"Provider request for {producer_role}"
                    )
            if resume_request_digest is not None:
                call_kwargs["resume_request_digest"] = resume_request_digest
            _resume_transport_kwargs(call_kwargs, resume_trace)
            prompt = call_kwargs["prompt"]
            result = self.provider_call(**call_kwargs)
            self._charge_output_tokens(context_identity, result)
            _record_trace(
                self,
                kind="declarative_research_producer",
                logical_call_id=logical_call_id,
                prompt=prompt,
                result=result,
                transport_ceiling={"total_tokens": call_kwargs["token_ceiling"],
                                   "output_tokens": call_kwargs["output_token_ceiling"]},
            )
            response = _require_success(result, kind="research proposal")
        else:
            response = replay["response"]
            _record_replay_trace(
                self,
                logical_call_id=logical_call_id,
                prompt=prompt,
                replay=replay,
            )
        if not isinstance(response, Mapping):
            raise fresh_r1.FreshR1Error(
                "research proposal response must be an object"
            )
        jsonschema.validate(canonical_value(response), schema)
        expected_parent_binding = _expected_parent_binding_for_proposal(
            context_view=context_view,
            provider_context_view=provider_context_view,
        )
        parent_program = _parent_program_for_proposal(
            provider_context_view=provider_context_view,
            expected_parent_binding=expected_parent_binding,
        )
        try:
            return self._decode(
                response,
                producer_role=producer_role,
                expected_parent_binding=expected_parent_binding,
                parent_program=parent_program,
                construction_parent_options=tuple(context_view.get("construction_parent_options", ())),
            )
        except _DeclarativeCompileError as error:
            if replay is not None:
                raise fresh_r1.FreshR1Error(
                    "replayed declarative proposal no longer compiles"
                ) from error
            repair_prompt = (
                prompt
                + "\n\nSEMANTIC COMPILER REPAIR (one bounded call only): "
                "the prior response passed JSON schema but failed the family "
                "compiler. Apply only the smallest semantic correction, preserve "
                "role, profile, parent_refs, hypothesis, and frozen execution "
                "envelope, and return one response under the same schema.\n\n"
                "Compiler report:\n"
                + canonical_json_bytes(error.report).decode("utf-8")
                + "\n\nOriginal response:\n"
                + canonical_json_bytes(response).decode("utf-8")
            )
            repair_id = logical_call_id + ":semantic-repair-01"
            repair_role = "mechanism_semantic_repair"
            repair_model_selection = select_provider_model(repair_role)
            repair_call_root = provider_call_root / "semantic_repair_01"
            repair_call_kwargs: dict[str, Any] = {
                "call_root": repair_call_root,
                "schema_path": schema_path,
                "logical_call_id": repair_id,
                "session_id": self.session_id,
                "prompt": repair_prompt,
                "provider_role": repair_role,
                "requested_model": repair_model_selection.requested_model,
                "token_ceiling": max(self.total_token_ceiling, 64_000),
                "output_token_ceiling": self._budgeted_output_ceiling(
                    context_identity
                ),
                "credential_config_path": self.credential_config_path,
                "expected_transport_release_digest": self.config_identity.get(
                    "release_digest"
                ),
                "maximum_physical_attempts": self.maximum_physical_attempts,
            }
            repair_request_digest = (
                self._resume_request_digest_by_logical_call.pop(repair_id, None)
            )
            if (
                repair_request_digest is None
                and logical_namespace is None
                and self._started_round_replay_context_ref
                == context_view.get("context_ref")
            ):
                (
                    request_started,
                    repair_request_digest,
                ) = _stored_provider_request_replay(
                    repair_call_root,
                    logical_call_id=repair_id,
                    session_id=self.session_id,
                )
                if request_started and repair_request_digest is None:
                    raise SealedProviderRequestUnavailableError(
                        "started Provider request is missing its exact sealed "
                        f"Provider request for {repair_role}"
                    )
            if repair_request_digest is not None:
                repair_call_kwargs["resume_request_digest"] = (
                    repair_request_digest
                )
            repair_result = self.provider_call(**repair_call_kwargs)
            self._charge_output_tokens(context_identity, repair_result)
            _record_trace(
                self,
                kind="declarative_research_producer_semantic_repair",
                logical_call_id=repair_id,
                prompt=repair_prompt,
                result=repair_result,
                transport_ceiling={"total_tokens": repair_call_kwargs["token_ceiling"],
                                   "output_tokens": repair_call_kwargs["output_token_ceiling"]},
            )
            repaired = _require_success(
                repair_result, kind="research proposal semantic repair"
            )
            if not isinstance(repaired, Mapping):
                raise fresh_r1.FreshR1Error(
                    "research proposal semantic repair must be an object"
                )
            jsonschema.validate(canonical_value(repaired), schema)
            return self._decode(
                repaired,
                producer_role=producer_role,
                expected_parent_binding=expected_parent_binding,
                parent_program=parent_program,
                construction_parent_options=tuple(context_view.get("construction_parent_options", ())),
            )


class _DeclarativeCompileError(fresh_r1.FreshR1Error):
    def __init__(self, report: Any) -> None:
        self.report = _compile_failure(report)
        super().__init__(
            "declarative mechanism_program did not compile as "
            "VALID_NEEDS_IMPLEMENTATION: "
            + canonical_json_bytes(self.report).decode("utf-8")
        )


__all__ = ["DeclarativeResearchProducer"]
