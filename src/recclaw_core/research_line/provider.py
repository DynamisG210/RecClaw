"""Provider-backed Research Line callables.

The adapters keep the external Provider edge injectable.  They reuse the
existing fresh-R1 prompt, schema, and bounded call contract without creating a
second retry, service, or receipt store.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import jsonschema

from recclaw_core.mechanism_space import (
    CompileReportV1,
    CompileStatus,
    compile_program,
    program_schema,
    prompt_projection,
)
from recclaw_core.experiments.helix_abc_v1 import fresh_r1
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    CampaignRuntimeError,
    campaign_scientific_profile_ref,
    executable_mechanism,
    executable_mechanisms,
    program_from_proposal as campaign_program_from_proposal,
    root_parent_mechanism_id,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.provider_model_routing import (
    EFFICIENT_MODEL,
    select_provider_model,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
    CandidateProposalV4,
    DiscriminativeExperimentPlanV1,
    DiscoveryCreditV1,
    ProposalIntentV1,
    SearchUtilityFeaturesV1,
)
from recclaw_core.experiments.helix_abc_v1.research_science import (
    DeterministicRouterFeatureBuilderV1,
    LineageIndexV1,
    matched_control_plan,
)
from recclaw_core.research_line.interfaces import (
    research_producer_roles,
    project_provider_context_view,
)
from recclaw_core.research_line.bl_icf_realization import (
    ProviderMechanismProgramProposalV1,
    parent_binding_for_mechanism_program,
)
from recclaw_core.research_line.effective_experiment import (
    effective_experiment_identity,
)
from recclaw_core.research_line.single_parent_search import (
    PARENT_BASE_MODEL_CONFIG,
    focused_mechanism_language,
    is_bl_icf_single_parent_context,
    is_single_parent_context,
)


ConfigSource: TypeAlias = Mapping[str, Any] | str | Path
ProviderCall: TypeAlias = Callable[..., fresh_r1.ProviderAttemptResult]

PROPOSAL_REPLAY_SCHEMA = "recclaw.research-line.provider-proposal-replay.v1"


class SealedProviderRequestUnavailableError(fresh_r1.FreshR1Error):
    """A started round cannot replay its exact sealed Provider request."""

    failure_code = "SEALED_PROVIDER_REQUEST_UNAVAILABLE"

_RESOURCE_ROOT = Path(fresh_r1.__file__).resolve().parent / "resources"
_DEFAULT_PROPOSAL_TEMPLATE = (
    _RESOURCE_ROOT / "research_line_open_spec_proposal_prompt_v1.txt"
)
_DEFAULT_PROPOSAL_SCHEMA = (
    _RESOURCE_ROOT / "research_line_open_spec_proposal_response_v1.schema.json"
)
_DEFAULT_BL_ICF_PROPOSAL_TEMPLATE = (
    _RESOURCE_ROOT / "research_line_bl_icf_proposal_prompt_v1.txt"
)
_DEFAULT_BL_ICF_PROPOSAL_SCHEMA = (
    _RESOURCE_ROOT / "campaign_proposal_response_v2.schema.json"
)
_DEFAULT_BL_ICF_PROGRAM_PROPOSAL_TEMPLATE = (
    _RESOURCE_ROOT / "research_line_bl_icf_program_proposal_prompt_v1.txt"
)
_DEFAULT_BL_ICF_PROGRAM_PROPOSAL_SCHEMA = (
    _RESOURCE_ROOT / "research_line_bl_icf_program_proposal_response_v1.schema.json"
)
_DEFAULT_IMPLEMENTATION_TEMPLATE = (
    _RESOURCE_ROOT / "research_line_implementer_prompt_v1.txt"
)
_DEFAULT_BL_ICF_IMPLEMENTATION_APPENDIX = (
    _RESOURCE_ROOT / "research_line_implementer_bl_icf_appendix_v1.txt"
)
_DEFAULT_DIFFUSION_FLOW_CF_IMPLEMENTATION_APPENDIX = (
    _RESOURCE_ROOT
    / "research_line_implementer_diffusion_flow_cf_appendix_v1.txt"
)
_DEFAULT_SEQUENTIAL_SCALING_IMPLEMENTATION_APPENDIX = (
    _RESOURCE_ROOT
    / "research_line_implementer_sequential_scaling_appendix_v1.txt"
)
_DEFAULT_SEMANTIC_ID_GENERATIVE_IMPLEMENTATION_APPENDIX = (
    _RESOURCE_ROOT
    / "research_line_implementer_semantic_id_generative_appendix_v1.txt"
)
_DEFAULT_IMPLEMENTATION_SCHEMA = (
    _RESOURCE_ROOT / "fresh_r1_implementation_response_v1.schema.json"
)
_PARENT_METHOD_PATCH_RESPONSE_MODE = "PARENT_METHOD_PATCH_V1"
_PARENT_LOCAL_SLOT_PATCH_RESPONSE_MODE = "PARENT_LOCAL_SLOT_PATCH_V1"
_PROFILE_MODEL_HOOKS_RESPONSE_MODE = "PROFILE_MODEL_HOOKS_V1"
_SCAFFOLDED_FULL_SOURCE_RESPONSE_MODE = "SCAFFOLDED_FULL_SOURCE_V1"


def _implementation_schema_for_request(
    schema_source: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Project the Provider wire schema to the request's source ownership."""

    schema = canonical_value(dict(schema_source))
    service_policy = request.get("service_policy")
    response_mode = (
        service_policy.get("response_mode")
        if isinstance(service_policy, Mapping)
        else None
    )
    if response_mode not in {
        _PARENT_METHOD_PATCH_RESPONSE_MODE,
        _PARENT_LOCAL_SLOT_PATCH_RESPONSE_MODE,
        _PROFILE_MODEL_HOOKS_RESPONSE_MODE,
        _SCAFFOLDED_FULL_SOURCE_RESPONSE_MODE,
    }:
        return schema
    files_contract = schema["properties"]["proposals"]["items"]["properties"][
        "files"
    ]
    allowed_paths = ["recclaw_ext/candidate.py"]
    if response_mode in {
        _PARENT_METHOD_PATCH_RESPONSE_MODE,
        _SCAFFOLDED_FULL_SOURCE_RESPONSE_MODE,
    }:
        allowed_paths.append("recclaw_ext/trainer.py")
    files_contract["minItems"] = 1
    files_contract["maxItems"] = len(allowed_paths)
    files_contract["items"]["properties"]["path"]["enum"] = allowed_paths
    return canonical_value(schema)


def _implementation_template_for_search_space(
    common_template: str,
    bl_icf_appendix: str,
    search_space_id: str | None,
    *,
    diffusion_flow_cf_appendix: str = "",
    sequential_scaling_appendix: str = "",
    semantic_id_generative_appendix: str = "",
) -> str:
    if search_space_id == "BL_ICF_MECHANISM_SPACE_V1":
        appendix = bl_icf_appendix
    elif search_space_id == "DIFFUSION_FLOW_CF_MECHANISM_SPACE_V1":
        appendix = diffusion_flow_cf_appendix
    elif search_space_id == "SEQUENTIAL_SCALING_MECHANISM_SPACE_V1":
        appendix = sequential_scaling_appendix
    elif search_space_id == "SEMANTIC_ID_GENERATIVE_MECHANISM_SPACE_V1":
        appendix = semantic_id_generative_appendix
    else:
        appendix = ""
    if not appendix:
        return common_template
    marker = "Shared blind request:"
    before, separator, after = common_template.partition(marker)
    if not separator:
        return (
            common_template.rstrip()
            + "\n\n"
            + appendix.strip()
            + "\n"
        )
    return (
        before.rstrip()
        + "\n\n"
        + appendix.strip()
        + "\n\n"
        + separator
        + after
    )

# Research proposals retain the original 6,000-token output capacity.  Their
# total-call budget also accounts for the measured ~1,750-token round-7 input
# plus operational headroom, instead of forcing input and output to share the
# old output-only value.
RESEARCH_PROPOSAL_TOTAL_TOKEN_CEILING = 9_000
RESEARCH_PROPOSAL_OUTPUT_TOKEN_CEILING = 6_000
# The direct-program lane embeds the canonical full BL-ICF language.  Its input
# is intentionally much larger than the compact OpenSpec/fixed-catalog prompt,
# so sharing the 9k total ceiling would turn exact language exposure into a
# deterministic token-ceiling failure before any proposal is produced.
RESEARCH_BL_ICF_PROGRAM_TOTAL_TOKEN_CEILING = 64_000
# Full research-model responses have a distinct resource envelope from the compact
# producer contract above.  The observed full-role tail reached 31,935 output /
# 76,896 total tokens.  The shared 32k/80k envelope covers that observed tail;
# it is only a maximum and does not force healthy shorter responses to spend it.
# The historical constant names are retained for artifact compatibility, but
# this envelope belongs to every full-model program call, not only Frontier.
RESEARCH_FRONTIER_PROGRAM_OUTPUT_TOKEN_CEILING = 32_000
RESEARCH_FRONTIER_PROGRAM_TOTAL_TOKEN_CEILING = 80_000

_MACHINE_EXECUTION_CONFIG_FIELDS = frozenset(
    {
        "candidate_id",
        "dataset",
        "entrypoint",
        "epochs",
        "evaluator",
        "model",
        "recclaw_trainer_entrypoint",
        "seed",
        "split",
        "timeout",
        "timeout_seconds",
    }
)


def shared_implementation_identity(
    *,
    search_seed: int,
    request: Mapping[str, Any],
    arm: str,
    implementer_model: str = EFFICIENT_MODEL,
    implementer_prompt_digest: str | None = None,
    implementer_schema_digest: str | None = None,
) -> Mapping[str, Any]:
    """Return the arm-blind identity for one canonical implementation request."""

    if arm not in {"A", "B", "C"}:
        raise ValueError("arm must be A, B, or C")
    if isinstance(search_seed, bool) or not isinstance(search_seed, int):
        raise ValueError("search_seed must be an integer")
    normalized = canonical_value(dict(request))
    compiled = normalized.get("compiled_mechanism")
    program_digest = (
        compiled.get("mechanism_program_digest")
        if isinstance(compiled, Mapping)
        else None
    )
    if (
        not isinstance(program_digest, str)
        or len(program_digest) != 64
        or any(character not in "0123456789abcdef" for character in program_digest)
    ):
        raise ValueError(
            "shared implementation request lacks canonical mechanism_program_digest"
        )
    request_digest = sha256_digest(normalized)
    if implementer_prompt_digest is None:
        space_identity = (
            compiled.get("space_identity")
            if isinstance(compiled, Mapping)
            else None
        )
        search_space_id = (
            str(space_identity.get("search_space_id"))
            if isinstance(space_identity, Mapping)
            and isinstance(space_identity.get("search_space_id"), str)
            else None
        )
        implementer_prompt_digest = sha256_digest(
            _implementation_template_for_search_space(
                _read_text(_DEFAULT_IMPLEMENTATION_TEMPLATE),
                _read_text(_DEFAULT_BL_ICF_IMPLEMENTATION_APPENDIX),
                search_space_id,
                diffusion_flow_cf_appendix=_read_text(
                    _DEFAULT_DIFFUSION_FLOW_CF_IMPLEMENTATION_APPENDIX
                ),
                sequential_scaling_appendix=_read_text(
                    _DEFAULT_SEQUENTIAL_SCALING_IMPLEMENTATION_APPENDIX
                ),
                semantic_id_generative_appendix=_read_text(
                    _DEFAULT_SEMANTIC_ID_GENERATIVE_IMPLEMENTATION_APPENDIX
                ),
            )
        )
    prompt_digest = implementer_prompt_digest
    schema_digest = implementer_schema_digest or sha256_digest(
        _implementation_schema_for_request(
            _read_json(_DEFAULT_IMPLEMENTATION_SCHEMA),
            normalized,
        )
    )
    for field_name, digest in (
        ("implementer_prompt_digest", prompt_digest),
        ("implementer_schema_digest", schema_digest),
    ):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{field_name} must be a SHA256 digest")
    identity = {
        "schema": "recclaw.shared-implementation-identity.v1",
        "search_seed": search_seed,
        "canonical_spec_digest": program_digest,
        "implementer_model": implementer_model,
        "implementer_prompt_digest": prompt_digest,
        "implementer_schema_digest": schema_digest,
        "request_digest": request_digest,
    }
    return canonical_value(
        {
            **identity,
            "shared_root_key": (
                f"search-seed-{search_seed}/identity-{sha256_digest(identity)}"
            ),
        }
    )
_FROZEN_QUALIFICATION_BASE_MODEL_CONFIGS = ("BPR", "LightGCN")

PROPOSAL_LANE_BL_ICF = "BL_ICF"
PROPOSAL_LANE_BL_ICF_PROGRAM = "BL_ICF_PROGRAM"
PROPOSAL_LANE_OPEN_SPEC = "OPEN_SPEC"
_PROPOSAL_LANES = frozenset(
    {PROPOSAL_LANE_BL_ICF, PROPOSAL_LANE_BL_ICF_PROGRAM, PROPOSAL_LANE_OPEN_SPEC}
)
_DEFAULT_PROPOSAL_LANE_BY_ROLE = {
    role: PROPOSAL_LANE_BL_ICF_PROGRAM for role in DISCOVERY_PRODUCERS
}
_FACTORED_BL_ICF_PRODUCER_ROLES = DISCOVERY_PRODUCERS
# Ordinary research stays on the compact, exact-identity program contract even
# when a stronger model is selected.  Representation complexity and model
# capability are independent choices; Frontier alone receives the full grammar.
_INDEPENDENT_MINI_PRODUCER_ROLES = frozenset(
    role for role in DISCOVERY_PRODUCERS if role != "frontier_architect"
)
_INDEPENDENT_MINI_LEGACY_TRANSPORT_NAMESPACE = (
    "bl-icf-program-mini-assembly-v2"
)
_INDEPENDENT_MINI_TRANSPORT_NAMESPACE = (
    "bl-icf-program-mini-responses-v1"
)
_INDEPENDENT_MINI_WIRE_API = "responses"
_FRONTIER_RESPONSES_TRANSPORT_NAMESPACE = (
    "bl-icf-program-frontier-responses-v1"
)
_FRONTIER_RESPONSES_WIRE_API = "responses"
_IMPLEMENTATION_RESPONSES_TRANSPORT_NAMESPACE = "implementation-responses-v1"
_IMPLEMENTATION_RESPONSES_WIRE_API = "responses"


def _successful_replay_call(call_root: Path) -> dict[str, Any]:
    reference_path = call_root / "PAIRED_PROVIDER_CALL_REF_V1.json"
    if not reference_path.is_file():
        raise ValueError(f"proposal replay reference is missing: {reference_path}")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if reference.get("call_status") != "SUCCESS":
        raise ValueError(f"proposal replay call is not successful: {reference_path}")
    canonical_call_root = Path(str(reference.get("canonical_call_root", ""))).resolve()
    databases = tuple(
        sorted(canonical_call_root.glob("physical_attempt_*/broker.sqlite3"))
    )
    if not databases:
        raise ValueError(
            f"proposal replay broker evidence is missing: {canonical_call_root}"
        )
    rows: list[tuple[Any, ...]] = []
    for database in databases:
        connection = sqlite3.connect(
            f"file:{database.as_posix()}?mode=ro",
            uri=True,
        )
        try:
            rows.extend(
                connection.execute(
                    "SELECT logical_call_id, request_digest, response_digest, "
                    "response_json, returned_model, status FROM calls "
                    "WHERE status = 'SUCCESS'"
                ).fetchall()
            )
        finally:
            connection.close()
    if len(rows) != 1:
        raise ValueError(
            "proposal replay evidence must contain exactly one successful call: "
            f"{canonical_call_root}"
        )
    (
        logical_call_id,
        request_digest,
        response_digest,
        response_json,
        returned_model,
        _status,
    ) = rows[0]
    response = json.loads(str(response_json))
    if not isinstance(response, Mapping):
        raise ValueError("proposal replay response must be an object")
    if reference.get("request_digest") != request_digest:
        raise ValueError("proposal replay request digest differs from its reference")
    if reference.get("response_digest") != response_digest:
        raise ValueError("proposal replay response digest differs from its reference")
    if sha256_digest(response) != response_digest:
        raise ValueError("proposal replay response bytes do not match their digest")
    return canonical_value(
        {
            "response": response,
            "source": {
                "schema": PROPOSAL_REPLAY_SCHEMA,
                "reference_path": str(reference_path.resolve()),
                "reference_digest": sha256_digest(reference),
                "canonical_call_root": str(canonical_call_root),
                "logical_call_id": str(logical_call_id),
                "request_digest": str(request_digest),
                "response_digest": str(response_digest),
                "returned_model": returned_model,
            },
        }
    )


def _stored_provider_request_replay(
    call_root: Path,
    *,
    logical_call_id: str,
    session_id: str,
) -> tuple[bool, str | None]:
    """Return whether a broker started and its unique clean success, if any."""

    databases = tuple(
        sorted(call_root.glob("physical_attempt_*/broker.sqlite3"))
    )
    if any(
        (wal := database.with_name("broker.sqlite3-wal")).is_file()
        and wal.stat().st_size > 0
        for database in databases
    ):
        # A live/unclean broker cannot be inspected without disturbing its
        # recovery boundary.  Treat the current request as ambiguously started
        # and fail closed before opening any database.
        return True, None

    request_started = False
    request_digests: set[str] = set()
    for database in databases:
        connection = sqlite3.connect(
            f"file:{database.as_posix()}?mode=ro&immutable=1",
            uri=True,
        )
        try:
            rows = connection.execute(
                "SELECT request_digest, status FROM calls "
                "WHERE logical_call_id=? "
                "AND proposal_generation_session_id=?",
                (logical_call_id, session_id),
            ).fetchall()
            request_started = request_started or bool(rows)
            request_digests.update(
                str(request_digest)
                for request_digest, status in rows
                if status == "SUCCESS"
            )
        finally:
            connection.close()
    if len(request_digests) > 1:
        raise fresh_r1.FreshR1Error(
            "stored Provider replay has conflicting request identities"
        )
    return request_started, next(iter(request_digests), None)


def _stored_success_request_digest(
    call_root: Path,
    *,
    logical_call_id: str,
    session_id: str,
) -> str | None:
    """Return the unique clean durable success for an already-started request."""

    return _stored_provider_request_replay(
        call_root,
        logical_call_id=logical_call_id,
        session_id=session_id,
    )[1]


def load_provider_proposal_replay(
    source_run_root: str | Path,
) -> dict[str, Mapping[str, Any]]:
    """Load one previously successful strict-program response per Producer role.

    Semantic-repair output is preferred when it exists, because that is the
    compiler-valid response actually consumed by the source run.  This replays
    proposal evidence only; candidate implementation is always regenerated.
    """

    source_run_root = Path(source_run_root).resolve()
    proposal_root = (
        source_run_root / "provider_calls" / "proposals" / "bl-icf-program"
    )
    replay: dict[str, Mapping[str, Any]] = {}
    for role in DISCOVERY_PRODUCERS:
        role_root = proposal_root / role
        candidates = (role_root / "semantic_repair_01", role_root)
        selected = next(
            (
                candidate
                for candidate in candidates
                if (candidate / "PAIRED_PROVIDER_CALL_REF_V1.json").is_file()
            ),
            None,
        )
        if selected is None:
            raise ValueError(f"proposal replay is missing role evidence: {role}")
        replay[role] = _successful_replay_call(selected)
    return replay


def provider_proposal_replay_identity(
    source_run_root: str | Path | None,
) -> Mapping[str, Any] | None:
    if source_run_root is None:
        return None
    replay = load_provider_proposal_replay(source_run_root)
    identity = canonical_value(
        {
            "schema": PROPOSAL_REPLAY_SCHEMA,
            "source_run_root": str(Path(source_run_root).resolve()),
            "roles": {
                role: replay[role]["source"] for role in DISCOVERY_PRODUCERS
            },
        }
    )
    return canonical_value({**identity, "digest": sha256_digest(identity)})

class ProviderRequestError(RuntimeError):
    """The invocation returned no structured result; retain stages for explicit resume."""

    failure_code = "PRODUCER_REQUEST_FAILED"


_DIRECTOR_ROLE_INSTRUCTION = (
    "Act as the sole research decision-maker for this step. Read the exact active "
    "construction parent and measured feedback, separating inherited root gains "
    "from the latest parent-relative and frontier-relative changes. Briefly compare "
    "the marginal value of another lineage refinement with a different direction "
    "using observed gains, failures and training cost. Choose according to expected "
    "real recommendation improvement. Structural distance is not a reward: retain "
    "useful parent behavior and pursue new mechanisms when their expected gain "
    "justifies the cost. Distinguish measured improvement from evidence of novelty. Compare "
    "a few complementary intervention ideas, choose one, and express its mechanism, "
    "competing explanation and cost rationale in the existing proposal fields. "
    "No separate peer proposals or reviews have been supplied; do not invent them. "
    "The chosen spec must stand on its own for the Implementer and next real experiment."
)


_RESEARCH_ROLE_INSTRUCTIONS = {
    **fresh_r1.ROLE_INSTRUCTIONS,
    "mechanism_composer": (
        "Develop one effect-directed scoring or learning mechanism. Start from "
        "the actual parent and measured feedback; coordinate multiple parts only "
        "when the causal hypothesis needs them. A focused change or useful "
        "subtraction is legitimate research. Preserve the supplied protocol and "
        "dependency requirements."
    ),
    "frontier_architect": (
        "Make the final portfolio-level research decision for this round. Review "
        "state.research_portfolio, then select, refine, combine or replace those "
        "ideas according to expected real recommendation improvement. Compare "
        "the marginal value of another lineage refinement with a different "
        "direction using observed gains, failures and training cost. Structural "
        "distance is not a reward: retain useful parent behavior, and use "
        "substantial new mechanisms when their expected gain justifies the cost. "
        "Keep the full innovation freedom and the supplied protocol and dependency "
        "requirements. Distinguish measured improvement from evidence of novelty."
    ),
}


_BL_ICF_ROLE_INSTRUCTIONS = {
    "mechanism_composer": (
        "Compose one broad-discovery mechanism from the complete BL-ICF language, "
        "including custom synthesis when scientifically justified. When recent "
        "measured evidence is weak, prefer the strongest measured parent/frontier "
        "and one interpretable causal-axis change before widening the architecture; "
        "structural novelty is secondary to expected effect. Treat pending control "
        "or falsification tasks as non-binding evidence context; do not turn this "
        "discovery lane into another matched-control proposal."
    ),
    "lineage_refiner": (
        "Author one compiler-valid descendant of the exact activated parent "
        "when its lineage binding is supplied. When state includes "
        "lineage_parent_mechanism_program, preserve that complete parent "
        "program field by field and change only the declared mechanism axis."
    ),
    "falsification_designer": (
        "Author one compiler-valid BL-ICF mechanism around a decisive "
        "same-protocol matched-control falsifier."
    ),
    "frontier_architect": (
        "Develop one effect-directed candidate for this round. Review "
        "state.research_portfolio from the three preceding roles as useful research "
        "context, then refine, combine, or depart from those ideas with one "
        "compiler-valid discovery composition having high expected measured effect "
        "in the complete BL-ICF language. The ordinary proposals remain peer "
        "executable search opportunities; do not act as their sole selector. "
        "Preserve a reliable "
        "measured parent/frontier when the recent result is weak, and use one "
        "interpretable mechanism-axis change where possible; structural distance "
        "is an anti-repeat aid, not the objective. Pending control or falsification "
        "tasks may inform the proposal but must not collapse this lane into the "
        "active parent-chain control."
    ),
}

_SINGLE_PARENT_BL_ICF_ROLE_INSTRUCTIONS = {
    "mechanism_composer": (
        "Develop one focused, effect-directed mechanism from a parent explicitly "
        "chosen from state.construction_parent_options when supplied; otherwise use "
        "the supplied construction parent. Coordinate as many mechanism zones as the causal thesis "
        "actually requires, but do not bundle unrelated changes. Known affordances "
        "are implementation material, not a candidate menu; custom synthesis remains "
        "available when the hypothesis genuinely requires it."
    ),
    "lineage_refiner": (
        "Refine a parent explicitly chosen from state.construction_parent_options "
        "when supplied; otherwise use the supplied construction parent. Preserve every undeclared parent "
        "slot and express the complete coordinated mechanism change faithfully."
    ),
    "falsification_designer": (
        "Author one parent-relative BL-ICF mechanism whose same-protocol matched "
        "control can distinguish the claimed causal effect from its strongest "
        "competing explanation."
    ),
    "frontier_architect": (
        "Develop one effect-directed candidate after reviewing "
        "state.research_portfolio from the three preceding roles as research "
        "context. Refine, combine, or depart from those ideas to seek a high "
        "expected real gain over the frozen LightGCN++ root; ordinary proposals "
        "remain peer executable search opportunities. Choose the construction parent "
        "from state.construction_parent_options when supplied, including the frozen "
        "root even when a better descendant exists. Retain the "
        "freedom to cross zones, introduce new typed components or learning "
        "signals, or use ARCHITECTURE_REWRITE or CUSTOM_MODEL when the causal thesis "
        "needs the structural distance and its expected gain justifies the added "
        "implementation risk. The focused known "
        "affordances are not a whitelist. State why they are insufficient, the "
        "actual activation path, matched-parent control, falsifier, and honest cost."
    ),
}


def _effect_first_recovery_guidance(
    provider_context_view: Mapping[str, Any],
) -> str:
    """Prefer measured-effect recovery after a weak result without constraining search."""

    state = provider_context_view.get("state")
    state = state if isinstance(state, Mapping) else {}
    latest = state.get("latest_result")
    latest = latest if isinstance(latest, Mapping) else {}
    delta = latest.get("delta")
    negative_delta = (
        isinstance(delta, (int, float))
        and not isinstance(delta, bool)
        and delta < 0
        and latest.get("frontier_updated") is False
    )
    objective = provider_context_view.get("objective")
    objective = objective if isinstance(objective, Mapping) else {}
    parent = objective.get("parent_anchor")
    if isinstance(parent, Mapping):
        if latest.get("baseline_position") != "BELOW_FROZEN_PARENT" and not negative_delta:
            return ""
        return (
            "\n\nEFFECT-FIRST RECOVERY (negative result): Choose the next construction "
            "parent from the available state.construction_parent_options when supplied; "
            "the frozen root remains available even when a better descendant exists. "
            "Otherwise use the supplied construction parent. Keep the best measured "
            "result independent of this choice and the frozen root as the paired comparator. "
            "Use the measured failure to choose a different causal delta or a "
            "simpler faithful realization; do not promote the failed candidate, "
            "relabel an implementation failure as mechanism "
            "evidence, or maximize structural distance for its own sake."
        )
    if (
        latest.get("baseline_position") != "BELOW_BASIC_BASELINE"
        and not negative_delta
    ):
        return ""
    basic = objective.get("basic_baseline")
    basic = basic if isinstance(basic, Mapping) else {}
    basic_name = basic.get("name")
    basic_name = (
        basic_name
        if isinstance(basic_name, str) and basic_name
        else "the basic baseline"
    )
    return (
        "\n\nEFFECT-FIRST RECOVERY (weak measured result): The latest result is "
        "below the basic floor or has a negative comparator delta. Restore "
        "expected measured effect before maximizing novelty: use the strongest "
        "measured parent/frontier or objective.basic_baseline ("
        + basic_name
        + ") as the measurable starting point. Choose the scale and combination "
        "of changes from actual variant results, not a blanket preference for "
        "smaller changes. Use structural distance to identify equivalent replay, "
        "not as the objective or a ban on related new variants. The complete "
        "configured mechanism language remains open, including architecture "
        "rewrites when their coherent mechanism offers a credible effect gain."
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


def _proposal_lanes(
    source: Mapping[str, str] | None,
) -> dict[str, str]:
    lanes = (
        dict(_DEFAULT_PROPOSAL_LANE_BY_ROLE)
        if source is None
        else {str(role): str(lane) for role, lane in source.items()}
    )
    if set(lanes) != set(DISCOVERY_PRODUCERS):
        raise ValueError(
            "proposal_lane_by_role must bind exactly the four Producer roles"
        )
    invalid = sorted(set(lanes.values()) - _PROPOSAL_LANES)
    if invalid:
        raise ValueError("unknown proposal lane: " + ", ".join(invalid))
    return canonical_value(lanes)


def _expected_intent(producer_role: str) -> ProposalIntentV1:
    return (
        ProposalIntentV1.FALSIFICATION
        if producer_role == "falsification_designer"
        else ProposalIntentV1.DISCOVERY
    )


def _write_role_bound_schema(schema: Mapping[str, Any], path: Path) -> Path:
    jsonschema.validators.validator_for(schema).check_schema(schema)
    schema_bytes = canonical_json_bytes(schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(schema_bytes)
    except FileExistsError:
        pass
    if path.read_bytes() != schema_bytes:
        raise fresh_r1.FreshR1Error(
            "sealed role-bound proposal schema does not match this call"
        )
    return path


def _provider_wire_program_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt the open-ended program maps to the Provider's closed-object wire.

    BL-ICF keeps parameter maps open at the mechanism-language boundary.  The
    laboratory Provider requires every response object to have a finite set of
    required properties, so those maps travel as canonical JSON object strings
    and are decoded before the mechanism compiler sees them.  Component source
    identity is likewise represented by two required nullable fields on the
    wire, then restored to the internal exactly-one-of shape.
    """

    adapted = canonical_value(dict(schema))
    payload = adapted["properties"]["program_payload"]
    payload_properties = payload["properties"]
    architecture = payload_properties["architecture_operators"]["items"]
    components = payload_properties["components"]["items"]
    for contract in (architecture, components):
        contract["properties"]["parameters"] = {
            "minLength": 2,
            "type": "string",
        }
    components.pop("oneOf", None)
    for field in ("primitive_id", "custom_component_id"):
        field_schema = components["properties"][field]
        components["properties"][field] = {
            "anyOf": [field_schema, {"type": "null"}]
        }
    components["required"] = list(components["properties"])
    return canonical_value(adapted)


def _provider_wire_program_schema_with_exact_component_identity(
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exactly one primitive/custom identity on the mini wire."""

    adapted = _provider_wire_program_schema(schema)
    components = adapted["properties"]["program_payload"]["properties"][
        "components"
    ]["items"]
    identity_contracts: dict[str, Mapping[str, Any]] = {}
    for field in ("primitive_id", "custom_component_id"):
        alternatives = components["properties"][field].get("anyOf", [])
        identity_contracts[field] = next(
            canonical_value(dict(contract))
            for contract in alternatives
            if contract.get("type") == "string"
        )

    branches: list[Mapping[str, Any]] = []
    for active_field, inactive_field in (
        ("primitive_id", "custom_component_id"),
        ("custom_component_id", "primitive_id"),
    ):
        branch = canonical_value(dict(components))
        branch["properties"][active_field] = identity_contracts[active_field]
        branch["properties"][inactive_field] = {"type": "null"}
        branches.append(branch)
    components.clear()
    components["anyOf"] = branches
    return canonical_value(adapted)


def _json_type_for_provider_wire(value: Any) -> str:
    """Return the explicit JSON type required by strict Provider schemas."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    raise TypeError(f"unsupported Provider wire JSON value: {type(value)!r}")


def _provider_wire_with_explicit_json_types(
    schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Add type annotations that do not change const/enum semantics.

    JSON Schema permits ``const`` and ``enum`` without ``type``.  The strict
    response-schema transport rejects that otherwise equivalent spelling, so
    the adaptation stays at the Provider wire boundary and leaves the BL-ICF
    mechanism language and compiler schema unchanged.
    """

    adapted = canonical_value(dict(schema))

    def enrich(node: Any) -> None:
        if isinstance(node, dict):
            # Structured-output transport does not accept this standard JSON
            # Schema annotation.  The decoded MechanismProgram is validated
            # again by the unchanged BL-ICF compiler schema, which retains the
            # uniqueness contract.
            node.pop("uniqueItems", None)
            if "oneOf" in node:
                # The only BL-ICF use is the DATA/COMPONENT source union.
                # Its branches are disjoint by a required ``kind`` constant,
                # so Provider-supported anyOf is equivalent on the wire.
                node["anyOf"] = node.pop("oneOf")
            if "type" not in node and "const" in node:
                node["type"] = _json_type_for_provider_wire(node["const"])
            if "type" not in node and isinstance(node.get("enum"), list):
                enum_types = sorted(
                    {
                        _json_type_for_provider_wire(value)
                        for value in node["enum"]
                    }
                )
                if enum_types:
                    node["type"] = (
                        enum_types[0] if len(enum_types) == 1 else enum_types
                    )
            for value in node.values():
                enrich(value)
        elif isinstance(node, list):
            for value in node:
                enrich(value)

    enrich(adapted)
    return canonical_value(adapted)


def _materialize_role_bound_proposal_schema(
    *,
    call_root: Path,
    proposal_schema: Mapping[str, Any],
    producer_role: str,
) -> Path:
    """Seal the proposal schema with the role assigned to this call."""

    schema = canonical_value(dict(proposal_schema))
    try:
        role_contract = schema["properties"]["proposals"]["items"][
            "properties"
        ]["producer_role"]
        allowed_roles = role_contract["enum"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "proposal schema lacks the producer_role enum contract"
        ) from error
    if not isinstance(allowed_roles, list) or producer_role not in allowed_roles:
        raise ValueError("producer_role is absent from the proposal schema")
    role_contract["enum"] = [producer_role]
    schema_path = (
        Path(call_root)
        / "proposal-schemas"
        / f"{producer_role}.schema.json"
    )
    # Open-spec consumers validate the response against the original schema.
    # Use the existing transport adaptation here as in the BL-ICF lane.
    return _write_role_bound_schema(
        _provider_wire_with_explicit_json_types(schema), schema_path
    )


def _materialize_role_bound_bl_icf_schema(
    *,
    call_root: Path,
    proposal_schema: Mapping[str, Any],
    producer_role: str,
) -> Path:
    """Narrow the existing campaign wire contract to one typed role call."""

    schema = canonical_value(dict(proposal_schema))
    try:
        proposals_contract = schema["properties"]["proposals"]
        proposal_contract = proposals_contract["items"]
        proposal_properties = proposal_contract["properties"]
        intent_contract = proposal_properties["proposal_intent"]
        allowed_intents = intent_contract["enum"]
        mechanism_contract = proposal_properties["mechanism_id"]
        primary_operator = proposal_properties["composition"]["properties"][
            "primary_operator_id"
        ]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "BL-ICF proposal schema lacks the campaign proposal contract"
        ) from error
    expected_intent = _expected_intent(producer_role).value
    if (
        not isinstance(allowed_intents, list)
        or expected_intent not in allowed_intents
    ):
        raise ValueError("Producer intent is absent from the BL-ICF schema")
    proposals_contract["minItems"] = 1
    proposals_contract["maxItems"] = 1
    intent_contract["enum"] = [expected_intent]
    executable = tuple(
        mechanism
        for mechanism in executable_mechanisms()
        if mechanism.operator_ids
    )
    mechanism_contract.pop("pattern", None)
    mechanism_contract["enum"] = [
        mechanism.mechanism_id for mechanism in executable
    ]
    primary_operator.pop("type", None)
    primary_operator["enum"] = sorted(
        {mechanism.operator_ids[0] for mechanism in executable}
    )
    secondary_operator = proposal_properties["composition"]["properties"][
        "secondary_operator_id"
    ]
    secondary_operator.pop("type", None)
    secondary_operator["enum"] = [
        None,
        *sorted(
            {
                mechanism.operator_ids[1]
                for mechanism in executable
                if len(mechanism.operator_ids) == 2
            }
        ),
    ]
    schema_path = (
        Path(call_root)
        / "proposal-schemas"
        / "bl-icf"
        / f"{producer_role}.schema.json"
    )
    return _write_role_bound_schema(schema, schema_path)


def _materialize_role_bound_bl_icf_program_schema(
    *,
    call_root: Path,
    proposal_schema: Mapping[str, Any],
    producer_role: str,
    frozen_profile_ref: Mapping[str, Any],
    required_base_model_config: str | None = None,
) -> Path:
    """Bind the full provider program schema to one role and real profile."""

    schema = canonical_value(dict(proposal_schema))
    try:
        proposal_contract = schema["properties"]["proposals"]["items"]
        properties = proposal_contract["properties"]
        properties["producer_role"]["const"] = producer_role
        wire_schema_factory = (
            _provider_wire_program_schema_with_exact_component_identity
            if producer_role in _INDEPENDENT_MINI_PRODUCER_ROLES
            else _provider_wire_program_schema
        )
        properties["mechanism_program"] = wire_schema_factory(
            program_schema("BL_ICF_MECHANISM_SPACE_V1")
        )
        properties["implementation_research"]["properties"]["base_model_config"] = {
            "enum": (
                [required_base_model_config]
                if required_base_model_config is not None
                else list(_FROZEN_QUALIFICATION_BASE_MODEL_CONFIGS)
            ),
            "type": "string",
        }
    except (KeyError, TypeError) as error:
        raise ValueError(
            "strict BL-ICF program proposal schema lacks its wire contract"
        ) from error
    profile = canonical_value(dict(frozen_profile_ref))
    properties["mechanism_program"]["properties"]["profile_ref"] = {
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
    schema = _provider_wire_with_explicit_json_types(schema)
    schema_namespace = (
        "bl-icf-program-mini-exact-identity-v1"
        if producer_role in _INDEPENDENT_MINI_PRODUCER_ROLES
        else "bl-icf-program"
    )
    schema_path = (
        Path(call_root)
        / "proposal-schemas"
        / schema_namespace
        / f"{producer_role}.schema.json"
    )
    return _write_role_bound_schema(schema, schema_path)


def _render_bl_icf_proposal_prompt(
    template: str,
    *,
    side_identity: str,
    logical_slot_id: str,
    proposal_seed: int,
    producer_role: str,
    frozen_profile_ref: Mapping[str, Any],
    single_parent: bool = False,
    role_instruction: str | None = None,
) -> str:
    scientific_profile = canonical_value(dict(frozen_profile_ref))
    replacements = {
        "{{SIDE_IDENTITY}}": side_identity,
        "{{LOGICAL_SLOT_ID}}": logical_slot_id,
        "{{PROPOSAL_SEED}}": str(proposal_seed),
        "{{PRODUCER_ROLE}}": producer_role,
        "{{PRODUCER_ROLE_INSTRUCTION}}": role_instruction or (
            _SINGLE_PARENT_BL_ICF_ROLE_INSTRUCTIONS
            if single_parent
            else _BL_ICF_ROLE_INSTRUCTIONS
        )[producer_role],
        "{{SCIENTIFIC_PROFILE_ID}}": scientific_profile["profile_id"],
        "{{SCIENTIFIC_PROFILE_DIGEST}}": scientific_profile[
            "profile_digest"
        ],
        "{{SCIENTIFIC_PROFILE_KIND}}": scientific_profile["profile_kind"],
    }
    rendered = template
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    if "{{" in rendered or "}}" in rendered:
        raise fresh_r1.FreshR1Error(
            "BL-ICF proposal prompt has an unresolved placeholder"
        )
    return rendered


def _factor_bl_icf_prompt_projection() -> Mapping[str, Any]:
    """Intern repeated primitive contracts without reducing the search space."""

    projection = canonical_value(prompt_projection("BL_ICF_MECHANISM_SPACE_V1"))
    axes = projection.get("axes")
    if not isinstance(axes, (tuple, list)):
        raise fresh_r1.FreshR1Error("BL-ICF prompt projection has no axes")
    contract_ref_by_wire: dict[bytes, str] = {}
    primitive_contracts: dict[str, Mapping[str, Any]] = {}
    factored_axes: list[Mapping[str, Any]] = []
    for axis in axes:
        if not isinstance(axis, Mapping):
            raise fresh_r1.FreshR1Error("BL-ICF prompt projection axis is invalid")
        slot_id = axis.get("slot_id")
        primitives = axis.get("primitives")
        if not isinstance(slot_id, str) or not isinstance(primitives, (tuple, list)):
            raise fresh_r1.FreshR1Error(
                "BL-ICF prompt projection axis contract is invalid"
            )
        refs: list[tuple[str, str]] = []
        for primitive in primitives:
            if not isinstance(primitive, Mapping):
                raise fresh_r1.FreshR1Error(
                    "BL-ICF prompt projection primitive is invalid"
                )
            primitive_id = primitive.get("primitive_id")
            if not isinstance(primitive_id, str) or primitive.get("slot_id") != slot_id:
                raise fresh_r1.FreshR1Error(
                    "BL-ICF prompt projection primitive identity drift"
                )
            contract = canonical_value(
                {
                    key: value
                    for key, value in primitive.items()
                    if key not in {"primitive_id", "slot_id"}
                }
            )
            wire = canonical_json_bytes(contract)
            contract_ref = contract_ref_by_wire.get(wire)
            if contract_ref is None:
                contract_ref = f"C{len(contract_ref_by_wire) + 1:03d}"
                contract_ref_by_wire[wire] = contract_ref
                primitive_contracts[contract_ref] = contract
            refs.append((primitive_id, contract_ref))
        factored_axes.append(
            canonical_value(
                {
                    "slot_id": slot_id,
                    "allow_multiple": axis.get("allow_multiple"),
                    "primitive_contract_refs": tuple(refs),
                }
            )
        )
    return canonical_value(
        {
            **{key: value for key, value in projection.items() if key != "axes"},
            "projection_encoding": "FACTORED_PRIMITIVE_CONTRACTS_V1",
            "source_projection_digest": sha256_digest(projection),
            "reconstruction_rule": (
                "For every factored_axes primitive_contract_refs pair "
                "[primitive_id, contract_ref], reconstruct one primitive by "
                "merging primitive_contracts[contract_ref] with that "
                "primitive_id and the enclosing slot_id, preserving pair order."
            ),
            "factored_axes": tuple(factored_axes),
            "primitive_contracts": primitive_contracts,
        }
    )


def _compact_bl_icf_prompt_projection() -> Mapping[str, Any]:
    """Co-locate mini primitive identity and its exact mechanical contract."""

    projection = canonical_value(prompt_projection("BL_ICF_MECHANISM_SPACE_V1"))
    factored = dict(_factor_bl_icf_prompt_projection())
    factored_axes = factored.pop("factored_axes")
    primitive_refs = {
        primitive_id: contract_ref
        for axis in factored_axes
        for primitive_id, contract_ref in axis["primitive_contract_refs"]
    }
    catalogs: dict[str, list[str]] = {
        "parameter_key": [],
        "port": [],
        "slot": [],
    }
    catalog_indices: dict[str, dict[str, int]] = {
        key: {} for key in catalogs
    }

    def intern(catalog: str, value: str) -> int:
        index = catalog_indices[catalog].get(value)
        if index is None:
            index = len(catalogs[catalog])
            catalogs[catalog].append(value)
            catalog_indices[catalog][value] = index
        return index

    rows: list[list[Any]] = []
    for axis in projection["axes"]:
        for primitive in axis["primitives"]:
            parameter_schema = primitive["parameter_schema"]
            rows.append(
                [
                    primitive["primitive_id"],
                    intern("slot", axis["slot_id"]),
                    primitive_refs[primitive["primitive_id"]],
                    [
                        intern("parameter_key", value)
                        for value in sorted(parameter_schema.get("properties", {}))
                    ],
                    [
                        intern("parameter_key", value)
                        for value in sorted(parameter_schema.get("required", []))
                    ],
                    [
                        [intern("port", item["port"]), item["minimum"]]
                        for item in primitive["input_ports"]
                    ],
                    [
                        intern("port", item["port"])
                        for item in primitive["output_ports"]
                    ],
                ]
            )
    factored.update(
        {
            "projection_encoding": (
                "FACTORED_PRIMITIVE_CONTRACTS_WITH_ASSEMBLY_INDEX_V2"
            ),
            "reconstruction_rule": (
                "Use assembly_index rows to bind every primitive_id to its "
                "exact slot_ref, contract_ref, parameter-key refs, and port "
                "refs. Merge primitive_contracts[contract_ref] for full type "
                "and capability details. Do not invent unlisted parameters "
                "or port names."
            ),
            "axis_cardinality": [
                [axis["slot_id"], axis["allow_multiple"]]
                for axis in factored_axes
            ],
            "program_reference_rules": {
                "component_references": (
                    "Every architecture-operator target/replacement and every "
                    "ablation remove_component_id must be an exact component_id "
                    "from this response, never a primitive_id, slot_id, or "
                    "operator_id."
                ),
                "required_exactly_once_slots": ["ENCODER", "SCORE_HEAD"],
                "changed_slots": (
                    "Declare the complete focused, causally coherent changed-slot "
                    "footprint relative to the selected construction parent. Do not "
                    "add unrelated mechanisms merely to increase distance."
                ),
            },
            "assembly_index": {
                "encoding": "PRIMITIVE_ASSEMBLY_INDEX_V1",
                "fields": [
                    "primitive_id",
                    "slot_ref",
                    "contract_ref",
                    "allowed_parameter_key_refs",
                    "required_parameter_key_refs",
                    "input_port_ref_minimum",
                    "output_port_refs",
                ],
                "slot_catalog": catalogs["slot"],
                "parameter_key_catalog": catalogs["parameter_key"],
                "port_catalog": catalogs["port"],
                "rows": rows,
            },
        }
    )
    return canonical_value(factored)


def _render_bl_icf_program_proposal_prompt(
    template: str,
    *,
    side_identity: str,
    logical_slot_id: str,
    proposal_seed: int,
    producer_role: str,
    frozen_profile_ref: Mapping[str, Any],
    focused_parent_language: bool = False,
    role_instruction: str | None = None,
) -> str:
    rendered = _render_bl_icf_proposal_prompt(
        template,
        side_identity=side_identity,
        logical_slot_id=logical_slot_id,
        proposal_seed=proposal_seed,
        producer_role=producer_role,
        frozen_profile_ref=frozen_profile_ref,
        single_parent=focused_parent_language,
        role_instruction=role_instruction,
    )
    frozen_marker = "Frozen inputs:\n"
    if rendered.count(frozen_marker) != 1:
        raise fresh_r1.FreshR1Error(
            "BL-ICF program prompt does not contain one frozen-input block"
        )
    stable_head, frozen_tail = rendered.split(frozen_marker, 1)
    frozen_body, stable_body = frozen_tail.split("\n\n", 1)
    frozen_inputs = frozen_marker + frozen_body
    stable_rendered = stable_head + stable_body
    if focused_parent_language:
        prompt_projection_value = focused_mechanism_language()
        language_introduction = (
            "\n\nThe following focused parent-relative mechanism language is "
            "the research map for this call. Known affordances are compact "
            "implementation material, not a whitelist or a candidate catalog. "
            "The unchanged v1 compiler still owns the full registry. Use a typed "
            "custom component or architecture rewrite when the causal hypothesis "
            "is not captured by the known affordances. Choose a parent from "
            "state.construction_parent_options when supplied, including the frozen root; "
            "otherwise use the supplied construction parent. Preserve undeclared "
            "parent slots. Declare every changed slot and custom-component port "
            "accurately. Source ownership is compiler-derived from that scientific "
            "delta; do not select it as a research mechanism. CUSTOM_MODEL keeps "
            "the narrowest available exact-parent seam, while "
            "ARCHITECTURE_REWRITE retains full-source freedom when the declared "
            "causal thesis genuinely requires it:\n"
        )
    else:
        prompt_projection_value = prompt_projection("BL_ICF_MECHANISM_SPACE_V1")
        if producer_role in _INDEPENDENT_MINI_PRODUCER_ROLES:
            prompt_projection_value = _compact_bl_icf_prompt_projection()
        elif producer_role in _FACTORED_BL_ICF_PRODUCER_ROLES:
            prompt_projection_value = _factor_bl_icf_prompt_projection()
        language_introduction = (
            "\n\nThe following canonical BL-ICF prompt projection is the "
            "complete search language for this legacy call. Use any valid "
            "primitive, architecture operator, architecture rewrite, or "
            "CUSTOM_MODEL escape; do not project it to an executable catalog. "
            "On the Provider response wire, encode every component and "
            "architecture-operator parameters map as a canonical JSON object "
            "string. Emit both primitive_id and custom_component_id for every "
            "component, using null for exactly the one that is not active. "
            "These are lossless wire encodings, not a smaller mechanism language:\n"
        )
    return (
        stable_rendered
        + "\n\nAuthoritative topology scaffold (choose the endpoints explicitly; "
        "the compiler will not change them): ID -> embedding -> "
        "representation -> score -> objective; relation/filter -> encoder; "
        "sampler -> pairwise supervision; objective -> training recipe."
        + "\n\nFrozen resolver budget ceilings for this call are shown below. "
        "Keep resolution_facts.required_budget at or below every corresponding "
        "ceiling, while reporting an honest implementation and qualification "
        "estimate. Do not inflate, reinterpret, or modify these frozen limits:\n"
        + canonical_json_bytes(fresh_r1.BUDGET_LIMITS).decode("utf-8")
        + language_introduction
        + canonical_json_bytes(prompt_projection_value).decode("utf-8")
        + "\n\n"
        + frozen_inputs
    )


def _compile_failure_summary(report: CompileReportV1) -> dict[str, Any]:
    serialized = report.to_dict()
    return canonical_value(
        {
            "diagnostics": serialized["diagnostics"],
            "status": serialized["status"],
        }
    )


class _BlIcfProgramCompileError(fresh_r1.FreshR1Error):
    def __init__(self, report: CompileReportV1) -> None:
        self.compiler_report = _compile_failure_summary(report)
        detail = canonical_json_bytes(self.compiler_report).decode("utf-8")
        super().__init__(
            "strict BL-ICF Provider mechanism_program does not compile as "
            f"VALID_NEEDS_IMPLEMENTATION: {detail}"
        )


def _render_bl_icf_program_semantic_repair_prompt(
    *,
    original_prompt: str,
    original_proposal: Mapping[str, Any],
    compiler_report: Mapping[str, Any],
) -> str:
    return (
        original_prompt
        + "\n\nSEMANTIC COMPILER REPAIR (one bounded call only): The prior "
        "response passed the strict Provider JSON schema and lossless wire "
        "decoding, but the package compiler rejected its mechanism semantics. "
        "Return exactly one corrected proposal under the same strict response "
        "schema. Apply only the smallest changes required by the compiler "
        "diagnostics. Preserve the producer role, scientific hypothesis, "
        "profile_ref, parent_refs, execution contract, resolution budget, and "
        "honestly declared estimated_cost unless a diagnostic directly requires "
        "a corresponding semantic correction. Do not relabel cost, reduce the "
        "frozen training envelope, use a fixed-catalog projection, or substitute "
        "a fallback. This repair receives no metric or outcome evidence. If the "
        "proposal cannot be repaired within these constraints, return your best "
        "strictly conforming correction; a second compiler failure will be "
        "preserved as a typed failure without another call.\n\n"
        "Exact compiler report:\n"
        + canonical_json_bytes(compiler_report).decode("utf-8")
        + "\n\nOriginal Provider proposal:\n"
        + canonical_json_bytes(original_proposal).decode("utf-8")
    )


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
    successful_attempt = next(
        (attempt for attempt in result.attempts if attempt.get("status") == "SUCCESS"),
        result.attempts[-1] if result.attempts else {},
    )
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
        "requested_model": successful_attempt.get("requested_model"),
        "provider_role": successful_attempt.get("provider_role"),
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
            "sealed_request": result.sealed_request,
        }
    )


def _latest_research_traces_by_role(
    traces: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Associate transport stages with the research role that owns the idea."""
    latest = {}
    for trace in traces:
        if not isinstance(trace, Mapping) or trace.get("kind") not in {
            "research_producer", "declarative_research_producer",
            "declarative_research_producer_semantic_repair",
        }:
            continue
        receipt = trace.get("receipt")
        role = receipt.get("provider_role") if isinstance(receipt, Mapping) else None
        if trace["kind"] == "declarative_research_producer_semantic_repair":
            # The transport role is mechanism_semantic_repair; the native call
            # identity retains the original research role immediately before this suffix.
            prefix, separator, _ = str(trace.get("logical_call_id", "")).rpartition(
                ":declarative-program-proposal"
            )
            role = prefix.rsplit(":", 1)[-1] if separator else None
        if role in DISCOVERY_PRODUCERS:
            latest[role] = trace
    return latest


def _resume_transport_kwargs(
    kwargs: dict[str, Any], trace: Mapping[str, Any] | None,
) -> None:
    """Reuse the saved wire request while only its physical identity advances."""
    if trace is None:
        return
    sealed = trace.get("sealed_request")
    payload = sealed.get("request_payload") if isinstance(sealed, Mapping) else None
    if not isinstance(payload, Mapping):
        return
    responses = "input" in payload
    kwargs["prompt"] = payload["input"] if responses else payload["messages"][0]["content"]
    ceiling = trace.get("transport_ceiling", {})
    kwargs["token_ceiling"] = ceiling.get("total_tokens", kwargs["token_ceiling"])
    kwargs["output_token_ceiling"] = ceiling.get(
        "output_tokens", payload.get("max_output_tokens", payload.get("max_completion_tokens"))
    )
    if "model" in payload:
        kwargs["requested_model"] = payload["model"]
    # A saved request also owns its schema and transport format. The source can
    # render other peers while this exact call is still awaiting a response.
    format_payload = payload.get("text" if responses else "response_format", {})
    format_schema = format_payload.get("format" if responses else "json_schema", {}).get("schema")
    if isinstance(format_schema, Mapping):
        kwargs["schema_path"] = _write_role_bound_schema(
            format_schema, Path(kwargs["call_root"]) / "proposal-schemas"
            / f"sealed-{sha256_digest(format_schema)}.json",
        )
        kwargs["wire_api"] = "responses" if responses else "chat_completions"


def _record_trace(
    owner: Any,
    *,
    kind: str,
    logical_call_id: str,
    prompt: str,
    result: fresh_r1.ProviderAttemptResult,
    transport_ceiling: Mapping[str, int] | None = None,
) -> None:
    trace = _call_trace(
        kind=kind,
        config_identity=owner.config_identity,
        logical_call_id=logical_call_id,
        session_id=owner.session_id,
        prompt=prompt,
        result=result,
    )
    if transport_ceiling is not None:
        trace = canonical_value({**trace, "transport_ceiling": transport_ceiling})
    owner.last_call_trace = trace
    owner.call_traces = (*owner.call_traces, trace)


def _record_replay_trace(
    owner: Any,
    *,
    logical_call_id: str,
    prompt: str,
    replay: Mapping[str, Any],
) -> None:
    source = canonical_value(dict(replay["source"]))
    prompt_digest = sha256_digest(prompt)
    trace = canonical_value(
        {
            "kind": "research_producer_replay",
            "config_identity": owner.config_identity,
            "logical_call_id": logical_call_id,
            "session_id": owner.session_id,
            "prompt_digest": prompt_digest,
            "receipt": {
                "logical_call_id": logical_call_id,
                "proposal_generation_session_id": owner.session_id,
                "request_digest": sha256_digest(
                    {
                        "logical_call_id": logical_call_id,
                        "prompt_digest": prompt_digest,
                        "replay_source_digest": source["response_digest"],
                    }
                ),
                "response_digest": source["response_digest"],
                "release_digest": owner.config_identity.get("release_digest"),
                "status": "SUCCESS",
                "latency_ms": 0,
                "returned_model": source.get("returned_model"),
            },
            "attempts": (),
            "usage": {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
                "billed_tokens": 0,
            },
            "failure": None,
            "replay_source": source,
        }
    )
    owner.last_call_trace = trace
    owner.call_traces = (*owner.call_traces, trace)


def _require_success(
    result: fresh_r1.ProviderAttemptResult,
    *,
    kind: str,
) -> Any:
    if result.call is None:
        if kind == "implementation" and fresh_r1.provider_failure_is_external(result.failure):
            raise fresh_r1.ProviderUnavailableError(result, kind=kind)
        raise fresh_r1.FreshR1Error(f"{kind} Provider call failed")
    return result.call.response


def _physical_attempt_output_tokens(
    result: fresh_r1.ProviderAttemptResult,
) -> int:
    return sum(
        int(attempt.get("output_tokens") or 0)
        for attempt in result.attempts
        if isinstance(attempt, Mapping)
    )


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


def _normalize_open_spec_proposal(
    proposal: Mapping[str, Any],
) -> dict[str, Any]:
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


def _normalize_implementation_research(
    proposal: Mapping[str, Any],
    *,
    required_base_model_config: str | None = None,
) -> dict[str, Any]:
    """Canonicalize only LLM-owned research choices, never stable runtime ABI."""

    normalized = dict(proposal)
    research = normalized.get("implementation_research")
    if not isinstance(research, Mapping):
        raise fresh_r1.FreshR1Error(
            "strict BL-ICF proposal implementation_research must be an object"
        )
    base_model_config = research.get("base_model_config")
    allowed_base_models = (
        (required_base_model_config,)
        if required_base_model_config is not None
        else _FROZEN_QUALIFICATION_BASE_MODEL_CONFIGS
    )
    if base_model_config not in allowed_base_models:
        raise fresh_r1.FreshR1Error(
            "implementation_research.base_model_config must select a frozen family"
        )
    config_json = research.get("mechanism_config_json")
    if not isinstance(config_json, str):
        raise fresh_r1.FreshR1Error(
            "implementation_research.mechanism_config_json must be a string"
        )
    try:
        mechanism_config = json.loads(config_json)
    except json.JSONDecodeError:
        # MechanismProgram is the authoritative research source.  This field is
        # only an optional, duplicate runtime-hyperparameter projection, so a
        # malformed copy cannot invalidate an otherwise exact program.
        mechanism_config = {}
    if not isinstance(mechanism_config, Mapping):
        raise fresh_r1.FreshR1Error(
            "implementation_research.mechanism_config_json must encode an object"
        )
    mechanism_config = {
        str(key): canonical_value(value)
        for key, value in mechanism_config.items()
        if key not in _MACHINE_EXECUTION_CONFIG_FIELDS
    }
    normalized["implementation_research"] = canonical_value(
        {
            "base_model_config": base_model_config,
            "mechanism_config": dict(mechanism_config),
        }
    )
    return canonical_value(normalized)


def _decode_provider_wire_program(program: Mapping[str, Any]) -> dict[str, Any]:
    decoded = canonical_value(dict(program))
    try:
        payload = decoded["program_payload"]
        architecture = payload["architecture_operators"]
        components = payload["components"]
    except (KeyError, TypeError) as error:
        raise fresh_r1.FreshR1Error(
            "strict BL-ICF Provider wire lacks the program payload"
        ) from error
    for kind, rows in (
        ("architecture_operators", architecture),
        ("components", components),
    ):
        if not isinstance(rows, list):
            raise fresh_r1.FreshR1Error(
                f"strict BL-ICF Provider wire {kind} must be an array"
            )
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise fresh_r1.FreshR1Error(
                    f"strict BL-ICF Provider wire {kind}[{index}] must be an object"
                )
            parameters = row.get("parameters")
            if isinstance(parameters, str):
                try:
                    parameters = json.loads(parameters)
                except json.JSONDecodeError as error:
                    if kind == "architecture_operators":
                        # Architecture-operator parameters are an open metadata
                        # map, not executable component configuration.  The
                        # strict Provider wire has to carry that open map as a
                        # string, and research agents naturally sometimes use
                        # the string for the operator instruction itself.  Keep
                        # that instruction losslessly instead of discarding the
                        # entire mechanism batch at this transport boundary.
                        parameters = {"description": parameters}
                    else:
                        raise fresh_r1.FreshR1Error(
                            "strict BL-ICF Provider parameters string is not JSON"
                        ) from error
            if not isinstance(parameters, Mapping):
                raise fresh_r1.FreshR1Error(
                    "strict BL-ICF Provider parameters must encode an object"
                )
            row["parameters"] = canonical_value(dict(parameters))
    for component in components:
        for field in ("primitive_id", "custom_component_id"):
            if component.get(field) is None:
                component.pop(field, None)
    return canonical_value(decoded)


def _fresh_compiler_component_id(
    component_ids: set[str],
    stem: str,
) -> str:
    candidate = stem
    suffix = 1
    while candidate in component_ids:
        suffix += 1
        candidate = f"{stem}_{suffix}"
    component_ids.add(candidate)
    return candidate


def _input_source(
    component: Mapping[str, Any],
    port: str,
) -> dict[str, Any] | None:
    matches = [
        item.get("source")
        for item in component.get("inputs", [])
        if isinstance(item, Mapping) and item.get("port") == port
    ]
    if len(matches) != 1 or not isinstance(matches[0], Mapping):
        return None
    return canonical_value(dict(matches[0]))


def _set_input_source(
    component: dict[str, Any],
    port: str,
    source: Mapping[str, Any],
) -> None:
    inputs = [dict(item) for item in component.get("inputs", [])]
    replacement = {"port": port, "source": canonical_value(dict(source))}
    indices = [index for index, item in enumerate(inputs) if item.get("port") == port]
    if len(indices) == 1:
        inputs[indices[0]] = replacement
    elif not indices:
        inputs.append(replacement)
    else:
        return
    component["inputs"] = inputs


def _compiler_primitive_contracts() -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, str],
]:
    """Return package-owned primitive and DATA type contracts."""

    projection = canonical_value(
        prompt_projection("BL_ICF_MECHANISM_SPACE_V1")
    )
    primitives: dict[str, Mapping[str, Any]] = {}
    for axis in projection.get("axes", []):
        if not isinstance(axis, Mapping):
            continue
        for primitive in axis.get("primitives", []):
            if isinstance(primitive, Mapping) and isinstance(
                primitive.get("primitive_id"), str
            ):
                primitives[str(primitive["primitive_id"])] = primitive
    data_types = {
        str(item["role_id"]): str(item["type_ref"])
        for item in projection.get("allowed_data_roles", [])
        if isinstance(item, Mapping)
        and isinstance(item.get("role_id"), str)
        and isinstance(item.get("type_ref"), str)
    }
    return primitives, data_types


def _canonicalize_registry_owned_component_slots(
    program: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve only the redundant slot owned by an exact primitive contract."""

    normalized = canonical_value(dict(program))
    payload = dict(normalized.get("program_payload", {}))
    components = [dict(item) for item in payload.get("components", [])]
    primitive_contracts, _ = _compiler_primitive_contracts()
    for component in components:
        primitive_id = component.get("primitive_id")
        contract = (
            primitive_contracts.get(primitive_id)
            if isinstance(primitive_id, str)
            else None
        )
        if contract is not None:
            component["slot_id"] = contract["slot_id"]
    payload["components"] = components
    normalized["program_payload"] = payload
    return canonical_value(normalized)


def _compiler_owned_interface_normalization(
    program: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonicalize only uniquely determined compiler interface syntax.

    Primitive choice, component topology, source targets, parameters, change
    roles, hypotheses, and ablations remain Provider-owned research content.
    This function normalizes a known ``primitive.`` wire prefix, structural
    DATA/COMPONENT kind tags, and unique source-output/target-input port
    bindings.  If more than one binding is possible, the original bytes are
    preserved so the strict compiler returns typed diagnostics.
    """

    normalized = canonical_value(dict(program))
    payload = dict(normalized.get("program_payload", {}))
    components = [dict(item) for item in payload.get("components", [])]
    primitive_contracts, data_types = _compiler_primitive_contracts()

    for component in components:
        primitive_id = component.get("primitive_id")
        if (
            isinstance(primitive_id, str)
            and primitive_id.startswith("primitive.")
        ):
            wire_id = primitive_id[len("primitive.") :]
            candidates = [
                candidate_id
                for candidate_id, contract in primitive_contracts.items()
                if contract.get("slot_id") == component.get("slot_id")
                and (
                    candidate_id == wire_id
                    or candidate_id.split(".", 1)[-1] == wire_id
                )
            ]
            if len(candidates) == 1:
                component["primitive_id"] = candidates[0]

    component_ids = [str(item.get("component_id")) for item in components]
    if len(set(component_ids)) == len(component_ids):
        components_by_id = {
            str(item.get("component_id")): item for item in components
        }
    else:
        components_by_id = {}
    custom_outputs = {
        str(item.get("custom_component_id")): tuple(item.get("output_ports", []))
        for item in payload.get("custom_components", [])
        if isinstance(item, Mapping)
    }
    changed_relation_slot = any(
        item.get("slot_id") == "RELATION_VIEW"
        for item in payload.get("changed_slots", [])
        if isinstance(item, Mapping)
    )
    removed_relation_slot = "RELATION_VIEW" in payload.get("removed_slots", [])
    operator_mutation_targets = {
        str(component_id)
        for operator in payload.get("architecture_operators", [])
        if isinstance(operator, Mapping)
        for field in ("targets", "replacements")
        for component_id in operator.get(field, [])
    }
    ablation_remove_component_ids = {
        str(component_id)
        for ablation in payload.get("ablation_plan", [])
        if isinstance(ablation, Mapping)
        for component_id in ablation.get("remove_component_ids", [])
    }

    def output_contracts(component: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
        primitive_id = component.get("primitive_id")
        if isinstance(primitive_id, str) and primitive_id in primitive_contracts:
            return tuple(primitive_contracts[primitive_id].get("output_ports", []))
        custom_id = component.get("custom_component_id")
        if isinstance(custom_id, str):
            return tuple(custom_outputs.get(custom_id, ()))
        return ()

    for component in components:
        primitive_id = component.get("primitive_id")
        contract = (
            primitive_contracts.get(str(primitive_id))
            if isinstance(primitive_id, str)
            else None
        )
        if contract is None:
            continue
        target_ports = tuple(contract.get("input_ports", []))
        input_port_counts = {
            str(port): sum(
                item.get("port") == port for item in component.get("inputs", [])
            )
            for port in {
                item.get("port")
                for item in component.get("inputs", [])
                if isinstance(item.get("port"), str)
            }
        }
        inputs: list[dict[str, Any]] = []
        for raw_input in component.get("inputs", []):
            input_item = dict(raw_input)
            raw_source = input_item.get("source")
            if not isinstance(raw_source, Mapping):
                inputs.append(input_item)
                continue
            source = dict(raw_source)
            current_port = input_item.get("port")
            exact_train_input_ports = tuple(
                item
                for item in target_ports
                if item.get("port") == current_port
                and tuple(item.get("accepted_types", ()))
                == ("bl_icf/train_interactions",)
            )
            source_component_id = source.get("component_id")
            source_component = (
                components_by_id.get(source_component_id)
                if isinstance(source_component_id, str)
                else None
            )
            source_component_inputs = (
                tuple(source_component.get("inputs", ()))
                if source_component is not None
                else ()
            )
            # A parameter-free user-item relation with one TRAIN_INTERACTIONS
            # input is only a typed view of those same training interactions.
            # When a uniquely identified target port accepts only the raw
            # interaction type, unwrap that redundant view.  Any transformed,
            # reweighted, multiply sourced, or research-owned relation remains
            # untouched so the strict compiler preserves its typed failure.
            if (
                len(exact_train_input_ports) == 1
                and input_port_counts.get(str(current_port)) == 1
                and set(source) == {"component_id", "kind", "output_port"}
                and source.get("kind") == "COMPONENT"
                and source.get("output_port") == "relation"
                and source_component is not None
                and source_component.get("primitive_id")
                == "relation.user_item_bipartite"
                and source_component.get("custom_component_id") is None
                and source_component.get("slot_id") == "RELATION_VIEW"
                and source_component.get("parameters") == {}
                and len(source_component_inputs) == 1
                and source_component_inputs[0].get("port") == "source"
                and source_component_inputs[0].get("source")
                == {"data_role": "TRAIN_INTERACTIONS", "kind": "DATA"}
                and data_types.get("TRAIN_INTERACTIONS")
                == "bl_icf/train_interactions"
                and not changed_relation_slot
                and not removed_relation_slot
                and "slot:RELATION_VIEW" not in operator_mutation_targets
                and source_component_id not in operator_mutation_targets
                and source_component_id not in ablation_remove_component_ids
            ):
                source = {"data_role": "TRAIN_INTERACTIONS", "kind": "DATA"}
            data_role = source.get("data_role")
            source_component_id = source.get("component_id")
            source_output_port = source.get("output_port")
            if (
                isinstance(data_role, str)
                and source_component_id is None
                and source_output_port is None
            ):
                source = {"data_role": data_role, "kind": "DATA"}
                source_options = ((None, data_types.get(data_role)),)
            elif (
                data_role is None
                and isinstance(source_component_id, str)
                and isinstance(source_output_port, str)
            ):
                source = {
                    "component_id": source_component_id,
                    "kind": "COMPONENT",
                    "output_port": source_output_port,
                }
                source_component = components_by_id.get(source_component_id)
                outputs = (
                    output_contracts(source_component)
                    if source_component is not None
                    else ()
                )
                source_options = tuple(
                    (str(item.get("port")), str(item.get("type")))
                    for item in outputs
                    if isinstance(item.get("port"), str)
                    and isinstance(item.get("type"), str)
                )
            else:
                input_item["source"] = canonical_value(source)
                inputs.append(input_item)
                continue

            compatible = [
                (target_port, output_port)
                for output_port, source_type in source_options
                if isinstance(source_type, str)
                for target_port in target_ports
                if source_type in target_port.get("accepted_types", [])
            ]
            current_port = input_item.get("port")
            current_is_compatible = any(
                target_port.get("port") == current_port
                and (
                    source.get("kind") == "DATA"
                    or output_port == source.get("output_port")
                )
                for target_port, output_port in compatible
            )
            if not current_is_compatible and len(compatible) == 1:
                target_port, output_port = compatible[0]
                input_item["port"] = str(target_port["port"])
                if source.get("kind") == "COMPONENT" and output_port is not None:
                    source["output_port"] = output_port
            input_item["source"] = canonical_value(source)
            inputs.append(input_item)
        component["inputs"] = inputs

    payload["components"] = components
    normalized["program_payload"] = payload
    return canonical_value(normalized)


def _compiler_owned_custom_marker_normalization(
    program: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize only custom-model ABI markers determined by the declared graph."""

    normalized = canonical_value(dict(program))
    payload = dict(normalized.get("program_payload", {}))
    components = [dict(item) for item in payload.get("components", [])]

    # ``synthesize_custom_model`` has no executable meaning without a declared
    # and instantiated custom component.  Some Provider responses nevertheless
    # append one with empty targets/replacements to an otherwise ordinary
    # registry-only COMPOSITION.  Removing that inert marker is uniquely
    # determined by the existing graph; changing construction mode or creating
    # a custom component would instead invent research content and remains a
    # typed failure.
    custom_declarations = tuple(payload.get("custom_components", []))
    custom_instances = tuple(
        item
        for item in components
        if isinstance(item.get("custom_component_id"), str)
    )
    operators = [dict(item) for item in payload.get("architecture_operators", [])]

    # A typed custom component already carries the complete research decision.
    # ``construction_mode`` and the empty synthesis operator are mechanical
    # compiler vocabulary.  When a rewrite or in-budget composition declares
    # and instantiates exactly the same custom identities, canonicalize only
    # those two ABI fields; preserve every research choice and change budget.
    declared_custom_ids = {
        str(item["custom_component_id"])
        for item in custom_declarations
        if isinstance(item, Mapping)
        and isinstance(item.get("custom_component_id"), str)
    }
    instantiated_custom_ids = {
        str(item["custom_component_id"])
        for item in custom_instances
    }
    has_custom_synthesis = any(
        item.get("operator_id") == "synthesize_custom_model"
        for item in operators
    )
    mode = payload.get("construction_mode")
    changes = payload.get("changed_slots", [])
    composition_in_budget = (
        mode == "COMPOSITION"
        and sum(item.get("change_role") == "CORE" for item in changes) == 1
        and sum(item.get("change_role") == "SUPPORT" for item in changes) <= 1
    )
    if (
        (mode == "ARCHITECTURE_REWRITE" or composition_in_budget)
        and declared_custom_ids
        and declared_custom_ids == instantiated_custom_ids
        and len(declared_custom_ids) == len(custom_declarations)
        and not has_custom_synthesis
    ):
        payload["construction_mode"] = "CUSTOM_MODEL"
        operators.append(
            {
                "operator_id": "synthesize_custom_model",
                "targets": [],
                "replacements": [],
                "parameters": {},
                "rationale": (
                    "Materialize the explicitly declared and instantiated "
                    "typed custom mechanism."
                ),
            }
        )
        payload["architecture_operators"] = operators
        normalized["program_payload"] = payload

    inert_synthesis = tuple(
        item
        for item in operators
        if item.get("operator_id") == "synthesize_custom_model"
        and not item.get("targets")
        and not item.get("replacements")
        and not item.get("parameters")
    )
    if (
        payload.get("construction_mode") != "CUSTOM_MODEL"
        and not custom_declarations
        and not custom_instances
        and inert_synthesis
        and len(inert_synthesis)
        == sum(
            item.get("operator_id") == "synthesize_custom_model"
            for item in operators
        )
    ):
        payload["architecture_operators"] = [
            item
            for item in operators
            if item.get("operator_id") != "synthesize_custom_model"
        ]
        normalized["program_payload"] = payload

    return normalized


def _compiler_owned_program_normalization(
    program: Mapping[str, Any],
) -> dict[str, Any]:
    """Close deterministic Provider component-scaffold contradictions.

    The Provider still owns every research choice.  This boundary only makes
    the already-declared encoder regime structurally executable.  Cases with
    more than one possible interpretation are left unchanged for the compiler
    to reject as typed evidence.
    """

    normalized = _compiler_owned_custom_marker_normalization(program)
    payload = dict(normalized.get("program_payload", {}))
    components = [dict(item) for item in payload.get("components", [])]

    encoders = [item for item in components if item.get("slot_id") == "ENCODER"]
    if len(encoders) != 1:
        return normalized
    encoder = encoders[0]
    encoder_index = next(
        index for index, item in enumerate(components) if item.get("component_id") == encoder.get("component_id")
    )
    component_ids = {str(item.get("component_id")) for item in components}
    changed_slots = [dict(item) for item in payload.get("changed_slots", [])]
    core_slots = {
        str(item.get("slot_id"))
        for item in changed_slots
        if item.get("change_role") == "CORE"
    }
    primitive_id = encoder.get("primitive_id")
    message_components = [
        item for item in components if item.get("slot_id") == "MESSAGE"
    ]
    propagation_components = [
        item
        for item in components
        if item.get("slot_id") == "PROPAGATION_AGGREGATION"
        and item.get("primitive_id") != "propagation.none"
    ]
    force_explicit_rewire = False

    if primitive_id == "encoder.none_mf" and (
        message_components or propagation_components
    ):
        if core_slots & {"MESSAGE", "PROPAGATION_AGGREGATION"}:
            if "ENCODER" in core_slots:
                return normalized
            encoder["primitive_id"] = "encoder.explicit_message_passing"
            primitive_id = encoder["primitive_id"]
            force_explicit_rewire = True
        else:
            removable_ids = {
                str(item["component_id"])
                for item in message_components + propagation_components
            }
            retained = [
                item
                for item in components
                if str(item.get("component_id")) not in removable_ids
            ]
            referenced = {
                str(source.get("component_id"))
                for item in retained
                for input_item in item.get("inputs", [])
                if isinstance(input_item, Mapping)
                for source in [input_item.get("source")]
                if isinstance(source, Mapping) and source.get("kind") == "COMPONENT"
            }
            operator_refs = {
                str(component_id)
                for operator in payload.get("architecture_operators", [])
                if isinstance(operator, Mapping)
                for field in ("targets", "replacements")
                for component_id in operator.get(field, [])
            }
            ablation_refs = {
                str(component_id)
                for ablation in payload.get("ablation_plan", [])
                if isinstance(ablation, Mapping)
                for component_id in ablation.get("remove_component_ids", [])
            }
            if removable_ids & (referenced | operator_refs | ablation_refs):
                return normalized
            components = retained
            changed_slots = [
                item
                for item in changed_slots
                if item.get("slot_id") not in {"MESSAGE", "PROPAGATION_AGGREGATION"}
                or item.get("change_role") == "CORE"
            ]
            payload["components"] = components
            payload["changed_slots"] = changed_slots
            normalized["program_payload"] = payload
            return canonical_value(normalized)

    if primitive_id == "encoder.explicit_message_passing":
        if len(message_components) > 1 or len(propagation_components) > 1:
            return normalized
        had_message = bool(message_components)
        had_propagation = bool(propagation_components)
        needs_scaffold = (
            force_explicit_rewire
            or not had_message
            or not had_propagation
        )
        if needs_scaffold:
            if (
                not force_explicit_rewire
                and core_slots & {"MESSAGE", "PROPAGATION_AGGREGATION"}
            ):
                return normalized
            representation_source = _input_source(encoder, "representation")
            relation_source = _input_source(encoder, "relation")
            if representation_source is None or relation_source is None:
                return normalized
            if message_components:
                message = dict(message_components[0])
            else:
                if propagation_components:
                    propagation_signal = _input_source(
                        propagation_components[0], "signal"
                    )
                    if propagation_signal is None:
                        return normalized
                    representation_source = propagation_signal
                message = {
                    "component_id": _fresh_compiler_component_id(
                        component_ids, "compiler_message_identity"
                    ),
                    "inputs": [
                        {"port": "representation", "source": representation_source},
                        {"port": "relation", "source": relation_source},
                    ],
                    "parameters": {},
                    "primitive_id": "message.identity",
                    "slot_id": "MESSAGE",
                }
                components.append(message)
            message_source = {
                "kind": "COMPONENT",
                "component_id": message["component_id"],
                "output_port": "message",
            }
            if propagation_components:
                propagation = dict(propagation_components[0])
                _set_input_source(propagation, "signal", message_source)
                propagation_index = next(
                    index
                    for index, item in enumerate(components)
                    if item.get("component_id") == propagation.get("component_id")
                )
                components[propagation_index] = propagation
            else:
                if had_message:
                    encoder_representation = _input_source(
                        encoder, "representation"
                    )
                    if (
                        encoder_representation is None
                        or encoder_representation.get("kind") != "COMPONENT"
                        or encoder_representation.get("component_id")
                        != message["component_id"]
                    ):
                        return normalized
                propagation = {
                    "component_id": _fresh_compiler_component_id(
                        component_ids, "compiler_propagation_symmetric"
                    ),
                    "inputs": [
                        {"port": "signal", "source": message_source},
                        {"port": "relation", "source": relation_source},
                    ],
                    "parameters": {},
                    "primitive_id": "propagation.symmetric_normalization",
                    "slot_id": "PROPAGATION_AGGREGATION",
                }
                components.append(propagation)
            if not had_propagation:
                _set_input_source(
                    encoder,
                    "representation",
                    {
                        "kind": "COMPONENT",
                        "component_id": propagation["component_id"],
                        "output_port": "representation",
                    },
                )
            components[encoder_index] = encoder

    primitive_id = encoder.get("primitive_id")
    if primitive_id in {
        "encoder.precomputed_graph_filter",
        "encoder.closed_form_filter",
    }:
        expected_efficiency = "efficiency.graph_filter_precomputation"
        efficiency_components = [
            item
            for item in components
            if item.get("primitive_id")
            in {
                "efficiency.graph_filter_precomputation",
                "efficiency.closed_form_solver",
            }
        ]
        if not efficiency_components:
            relation_source = _input_source(encoder, "relation")
            if relation_source is None:
                return normalized
            components.append(
                {
                    "component_id": _fresh_compiler_component_id(
                        component_ids, "compiler_filter_precompute"
                    ),
                    "inputs": [{"port": "target", "source": relation_source}],
                    "parameters": {},
                    "primitive_id": expected_efficiency,
                    "slot_id": "EFFICIENCY_APPROXIMATION",
                }
            )

    payload["components"] = components
    normalized["program_payload"] = payload
    return canonical_value(normalized)


def _decode_bl_icf_program_proposal(
    proposal: Mapping[str, Any],
    *,
    producer_role: str,
    required_base_model_config: str | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, str] | tuple[dict[str, str], ...] | None,
]:
    if proposal.get("producer_role") != producer_role:
        raise fresh_r1.FreshR1Error(
            "Provider changed the preassigned Producer role"
        )
    normalized = _normalize_implementation_research(
        proposal,
        required_base_model_config=required_base_model_config,
    )
    wire_program = normalized.get("mechanism_program")
    if not isinstance(wire_program, Mapping):
        raise fresh_r1.FreshR1Error(
            "strict BL-ICF proposal lacks mechanism_program"
        )
    program = _decode_provider_wire_program(wire_program)
    return normalized, program, parent_binding_for_mechanism_program(program)


def _expected_parent_binding_for_proposal(
    *,
    context_view: Mapping[str, Any],
    provider_context_view: Mapping[str, Any],
) -> dict[str, str] | tuple[dict[str, str], ...] | None:
    provider_state = provider_context_view.get("state")
    provider_state = provider_state if isinstance(provider_state, Mapping) else {}
    active_task = provider_state.get("task")
    if (
        isinstance(active_task, Mapping)
        and active_task.get("execution_eligible_this_round") is True
        and active_task.get("binding_requirement")
        == "EXACT_EFFECTIVE_IDENTITY"
    ):
        target_program = active_task.get("target_mechanism_program")
        if isinstance(target_program, Mapping):
            return parent_binding_for_mechanism_program(target_program)
    lineage_parent = provider_state.get("lineage_parent_binding")
    if (
        isinstance(provider_state.get("lineage_parent_mechanism_program"), Mapping)
        and isinstance(lineage_parent, Mapping)
    ):
        return canonical_value(dict(lineage_parent))
    frozen_parent = provider_state.get("frozen_parent_binding")
    if isinstance(frozen_parent, Mapping):
        return canonical_value(dict(frozen_parent))
    return None


def _selected_construction_parent(
    program: Mapping[str, Any],
    options: tuple[Mapping[str, Any], ...],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    refs = program.get("program_payload", {}).get("parent_refs", ())
    if len(refs) != 1 or not isinstance(refs[0], Mapping):
        raise fresh_r1.FreshR1Error("choose one available construction parent")
    matches = [option for option in options if all(
        option[key] == refs[0].get(key) for key in ("candidate_id", "program_digest")
    )]
    if len(matches) != 1:
        raise fresh_r1.FreshR1Error("selected construction parent is not available")
    option = matches[0]
    return ({key: option[key] for key in ("candidate_id", "program_digest")},
            option["mechanism_program"])


def _parent_program_for_proposal(
    *,
    provider_context_view: Mapping[str, Any],
    expected_parent_binding: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(expected_parent_binding, Mapping):
        return None
    state = provider_context_view.get("state")
    state = state if isinstance(state, Mapping) else {}
    candidates = (
        ("lineage parent", state.get("lineage_parent_mechanism_program")),
        ("frozen root parent", state.get("frozen_parent_mechanism_program")),
    )
    saw_program = False
    for _parent_kind, program in candidates:
        if not isinstance(program, Mapping):
            continue
        saw_program = True
        report = compile_program(program)
        if (
            report.candidate_id == expected_parent_binding.get("candidate_id")
            and report.mechanism_program_digest
            == expected_parent_binding.get("program_digest")
        ):
            return program
    if saw_program:
        raise fresh_r1.FreshR1Error(
            "active construction parent mechanism program identity drift"
        )
    return None


def _inherit_unchanged_parent_slots(
    program: Mapping[str, Any],
    parent_program: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Apply one declared child delta to the exact active construction parent."""

    child = canonical_value(dict(program))
    if parent_program is None:
        return child
    parent = canonical_value(dict(parent_program))
    child_payload = dict(child["program_payload"])
    parent_payload = dict(parent["program_payload"])
    changed_slots = {
        str(item["slot_id"])
        for item in child_payload.get("changed_slots", [])
        if isinstance(item, Mapping) and isinstance(item.get("slot_id"), str)
    }

    def rows(payload: Mapping[str, Any], field: str) -> list[dict[str, Any]]:
        return [
            dict(item)
            for item in payload.get(field, [])
            if isinstance(item, Mapping)
        ]

    parent_components = rows(parent_payload, "components")
    child_components = rows(child_payload, "components")
    parent_custom = rows(parent_payload, "custom_components")
    child_custom = rows(child_payload, "custom_components")

    # Provider programs are complete wire programs, so the model may restate an
    # unchanged parent component with a fresh local id.  Those restatements are
    # not research choices: the overlay below always restores the exact frozen
    # parent row.  Resolve the unambiguous local ids here so components in a
    # declared changed slot can still read the inherited parent component.
    parent_by_component_id = {
        str(item["component_id"]): item
        for item in parent_components
        if isinstance(item.get("component_id"), str)
    }
    changed_component_aliases: dict[str, str] = {}
    for slot in changed_slots:
        parent_slot_rows = [
            item for item in parent_components if item.get("slot_id") == slot
        ]
        child_slot_rows = [
            item for item in child_components if item.get("slot_id") == slot
        ]
        if len(parent_slot_rows) != 1 or len(child_slot_rows) != 1:
            continue
        parent_id = parent_slot_rows[0].get("component_id")
        child_id = child_slot_rows[0].get("component_id")
        if (
            isinstance(parent_id, str)
            and isinstance(child_id, str)
            and child_id != parent_id
            and child_id not in parent_by_component_id
        ):
            # A one-for-one slot replacement keeps the machine-owned parent id.
            # This preserves downstream frozen-parent wiring while the primitive,
            # parameters and mechanism semantics remain the Provider's choice.
            changed_component_aliases[child_id] = parent_id
    unchanged_component_aliases: dict[str, str] = {}
    for item in child_components:
        component_id = item.get("component_id")
        slot = item.get("slot_id")
        primitive_id = item.get("primitive_id")
        if (
            not isinstance(component_id, str)
            or not isinstance(slot, str)
            or slot in changed_slots
        ):
            continue
        exact = parent_by_component_id.get(component_id)
        if exact is not None and exact.get("slot_id") == slot:
            unchanged_component_aliases[component_id] = component_id
            continue
        candidates = [
            parent_item
            for parent_item in parent_components
            if parent_item.get("slot_id") == slot
            and parent_item.get("primitive_id") == primitive_id
            and isinstance(parent_item.get("component_id"), str)
        ]
        if len(candidates) == 1:
            unchanged_component_aliases[component_id] = str(
                candidates[0]["component_id"]
            )

    component_reference_aliases = {
        **unchanged_component_aliases,
        **changed_component_aliases,
    }
    if component_reference_aliases:
        rewired_components: list[dict[str, Any]] = []
        for item in child_components:
            rewritten = dict(item)
            component_id = rewritten.get("component_id")
            if isinstance(component_id, str):
                rewritten["component_id"] = changed_component_aliases.get(
                    component_id, component_id
                )
            rewritten_inputs: list[dict[str, Any]] = []
            for input_item in item.get("inputs", []):
                normalized_input = dict(input_item)
                source = input_item.get("source")
                if isinstance(source, Mapping) and source.get("kind") == "COMPONENT":
                    normalized_source = dict(source)
                    source_id = normalized_source.get("component_id")
                    if isinstance(source_id, str):
                        normalized_source["component_id"] = (
                            component_reference_aliases.get(source_id, source_id)
                        )
                    normalized_input["source"] = normalized_source
                rewritten_inputs.append(normalized_input)
            rewritten["inputs"] = rewritten_inputs
            rewired_components.append(rewritten)
        child_components = rewired_components

        rewritten_operators: list[dict[str, Any]] = []
        for operator in rows(child_payload, "architecture_operators"):
            normalized_operator = dict(operator)
            for field in ("targets", "replacements"):
                normalized_operator[field] = [
                    component_reference_aliases.get(item, item)
                    if isinstance(item, str) and not item.startswith("slot:")
                    else item
                    for item in operator.get(field, [])
                ]
            rewritten_operators.append(normalized_operator)
        child_payload["architecture_operators"] = rewritten_operators

        rewritten_ablations: list[dict[str, Any]] = []
        for ablation in rows(child_payload, "ablation_plan"):
            normalized_ablation = dict(ablation)
            normalized_ablation["remove_component_ids"] = [
                component_reference_aliases.get(item, item)
                for item in ablation.get("remove_component_ids", [])
            ]
            rewritten_ablations.append(normalized_ablation)
        child_payload["ablation_plan"] = rewritten_ablations

    def reject_undeclared_row_changes(
        parent_rows: list[dict[str, Any]],
        child_rows: list[dict[str, Any]],
        *,
        identity_field: str,
        field_name: str,
    ) -> None:
        parent_by_id = {
            str(item[identity_field]): item
            for item in parent_rows
            if isinstance(item.get(identity_field), str)
        }
        for item in child_rows:
            identity = item.get(identity_field)
            slot = item.get("slot_id")
            if not isinstance(identity, str) or not isinstance(slot, str):
                continue
            original = parent_by_id.get(identity)
            if slot not in changed_slots:
                if (
                    original is not None
                    and original.get("slot_id") in changed_slots
                ):
                    raise fresh_r1.FreshR1Error(
                        f"single-parent child moves {field_name} outside declared "
                        "changed_slots"
                    )
                # This row is a non-authoritative restatement.  It is discarded
                # by overlay_rows in favor of the exact frozen-parent row.
                continue
            if original is None:
                continue
            original_slot = original.get("slot_id")
            unchanged = canonical_value(item) == canonical_value(original)
            if (
                not unchanged
                and (
                    slot not in changed_slots
                    or original_slot not in changed_slots
                )
            ):
                raise fresh_r1.FreshR1Error(
                    f"single-parent child changes {field_name} outside declared "
                    "changed_slots"
                )

    reject_undeclared_row_changes(
        parent_components,
        child_components,
        identity_field="component_id",
        field_name="component",
    )
    reject_undeclared_row_changes(
        parent_custom,
        child_custom,
        identity_field="custom_component_id",
        field_name="custom component",
    )

    parent_removed = {
        str(item)
        for item in parent_payload.get("removed_slots", [])
        if isinstance(item, str)
    }
    child_removed = {
        str(item)
        for item in child_payload.get("removed_slots", [])
        if isinstance(item, str)
    }
    undeclared_removals = child_removed - parent_removed - changed_slots
    if undeclared_removals:
        raise fresh_r1.FreshR1Error(
            "single-parent child removes slots outside declared changed_slots: "
            + ", ".join(sorted(undeclared_removals))
        )

    def overlay_rows(
        parent_rows: list[dict[str, Any]],
        child_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        inserted: set[str] = set()
        for item in parent_rows:
            slot = str(item.get("slot_id", ""))
            if slot in changed_slots:
                if slot not in inserted:
                    result.extend(
                        row
                        for row in child_rows
                        if row.get("slot_id") == slot
                    )
                    inserted.add(slot)
                continue
            result.append(item)
        for item in child_rows:
            slot = str(item.get("slot_id", ""))
            if slot in changed_slots and slot not in inserted:
                result.extend(
                    row for row in child_rows if row.get("slot_id") == slot
                )
                inserted.add(slot)
        return result

    components = overlay_rows(parent_components, child_components)
    custom_components = overlay_rows(parent_custom, child_custom)

    def component_slot_map(
        component_rows: list[dict[str, Any]],
        custom_rows: list[dict[str, Any]],
    ) -> dict[str, str]:
        return {
            **{
                str(item["component_id"]): str(item["slot_id"])
                for item in component_rows
                if isinstance(item.get("component_id"), str)
                and isinstance(item.get("slot_id"), str)
            },
            **{
                str(item["custom_component_id"]): str(item["slot_id"])
                for item in custom_rows
                if isinstance(item.get("custom_component_id"), str)
                and isinstance(item.get("slot_id"), str)
            },
        }

    parent_component_slots = component_slot_map(
        parent_components,
        parent_custom,
    )
    child_component_slots = component_slot_map(
        components,
        custom_components,
    )

    def changed_row_slots(
        before: list[dict[str, Any]],
        after: list[dict[str, Any]],
        *,
        identity_field: str,
    ) -> set[str]:
        before_by_id = {
            str(item[identity_field]): item
            for item in before
            if isinstance(item.get(identity_field), str)
        }
        after_by_id = {
            str(item[identity_field]): item
            for item in after
            if isinstance(item.get(identity_field), str)
        }
        result: set[str] = set()
        for identity in set(before_by_id) | set(after_by_id):
            old = before_by_id.get(identity)
            new = after_by_id.get(identity)
            if canonical_value(old) == canonical_value(new):
                continue
            for item in (old, new):
                if isinstance(item, Mapping) and isinstance(
                    item.get("slot_id"), str
                ):
                    result.add(str(item["slot_id"]))
        return result

    row_delta_slots = changed_row_slots(
        parent_components,
        components,
        identity_field="component_id",
    ) | changed_row_slots(
        parent_custom,
        custom_components,
        identity_field="custom_component_id",
    )

    def operator_write_slots(
        operator: Mapping[str, Any],
        *,
        component_slots: Mapping[str, str],
        row_delta_slots: set[str] | None = None,
    ) -> set[str]:
        # ``add_component`` writes the new component row.  Its targets are
        # descriptive operator metadata, not graph wiring; the complete child
        # graph below is authoritative for dataflow.
        if (
            operator.get("operator_id") == "add_component"
            and row_delta_slots is not None
        ):
            return set(row_delta_slots)
        writes: set[str] = set()
        for field in ("targets", "replacements"):
            for item in operator.get(field, []):
                if not isinstance(item, str):
                    continue
                if item.startswith("slot:"):
                    writes.add(item.removeprefix("slot:"))
                elif item in component_slots:
                    writes.add(component_slots[item])
        if not writes and row_delta_slots is not None:
            writes.update(row_delta_slots)
        return writes

    parent_operators = rows(parent_payload, "architecture_operators")
    child_operators = rows(child_payload, "architecture_operators")
    novel_child_operators = [
        item for item in child_operators if item not in parent_operators
    ]
    for item in novel_child_operators:
        writes = operator_write_slots(
            item,
            component_slots=child_component_slots,
            row_delta_slots=(
                row_delta_slots
                if item.get("operator_id") == "add_component"
                or (not item.get("targets") and not item.get("replacements"))
                else None
            ),
        )
        if not writes:
            raise fresh_r1.FreshR1Error(
                "single-parent child architecture operator has no executable "
                "write footprint"
            )
        undeclared_writes = writes - changed_slots
        if undeclared_writes:
            raise fresh_r1.FreshR1Error(
                "single-parent child architecture operator writes outside "
                "declared changed_slots: "
                + ", ".join(sorted(undeclared_writes))
            )

    def splice_child_graph_edges_into_parent_consumers(
        component_rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Restore child dataflow while retaining exact frozen-parent rows.

        A Provider describes a complete graph, but unchanged parent rows are
        deliberately restored above.  Copy only a real child edge from a new
        changed-slot component into the matching input of an unchanged parent
        consumer.  Operator targets never determine dataflow.
        """

        materialized = canonical_value(component_rows)
        materialized_by_id = {
            str(item["component_id"]): item
            for item in materialized
            if isinstance(item.get("component_id"), str)
        }
        parent_by_id = {
            str(item["component_id"]): item
            for item in parent_components
            if isinstance(item.get("component_id"), str)
        }
        added_ids = {
            str(item["component_id"])
            for item in child_components
            if isinstance(item.get("component_id"), str)
            and item["component_id"] not in parent_by_id
            and item.get("slot_id") in changed_slots
            and item["component_id"] in materialized_by_id
        }
        if not added_ids:
            return materialized

        for child_consumer in child_components:
            child_consumer_id = child_consumer.get("component_id")
            if not isinstance(child_consumer_id, str):
                continue
            consumer_id = unchanged_component_aliases.get(
                child_consumer_id,
                child_consumer_id,
            )
            parent_consumer = parent_by_id.get(consumer_id)
            materialized_consumer = materialized_by_id.get(consumer_id)
            if (
                parent_consumer is None
                or materialized_consumer is None
                or parent_consumer.get("slot_id") in changed_slots
            ):
                continue

            parent_inputs = [
                dict(item)
                for item in parent_consumer.get("inputs", [])
                if isinstance(item, Mapping)
            ]
            materialized_inputs = canonical_value(parent_inputs)
            child_edges_by_port: dict[str, list[dict[str, Any]]] = {}
            for child_input in child_consumer.get("inputs", []):
                if not isinstance(child_input, Mapping):
                    continue
                port = child_input.get("port")
                source = child_input.get("source")
                if (
                    not isinstance(port, str)
                    or not isinstance(source, Mapping)
                    or source.get("kind") != "COMPONENT"
                    or source.get("component_id") not in added_ids
                ):
                    continue
                child_edges_by_port.setdefault(port, []).append(dict(source))

            changed = False
            for port, child_sources in child_edges_by_port.items():
                parent_matches = [
                    index
                    for index, parent_input in enumerate(parent_inputs)
                    if parent_input.get("port") == port
                ]
                if len(child_sources) != 1 or len(parent_matches) != 1:
                    continue
                parent_index = parent_matches[0]
                parent_source = parent_inputs[parent_index].get("source")
                child_source = child_sources[0]
                added_component = materialized_by_id.get(
                    str(child_source["component_id"])
                )
                if not isinstance(parent_source, Mapping) or added_component is None:
                    continue
                consumes_parent_source = any(
                    isinstance(added_input, Mapping)
                    and isinstance(added_input.get("source"), Mapping)
                    and canonical_value(dict(added_input["source"]))
                    == canonical_value(dict(parent_source))
                    for added_input in added_component.get("inputs", [])
                )
                if not consumes_parent_source:
                    continue
                spliced_input = dict(materialized_inputs[parent_index])
                spliced_input["source"] = canonical_value(child_source)
                materialized_inputs[parent_index] = spliced_input
                changed = True
            if changed:
                materialized_consumer["inputs"] = materialized_inputs
        return materialized

    components = splice_child_graph_edges_into_parent_consumers(components)

    architecture_operators: list[dict[str, Any]] = []
    for item in parent_operators:
        if item in child_operators:
            if item not in architecture_operators:
                architecture_operators.append(item)
            continue
        parent_writes = operator_write_slots(
            item,
            component_slots=parent_component_slots,
        )
        if parent_writes & changed_slots:
            continue
        if item not in architecture_operators:
            architecture_operators.append(item)
    for item in novel_child_operators:
        if item not in architecture_operators:
            architecture_operators.append(item)

    child_payload["components"] = components
    child_payload["custom_components"] = custom_components
    child_payload["architecture_operators"] = architecture_operators
    inherited_removed = [
        item
        for item in parent_payload.get("removed_slots", [])
        if item not in changed_slots
    ]
    declared_removed = [
        item
        for item in child_payload.get("removed_slots", [])
        if item in changed_slots
    ]
    child_payload["removed_slots"] = list(
        dict.fromkeys((*inherited_removed, *declared_removed))
    )
    # Construction mode is the Provider's research choice. Stable family,
    # input, and protocol contracts remain exact-parent facts.
    for field_name in (
        "schema_version",
        "family_contract_id",
        "input_semantics",
        "protocol_impact",
    ):
        if field_name in parent_payload:
            child_payload[field_name] = parent_payload[field_name]
        else:
            child_payload.pop(field_name, None)
    child["program_payload"] = child_payload
    return canonical_value(child)


def _require_effective_single_parent_delta(
    program: Mapping[str, Any],
    parent_program: Mapping[str, Any],
) -> None:
    """Reject parameter-only, label-only, and no-op parent descendants."""

    candidate_identity = effective_experiment_identity(program)
    parent_identity = effective_experiment_identity(parent_program)
    if (
        candidate_identity["effective_family_digest"]
        == parent_identity["effective_family_digest"]
    ):
        raise fresh_r1.FreshR1Error(
            "single-parent proposal must change the active parent's effective "
            "mechanism family, not only parameters or labels"
        )


def _bl_icf_program_proposal_v1(
    proposal: Mapping[str, Any],
    *,
    producer_role: str,
    expected_parent_binding: Mapping[str, Any] | None,
    parent_program: Mapping[str, Any] | None = None,
    force_parent_binding: bool = False,
) -> ProviderMechanismProgramProposalV1:
    normalized, program, declared_parent_binding = (
        _decode_bl_icf_program_proposal(
            proposal,
            producer_role=producer_role,
            required_base_model_config=(
                PARENT_BASE_MODEL_CONFIG if force_parent_binding else None
            ),
        )
    )
    # The Provider chooses the mechanism and construction parent. The caller
    # resolves that choice to exact campaign-owned identifiers and program;
    # this graft must inherit that selected parent rather than the current best.
    effective_parent_binding = (
        expected_parent_binding
        if force_parent_binding or declared_parent_binding is not None
        else None
    )
    parent_refs = (
        []
        if effective_parent_binding is None
        else [dict(effective_parent_binding)]
        if isinstance(effective_parent_binding, Mapping)
        else [dict(item) for item in effective_parent_binding]
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
    program = _inherit_unchanged_parent_slots(
        program,
        parent_program if effective_parent_binding is not None else None,
    )
    if producer_role in _INDEPENDENT_MINI_PRODUCER_ROLES:
        program = _canonicalize_registry_owned_component_slots(program)
    program = _compiler_owned_interface_normalization(program)
    program = _compiler_owned_program_normalization(program)
    report = compile_program(program)
    if report.status is CompileStatus.INVALID:
        raise _BlIcfProgramCompileError(report)
    if (
        report.status is not CompileStatus.VALID_NEEDS_IMPLEMENTATION
        or report.candidate_id is None
        or report.mechanism_program_digest is None
    ):
        raise fresh_r1.FreshR1Error(
            "strict BL-ICF Provider mechanism_program does not compile as "
            "VALID_NEEDS_IMPLEMENTATION: "
            + canonical_json_bytes(_compile_failure_summary(report)).decode(
                "utf-8"
            )
        )
    if force_parent_binding and parent_program is not None:
        _require_effective_single_parent_delta(program, parent_program)
    return ProviderMechanismProgramProposalV1(
        producer_role=producer_role,
        mechanism_program=program,
        implementation_research=normalized["implementation_research"],
        resolution_facts=normalized["resolution_facts"],
        parent_binding=effective_parent_binding,
    )


def _bl_icf_candidate_proposal_v4(
    proposal: Mapping[str, Any],
    *,
    producer_role: str,
    context_view: Mapping[str, Any],
    logical_call_id: str,
) -> CandidateProposalV4:
    """Convert the existing campaign wire proposal into the formal V4 type."""

    try:
        program = campaign_program_from_proposal(proposal)
        compiled = compile_program(program)
        if (
            not compiled.is_valid
            or compiled.mechanism_program_digest is None
            or compiled.mechanism_semantics_digest is None
        ):
            raise fresh_r1.FreshR1Error(
                "BL-ICF Provider proposal does not compile"
            )
        mechanism = executable_mechanism(str(proposal["mechanism_id"]))
        if mechanism.mechanism_program_digest != compiled.mechanism_program_digest:
            raise fresh_r1.FreshR1Error(
                "BL-ICF Provider proposal materialized another program"
            )
        intent = ProposalIntentV1(str(proposal["proposal_intent"]))
        expected_intent = _expected_intent(producer_role)
        if intent is not expected_intent:
            raise fresh_r1.FreshR1Error(
                "BL-ICF Provider changed the role-bound proposal intent"
            )
        protocol_digest = context_view.get("protocol_digest")
        if not isinstance(protocol_digest, str) or not protocol_digest:
            raise fresh_r1.FreshR1Error(
                "BL-ICF Provider context lacks protocol_digest"
            )

        candidate_id = "cand-provider-" + sha256_digest(
            {
                "logical_call_id": logical_call_id,
                "mechanism_program_digest": compiled.mechanism_program_digest,
                "producer_role": producer_role,
            }
        )[:24]
        lineage = LineageIndexV1()
        estimated_cost = min(
            1.0,
            0.15 + 0.05 * len(mechanism.operator_ids),
        )
        wire_utility = proposal["utility_features"]
        diagnostic = SearchUtilityFeaturesV1(
            runnable_probability=1.0,
            useful_signal=float(wire_utility["useful_signal"]),
            frontier_potential=float(wire_utility["frontier_potential"]),
            information_gain=float(wire_utility["information_gain"]),
            cost=estimated_cost,
            blocker_risk=0.0,
        )
        utility, evidence = DeterministicRouterFeatureBuilderV1().build(
            compile_valid=True,
            handler_available=bool(mechanism.entrypoint),
            materializer_available=True,
            mechanism_id=mechanism.mechanism_id,
            mechanism_depth=0,
            estimated_cost=estimated_cost,
            semantics_digest=compiled.mechanism_semantics_digest,
            parent_available=True,
            lineage=lineage,
            llm_diagnostic=diagnostic,
        )

        root_program = campaign_program_from_proposal(
            {
                "mechanism_id": root_parent_mechanism_id(
                    mechanism.mechanism_id
                )
            }
        )
        root_report = compile_program(root_program)
        if (
            not root_report.is_valid
            or root_report.candidate_id is None
            or root_report.mechanism_program_digest is None
        ):
            raise fresh_r1.FreshR1Error(
                "BL-ICF matched-control program does not compile"
            )
        control = matched_control_plan(
            lineage=lineage,
            primary_candidate_id=candidate_id,
            parent_candidate_id=None,
            changed_axis=mechanism.mechanism_axis,
            mechanism_hypothesis=str(proposal["mechanism_hypothesis"]),
            protocol_digest=protocol_digest,
            queued_comparator_candidate_id=root_report.candidate_id,
            queued_comparator_program_digest=(
                root_report.mechanism_program_digest
            ),
        )
        discriminative_plan = (
            DiscriminativeExperimentPlanV1(
                competing_hypotheses=(
                    str(proposal["mechanism_hypothesis"]),
                    str(proposal["competing_hypothesis"]),
                ),
                predicted_outcome_signature=str(
                    proposal["predicted_outcome_signature"]
                ),
                primary_candidate=candidate_id,
                matched_control_plan=control,
                falsifier=str(proposal["failure_mode"]),
                next_decision_rule=(
                    "retain the mechanism only if the exact matched control "
                    "does not reproduce the signature"
                ),
            )
            if producer_role == "falsification_designer"
            else None
        )
        return CandidateProposalV4(
            candidate_id=candidate_id,
            producer_id=f"producer-provider-{producer_role}",
            producer_role=producer_role,
            proposal_intent=intent,
            discovery_credit=DiscoveryCreditV1.DISCOVERY,
            mechanism_id=mechanism.mechanism_id,
            mechanism_axis=mechanism.mechanism_axis,
            mechanism_program=program,
            candidate_label=str(proposal["candidate_label"]),
            mechanism_hypothesis=str(proposal["mechanism_hypothesis"]),
            competing_hypothesis=str(proposal["competing_hypothesis"]),
            predicted_outcome_signature=str(
                proposal["predicted_outcome_signature"]
            ),
            failure_mode=str(proposal["failure_mode"]),
            utility_features=utility,
            feature_evidence=evidence,
            matched_control_plan=control,
            discriminative_plan=discriminative_plan,
            parent_candidate_id=None,
            assigned_before_call=True,
            post_hoc_relabel=False,
        )
    except fresh_r1.FreshR1Error:
        raise
    except (CampaignRuntimeError, KeyError, TypeError, ValueError) as error:
        raise fresh_r1.FreshR1Error(
            "BL-ICF Provider proposal violates the executable campaign contract: "
            f"{type(error).__name__}: {error}"
        ) from error


class ProviderResearchProducer:
    """Typed Producer with a deterministic strict-program compiler boundary."""

    def __init__(
        self,
        *,
        config_source: ConfigSource,
        call_root: Path,
        session_id: str,
        total_token_ceiling: int | None = None,
        provider_call: ProviderCall | None = None,
        proposal_template_source: str | Path = _DEFAULT_PROPOSAL_TEMPLATE,
        proposal_protocol_requirements: Sequence[str] = fresh_r1.PROTOCOL_REQUIREMENTS,
        proposal_context_appendix: str = "",
        proposal_schema_source: Mapping[str, Any] | str | Path = _DEFAULT_PROPOSAL_SCHEMA,
        proposal_schema_delta_source: Mapping[str, Any] | str | Path | None = None,
        proposal_schema_path: Path = _DEFAULT_PROPOSAL_SCHEMA,
        bl_icf_proposal_template_source: str
        | Path = _DEFAULT_BL_ICF_PROPOSAL_TEMPLATE,
        bl_icf_proposal_schema_source: Mapping[str, Any]
        | str
        | Path = _DEFAULT_BL_ICF_PROPOSAL_SCHEMA,
        bl_icf_program_proposal_template_source: str
        | Path = _DEFAULT_BL_ICF_PROGRAM_PROPOSAL_TEMPLATE,
        bl_icf_program_proposal_schema_source: Mapping[str, Any]
        | str
        | Path = _DEFAULT_BL_ICF_PROGRAM_PROPOSAL_SCHEMA,
        proposal_lane_by_role: Mapping[str, str] | None = None,
        proposal_seed: int = 0,
        maximum_physical_attempts: int | None = fresh_r1.MAX_PHYSICAL_ATTEMPTS,
        frozen_profile_ref: Mapping[str, Any] | None = None,
        allowed_frozen_profile_kinds: Sequence[str] = ("OFFLINE_TOPN",),
        proposal_replay_by_role: Mapping[str, Mapping[str, Any]] | None = None,
        proposal_output_token_ceiling_total_per_context: int | None = None,
        proposal_output_token_ceiling_per_physical_call: int = (
            RESEARCH_PROPOSAL_OUTPUT_TOKEN_CEILING
        ),
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
            RESEARCH_PROPOSAL_TOTAL_TOKEN_CEILING
            if total_token_ceiling is None
            else total_token_ceiling
        )
        if self.total_token_ceiling < RESEARCH_PROPOSAL_OUTPUT_TOKEN_CEILING:
            raise ValueError("total_token_ceiling is below the proposal output ceiling")
        self.provider_call = provider_call or fresh_r1.bounded_provider_call
        self.proposal_template = _read_text(proposal_template_source)
        self.proposal_protocol_requirements = tuple(proposal_protocol_requirements)
        self.proposal_context_appendix = proposal_context_appendix
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
        self.bl_icf_proposal_template = _read_text(
            bl_icf_proposal_template_source
        )
        self.bl_icf_proposal_schema = _read_json(
            bl_icf_proposal_schema_source
        )
        jsonschema.validators.validator_for(
            self.bl_icf_proposal_schema
        ).check_schema(self.bl_icf_proposal_schema)
        self.bl_icf_program_proposal_template = _read_text(
            bl_icf_program_proposal_template_source
        )
        self.bl_icf_program_proposal_schema = _read_json(
            bl_icf_program_proposal_schema_source
        )
        jsonschema.validators.validator_for(
            self.bl_icf_program_proposal_schema
        ).check_schema(self.bl_icf_program_proposal_schema)
        self.proposal_lane_by_role = _proposal_lanes(proposal_lane_by_role)
        self.proposal_seed = int(proposal_seed)
        self.maximum_physical_attempts = (
            None
            if maximum_physical_attempts is None
            else int(maximum_physical_attempts)
        )
        if (
            isinstance(maximum_physical_attempts, bool)
            or (
                self.maximum_physical_attempts is not None
                and self.maximum_physical_attempts < 1
            )
        ):
            raise ValueError("maximum_physical_attempts must be positive or None")
        self.proposal_output_token_ceiling_per_physical_call = int(
            proposal_output_token_ceiling_per_physical_call
        )
        if self.proposal_output_token_ceiling_per_physical_call < 1:
            raise ValueError(
                "proposal_output_token_ceiling_per_physical_call must be positive"
            )
        self.proposal_output_token_ceiling_total_per_context = (
            None
            if proposal_output_token_ceiling_total_per_context is None
            else int(proposal_output_token_ceiling_total_per_context)
        )
        if (
            self.proposal_output_token_ceiling_total_per_context is not None
            and self.proposal_output_token_ceiling_total_per_context < 1
        ):
            raise ValueError(
                "proposal_output_token_ceiling_total_per_context must be positive"
            )
        self._proposal_output_tokens_by_context: dict[str, int] = {}
        self.frozen_profile_ref = canonical_value(
            dict(
                campaign_scientific_profile_ref()
                if frozen_profile_ref is None
                else frozen_profile_ref
            )
        )
        allowed_profile_kinds = tuple(allowed_frozen_profile_kinds)
        if (
            not allowed_profile_kinds
            or len(set(allowed_profile_kinds)) != len(allowed_profile_kinds)
            or any(
                not isinstance(item, str)
                or not item
                or item != item.strip()
                for item in allowed_profile_kinds
            )
        ):
            raise ValueError(
                "allowed_frozen_profile_kinds must contain unique non-empty strings"
            )
        self.allowed_frozen_profile_kinds = allowed_profile_kinds
        if set(self.frozen_profile_ref) != {
            "profile_id",
            "profile_digest",
            "profile_kind",
        } or self.frozen_profile_ref["profile_kind"] not in allowed_profile_kinds:
            if allowed_profile_kinds == ("OFFLINE_TOPN",):
                raise ValueError("frozen_profile_ref is not an OFFLINE_TOPN identity")
            raise ValueError(
                "frozen_profile_ref profile_kind is outside the adapter-owned family"
            )
        self.last_call_trace: Mapping[str, Any] | None = None
        self.call_traces: tuple[Mapping[str, Any], ...] = ()
        replay = dict(proposal_replay_by_role or {})
        if replay and set(replay) != set(DISCOVERY_PRODUCERS):
            raise ValueError(
                "proposal replay must contain exactly one response per Producer role"
            )
        normalized_replay: dict[str, Mapping[str, Any]] = {}
        for role, record in replay.items():
            if not isinstance(record, Mapping):
                raise ValueError("proposal replay records must be mappings")
            response = record.get("response")
            source = record.get("source")
            if not isinstance(response, Mapping) or not isinstance(source, Mapping):
                raise ValueError(
                    "proposal replay records require response and source mappings"
                )
            normalized_replay[role] = canonical_value(
                {"response": dict(response), "source": dict(source)}
            )
        self._proposal_replay_remaining = normalized_replay
        self._resume_request_digest_by_logical_call: dict[str, str] = {}
        self._resume_request_by_producer_role: dict[str, tuple[str, str]] = {}
        self._started_round_replay_context_ref: str | None = None

    def configure_started_round_provider_replay(self, context_ref: str) -> bool:
        """Replay started role brokers exactly; let unstarted roles begin once."""

        if not isinstance(context_ref, str) or not context_ref:
            raise ValueError("started-round replay requires a context_ref")
        self._started_round_replay_context_ref = context_ref
        return True

    def configure_resume_provider_replay(
        self,
        traces: Sequence[Mapping[str, Any]],
    ) -> None:
        """Bind sealed checkpoint identities to stored broker responses."""

        replay: dict[str, str] = {}
        replay_by_role: dict[str, tuple[str, str]] = {}
        for trace in traces:
            if not isinstance(trace, Mapping):
                continue
            logical_call_id = trace.get("logical_call_id")
            receipt = trace.get("receipt")
            if not isinstance(logical_call_id, str) or not isinstance(
                receipt, Mapping
            ):
                continue
            request_digest = receipt.get("request_digest")
            if isinstance(request_digest, str) and len(request_digest) == 64:
                replay[logical_call_id] = request_digest
                producer_role = receipt.get("provider_role")
                if producer_role in DISCOVERY_PRODUCERS:
                    sealed_identity = (logical_call_id, request_digest)
                    prior = replay_by_role.get(producer_role)
                    if prior is not None and prior != sealed_identity:
                        raise fresh_r1.FreshR1Error(
                            "prepared checkpoint contains conflicting sealed "
                            f"Provider identities for {producer_role}"
                        )
                    replay_by_role[producer_role] = sealed_identity
        self._resume_request_digest_by_logical_call = replay
        self._resume_request_by_producer_role = replay_by_role

    def _budgeted_output_ceiling(self, context_ref: str) -> int:
        total = self.proposal_output_token_ceiling_total_per_context
        if total is None:
            return self.proposal_output_token_ceiling_per_physical_call
        consumed = self._proposal_output_tokens_by_context.get(context_ref, 0)
        remaining = total - consumed
        if remaining < 1:
            raise fresh_r1.FreshR1Error(
                "proposal output token budget is exhausted for this Research Context"
            )
        if self.maximum_physical_attempts is None:
            return min(
                self.proposal_output_token_ceiling_per_physical_call,
                remaining,
            )
        return min(
            self.proposal_output_token_ceiling_per_physical_call,
            max(1, remaining // self.maximum_physical_attempts),
        )

    def _charge_output_tokens(
        self,
        context_ref: str,
        result: fresh_r1.ProviderAttemptResult,
    ) -> None:
        total = self.proposal_output_token_ceiling_total_per_context
        if total is None:
            return
        consumed = self._proposal_output_tokens_by_context.get(context_ref, 0)
        consumed += _physical_attempt_output_tokens(result)
        if consumed > total:
            raise fresh_r1.FreshR1Error(
                "proposal output token budget exceeded for this Research Context"
            )
        self._proposal_output_tokens_by_context[context_ref] = consumed

    def __call__(
        self,
        producer_role: str,
        context_view: Mapping[str, Any],
    ) -> CandidateProposalV4 | ProviderMechanismProgramProposalV1 | Mapping[str, Any]:
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
    ) -> CandidateProposalV4 | ProviderMechanismProgramProposalV1 | Mapping[str, Any]:
        """Run a policy-shadow call without colliding with the live call identity."""

        return self._call(
            producer_role,
            context_view,
            logical_namespace=logical_namespace,
        )

    def resume_transport_call(
        self, producer_role: str, context_view: Mapping[str, Any], *,
        logical_namespace: str, trace: Mapping[str, Any],
    ) -> CandidateProposalV4 | ProviderMechanismProgramProposalV1 | Mapping[str, Any]:
        """Continue a missing response using the saved request, not a new idea."""
        return self._call(
            producer_role, context_view, logical_namespace=logical_namespace,
            resume_trace=trace,
        )

    def _call(
        self,
        producer_role: str,
        context_view: Mapping[str, Any],
        *,
        logical_namespace: str | None,
        resume_trace: Mapping[str, Any] | None = None,
    ) -> CandidateProposalV4 | ProviderMechanismProgramProposalV1 | Mapping[str, Any]:
        if producer_role not in DISCOVERY_PRODUCERS:
            raise ValueError("producer_role is outside the four-role portfolio")
        if not isinstance(context_view, Mapping):
            raise TypeError("context_view must be a mapping")
        context_ref = context_view.get("context_ref")
        if not isinstance(context_ref, str) or not context_ref:
            raise ValueError("context_view must contain context_ref")

        knowledge_base = context_view.get("knowledge_base")
        knowledge_base = (
            knowledge_base if isinstance(knowledge_base, Mapping) else {}
        )
        baseline_context = knowledge_base.get("baseline_context")
        generic_single_parent_mode = is_single_parent_context(baseline_context)
        bl_icf_single_parent_mode = is_bl_icf_single_parent_context(
            baseline_context
        )

        provider_context_view = project_provider_context_view(context_view)
        director_instruction = (
            _DIRECTOR_ROLE_INSTRUCTION
            if research_producer_roles(context_view.get("budget", {}))
            == ("frontier_architect",) else None
        )
        if generic_single_parent_mode:
            provider_state = provider_context_view.get("state")
            provider_state = (
                provider_state if isinstance(provider_state, Mapping) else {}
            )
            if not (
                isinstance(provider_state.get("frozen_parent_binding"), Mapping)
                and isinstance(
                    provider_state.get("frozen_parent_mechanism_program"),
                    Mapping,
                )
            ):
                raise fresh_r1.FreshR1Error(
                    "single-parent research requires the exact executable "
                    "frozen parent binding and mechanism program"
                )
        proposal_lane = self.proposal_lane_by_role[producer_role]
        if bl_icf_single_parent_mode:
            # BL-ICF uses its native typed MechanismProgram path. Ordinary
            # roles stay focused while frontier_architect retains the full
            # custom-component and architecture-rewrite grammar.
            proposal_lane = PROPOSAL_LANE_BL_ICF_PROGRAM
        elif (
            generic_single_parent_mode
            and proposal_lane != PROPOSAL_LANE_OPEN_SPEC
        ):
            raise fresh_r1.FreshR1Error(
                "non-BL single-parent research requires an explicit OPEN_SPEC "
                "proposal lane from its search-space adapter"
            )
        context_identity = (
            f"{context_ref}:context-{sha256_digest(provider_context_view)[:16]}"
        )
        if proposal_lane == PROPOSAL_LANE_BL_ICF:
            prompt = _render_bl_icf_proposal_prompt(
                self.bl_icf_proposal_template,
                side_identity=context_identity,
                logical_slot_id=f"{context_identity}:{producer_role}:bl-icf",
                proposal_seed=self.proposal_seed,
                producer_role=producer_role,
                frozen_profile_ref=self.frozen_profile_ref,
                single_parent=bl_icf_single_parent_mode,
                role_instruction=director_instruction,
            )
        elif proposal_lane == PROPOSAL_LANE_BL_ICF_PROGRAM:
            prompt = _render_bl_icf_program_proposal_prompt(
                self.bl_icf_program_proposal_template,
                side_identity=context_identity,
                logical_slot_id=(
                    f"{context_identity}:{producer_role}:bl-icf-program"
                ),
                proposal_seed=self.proposal_seed,
                producer_role=producer_role,
                frozen_profile_ref=self.frozen_profile_ref,
                focused_parent_language=bl_icf_single_parent_mode,
                role_instruction=director_instruction,
            )
        else:
            prompt = fresh_r1.render_proposal_prompt(
                self.proposal_template,
                side_identity=context_identity,
                logical_slot_id=f"{context_identity}:{producer_role}",
                proposal_seed=self.proposal_seed,
                producer_role=producer_role,
                protocol_requirements=self.proposal_protocol_requirements,
                role_instruction=director_instruction or _RESEARCH_ROLE_INSTRUCTIONS[producer_role],
            )
        # Prior evidence is data, not template syntax (JSON can contain "}}").
        prompt += self.proposal_context_appendix
        if proposal_lane == PROPOSAL_LANE_BL_ICF_PROGRAM:
            contract = provider_context_view.get("contract", {})
            execution_envelope = {
                key: contract[key]
                for key in (
                    "dataset",
                    "evaluation_split",
                    "candidate_universe",
                    "heldout_access",
                    "epochs_requested",
                    "eval_step",
                    "stopping_step",
                    "worker_ceiling_seconds",
                )
                if isinstance(contract, Mapping) and contract.get(key) is not None
            }
            if execution_envelope:
                prompt += (
                    "\n\nThe hard frozen experiment envelope for this call is "
                    "shown below. epochs_requested is the maximum training request; "
                    "the configured eval_step/stopping_step native early stopping "
                    "remains active and is the normal completion path. Design the "
                    "candidate so that a legitimate native early-stopped training "
                    "and evaluation run fits within worker_ceiling_seconds. Any "
                    "preprocessing or precompute is "
                    "charged inside the same ceiling. Do not reduce epochs, "
                    "alter the dataset/split/candidate universe, access a "
                    "forbidden partition, relabel estimated cost, or use a "
                    "fallback to claim compatibility. The fixed-batch resource "
                    "probe is authoritative and an incompatible candidate is "
                    "a typed rejection:\n"
                    + canonical_json_bytes(execution_envelope).decode("utf-8")
                )
        if proposal_lane == PROPOSAL_LANE_OPEN_SPEC:
            prompt += (
                "\n\nUse implementation_requirements as the single operational definition "
                "of this candidate's changed mathematics and active parameters: specify "
                "the relevant operations, dimensions, coefficients, reductions and gradient "
                "destinations, including what is replaced and what is inherited. Use the "
                "other existing fields for motivation, predictions and alternative tests; "
                "refer to that definition instead of restating incompatible formulas or "
                "preservation clauses. A mechanism-off or matched control is a different "
                "experiment, not an extra component of this candidate."
            )
        provider_state = provider_context_view.get("state")
        provider_state = (
            provider_state if isinstance(provider_state, Mapping) else {}
        )
        active_task = provider_state.get("task")
        if (
            isinstance(active_task, Mapping)
            and active_task.get("execution_eligible_this_round") is True
            and active_task.get("binding_requirement")
            == "EXACT_EFFECTIVE_IDENTITY"
        ):
            prompt += (
                "\n\nACTIVE EXACT FEEDBACK TASK (highest priority): This "
                "round is not ordinary discovery. The role-specific discovery "
                "default is superseded by state.task. Return only a proposal "
                "that resolves its required operation against the exact target "
                "semantic identity, effective experiment, and effective family "
                "shown there. A parent reference, prose-only control claim, or "
                "unrelated innovation does not satisfy the task and must not be "
                "proposed. Copy state.task.target_mechanism_program exactly into "
                "the proposal mechanism_program; do not reconstruct, paraphrase, "
                "or independently redesign it. Preserve the task protocol and "
                "exact target identity.\n"
                + canonical_json_bytes(active_task).decode("utf-8")
            )
        elif (
            isinstance(active_task, Mapping)
            and active_task.get("execution_eligible_this_round") is True
            and active_task.get("binding_requirement") == "NEEDS_PROPOSAL"
            and active_task.get("operation")
            in {"MATCHED_CONTROL", "MECHANISM_OFF"}
        ):
            prompt += (
                "\n\nACTIVE STRUCTURAL FEEDBACK TASK (highest priority): "
                "Design a new candidate that performs the requested "
                "state.task.operation as a genuine matched control or "
                "mechanism-off experiment. Treat "
                "state.task.frontier_mechanism_program only as the comparison "
                "anchor; do not copy it verbatim as the proposed executable "
                "program. Preserve the fixed protocol and unrelated mechanism "
                "slots, and express the structural control relation in the "
                "existing mechanism_program fields. The SearchSpaceAdapter "
                "will confirm that relation before execution.\n"
                + canonical_json_bytes(active_task).decode("utf-8")
            )
        baseline_objective = provider_context_view.get("objective")
        parent_label = "the configured frozen parent"
        if (
            isinstance(baseline_objective, Mapping)
            and isinstance(baseline_objective.get("parent_anchor"), Mapping)
        ):
            parent_anchor = baseline_objective["parent_anchor"]
            parent_name = parent_anchor.get("name")
            parent_label = (
                parent_name.strip()
                if isinstance(parent_name, str) and parent_name.strip()
                else parent_label
            )
            metric_name = baseline_objective.get("metric")
            metric_label = (
                f"real fixed-protocol dev {metric_name}"
                if isinstance(metric_name, str) and metric_name.strip()
                else "the real fixed-protocol dev metric"
            )
            prompt += (
                f"\n\nSINGLE-ROOT SEARCH-TREE OBJECTIVE: {parent_label} remains "
                "the frozen root and paired comparator. Build the next proposal "
                "from state.lineage_parent_mechanism_program when it identifies "
                "the active genuinely improved descendant; otherwise build from "
                f"the exact {parent_label} root. Both ordinary search and the "
                "frontier_architect innovation lane may continue from that active "
                "successful node, while every descendant remains in the same "
                f"{parent_label}-rooted tree. Propose a focused, causally coherent "
                f"mechanism with a credible path to higher {metric_label} than "
                f"{parent_label}; negative candidates never replace the active "
                "construction parent."
            )
        elif (
            isinstance(baseline_objective, Mapping)
            and isinstance(
                baseline_objective.get("basic_baseline"), Mapping
            )
            and isinstance(
                baseline_objective.get("strong_baseline"), Mapping
            )
        ):
            prompt += (
                "\n\nBASELINE INTERPRETATION: objective.basic_baseline is the "
                "same-protocol basic floor and initial comparator; "
                "state.incumbent is the dynamic measured frontier; and "
                "objective.strong_baseline is an aspirational development "
                "target, not the paired runtime comparator. Baseline-pack "
                "milestones are informative, not hard gates. Treat a result "
                "below the basic floor as negative evidence, advance the "
                "frontier only with a genuine improvement over the current "
                "incumbent, and treat reaching the strong target as an ideal "
                "KEEP/confirmation/ablation priority without stopping the "
                "fixed search budget or claiming causality automatically."
            )
        prompt += _effect_first_recovery_guidance(provider_context_view)
        prompt += (
            "\n\nUse memory.recent_experiments and memory.related_mechanism_experiments "
            "to compare actual variants, their measured effects and costs. "
            "An exact or algebraically equivalent replay is not a new mechanism; "
            "a related but meaningfully changed variant is a distinct experiment. "
            "An earlier implementation failure does not refute the mechanism family. "
            "Use concrete positive and negative evidence to decide what to retain "
            "or change within the allowed search space. Only when the current "
            "selected task is IMPLEMENTATION_EFFICIENCY_ONLY must its exact "
            "mechanism be preserved while improving the realization. A historical "
            "efficiency-only attempt does not force that choice now; a new "
            "hypothesis remains normal discovery. Actual slow/censored execution "
            "is cost and feasibility evidence, not a fabricated final metric. "
            "Treat state.task as a binding round-local proposal contract only when "
            "execution_eligible_this_round is true: satisfy its operation, parent, "
            "comparator, protocol, seed, and exact binding requirements. Otherwise it "
            "is deferred scientific evidence, not executable work. "
            + (
                "After a negative result, keep the current active genuinely "
                "successful construction parent (or fall back to the frozen "
                f"{parent_label} root when no successful lineage exists), and "
                "choose a different high-value causal delta. Keep the frozen root "
                "as the paired comparator and never promote the failed candidate. "
                if generic_single_parent_mode
                else "After a negative result, use the strongest relevant measured "
                "parent and related evidence to choose a promising change; "
                "a coherent multi-axis mechanism remains legitimate exploration. "
            )
            + "Evidence feedback informs candidate design; it is not a validation "
            "prerequisite or a reason to abandon exploration. Identify "
            "the chosen construction parent and frozen-root comparator, the focused "
            "causal mechanism change, and make its complete changed-axis footprint, expected "
            "observable, falsifier, and next discriminative test explicit in "
            "the existing mechanism_program/OpenSpec fields. A multi-axis result "
            "supports learning about that measured combination immediately, not "
            "a causal claim about every component. Declared alternative explanations "
            "are hypotheses. Use them with related observations to design a better "
            "candidate, including a useful subtraction or new combination when "
            "justified; do not turn them into compulsory control experiments."
            "\n\nUse this complete role-scoped Research Context JSON as the "
            "authoritative input for this call:\n"
            + canonical_json_bytes(provider_context_view).decode("utf-8")
        )
        namespace = f":{logical_namespace}" if logical_namespace else ""
        call_role = (
            "meta_strategy_synthesis"
            if logical_namespace is not None
            and logical_namespace.startswith("offline-replay:")
            else producer_role
        )
        is_mini_program_call = (
            proposal_lane == PROPOSAL_LANE_BL_ICF_PROGRAM
            and call_role in _INDEPENDENT_MINI_PRODUCER_ROLES
        )
        is_frontier_program_call = (
            proposal_lane == PROPOSAL_LANE_BL_ICF_PROGRAM
            and call_role == "frontier_architect"
        )
        model_selection = select_provider_model(call_role)
        selected_model = model_selection.requested_model
        is_full_model_program_call = (
            proposal_lane == PROPOSAL_LANE_BL_ICF_PROGRAM
            and model_selection.tier == "STRONG"
        )
        logical_suffix = (
            "bl-icf-proposal"
            if proposal_lane == PROPOSAL_LANE_BL_ICF
            else "bl-icf-program-proposal"
            if proposal_lane == PROPOSAL_LANE_BL_ICF_PROGRAM
            else "proposal"
        )
        logical_call_id = (
            f"{self.session_id}{namespace}:{context_identity}:"
            f"{producer_role}:{logical_suffix}"
        )
        resume_request_digest: str | None = None
        if logical_namespace is None:
            sealed_identity = self._resume_request_by_producer_role.pop(
                producer_role, None
            )
            if sealed_identity is not None:
                logical_call_id, resume_request_digest = sealed_identity
        if proposal_lane == PROPOSAL_LANE_BL_ICF:
            role_schema_path = _materialize_role_bound_bl_icf_schema(
                call_root=self.call_root,
                proposal_schema=self.bl_icf_proposal_schema,
                producer_role=producer_role,
            )
            provider_call_root = (
                self.call_root / "proposals" / "bl-icf" / producer_role
            )
        elif proposal_lane == PROPOSAL_LANE_BL_ICF_PROGRAM:
            role_schema_path = _materialize_role_bound_bl_icf_program_schema(
                call_root=self.call_root,
                proposal_schema=self.bl_icf_program_proposal_schema,
                producer_role=producer_role,
                frozen_profile_ref=self.frozen_profile_ref,
                required_base_model_config=(
                    PARENT_BASE_MODEL_CONFIG
                    if bl_icf_single_parent_mode
                    else None
                ),
            )
            transport_namespace = "bl-icf-program"
            if is_frontier_program_call:
                transport_namespace = _FRONTIER_RESPONSES_TRANSPORT_NAMESPACE
            elif producer_role in _INDEPENDENT_MINI_PRODUCER_ROLES:
                # The compact prompt/schema are one transport contract.  A
                # sealed call must resume in this root or fail closed; never
                # reinterpret an older root under the V2 representation.
                transport_namespace = (
                    _INDEPENDENT_MINI_TRANSPORT_NAMESPACE
                    if is_mini_program_call
                    else _INDEPENDENT_MINI_LEGACY_TRANSPORT_NAMESPACE
                )
            provider_call_root = (
                self.call_root / "proposals" / transport_namespace / producer_role
            )
        else:
            role_schema_path = _materialize_role_bound_proposal_schema(
                call_root=self.call_root,
                proposal_schema=self.proposal_schema,
                producer_role=producer_role,
            )
            provider_call_root = self.call_root / "proposals" / producer_role
        replay = self._proposal_replay_remaining.pop(producer_role, None)
        if replay is not None and proposal_lane == PROPOSAL_LANE_BL_ICF_PROGRAM:
            replay_response = replay["response"]
            if not isinstance(replay_response, Mapping):
                raise fresh_r1.FreshR1Error(
                    "replayed research proposal response must be an object"
                )
            jsonschema.validate(
                canonical_value(replay_response),
                json.loads(role_schema_path.read_text(encoding="utf-8")),
            )
            replay_proposal = _one_proposal(
                replay_response,
                kind="replayed strict BL-ICF program research proposal",
            )
            _, _, replay_parent_binding = _decode_bl_icf_program_proposal(
                replay_proposal,
                producer_role=producer_role,
            )
            expected_parent_binding = _expected_parent_binding_for_proposal(
                context_view=context_view,
                provider_context_view=provider_context_view,
            )
            if (
                not bl_icf_single_parent_mode
                and replay_parent_binding != expected_parent_binding
            ):
                replay = None
        if replay is None:
            call_kwargs = {
                "call_root": provider_call_root,
                "schema_path": role_schema_path,
                "logical_call_id": logical_call_id,
                "session_id": self.session_id,
                "prompt": prompt,
                "provider_role": call_role,
                "requested_model": selected_model,
                "token_ceiling": (
                    RESEARCH_FRONTIER_PROGRAM_TOTAL_TOKEN_CEILING
                    if is_full_model_program_call
                    else max(
                        self.total_token_ceiling,
                        RESEARCH_BL_ICF_PROGRAM_TOTAL_TOKEN_CEILING,
                    )
                    if proposal_lane == PROPOSAL_LANE_BL_ICF_PROGRAM
                    else self.total_token_ceiling
                ),
                "output_token_ceiling": (
                    RESEARCH_FRONTIER_PROGRAM_OUTPUT_TOKEN_CEILING
                    if is_full_model_program_call
                    else self._budgeted_output_ceiling(context_identity)
                ),
                "credential_config_path": self.credential_config_path,
                "expected_transport_release_digest": self.config_identity.get(
                    "release_digest"
                ),
                "maximum_physical_attempts": self.maximum_physical_attempts,
            }
            if is_mini_program_call:
                call_kwargs["wire_api"] = _INDEPENDENT_MINI_WIRE_API
            elif is_frontier_program_call:
                call_kwargs["wire_api"] = _FRONTIER_RESPONSES_WIRE_API
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
            try:
                result = self.provider_call(**call_kwargs)
            except Exception as error:
                # Bounded transport failures return their receipt below. An
                # exception without that result must retain its actual cause;
                # do not buy new proposals or assume a retryable transport fault.
                raise ProviderRequestError(f"{type(error).__name__}: {error}") from error
            self._charge_output_tokens(context_identity, result)
            _record_trace(
                self,
                kind="research_producer",
                logical_call_id=logical_call_id,
                prompt=prompt,
                result=result,
                transport_ceiling={
                    "total_tokens": call_kwargs["token_ceiling"],
                    "output_tokens": call_kwargs["output_token_ceiling"],
                },
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
            raise fresh_r1.FreshR1Error("research proposal response must be an object")
        if proposal_lane == PROPOSAL_LANE_BL_ICF:
            jsonschema.validate(
                canonical_value(response),
                json.loads(role_schema_path.read_text(encoding="utf-8")),
            )
            proposal = _one_proposal(response, kind="BL-ICF research proposal")
            return _bl_icf_candidate_proposal_v4(
                proposal,
                producer_role=producer_role,
                context_view=context_view,
                logical_call_id=logical_call_id,
            )
        if proposal_lane == PROPOSAL_LANE_BL_ICF_PROGRAM:
            jsonschema.validate(
                canonical_value(response),
                json.loads(role_schema_path.read_text(encoding="utf-8")),
            )
            proposal = _one_proposal(
                response,
                kind="strict BL-ICF program research proposal",
            )
            expected_parent_binding = _expected_parent_binding_for_proposal(
                context_view=context_view,
                provider_context_view=provider_context_view,
            )
            parent_program = _parent_program_for_proposal(
                provider_context_view=provider_context_view,
                expected_parent_binding=expected_parent_binding,
            )
            parent_options = context_view.get("construction_parent_options", ())
            if parent_options:
                expected_parent_binding, parent_program = _selected_construction_parent(
                    proposal["mechanism_program"], parent_options,
                )
            provider_state = provider_context_view.get("state")
            force_active_parent = (
                isinstance(provider_state, Mapping)
                and isinstance(provider_state.get("frozen_parent_binding"), Mapping)
            )
            try:
                return _bl_icf_program_proposal_v1(
                    proposal,
                    producer_role=producer_role,
                    expected_parent_binding=expected_parent_binding,
                    parent_program=parent_program,
                    force_parent_binding=force_active_parent,
                )
            except _BlIcfProgramCompileError:
                # Compiler syntax may be normalized only when the registry has
                # one exact mechanical binding.  A second model call could
                # instead replace topology, mechanism, hypothesis, or ablation.
                # Preserve the first proposal and its typed compiler report.
                raise

        fresh_r1.validate_v4_response_contract(
            response,
            provider_schema=self.proposal_schema,
        )
        proposal = _one_proposal(response, kind="research proposal")
        if proposal.get("producer_role") != producer_role:
            raise fresh_r1.FreshR1Error(
                "Provider changed the preassigned Producer role"
            )
        return _normalize_open_spec_proposal(proposal)


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
        bl_icf_implementation_appendix_source: str | Path | None = (
            _DEFAULT_BL_ICF_IMPLEMENTATION_APPENDIX
        ),
        diffusion_flow_cf_implementation_appendix_source: str | Path | None = (
            _DEFAULT_DIFFUSION_FLOW_CF_IMPLEMENTATION_APPENDIX
        ),
        sequential_scaling_implementation_appendix_source: str | Path | None = (
            _DEFAULT_SEQUENTIAL_SCALING_IMPLEMENTATION_APPENDIX
        ),
        semantic_id_generative_implementation_appendix_source: str | Path | None = (
            _DEFAULT_SEMANTIC_ID_GENERATIVE_IMPLEMENTATION_APPENDIX
        ),
        implementation_schema_source: Mapping[str, Any] | str | Path = _DEFAULT_IMPLEMENTATION_SCHEMA,
        implementation_schema_path: Path = _DEFAULT_IMPLEMENTATION_SCHEMA,
        maximum_physical_attempts: int | None = fresh_r1.MAX_PHYSICAL_ATTEMPTS,
        implementation_output_token_ceiling_total_per_candidate: int | None = None,
        shared_implementation_root: Path | None = None,
        paired_search_seed: int | None = None,
        arm: str | None = None,
        implementation_output_token_ceiling_per_physical_call: int | None = None,
    ) -> None:
        self.config_identity = _config_identity(config_source)
        self.credential_config_path = (
            None
            if isinstance(config_source, Mapping)
            else Path(config_source).resolve()
        )
        self.call_root = Path(call_root)
        self.session_id = str(session_id)
        self.shared_implementation_root = (
            None
            if shared_implementation_root is None
            else Path(shared_implementation_root).resolve()
        )
        self.paired_search_seed = paired_search_seed
        self.arm = arm
        if self.shared_implementation_root is not None:
            if isinstance(paired_search_seed, bool) or not isinstance(
                paired_search_seed, int
            ):
                raise ValueError(
                    "shared implementation root requires paired_search_seed"
                )
            if arm not in {"A", "B", "C"}:
                raise ValueError(
                    "shared implementation root requires arm A, B, or C"
                )
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
        self.bl_icf_implementation_appendix = (
            ""
            if bl_icf_implementation_appendix_source is None
            else _read_text(bl_icf_implementation_appendix_source)
        )
        self.diffusion_flow_cf_implementation_appendix = (
            ""
            if diffusion_flow_cf_implementation_appendix_source is None
            else _read_text(diffusion_flow_cf_implementation_appendix_source)
        )
        self.sequential_scaling_implementation_appendix = (
            ""
            if sequential_scaling_implementation_appendix_source is None
            else _read_text(sequential_scaling_implementation_appendix_source)
        )
        self.semantic_id_generative_implementation_appendix = (
            ""
            if semantic_id_generative_implementation_appendix_source is None
            else _read_text(semantic_id_generative_implementation_appendix_source)
        )
        self.implementation_schema = _read_json(implementation_schema_source)
        jsonschema.validators.validator_for(
            self.implementation_schema
        ).check_schema(self.implementation_schema)
        self.implementation_schema_path = Path(implementation_schema_path)
        self.maximum_physical_attempts = (
            None
            if maximum_physical_attempts is None
            else int(maximum_physical_attempts)
        )
        if (
            isinstance(maximum_physical_attempts, bool)
            or (
                self.maximum_physical_attempts is not None
                and self.maximum_physical_attempts < 1
            )
        ):
            raise ValueError("maximum_physical_attempts must be positive or None")
        self.implementation_output_token_ceiling_per_physical_call = int(
            self.total_token_ceiling
            if implementation_output_token_ceiling_per_physical_call is None
            else implementation_output_token_ceiling_per_physical_call
        )
        if self.implementation_output_token_ceiling_per_physical_call < 1:
            raise ValueError(
                "implementation_output_token_ceiling_per_physical_call "
                "must be positive"
            )
        self.implementation_output_token_ceiling_total_per_candidate = (
            None
            if implementation_output_token_ceiling_total_per_candidate is None
            else int(implementation_output_token_ceiling_total_per_candidate)
        )
        if (
            self.implementation_output_token_ceiling_total_per_candidate is not None
            and self.implementation_output_token_ceiling_total_per_candidate < 1
        ):
            raise ValueError(
                "implementation_output_token_ceiling_total_per_candidate "
                "must be positive"
            )
        self._implementation_output_tokens_by_candidate: dict[str, int] = {}
        self.last_call_trace: Mapping[str, Any] | None = None
        self.call_traces: tuple[Mapping[str, Any], ...] = ()

    def _budgeted_output_ceiling(self, candidate_id: str) -> int:
        total = self.implementation_output_token_ceiling_total_per_candidate
        if total is None:
            return self.implementation_output_token_ceiling_per_physical_call
        consumed = self._implementation_output_tokens_by_candidate.get(
            candidate_id,
            0,
        )
        remaining = total - consumed
        if remaining < 1:
            raise fresh_r1.FreshR1Error(
                "implementation output token budget is exhausted for this candidate"
            )
        # Retries are the same logical call.  Charge their observed usage to the
        # aggregate ledger; do not pre-split a healthy first response by the
        # maximum number of physical attempts that might never occur.
        return min(
            self.implementation_output_token_ceiling_per_physical_call,
            remaining,
        )

    def _charge_output_tokens(
        self,
        candidate_id: str,
        result: fresh_r1.ProviderAttemptResult,
    ) -> None:
        total = self.implementation_output_token_ceiling_total_per_candidate
        if total is None:
            return
        consumed = self._implementation_output_tokens_by_candidate.get(
            candidate_id,
            0,
        )
        consumed += _physical_attempt_output_tokens(result)
        if consumed > total:
            raise fresh_r1.FreshR1Error(
                "implementation output token budget exceeded for this candidate"
            )
        self._implementation_output_tokens_by_candidate[candidate_id] = consumed

    def implementation_template_for_search_space(
        self,
        search_space_id: str | None,
    ) -> str:
        return _implementation_template_for_search_space(
            self.implementation_template,
            self.bl_icf_implementation_appendix,
            search_space_id,
            diffusion_flow_cf_appendix=(
                self.diffusion_flow_cf_implementation_appendix
            ),
            sequential_scaling_appendix=(
                self.sequential_scaling_implementation_appendix
            ),
            semantic_id_generative_appendix=(
                self.semantic_id_generative_implementation_appendix
            ),
        )

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._call(request, logical_namespace=None)

    def call_with_namespace(
        self, request: Mapping[str, Any], *, logical_namespace: str,
    ) -> Mapping[str, Any]:
        return self._call(request, logical_namespace=logical_namespace)

    def _call(
        self, request: Mapping[str, Any], *, logical_namespace: str | None,
    ) -> Mapping[str, Any]:
        if not isinstance(request, Mapping):
            raise TypeError("implementation request must be a mapping")
        candidate_id = str(request.get("blind_candidate_id", "request"))
        repair_attempt = int(request.get("repair_attempt", 0))
        compiled_mechanism = request.get("compiled_mechanism")
        space_identity = (
            compiled_mechanism.get("space_identity")
            if isinstance(compiled_mechanism, Mapping)
            else None
        )
        search_space_id = (
            str(space_identity.get("search_space_id"))
            if isinstance(space_identity, Mapping)
            and isinstance(space_identity.get("search_space_id"), str)
            else None
        )
        effective_template = self.implementation_template_for_search_space(
            search_space_id
        )
        if compiled_mechanism is None and isinstance(request.get("blind_research_spec"), Mapping):
            effective_template = (
                "For this OpenSpec, implementation_requirements defines the current "
                "candidate's research-owned operations and parameters under the fixed "
                "service policy. The other scientific fields explain motivation, expected "
                "observations and alternative experiments. Implement that operational "
                "definition; do not add a control's losses or components to this candidate. "
                "Explain any realization choices left open by the definition in the "
                "existing implementation_summary.\n\n" + effective_template
            )
        effective_schema = _implementation_schema_for_request(
            self.implementation_schema,
            request,
        )
        if effective_schema == self.implementation_schema:
            effective_schema_path = self.implementation_schema_path
        else:
            response_mode = str(request["service_policy"]["response_mode"])
            effective_schema_path = _write_role_bound_schema(
                effective_schema,
                self.call_root
                / "implementation-schemas"
                / (
                    f"{response_mode.lower()}-"
                    f"{sha256_digest(effective_schema)}.schema.json"
                ),
            )
        prompt = fresh_r1.render_implementation_prompt(
            effective_template,
            request,
        )
        request_identity = sha256_digest(
            {
                "blind_candidate_id": candidate_id,
                "repair_attempt": repair_attempt,
                "prompt_digest": sha256_digest(prompt),
            }
        )
        if self.shared_implementation_root is None:
            logical_session_id = self.session_id
            implementation_root = (
                self.call_root
                / "implementations"
                / _IMPLEMENTATION_RESPONSES_TRANSPORT_NAMESPACE
                / candidate_id
            )
        else:
            shared_identity = shared_implementation_identity(
                search_seed=int(self.paired_search_seed),
                request=request,
                arm=str(self.arm),
                implementer_model=select_provider_model(
                    "implementation_revision" if repair_attempt else "implementer"
                ).requested_model,
                implementer_prompt_digest=sha256_digest(effective_template),
                implementer_schema_digest=sha256_digest(effective_schema),
            )
            logical_session_id = (
                f"shared-implementation:search-seed-{self.paired_search_seed}"
            )
            implementation_root = (
                self.shared_implementation_root
                / _IMPLEMENTATION_RESPONSES_TRANSPORT_NAMESPACE
                / str(shared_identity["shared_root_key"])
            )
        logical_call_id = (
            f"{logical_session_id}:{candidate_id}:request-{request_identity}:"
            f"implementation:{repair_attempt}"
        )
        if logical_namespace is not None:
            if not logical_namespace.startswith("transport-retry-") or not (
                logical_namespace.removeprefix("transport-retry-").isdigit()
            ):
                raise ValueError("invalid implementation transport retry namespace")
            logical_call_id = f"{logical_call_id}:{logical_namespace}"
            implementation_root = implementation_root / logical_namespace
        result = self.provider_call(
            call_root=implementation_root / f"revision_{repair_attempt:02d}",
            schema_path=effective_schema_path,
            logical_call_id=logical_call_id,
            session_id=logical_session_id,
            prompt=prompt,
            provider_role=(
                "implementation_revision" if repair_attempt else "implementer"
            ),
            requested_model=select_provider_model(
                "implementation_revision" if repair_attempt else "implementer"
            ).requested_model,
            token_ceiling=self.total_token_ceiling,
            output_token_ceiling=self._budgeted_output_ceiling(candidate_id),
            credential_config_path=self.credential_config_path,
            expected_transport_release_digest=self.config_identity.get(
                "release_digest"
            ),
            maximum_physical_attempts=self.maximum_physical_attempts,
            wire_api=_IMPLEMENTATION_RESPONSES_WIRE_API,
        )
        self._charge_output_tokens(candidate_id, result)
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
            canonical_value(effective_schema),
        )
        return dict(_one_proposal(response, kind="implementation"))


__all__ = [
    "PROPOSAL_LANE_BL_ICF",
    "PROPOSAL_LANE_BL_ICF_PROGRAM",
    "PROPOSAL_LANE_OPEN_SPEC",
    "ProviderImplementerGateway",
    "ProviderResearchProducer",
    "RESEARCH_BL_ICF_PROGRAM_TOTAL_TOKEN_CEILING",
    "RESEARCH_FRONTIER_PROGRAM_OUTPUT_TOKEN_CEILING",
    "RESEARCH_FRONTIER_PROGRAM_TOTAL_TOKEN_CEILING",
    "load_provider_proposal_replay",
    "provider_proposal_replay_identity",
    "shared_implementation_identity",
]
