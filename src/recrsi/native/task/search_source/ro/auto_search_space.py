"""Task-generic AutoSpace validation and conceptual proposal binding.

The module ends at the conceptual boundary.  Executable identity is created only
by RecClaw's existing Implementer, qualification, and capability-registry path.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator, FormatChecker
from referencing import Registry, Resource
from referencing.exceptions import NoSuchResource


CORE_FILES = (
    "auto_space_manifest_v1.json",
    "auto_space_family_v1.json",
    "auto_space_primitive_registry_v1.json",
    "auto_space_grammar_v1.json",
    "auto_space_capability_policy_v1.json",
)
LEGACY_CORE_FILES = (
    "search_space_manifest_v1.json",
    "family_contract_v1.json",
    "primitive_registry_v1.json",
    "mechanism_grammar_v1.json",
    "capability_policy_v1.json",
)
SCHEMA_BY_CORE_FILE = {
    "auto_space_manifest_v1.json": "auto_space_manifest.schema.json",
    "auto_space_family_v1.json": "auto_space_family.schema.json",
    "auto_space_primitive_registry_v1.json": "auto_space_primitive_registry.schema.json",
    "auto_space_grammar_v1.json": "auto_space_grammar.schema.json",
    "auto_space_capability_policy_v1.json": "auto_space_capability_policy.schema.json",
}
SCHEMA_ROOT = Path(__file__).resolve().parents[1] / "schemas" / "auto_space_v1"
FROZEN_MANIFEST = "AUTO_SPACE_FROZEN_MANIFEST.json"
MANUAL_FROZEN_MANIFEST = "MANUAL_SPACE_FROZEN_MANIFEST.json"
MANUAL_CONTRACT_FILE = "MANUAL_SPACE_CONTRACT.json"
MANUAL_PROVENANCE_FILE = "MANUAL_SPACE_PROVENANCE.json"
MANUAL_PACKAGE_MANIFEST_FILE = "MANUAL_SPACE_MANIFEST.json"
CONFORMANCE_REPORT = "AUTO_SPACE_CONFORMANCE_REPORT.json"
TRUSTED_CONTRACT_FILE = "AUTO_SPACE_TRUSTED_CONTRACT.json"
GENESIS_CALL_AUTHORIZATION_FILE = "AUTO_SPACE_GENESIS_CALL_AUTHORIZATION.json"
PROVIDER_RECEIPT_FILES = (
    "AUTO_SPACE_GENERATOR_PROVIDER_CALL_RECEIPT.json",
    "AUTO_SPACE_CRITIC_PROVIDER_CALL_RECEIPT.json",
)
POST_FREEZE_REPORT = "AUTO_SPACE_CONTROLLER_INTEGRATION_REPORT.json"
BASE_PROTOCOL_ID = "recclaw-external-comparison-v1"
BASE_PROTOCOL_VERSION = "v1.0-Final-Prefreeze"
BASE_PROTOCOL_SHA256 = "71400a344d46e14a10feb60cf4cf82dac603faf0c59dcbaaccfb204760176762"
V12_SIGNED_AUTOSPACE_AMENDMENT_VERSION = "v1.2-SIGNED"
# Backward-compatible public alias for existing callers that imported the old constant.
SIGNED_AUTOSPACE_AMENDMENT_VERSION = V12_SIGNED_AUTOSPACE_AMENDMENT_VERSION
V13_SIGNED_AUTOSPACE_AMENDMENT_VERSION = "v1.3-SIGNED"
V14_SIGNED_AUTOSPACE_AMENDMENT_VERSION = "v1.4-SIGNED"
V15_SIGNED_AUTOSPACE_AMENDMENT_VERSION = "v1.5-SIGNED"
V16_SIGNED_AUTOSPACE_AMENDMENT_VERSION = "v1.6-SIGNED"
V16_TRANSPORT_RETRY1_SIGNED_AUTOSPACE_AMENDMENT_VERSION = "v1.6-TRANSPORT-RETRY1-SIGNED"
V161_SIGNED_AUTOSPACE_AMENDMENT_VERSION = "v1.6.1-SIGNED"
SIGNED_AUTOSPACE_AMENDMENT_FILES = {
    V12_SIGNED_AUTOSPACE_AMENDMENT_VERSION: "PROTOCOL_AMENDMENT_v1.2_AUTOSPACE_SIGNED.md",
    V13_SIGNED_AUTOSPACE_AMENDMENT_VERSION: "PROTOCOL_AMENDMENT_v1.3_AUTOSPACE_SIGNED.md",
    V14_SIGNED_AUTOSPACE_AMENDMENT_VERSION: "PROTOCOL_AMENDMENT_v1.4_AUTOSPACE_SIGNED.md",
    V15_SIGNED_AUTOSPACE_AMENDMENT_VERSION: "PROTOCOL_AMENDMENT_v1.5_AUTOSPACE_SIGNED.md",
    V16_SIGNED_AUTOSPACE_AMENDMENT_VERSION: "PROTOCOL_AMENDMENT_v1.6_AUTOSPACE_SIGNED.md",
    V16_TRANSPORT_RETRY1_SIGNED_AUTOSPACE_AMENDMENT_VERSION: "PROTOCOL_AMENDMENT_v1.6_AUTOSPACE_TRANSPORT_RETRY1_SIGNED.md",
    V161_SIGNED_AUTOSPACE_AMENDMENT_VERSION: "PROTOCOL_AMENDMENT_v1.6.1_AUTOSPACE_SIGNED.md",
}
SIGNED_AUTOSPACE_AMENDMENT_VERSIONS = frozenset(SIGNED_AUTOSPACE_AMENDMENT_FILES)
V13_DRAFT_AUTOSPACE_AMENDMENT_VERSION = "v1.3-DRAFT-UNSIGNED"
DRAFT_AUTOSPACE_AMENDMENT_VERSIONS = frozenset({
    "v1.2-DRAFT-UNSIGNED",
    "v2.0-DRAFT-UNSIGNED",
    "v1.6-DRAFT-UNSIGNED",
    "v1.6.1-DRAFT-UNSIGNED",
    V13_DRAFT_AUTOSPACE_AMENDMENT_VERSION,
})
REAL_GENERATION_STATUS = "SIGNED_AUTHORIZED_FOR_ONE_REAL_GENERATION"
DRAFT_GENERATION_STATUS = "DRAFT_NOT_AUTHORIZED_FOR_REAL_GENERATION"
PROVIDER_OUTPUT_MODE_CANONICAL = "CANONICAL_RESPONSE"
PROVIDER_OUTPUT_MODE_DRAFT_ASSEMBLY = "SCIENTIFIC_DRAFT_TRUSTED_ASSEMBLY_V1"
PROVIDER_OUTPUT_MODE_V16 = "SCIENTIFIC_PLAN_PRIMITIVES_TRUSTED_ASSEMBLY_V2"
PROVIDER_OUTPUT_MODE_V161 = "SCIENTIFIC_PLAN_PRIMITIVES_TRUSTED_ASSEMBLY_V3"
CANONICAL_RESPONSE_SCHEMA = "auto_space_generator_response.schema.json"
PROVIDER_DRAFT_SCHEMA = "auto_space_provider_draft.schema.json"
BROKER_RESPONSE_FORMAT_NAME = "recclaw_campaign_proposals"
DIRECT_RESPONSE_FORMAT_NAME = "auto_space"
GENESIS_PROVIDER_GUIDANCE_V2_MARKER = "AUTOSPACE_PROVIDER_GUIDANCE_V2"
GENESIS_PROVIDER_GUIDANCE_V16_MARKER = "AUTOSPACE_PROVIDER_GUIDANCE_V16"
GENESIS_PROVIDER_GUIDANCE_V161_MARKER = "AUTOSPACE_PROVIDER_GUIDANCE_V161"
CONCEPTUAL_MUTATION_TARGET_ANCHORS = {
    "main_code.py:train_update_kernel": {
        "path": "main_code.py",
        "symbol": "main",
        "kind": "function",
        "meaning": (
            "conceptual mutation-region identifier for the train-update statements "
            "inside the supplied main() training loop; it is not a literal Python symbol"
        ),
    }
}
E2_SEARCH_SEEDS = tuple(range(41001, 41009))

MANUAL_SCHEMA_BY_CORE_FILE = {
    **SCHEMA_BY_CORE_FILE,
    "auto_space_manifest_v1.json": "manual_space_core_manifest.schema.json",
}
MANUAL_AUTO_EVIDENCE_FILES = frozenset(
    {
        FROZEN_MANIFEST,
        CONFORMANCE_REPORT,
        TRUSTED_CONTRACT_FILE,
        GENESIS_CALL_AUTHORIZATION_FILE,
        "AUTO_SPACE_GENESIS_MANIFEST.json",
        "AUTO_SPACE_GENESIS_REQUEST.json",
        "AUTO_SPACE_GENERATOR_PROVIDER_CALL_RECEIPT.json",
        "AUTO_SPACE_CRITIC_PROVIDER_CALL_RECEIPT.json",
        "AUTO_SPACE_GENERATOR_RAW_PROVIDER_RESPONSE.json",
        "AUTO_SPACE_CRITIC_RAW_PROVIDER_RESPONSE.json",
        "AUTO_SPACE_GENERATOR_SCIENTIFIC_RESPONSE.json",
        "AUTO_SPACE_CRITIC_SCIENTIFIC_RESPONSE.json",
        "AUTO_SPACE_GENERATOR_TRANSPORT_REQUEST.json",
        "AUTO_SPACE_CRITIC_TRANSPORT_REQUEST.json",
        POST_FREEZE_REPORT,
    }
)


FORBIDDEN_CONCEPTUAL_FIELDS = frozenset(
    {
        "capability_digest",
        "capability_ref",
        "capability_envelope",
        "executable_entrypoint",
        "implementation_ref",
        "qualification_receipt",
        "qualification_receipt_digest",
        "qualification_receipt_ref",
    }
)


class AutoSpaceValidationError(ValueError):
    """The package cannot be admitted as a conceptual search space."""


class SearchSpaceSource(str, Enum):
    MANUAL = "MANUAL"
    AUTO_GENERATED = "AUTO_GENERATED"


def canonical_bytes(value: Any) -> bytes:
    """Use RecClaw's RFC 8785 canonicalization, never a local JSON variant."""

    try:
        from recclaw_core.mechanism_space.canonical import canonical_bytes as project_bytes

        return project_bytes(value)
    except ImportError:
        try:
            import rfc8785
        except ImportError as error:
            raise RuntimeError(
                "AutoSpace requires RecClaw's rfc8785 canonical dependency"
            ) from error
        try:
            return rfc8785.dumps(value)
        except Exception as error:
            raise AutoSpaceValidationError(
                f"value is outside the RFC 8785 domain: {error}"
            ) from error


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




def signed_amendment_filename_for_version(amendment_version: Any) -> str:
    if (
        not isinstance(amendment_version, str)
        or amendment_version not in SIGNED_AUTOSPACE_AMENDMENT_FILES
    ):
        raise AutoSpaceValidationError("AutoSpace amendment is not in a supported signed state")
    return SIGNED_AUTOSPACE_AMENDMENT_FILES[amendment_version]


def signed_amendment_path_for_identity(
    identity: Mapping[str, Any], *, project_root: Path | None = None
) -> Path:
    if not isinstance(identity, Mapping):
        raise AutoSpaceValidationError("protocol_identity must be an object")
    filename = signed_amendment_filename_for_version(
        identity.get("autospace_amendment_version")
    )
    root = (
        Path(project_root)
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    return root / "protocol" / filename


def provider_output_mode(contract: Mapping[str, Any]) -> str:
    generation = contract.get("generation_policy")
    if not isinstance(generation, Mapping):
        raise AutoSpaceValidationError("generation_policy must be an object")
    mode = generation.get("provider_output_mode", PROVIDER_OUTPUT_MODE_CANONICAL)
    if mode not in {
        PROVIDER_OUTPUT_MODE_CANONICAL,
        PROVIDER_OUTPUT_MODE_DRAFT_ASSEMBLY,
        PROVIDER_OUTPUT_MODE_V16,
        PROVIDER_OUTPUT_MODE_V161,
    }:
        raise AutoSpaceValidationError("unsupported provider output mode")
    return str(mode)


def uses_deterministic_provider_assembly(contract: Mapping[str, Any]) -> bool:
    return provider_output_mode(contract) in {
        PROVIDER_OUTPUT_MODE_DRAFT_ASSEMBLY,
        PROVIDER_OUTPUT_MODE_V16,
        PROVIDER_OUTPUT_MODE_V161,
    }


def provider_design_guidance(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return deterministic provider guidance derived from the trusted design.

    The guidance deliberately distinguishes registry coverage from per-candidate
    composition limits and distinguishes conceptual mutation-region identifiers
    from structural Python anchors.  Those two distinctions are required for E2
    v3, whose trusted design has four required slots while candidates may compose
    at most three primitives, and whose train-update target is a conceptual region
    within ``main_code.py:main`` rather than a literal function named
    ``train_update_kernel``.
    """

    design = contract.get("required_search_space_design")
    if not isinstance(design, Mapping):
        return {}
    raw_slots = design.get("slots")
    if not isinstance(raw_slots, list) or not raw_slots:
        return {}

    required_slots: list[str] = []
    conceptual_targets: list[dict[str, Any]] = []
    seen_conceptual_targets: set[str] = set()
    for raw_slot in raw_slots:
        if not isinstance(raw_slot, Mapping):
            continue
        slot_id = raw_slot.get("slot_id")
        if isinstance(slot_id, str) and slot_id.strip():
            required_slots.append(slot_id.strip())
        targets = raw_slot.get("legal_mutation_targets")
        if not isinstance(targets, list):
            continue
        for target in targets:
            if (
                isinstance(target, str)
                and target in CONCEPTUAL_MUTATION_TARGET_ANCHORS
                and target not in seen_conceptual_targets
            ):
                seen_conceptual_targets.add(target)
                anchor = CONCEPTUAL_MUTATION_TARGET_ANCHORS[target]
                conceptual_targets.append(
                    {
                        "mutation_target": target,
                        "meaning": anchor["meaning"],
                        "required_real_baseline_anchor": {
                            "path": anchor["path"],
                            "symbol": anchor["symbol"],
                            "kind": anchor["kind"],
                        },
                        "anchor_rule": (
                            "Do not use the conceptual mutation-target identifier itself "
                            "as a baseline_anchors symbol. Anchor the primitive to the real "
                            "supplied Python symbol above."
                        ),
                    }
                )

    guidance: dict[str, Any] = {
        "required_slot_ids": required_slots,
        "minimum_registry_primitive_records": len(required_slots),
        "registry_coverage_requirement": (
            "The provider scientific draft must contain at least one primitive record "
            "for every required_slot_id. A required slot may not be omitted."
        ),
        "candidate_cardinality_semantics": (
            "max_primitives_per_candidate limits how many registered primitives one "
            "candidate may compose; it does NOT cap the number of primitive records in "
            "the registry or scientific draft."
        ),
    }
    if "max_primitives_per_candidate" in design:
        guidance["max_primitives_per_candidate"] = design[
            "max_primitives_per_candidate"
        ]
    for name in (
        "target_primitive_count",
        "proposal_pool_primitive_count",
        "proposal_mechanism_families_per_axis",
        "minimum_distinct_mechanism_families_per_axis",
        "max_mutation_targets",
        "effective_width_guard",
    ):
        if name in design:
            guidance[name] = design[name]
    if provider_output_mode(contract) in {PROVIDER_OUTPUT_MODE_V16, PROVIDER_OUTPUT_MODE_V161}:
        guidance["generation_stages"] = [
            "SPACE_PLANNER",
            "PRIMITIVE_GENERATION",
            "CRITIC_MERGER",
        ]
        guidance["parameter_policy"] = "FIXED_OR_BOUNDED_DISCRETE_ONLY"
        guidance["mechanism_family_diversity"] = (
            "Each axis needs at least two semantically distinct mechanism families; "
            "renaming, parameter-only variants, cosmetic prose, and equivalent effect "
            "paths do not count."
        )
    if conceptual_targets:
        guidance["conceptual_mutation_targets"] = conceptual_targets
    return guidance


def genesis_provider_instructions(contract: Mapping[str, Any]) -> str:
    """Build the request-level provider instruction profile.

    Canonical-response contracts retain the historical instruction byte-for-byte.
    Draft-assembly contracts use the v2 guidance profile so future provider calls
    cannot confuse registry size with candidate composition limits or conceptual
    mutation regions with Python symbols.
    """

    historical = (
        "Analyze only the supplied baseline and trusted contract. Produce one task-specific "
        "conceptual family, primitive registry, behavioral grammar, and capability policy. "
        "Generated permissions may only narrow the trusted contract, and any declared "
        "required_search_space_design must be followed exactly. Do not emit candidate "
        "source, executable identity, qualification claims, usage, model identity, or billing "
        "metadata. Every anchor must identify a real supplied Python class, function, or method."
    )
    mode = provider_output_mode(contract)
    if mode == PROVIDER_OUTPUT_MODE_CANONICAL:
        return historical

    if mode == PROVIDER_OUTPUT_MODE_V161:
        guidance = provider_design_guidance(contract)
        slots = ", ".join(guidance.get("required_slot_ids", []))
        return (
            "AUTOSPACE_PROVIDER_GUIDANCE_V161. Analyze only the supplied baseline and trusted contract. "
            "Stage A: plan exactly three semantically distinct mechanism families for each required axis. "
            "Stage B: author exactly 18 bounded conceptual proposal primitives, exactly three per axis and "
            "exactly one primitive for each planned family. Every parameter must be fully fixed with const. "
            "Do not leave any implementation choice unresolved: no either/or loss definitions, no unspecified "
            "bucket boundaries, probabilities, weighting schedules, or other latent choices. Architecture proposals "
            "must change structure or information flow rather than merely multiply an existing residual by one "
            "fixed scalar. A ranking-preserving scoring transform may be proposed when it has a substantive "
            "training-time effect; mark it low_impact_risk=true and RANKING_PRESERVING_INFERENCE, but do not rely "
            "on low-impact proposals alone for scoring coverage. Required axes are ["
            f"{slots}]. The Stage C critic is expected to prune the 18-proposal pool to a final 12 to 16 primitive "
            "space while retaining at least two distinct accepted families per axis. Do not emit candidate source, "
            "qualification claims, usage, billing, prior-result details, comparator-specific mechanisms, the v1.6 "
            "rejected mechanisms, or any known winning mechanism. Every baseline anchor must identify a real supplied "
            "Python class, function, or method."
        )

    if mode == PROVIDER_OUTPUT_MODE_V16:
        guidance = provider_design_guidance(contract)
        slots = ", ".join(guidance.get("required_slot_ids", []))
        return (
            "AUTOSPACE_PROVIDER_GUIDANCE_V16. Analyze only the supplied baseline and "
            "trusted contract. Stage A: plan two or three semantically distinct "
            "mechanism families for each required axis, with an explicit effect, phase, "
            "real baseline anchor, legal mutation target, and reachability rationale; "
            "do not emit code. Stage B: author 12 to 16 bounded conceptual primitives "
            "implementing that plan. Every primitive parameter must be fixed with const; "
            "do not leave unresolved enum or continuous choices. Do not use renamed, "
            "parameter-only, cosmetic, unreachable, or low-impact-only alternatives as "
            "diversity. Required axes are ["
            f"{slots}]. Trusted grammar, policy, frozen boundaries, executable identity, "
            "and novelty signatures are assembled locally. Mark an inference-only "
            "ranking-preserving scoring operation with the RANKING_PRESERVING_INFERENCE "
            "compatibility tag and low_impact_risk=true. Emit no candidate source, "
            "qualification claim, usage, billing, prior-result details, comparator-specific "
            "mechanism content, or known winning mechanism. Every baseline anchor must "
            "identify a real supplied Python class, function, or method."
        )

    guidance = provider_design_guidance(contract)
    slots = guidance.get("required_slot_ids", [])
    slot_text = ", ".join(str(item) for item in slots)
    conceptual_note = ""
    if guidance.get("conceptual_mutation_targets"):
        conceptual_note = (
            " The mutation target main_code.py:train_update_kernel is a conceptual "
            "train-update region inside the real supplied function main_code.py:main; "
            "it is not a literal Python symbol. For a primitive using that mutation "
            "target, use main_code.py:main (kind=function) as its structural baseline "
            "anchor rather than omitting the slot."
        )
    return (
        f"{GENESIS_PROVIDER_GUIDANCE_V2_MARKER}. Analyze only the supplied baseline and "
        "trusted contract. Author one task-specific scientific family and primitive "
        "registry draft; trusted grammar, capability policy, and other control-plane "
        "fields are assembled deterministically after the provider call. Generated "
        "permissions may only narrow the trusted contract, and required_search_space_design "
        f"must be followed exactly. Required slots are [{slot_text}], and the scientific "
        "draft must include at least one primitive for every required slot. "
        "max_primitives_per_candidate is a per-candidate composition limit, not a limit "
        "on registry or draft size."
        f"{conceptual_note} Do not emit candidate source, executable identity, qualification "
        "claims, usage, model identity, or billing metadata. Every baseline anchor must "
        "identify a real supplied Python class, function, or method."
    )


def _embedded_task_contract_for_prompt(
    genesis_request: Mapping[str, Any],
) -> dict[str, Any]:
    inputs = genesis_request.get("inputs")
    if not isinstance(inputs, list):
        raise AutoSpaceValidationError("guided Genesis request has no input list")
    matches = [
        item
        for item in inputs
        if isinstance(item, Mapping) and item.get("role") == "task_contract"
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("content"), str):
        raise AutoSpaceValidationError(
            "guided Genesis request must contain exactly one task contract"
        )
    try:
        value = json.loads(matches[0]["content"])
    except json.JSONDecodeError as error:
        raise AutoSpaceValidationError(
            "guided Genesis task contract is not valid JSON"
        ) from error
    if not isinstance(value, dict):
        raise AutoSpaceValidationError("guided Genesis task contract must be an object")
    return value


def scientific_response_schema_name(
    contract: Mapping[str, Any], *, call_role: str = "GENERATOR"
) -> str:
    mode = provider_output_mode(contract)
    if mode == PROVIDER_OUTPUT_MODE_V161:
        from auto_space_v161 import CRITIC_SCHEMA, PROVIDER_DRAFT_SCHEMA as V161_DRAFT_SCHEMA

        if call_role == "GENERATOR":
            return V161_DRAFT_SCHEMA
        if call_role == "CRITIC":
            return CRITIC_SCHEMA
        raise AutoSpaceValidationError(f"unsupported provider call role: {call_role}")
    if mode == PROVIDER_OUTPUT_MODE_V16:
        from auto_space_v16 import CRITIC_SCHEMA, PROVIDER_DRAFT_SCHEMA as V16_DRAFT_SCHEMA

        if call_role == "GENERATOR":
            return V16_DRAFT_SCHEMA
        if call_role == "CRITIC":
            return CRITIC_SCHEMA
        raise AutoSpaceValidationError(f"unsupported provider call role: {call_role}")
    if mode == PROVIDER_OUTPUT_MODE_DRAFT_ASSEMBLY:
        return PROVIDER_DRAFT_SCHEMA
    if call_role not in {"GENERATOR", "CRITIC"}:
        raise AutoSpaceValidationError(f"unsupported provider call role: {call_role}")
    return CANONICAL_RESPONSE_SCHEMA


def _json_pointer(value: Any, fragment: str) -> Any:
    for part in fragment.lstrip("/").split("/") if fragment else ():
        value = value[part.replace("~1", "/").replace("~0", "~")]
    return value


def resolved_schema(name: str) -> dict[str, Any]:
    """Resolve a local AutoSpace schema exactly as provider callers send it."""

    cache = {path.name: load_json(path) for path in SCHEMA_ROOT.glob("*.schema.json")}
    if name not in cache:
        raise AutoSpaceValidationError(f"unknown local response schema: {name}")

    def resolve(value: Any, current: str) -> Any:
        if isinstance(value, list):
            return [resolve(item, current) for item in value]
        if not isinstance(value, Mapping):
            return value
        if set(value) == {"$ref"}:
            reference = value["$ref"]
            target_name, _, fragment = reference.partition("#")
            target_name = target_name or current
            if target_name not in cache:
                raise AutoSpaceValidationError(
                    f"non-local response-schema reference: {reference}"
                )
            return resolve(_json_pointer(cache[target_name], fragment), target_name)
        return {
            key: resolve(item, current)
            for key, item in value.items()
            if key != "$id"
        }

    return resolve(cache[name], name)


def broker_scientific_prompt(
    *,
    genesis_request: Mapping[str, Any],
    target_schema_name: str,
    target_schema: Mapping[str, Any],
    draft: Mapping[str, Any] | None,
) -> str:
    """Build the canonical prompt shared by broker caller and verifier.

    Historical requests keep the historical prompt byte-for-byte so already-sealed
    provider evidence remains verifiable. Requests built with the v2 guidance marker
    receive explicit trusted-design guidance and critic correction hints.
    """

    from auto_space_v16 import CRITIC_SCHEMA as V16_CRITIC_SCHEMA
    from auto_space_v16 import PROVIDER_DRAFT_SCHEMA as V16_DRAFT_SCHEMA
    from auto_space_v161 import CRITIC_SCHEMA as V161_CRITIC_SCHEMA
    from auto_space_v161 import PROVIDER_DRAFT_SCHEMA as V161_DRAFT_SCHEMA

    if target_schema_name == V161_DRAFT_SCHEMA:
        response_instruction = (
            "Perform v1.6.1 Stage A and Stage B exactly as the schema requires: propose exactly "
            "18 primitives, three per axis, all fully specified. Do not author trusted control-plane fields."
        )
    elif target_schema_name == V161_CRITIC_SCHEMA:
        response_instruction = (
            "Perform v1.6.1 Stage C only. Prune the 18-proposal pool to 12..16 accepted primitives, "
            "retaining two or three distinct mechanisms per axis. Rejected proposals are expected and do not "
            "by themselves make the audit fail; status PASS means the accepted subset passes all five audits. "
            "Do not add, rewrite, or silently repair mechanisms."
        )
    elif target_schema_name == V16_DRAFT_SCHEMA:
        response_instruction = (
            "Perform Stage A planning and Stage B primitive authoring exactly as the "
            "schema requires. Do not author trusted control-plane fields."
        )
    elif target_schema_name == V16_CRITIC_SCHEMA:
        response_instruction = (
            "Perform Stage C only. Audit coverage, redundancy, runtime reachability, "
            "effect strength, legality, and composability; classify every supplied "
            "primitive without adding or rewriting scientific mechanisms."
        )
    elif target_schema_name == PROVIDER_DRAFT_SCHEMA:
        response_instruction = (
            "Author only the scientific draft fields in that schema; trusted "
            "control-plane fields are assembled deterministically after this call."
        )
    elif target_schema_name == CANONICAL_RESPONSE_SCHEMA:
        response_instruction = (
            "Author the complete canonical generator response required by that schema."
        )
    else:
        raise AutoSpaceValidationError(
            f"unsupported scientific response schema: {target_schema_name}"
        )

    instructions = str(genesis_request.get("instructions", ""))
    v161_guided = GENESIS_PROVIDER_GUIDANCE_V161_MARKER in instructions
    v16_guided = GENESIS_PROVIDER_GUIDANCE_V16_MARKER in instructions
    guided = GENESIS_PROVIDER_GUIDANCE_V2_MARKER in instructions or v16_guided or v161_guided
    instruction = (
        "Return exactly one proposal. The proposal field scientific_response_json "
        "must be a JSON-encoded STRING whose decoded value is one complete JSON "
        f"object satisfying target_scientific_response_schema. {response_instruction} "
        "Do not put Markdown, commentary, or code fences inside or outside that "
        "JSON string. Preserve the signed Genesis request constraints exactly."
    )
    payload: dict[str, Any] = {
        "instruction": instruction,
        "genesis_request": genesis_request,
        "target_scientific_response_schema": target_schema,
    }

    guidance: dict[str, Any] | None = None
    if guided:
        contract = _embedded_task_contract_for_prompt(genesis_request)
        guidance = provider_design_guidance(contract)
        if not guidance:
            raise AutoSpaceValidationError(
                "guided Genesis request has no required search-space design"
            )
        marker = (
            GENESIS_PROVIDER_GUIDANCE_V161_MARKER
            if v161_guided
            else GENESIS_PROVIDER_GUIDANCE_V16_MARKER
            if v16_guided
            else GENESIS_PROVIDER_GUIDANCE_V2_MARKER
        )
        payload["instruction"] = (
            instruction
            + f" The Genesis request uses {marker}; obey trusted_design_guidance "
            "exactly. Registry coverage and per-candidate composition limits are "
            "different constraints, and conceptual mutation targets are not "
            "necessarily Python symbols."
        )
        payload["trusted_design_guidance"] = guidance

    if draft is not None:
        payload["draft"] = draft
        if target_schema_name == V161_CRITIC_SCHEMA:
            payload["critic_instruction"] = (
                "Return the Stage C audit and exact accepted/rejected partition. Prune to 12..16 accepted "
                "primitives and keep at least two accepted families per axis. Rejections are expected: audit PASS "
                "means the retained subset is valid. Reject unresolved implementation choices, incompletely specified "
                "sampling distributions, and scalar-only architecture variants. Do not reject a TRAIN_AND_INFERENCE "
                "ranking-preserving scoring transform solely for preserving inference ranking when it has a declared "
                "training effect; it remains low-impact and cannot be the axis's only meaningful mechanism. "
                "Do not add, rewrite, or silently repair a primitive."
            )
        elif target_schema_name == V16_CRITIC_SCHEMA:
            payload["critic_instruction"] = (
                "Return the Stage C audit and exact accepted/rejected partition. "
                "Do not add, rewrite, or silently repair a primitive."
            )
        else:
            payload["critic_instruction"] = (
                "Return one complete corrected package, not commentary and not a patch. "
                "Do not cherry-pick. The decoded scientific_response_json must itself be "
                "the complete corrected scientific response."
            )
        if guided and guidance is not None and target_schema_name not in {V16_CRITIC_SCHEMA, V161_CRITIC_SCHEMA}:
            required = {str(item) for item in guidance.get("required_slot_ids", [])}
            present = {
                str(item.get("slot_id"))
                for item in draft.get("primitives", [])
                if isinstance(item, Mapping) and isinstance(item.get("slot_id"), str)
            }
            missing = sorted(required - present)
            payload["critic_required_corrections"] = {
                "missing_required_slots": missing,
                "instruction": (
                    "The corrected complete package must include at least one primitive "
                    "for every trusted required slot. Missing slots listed here are a "
                    "hard correction requirement; do not preserve an omission from the "
                    "generator draft."
                ),
            }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def broker_request_digest(
    transport_request: Mapping[str, Any], genesis_request_digest: str
) -> str:
    return canonical_sha256(
        {
            "expected_proposal_count": 1,
            "proposal_generation_session_id": (
                f"autospace-genesis:{genesis_request_digest}"
            ),
            "request_payload": transport_request,
        }
    )


def protocol_identity_payload(identity: Mapping[str, Any]) -> dict[str, str]:
    required = (
        "base_protocol_id",
        "base_protocol_version",
        "base_protocol_sha256",
        "autospace_amendment_version",
        "autospace_amendment_sha256",
    )
    payload: dict[str, str] = {}
    for key in required:
        value = identity.get(key)
        if not isinstance(value, str) or not value.strip():
            raise AutoSpaceValidationError(f"protocol identity field {key!r} is missing")
        payload[key] = value.strip()
    if (
        payload["base_protocol_id"] != BASE_PROTOCOL_ID
        or payload["base_protocol_version"] != BASE_PROTOCOL_VERSION
        or payload["base_protocol_sha256"] != BASE_PROTOCOL_SHA256
    ):
        raise AutoSpaceValidationError("base protocol identity mismatch")
    return payload


def recompute_composite_protocol_identity(identity: Mapping[str, Any]) -> str:
    return canonical_sha256(protocol_identity_payload(identity))


def validate_protocol_identity(
    identity: Mapping[str, Any],
    *,
    require_signed: bool = False,
    signed_amendment_path: Path | None = None,
) -> str | None:
    if not isinstance(identity, Mapping):
        raise AutoSpaceValidationError("protocol_identity must be an object")
    base = {
        "base_protocol_id": identity.get("base_protocol_id"),
        "base_protocol_version": identity.get("base_protocol_version"),
        "base_protocol_sha256": identity.get("base_protocol_sha256"),
    }
    if base != {
        "base_protocol_id": BASE_PROTOCOL_ID,
        "base_protocol_version": BASE_PROTOCOL_VERSION,
        "base_protocol_sha256": BASE_PROTOCOL_SHA256,
    }:
        raise AutoSpaceValidationError("base protocol identity mismatch")
    amendment_version = identity.get("autospace_amendment_version")
    amendment_sha = identity.get("autospace_amendment_sha256")
    composite = identity.get("composite_protocol_identity")
    if not require_signed:
        if amendment_version in DRAFT_AUTOSPACE_AMENDMENT_VERSIONS:
            if amendment_sha is not None or composite is not None:
                raise AutoSpaceValidationError("draft AutoSpace protocol identity must not carry signed digests")
            return None
        # A signed identity may also be validated in non-live/offline paths.
    expected_amendment_filename = signed_amendment_filename_for_version(amendment_version)
    expected_composite = recompute_composite_protocol_identity(identity)
    if composite != expected_composite:
        raise AutoSpaceValidationError("composite protocol identity mismatch")
    if signed_amendment_path is not None:
        signed = Path(signed_amendment_path)
        if signed.name != expected_amendment_filename:
            raise AutoSpaceValidationError("signed AutoSpace amendment filename/version mismatch")
        if not signed.is_file() or signed.is_symlink():
            raise AutoSpaceValidationError("signed AutoSpace amendment file is missing or symlinked")
        if file_sha256(signed) != amendment_sha:
            raise AutoSpaceValidationError("signed AutoSpace amendment digest mismatch")
        if "Status: SIGNED" not in signed.read_text(encoding="utf-8"):
            raise AutoSpaceValidationError("AutoSpace amendment file is not signed")
    return expected_composite



def load_json_text(text: str, *, label: str = "JSON text") -> Any:
    """Parse strict JSON text with duplicate-key and non-finite rejection."""

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AutoSpaceValidationError(f"duplicate JSON member {key!r} in {label}")
            result[key] = value
        return result

    try:
        return json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                AutoSpaceValidationError(f"non-finite JSON number {token!r} in {label}")
            ),
        )
    except json.JSONDecodeError as error:
        raise AutoSpaceValidationError(f"invalid JSON in {label}: {error}") from error


def extract_responses_api_scientific_response(raw_response: Mapping[str, Any]) -> dict[str, Any]:
    """Extract exactly one structured-output JSON object from a Responses API envelope."""

    texts: list[str] = []
    output = raw_response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                    texts.append(part["text"])
    if len(texts) != 1:
        raise AutoSpaceValidationError(
            f"Responses API envelope must contain exactly one scientific JSON text, got {len(texts)}"
        )
    value = load_json_text(texts[0], label="provider scientific response")
    if not isinstance(value, Mapping):
        raise AutoSpaceValidationError("provider scientific response must be a JSON object")
    return dict(value)


def _extract_broker_normalized_scientific_response(raw_response: Mapping[str, Any]) -> dict[str, Any]:
    response = raw_response.get("response")
    if raw_response.get("transport") != "LAB_API_CANARY_BROKER_V1" or not isinstance(response, Mapping):
        raise AutoSpaceValidationError("broker normalized response identity mismatch")
    proposals = response.get("proposals")
    if not isinstance(proposals, list) or len(proposals) != 1 or not isinstance(proposals[0], Mapping):
        raise AutoSpaceValidationError("broker normalized response must contain exactly one proposal")
    encoded = proposals[0].get("scientific_response_json")
    if not isinstance(encoded, str) or not encoded:
        raise AutoSpaceValidationError("broker normalized proposal has no scientific_response_json")
    if "local_salvage" not in raw_response:
        value = load_json_text(encoded, label="broker scientific response")
        if not isinstance(value, Mapping):
            raise AutoSpaceValidationError(
                "broker scientific response must decode to an object"
            )
        return dict(value)
    salvage = raw_response.get("local_salvage")
    if not isinstance(salvage, Mapping):
        raise AutoSpaceValidationError("broker local salvage record is malformed")
    if salvage.get("source_raw_sha256") != __import__("hashlib").sha256(encoded.encode("utf-8")).hexdigest():
        raise AutoSpaceValidationError("broker local salvage source identity mismatch")
    repairs = salvage.get("repairs")
    if not isinstance(repairs, list) or len(repairs) != 2 or not all(isinstance(x, Mapping) for x in repairs):
        raise AutoSpaceValidationError("broker local salvage repair list is malformed")
    first, second = repairs
    pattern, replacement, positions = first.get("raw_pattern"), first.get("replacement"), first.get("raw_positions")
    if not isinstance(pattern, str) or not isinstance(replacement, str) or not isinstance(positions, list):
        raise AutoSpaceValidationError("broker repeated-boundary salvage metadata is malformed")
    observed = [i for i in range(len(encoded)) if encoded.startswith(pattern, i)]
    if observed != positions:
        raise AutoSpaceValidationError("broker repeated-boundary salvage positions differ from evidence")
    intermediate = encoded.replace(pattern, replacement)
    pos, removed = second.get("post_axis_repair_position"), second.get("removed")
    if not isinstance(pos, int) or not isinstance(removed, str) or intermediate[pos:pos+len(removed)] != removed:
        raise AutoSpaceValidationError("broker pre-grammar salvage bytes differ from evidence")
    repaired = intermediate[:pos] + intermediate[pos+len(removed):]
    value = load_json_text(repaired, label="salvaged broker scientific response")
    if not isinstance(value, Mapping):
        raise AutoSpaceValidationError("broker scientific response must decode to an object")
    value = dict(value)
    semantic_normalizations = salvage.get("semantic_normalizations", [])
    if not isinstance(semantic_normalizations, list):
        raise AutoSpaceValidationError("broker semantic normalization record is malformed")
    for normalization in semantic_normalizations:
        if not isinstance(normalization, Mapping) or normalization.get("kind") != "ALIGN_SINGLE_GENERATED_FAMILY_ID":
            raise AutoSpaceValidationError("unsupported broker semantic normalization")
        family = value.get("family")
        policy = value.get("capability_policy")
        registry = value.get("primitive_registry")
        if not isinstance(family, dict) or not isinstance(policy, dict) or not isinstance(registry, dict):
            raise AutoSpaceValidationError("broker family normalization target is malformed")
        axes = registry.get("axes")
        if not isinstance(axes, list):
            raise AutoSpaceValidationError("broker primitive axes are malformed")
        primitive_family_ids = sorted({
            primitive.get("family_id")
            for axis in axes if isinstance(axis, Mapping)
            for primitive in axis.get("primitives", [])
            if isinstance(primitive, Mapping)
        })
        observed = {
            "family_id": family.get("family_id"),
            "policy_families": policy.get("legal_capability_families"),
            "primitive_family_ids": primitive_family_ids,
        }
        if observed != normalization.get("before"):
            raise AutoSpaceValidationError("broker semantic normalization preimage differs from evidence")
        target = normalization.get("target_family_id")
        if not isinstance(target, str) or not target:
            raise AutoSpaceValidationError("broker semantic normalization target is invalid")
        family["family_id"] = target
        policy["legal_capability_families"] = [target]
        for axis in axes:
            if not isinstance(axis, dict):
                raise AutoSpaceValidationError("broker primitive axis is malformed")
            primitives = axis.get("primitives")
            if not isinstance(primitives, list):
                raise AutoSpaceValidationError("broker primitive list is malformed")
            for primitive in primitives:
                if not isinstance(primitive, dict):
                    raise AutoSpaceValidationError("broker primitive is malformed")
                primitive["family_id"] = target
    if salvage.get("salvaged_scientific_response_canonical_sha256") != canonical_sha256(value):
        raise AutoSpaceValidationError("broker salvaged scientific response identity mismatch")
    return value


def verify_provider_receipt_bundle(
    *,
    receipt: Mapping[str, Any],
    transport_request: Mapping[str, Any],
    raw_response: Mapping[str, Any],
    scientific_response: Mapping[str, Any],
    pricing: Mapping[str, Any] | None = None,
) -> None:
    """Recompute the local transport receipt from immutable request/response evidence."""

    if raw_response.get("schema") == "recclaw.auto-space.broker-normalized-provider-response.v1":
        genesis_request_digest = str(receipt.get("genesis_request_digest") or "")
        if receipt.get("request_digest") != broker_request_digest(
            transport_request, genesis_request_digest
        ):
            raise AutoSpaceValidationError("provider receipt transport request identity mismatch")
        verify_genesis_transport_request(
            transport_request,
            genesis_request_digest=genesis_request_digest,
            input_scientific_response_canonical_sha256=receipt.get(
                "input_scientific_response_canonical_sha256"
            ),
            call_role=str(receipt.get("call_role") or ""),
        )
        extracted = _extract_broker_normalized_scientific_response(raw_response)
        if canonical_bytes(extracted) != canonical_bytes(scientific_response):
            raise AutoSpaceValidationError("provider raw response and scientific response differ")
        if receipt.get("scientific_response_canonical_sha256") != canonical_sha256(extracted):
            raise AutoSpaceValidationError("provider receipt scientific response canonical identity mismatch")
        model = raw_response.get("returned_model")
        if receipt.get("effective_model") != model or not isinstance(model, str) or not model.endswith("gpt-5.6-terra"):
            raise AutoSpaceValidationError("broker model identity mismatch")
        usage = raw_response.get("usage")
        if not isinstance(usage, Mapping):
            raise AutoSpaceValidationError("broker normalized response has no usage object")
        try:
            input_tokens, output_tokens = int(usage["input_tokens"]), int(usage["output_tokens"])
        except (KeyError, TypeError, ValueError) as error:
            raise AutoSpaceValidationError("broker normalized usage is incomplete") from error
        if receipt.get("input_tokens") != input_tokens or receipt.get("output_tokens") != output_tokens:
            raise AutoSpaceValidationError("provider receipt token usage differs from broker evidence")
        request_id = raw_response.get("broker_request_digest")
        if receipt.get("provider_request_id") != request_id:
            raise AutoSpaceValidationError("provider receipt request ID differs from broker evidence")
        if pricing is not None:
            rates=pricing["models"]["gpt-5.6-terra"]; unit=float(pricing["normalized_unit"]["usd"])
            cost=input_tokens*float(rates["input"])/1_000_000 + output_tokens*float(rates["output"])/1_000_000
            normalized=cost/unit
            if not math.isclose(float(receipt.get("cost_usd",-1)),cost,rel_tol=1e-12,abs_tol=1e-12) or not math.isclose(float(receipt.get("normalized_u",-1)),normalized,rel_tol=1e-12,abs_tol=1e-12):
                raise AutoSpaceValidationError("broker receipt pricing differs from frozen pricing")
        return


    if receipt.get("request_digest") != canonical_sha256(transport_request):
        raise AutoSpaceValidationError("provider receipt transport request identity mismatch")
    verify_genesis_transport_request(
        transport_request,
        genesis_request_digest=str(receipt.get("genesis_request_digest") or ""),
        input_scientific_response_canonical_sha256=receipt.get(
            "input_scientific_response_canonical_sha256"
        ),
        call_role=str(receipt.get("call_role") or ""),
    )
    extracted = extract_responses_api_scientific_response(raw_response)
    if canonical_bytes(extracted) != canonical_bytes(scientific_response):
        raise AutoSpaceValidationError("provider raw response and scientific response differ")
    if receipt.get("scientific_response_canonical_sha256") != canonical_sha256(extracted):
        raise AutoSpaceValidationError("provider receipt scientific response canonical identity mismatch")
    effective_model = raw_response.get("model")
    if receipt.get("effective_model") != effective_model:
        raise AutoSpaceValidationError("provider receipt effective model differs from raw response")
    if not isinstance(effective_model, str) or not effective_model.endswith("gpt-5.6-terra"):
        raise AutoSpaceValidationError("provider raw response did not execute the authorized Terra model")
    usage = raw_response.get("usage")
    if not isinstance(usage, Mapping):
        raise AutoSpaceValidationError("provider raw response has no usage object")
    try:
        input_tokens = int(usage["input_tokens"])
        output_tokens = int(usage["output_tokens"])
    except (KeyError, TypeError, ValueError) as error:
        raise AutoSpaceValidationError("provider raw response usage is incomplete") from error
    if input_tokens < 0 or output_tokens < 0:
        raise AutoSpaceValidationError("provider raw response usage is negative")
    if receipt.get("input_tokens") != input_tokens or receipt.get("output_tokens") != output_tokens:
        raise AutoSpaceValidationError("provider receipt token usage differs from raw response")
    raw_request_id = raw_response.get("id")
    if not isinstance(raw_request_id, str) or not raw_request_id:
        raise AutoSpaceValidationError("provider raw response has no stable request ID")
    if receipt.get("provider_request_id") != raw_request_id:
        raise AutoSpaceValidationError("provider receipt request ID differs from raw response")
    if pricing is not None:
        try:
            rates = pricing["models"]["gpt-5.6-terra"]
            unit = float(pricing["normalized_unit"]["usd"])
            cost = input_tokens * float(rates["input"]) / 1_000_000
            cost += output_tokens * float(rates["output"]) / 1_000_000
        except (KeyError, TypeError, ValueError) as error:
            raise AutoSpaceValidationError("pricing manifest is malformed") from error
        normalized = cost / unit
        if not math.isclose(float(receipt.get("cost_usd", -1)), cost, rel_tol=1e-12, abs_tol=1e-12):
            raise AutoSpaceValidationError("provider receipt cost differs from frozen pricing")
        if not math.isclose(float(receipt.get("normalized_u", -1)), normalized, rel_tol=1e-12, abs_tol=1e-12):
            raise AutoSpaceValidationError("provider receipt normalized U differs from frozen pricing")

def validate_genesis_call_authorization_document(authorization: Mapping[str, Any]) -> str:
    """Validate a signed call-authorization document and return its canonical digest."""

    validate_schema(
        authorization,
        "auto_space_genesis_call_authorization.schema.json",
        label="Genesis call authorization",
    )
    payload = dict(authorization)
    observed = payload.pop("authorization_digest", None)
    expected = canonical_sha256(payload)
    if observed != expected:
        raise AutoSpaceValidationError("Genesis authorization digest is stale")
    return expected


def verify_genesis_transport_request(
    transport_request: Mapping[str, Any],
    *,
    genesis_request_digest: str,
    input_scientific_response_canonical_sha256: str | None,
    call_role: str,
) -> None:
    expected_policy = {
        "model": "gpt-5.6-terra",
        "reasoning": {"effort": "medium"},
        "store": False,
    }
    for key, expected in expected_policy.items():
        if transport_request.get(key) != expected:
            raise AutoSpaceValidationError(f"provider transport policy drift: {key}")
    if "tools" in transport_request and transport_request.get("tools") != []:
        raise AutoSpaceValidationError("provider transport policy drift: tools")
    text = transport_request.get("text")
    if not isinstance(text, Mapping):
        raise AutoSpaceValidationError("provider transport request lacks structured-output policy")
    fmt = text.get("format")
    if (
        not isinstance(fmt, Mapping)
        or fmt.get("type") != "json_schema"
        or fmt.get("strict") is not True
        or not isinstance(fmt.get("schema"), Mapping)
    ):
        raise AutoSpaceValidationError("provider transport structured-output policy drift")

    input_text = transport_request.get("input")
    if not isinstance(input_text, str):
        raise AutoSpaceValidationError("provider transport input must be serialized JSON text")
    embedded = load_json_text(input_text, label="provider transport input")
    format_name = fmt.get("name")

    if format_name == BROKER_RESPONSE_FORMAT_NAME:
        if int(transport_request.get("max_output_tokens", -1)) != 20000:
            raise AutoSpaceValidationError(
                "provider transport policy drift: max_output_tokens"
            )
        if not isinstance(embedded, Mapping):
            raise AutoSpaceValidationError("broker Genesis prompt is malformed")
        genesis = embedded.get("genesis_request")
        target_schema = embedded.get("target_scientific_response_schema")
        if not isinstance(genesis, Mapping) or not isinstance(target_schema, Mapping):
            raise AutoSpaceValidationError("broker Genesis prompt is malformed")
        contract = validate_embedded_genesis_request(genesis)
        if genesis.get("request_digest") != genesis_request_digest:
            raise AutoSpaceValidationError("broker transport binds another Genesis request")
        target_schema_name = scientific_response_schema_name(
            contract, call_role=call_role
        )
        expected_schema = resolved_schema(target_schema_name)
        if canonical_bytes(target_schema) != canonical_bytes(expected_schema):
            raise AutoSpaceValidationError("broker target scientific schema drift")
        draft = embedded.get("draft")
        if input_scientific_response_canonical_sha256 is None:
            if draft is not None:
                raise AutoSpaceValidationError("broker generator prompt contains a draft")
        else:
            if not isinstance(draft, Mapping):
                raise AutoSpaceValidationError("broker critic prompt lacks generator draft")
            if canonical_sha256(draft) != input_scientific_response_canonical_sha256:
                raise AutoSpaceValidationError(
                    "broker critic prompt uses a different generator draft"
                )
        expected_prompt = broker_scientific_prompt(
            genesis_request=genesis,
            target_schema_name=target_schema_name,
            target_schema=expected_schema,
            draft=draft,
        )
        if input_text != expected_prompt:
            raise AutoSpaceValidationError("broker scientific prompt drift")
        return

    if format_name != DIRECT_RESPONSE_FORMAT_NAME:
        raise AutoSpaceValidationError("provider transport response format drift")
    if int(transport_request.get("max_output_tokens", -1)) != 16000:
        raise AutoSpaceValidationError("provider transport policy drift: max_output_tokens")

    if input_scientific_response_canonical_sha256 is None:
        if not isinstance(embedded, Mapping):
            raise AutoSpaceValidationError("generator transport input must be the Genesis request")
        validate_embedded_genesis_request(embedded)
        if embedded.get("request_digest") != genesis_request_digest:
            raise AutoSpaceValidationError("generator transport input binds another Genesis request")
        return

    if not isinstance(embedded, Mapping) or set(embedded) != {
        "genesis_request",
        "draft",
        "instruction",
    }:
        raise AutoSpaceValidationError("critic transport input wrapper is malformed")
    genesis = embedded.get("genesis_request")
    draft = embedded.get("draft")
    if not isinstance(genesis, Mapping) or not isinstance(draft, Mapping):
        raise AutoSpaceValidationError("critic transport input lacks Genesis request or draft")
    contract = validate_embedded_genesis_request(genesis)
    if genesis.get("request_digest") != genesis_request_digest:
        raise AutoSpaceValidationError("critic transport input binds another Genesis request")
    if canonical_sha256(draft) != input_scientific_response_canonical_sha256:
        raise AutoSpaceValidationError("critic transport input uses a different generator draft")
    expected_instruction = (
        "Return the Stage C audit and exact accepted/rejected primitive partition; "
        "do not add, rewrite, or silently repair mechanisms."
        if provider_output_mode(contract) == PROVIDER_OUTPUT_MODE_V16
        else "Return one complete corrected package; do not cherry-pick or emit commentary."
    )
    if embedded.get("instruction") != expected_instruction:
        raise AutoSpaceValidationError("critic transport instruction drift")

def load_json(path: Path) -> Any:
    resolved = path.resolve(strict=True)
    try:
        text = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise AutoSpaceValidationError(f"invalid UTF-8 in {path.name}: {error}") from error
    return load_json_text(text, label=path.name)


def _schema_registry() -> Registry:
    def deny_remote(uri: str) -> Resource:
        raise NoSuchResource(ref=uri)

    registry = Registry(retrieve=deny_remote)
    for path in sorted(SCHEMA_ROOT.glob("*.schema.json")):
        schema = load_json(path)
        resource = Resource.from_contents(schema)
        registry = registry.with_resource(schema["$id"], resource)
    return registry


def validate_schema(value: Any, schema_name: str, *, label: str | None = None) -> None:
    schema_path = SCHEMA_ROOT / schema_name
    schema = load_json(schema_path)
    validator = Draft202012Validator(
        schema,
        registry=_schema_registry(),
        format_checker=FormatChecker(),
    )
    errors = sorted(
        validator.iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(item) for item in error.absolute_path) or "<root>"
        raise AutoSpaceValidationError(
            f"{label or schema_name} schema violation at {location}: {error.message}"
        )


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AutoSpaceValidationError(f"{label} must be an object")
    return dict(value)


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AutoSpaceValidationError(f"{label} must be a non-empty string")
    return value.strip()


def _string_tuple(value: Any, label: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise AutoSpaceValidationError(f"{label} must be an array of non-empty strings")
    result = tuple(item.strip() for item in value)
    if nonempty and not result:
        raise AutoSpaceValidationError(f"{label} must not be empty")
    if len(set(result)) != len(result):
        raise AutoSpaceValidationError(f"{label} contains duplicates")
    return result


def _assert_no_executable_claims(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in FORBIDDEN_CONCEPTUAL_FIELDS:
                raise AutoSpaceValidationError(
                    f"{label} contains executable-only field {key!r}"
                )
            if key == "origin" and item == "QUALIFIED_REGISTRY":
                raise AutoSpaceValidationError(
                    f"{label} falsely claims QUALIFIED_REGISTRY origin"
                )
            _assert_no_executable_claims(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_executable_claims(item, f"{label}[{index}]")


def _symbol_index(source: str, path: str) -> set[tuple[str, str]]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as error:
        raise AutoSpaceValidationError(f"baseline source {path} is not valid Python") from error
    symbols: set[tuple[str, str]] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.add((node.name, "function"))
        elif isinstance(node, ast.ClassDef):
            symbols.add((node.name, "class"))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.add((f"{node.name}.{child.name}", "method"))
    return symbols


def validate_baseline_anchors(
    primitive_records: Sequence[Mapping[str, Any]], request: Mapping[str, Any]
) -> None:
    sources = {
        item["path"]: _symbol_index(item["content"], item["path"])
        for item in request.get("inputs", [])
        if item.get("role") == "baseline_source"
    }
    for primitive in primitive_records:
        for anchor in primitive.get("baseline_anchors", []):
            key = (anchor.get("symbol"), anchor.get("kind"))
            if anchor.get("path") not in sources or key not in sources[anchor["path"]]:
                raise AutoSpaceValidationError(
                    f"primitive {primitive.get('primitive_id')!r} has an unverified structural baseline anchor"
                )


def canonical_anchor_identity(anchor: Mapping[str, Any]) -> str:
    """Return the family-contract identity of a structured primitive anchor."""

    path = _nonempty_string(anchor.get("path"), "anchor.path")
    symbol = anchor.get("symbol")
    if symbol is None:
        return path
    return f"{path}:{_nonempty_string(symbol, 'anchor.symbol')}"


def assemble_provider_draft(
    draft: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    """Assemble a provider scientific draft into the trusted canonical response."""

    contract = validate_trusted_contract(contract)
    if provider_output_mode(contract) != PROVIDER_OUTPUT_MODE_DRAFT_ASSEMBLY:
        raise AutoSpaceValidationError(
            "provider-draft assembly requires explicit provider output mode"
        )
    design = contract.get("required_search_space_design")
    if not isinstance(design, Mapping):
        raise AutoSpaceValidationError(
            "provider-draft assembly requires a trusted search-space design"
        )
    value = _object(draft, "provider scientific draft")
    family_draft = _object(value.get("family"), "provider family draft")
    raw_primitives = value.get("primitives")
    if not isinstance(raw_primitives, list) or not raw_primitives:
        raise AutoSpaceValidationError("provider draft must contain primitives")

    family_id = _nonempty_string(family_draft.get("family_id"), "family.family_id")
    content_version = _nonempty_string(design.get("content_version"), "content_version")
    slots = [_object(item, "trusted slot") for item in design["slots"]]
    slot_by_id = {item["slot_id"]: item for item in slots}
    grouped: dict[str, list[dict[str, Any]]] = {item["slot_id"]: [] for item in slots}
    primitive_ids: list[str] = []
    anchor_identities: set[str] = set()

    for raw in raw_primitives:
        primitive = _object(raw, "provider primitive draft")
        primitive_id = _nonempty_string(
            primitive.get("primitive_id"), "primitive.primitive_id"
        )
        if primitive_id in primitive_ids:
            raise AutoSpaceValidationError(
                f"provider draft contains duplicate primitive_id {primitive_id!r}"
            )
        primitive_ids.append(primitive_id)
        slot_id = _nonempty_string(primitive.get("slot_id"), "primitive.slot_id")
        if slot_id not in slot_by_id:
            raise AutoSpaceValidationError(
                f"provider draft uses slot outside the trusted design: {slot_id!r}"
            )
        slot = slot_by_id[slot_id]
        anchors = list(primitive["baseline_anchors"])
        anchor_keys = {
            (anchor.get("path"), anchor.get("symbol"), anchor.get("kind"))
            for anchor in anchors
            if isinstance(anchor, Mapping)
        }
        slot_targets = list(slot.get("legal_mutation_targets", []))
        conceptual_requirements = [
            (target, CONCEPTUAL_MUTATION_TARGET_ANCHORS[target])
            for target in slot_targets
            if target in CONCEPTUAL_MUTATION_TARGET_ANCHORS
        ]
        # Require the mapped real anchor only when the trusted slot is confined to
        # conceptual mutation regions. A mixed-target slot (for example the
        # training-objective slot, which may legally stay entirely in models.py)
        # must not be forced to anchor every optional legal target.
        if conceptual_requirements and len(conceptual_requirements) == len(slot_targets):
            for target, required_anchor in conceptual_requirements:
                required_key = (
                    required_anchor["path"],
                    required_anchor["symbol"],
                    required_anchor["kind"],
                )
                if required_key not in anchor_keys:
                    raise AutoSpaceValidationError(
                        f"provider primitive {primitive_id!r} for slot {slot_id!r} "
                        f"must anchor conceptual mutation target {target!r} to the real "
                        f"baseline symbol {required_anchor['path']}:{required_anchor['symbol']}"
                    )
        anchor_identities.update(canonical_anchor_identity(anchor) for anchor in anchors)
        assembled = {
            "primitive_id": primitive_id,
            "version": content_version,
            "family_id": family_id,
            "slot_id": slot_id,
            "implementation_status": "CONCEPTUAL_UNIMPLEMENTED",
            "semantic_role": primitive["semantic_role"],
            "baseline_anchors": anchors,
            "input_ports": list(primitive["input_ports"]),
            "output_ports": list(primitive["output_ports"]),
            "parameter_schema": dict(primitive["parameter_schema"]),
            "conceptual_capabilities": [slot["conceptual_capability"]],
            "allowed_data_roles": list(primitive["allowed_data_roles"]),
            "composition_roles": [slot["composition_role"]],
            "legal_mutation_targets": list(slot["legal_mutation_targets"]),
            "prohibited_mutation_targets": list(
                contract["trusted_upper_bound"]["frozen_mutation_targets"]
            ),
            "implementation_requirements": list(
                primitive["implementation_requirements"]
            ),
            "required_invariants": list(primitive["required_invariants"]),
            "rationale": primitive["rationale"],
        }
        if "family_boundary_justification" in primitive:
            assembled["family_boundary_justification"] = primitive[
                "family_boundary_justification"
            ]
        grouped[slot_id].append(assembled)

    missing_slots = [slot_id for slot_id, records in grouped.items() if not records]
    if missing_slots:
        raise AutoSpaceValidationError(
            f"provider draft omits trusted slots: {missing_slots}"
        )

    boundary = contract["trusted_upper_bound"]
    family = {
        "record_type": "search_space_family",
        "family_id": family_id,
        "family_version": content_version,
        "description": family_draft["description"],
        "scientific_boundary": family_draft["scientific_boundary"],
        "baseline_contract": {
            "task_id": contract["task_id"],
            "baseline_model": contract["baseline"],
            "required_anchor_paths": sorted(anchor_identities),
        },
        "mutable_dimensions": list(design["mutable_dimensions"]),
        "frozen_protocol_dimensions": list(boundary["frozen_protocol_dimensions"]),
        "allowed_data_roles": list(boundary["allowed_data_roles"]),
        "forbidden_data_roles": list(boundary["forbidden_data_roles"]),
        "output_contract": {
            "type_ref": contract["candidate_abi"]["type_ref"],
            "candidate_universe_effect": "PRESERVE",
            "semantics": "Produce candidates through the trusted frozen candidate ABI.",
        },
        "type_compatibility": list(family_draft["type_compatibility"]),
        "invariants": list(family_draft["invariants"]),
    }
    search_space_id = _nonempty_string(
        value.get("search_space_id"), "search_space_id"
    )
    axes = [
        {
            "slot_id": slot["slot_id"],
            "allow_multiple": False,
            "primitives": grouped[slot["slot_id"]],
        }
        for slot in slots
    ]
    registry = {
        "record_type": "search_space_primitive_registry",
        "registry_id": f"{search_space_id}.primitive-registry",
        "registry_version": content_version,
        "axis_count": len(axes),
        "primitive_count": len(primitive_ids),
        "axes": axes,
    }

    composition_roles = list(
        dict.fromkeys(slot["composition_role"] for slot in slots)
    )
    operators = [
        {
            "operator_id": item["operator_id"],
            "kind": item["kind"],
            "min_arity": item["min_arity"],
            "max_arity": item["max_arity"],
            "requires_target": False,
            "requires_replacement": False,
            "required_capabilities": [],
            "allowed_composition_roles": composition_roles,
            "port_compatibility": "NOT_REQUIRED",
        }
        for item in design["operators"]
    ]
    custom_design = design["custom_synthesis"]
    custom_operator = next(
        (
            item["operator_id"]
            for item in design["operators"]
            if item["kind"] == "CUSTOM_SYNTHESIS"
        ),
        None,
    )
    custom_enabled = custom_design["max_components"] > 0
    grammar = {
        "record_type": "search_space_grammar",
        "grammar_id": f"{search_space_id}.grammar",
        "grammar_version": content_version,
        "operators": operators,
        "primitive_ids": primitive_ids,
        "composition_policy": {
            "max_primitives_per_candidate": design["max_primitives_per_candidate"],
            "cross_family_composition": "DENY",
            "incompatible_primitive_sets": list(
                design.get("incompatible_primitive_sets", [])
            ),
            "incompatible_composition_role_pairs": list(
                design.get("incompatible_composition_role_pairs", [])
            ),
            "max_mutation_targets": design["max_mutation_targets"],
            "custom_synthesis": {
                "enabled": custom_enabled,
                "operator_id": custom_operator if custom_enabled else None,
                "max_components": custom_design["max_components"],
                "allowed_slots": list(custom_design["allowed_slots"]),
            },
        },
    }

    policy_targets = list(
        dict.fromkeys(
            target
            for slot in slots
            for target in slot["legal_mutation_targets"]
        )
    )
    policy = {
        "record_type": "search_space_capability_policy",
        "policy_id": f"{search_space_id}.capability-policy",
        "policy_version": content_version,
        "legal_capability_families": [family_id],
        "allowed_mutation_targets": policy_targets,
        "frozen_mutation_targets": list(boundary["frozen_mutation_targets"]),
        "allowed_data_roles": list(boundary["allowed_data_roles"]),
        "forbidden_data_roles": list(boundary["forbidden_data_roles"]),
        "allowed_dependencies": list(boundary["allowed_dependencies"]),
        "network_access": False,
        "conceptual_admission": {
            "requires_baseline_anchor": True,
            "requires_mutation_boundary_conformance": True,
            "requires_grammar_conformance": True,
            "requires_trusted_contract_conformance": True,
            "executable_claims_forbidden": True,
        },
        "executable_admission": {
            "requires_implementation": True,
            "requires_qualification_receipt": True,
            "requires_qualified_registry_admission": True,
            "initial_registry_count": 0,
        },
    }
    return {
        "schema": "recclaw.auto-space.generator-response.v1",
        "search_space_id": search_space_id,
        "analysis": dict(value["analysis"]),
        "family": family,
        "primitive_registry": registry,
        "grammar": grammar,
        "capability_policy": policy,
    }



E2_V2_REQUIRED_SLOTS = {
    "REPRESENTATION_TRANSFORM": {
        "composition_role": "representation",
        "conceptual_capability": "representation_transform",
        "legal_mutation_targets": ("models.py",),
    },
    "SCORING_TRANSFORM": {
        "composition_role": "candidate_scoring",
        "conceptual_capability": "candidate_scoring_transform",
        "legal_mutation_targets": ("models.py",),
    },
    "TRAINING_OBJECTIVE_TRANSFORM": {
        "composition_role": "training_objective",
        "conceptual_capability": "train_only_objective",
        "legal_mutation_targets": ("models.py", "main_code.py:train_update_kernel"),
    },
    "TRAIN_UPDATE_TRANSFORM": {
        "composition_role": "train_update",
        "conceptual_capability": "train_update_transform",
        "legal_mutation_targets": ("main_code.py:train_update_kernel",),
    },
}
E2_V2_REQUIRED_MUTABLE_DIMENSIONS = (
    "candidate_scoring",
    "representation",
    "train_only_objective",
    "train_update_rule",
)
E2_V2_REQUIRED_OPERATORS = {
    "SELECT": ("select", 1, 1),
    "COMPOSE": ("compose", 2, 3),
    "CUSTOM_SYNTHESIS": ("custom_synthesis", 1, 1),
}


def _validate_required_search_space_design(
    contract: Mapping[str, Any], amendment_version: Any
) -> Mapping[str, Any] | None:
    if provider_output_mode(contract) == PROVIDER_OUTPUT_MODE_V161:
        from auto_space_v161 import validate_contract_design

        return validate_contract_design(contract)
    if provider_output_mode(contract) == PROVIDER_OUTPUT_MODE_V16:
        from auto_space_v16 import validate_contract_design

        return validate_contract_design(contract)
    design = contract.get("required_search_space_design")
    is_v2_draft = (
        isinstance(amendment_version, str)
        and amendment_version.startswith("v2.0-")
    ) or amendment_version == V13_DRAFT_AUTOSPACE_AMENDMENT_VERSION
    is_v2_signed = amendment_version in SIGNED_AUTOSPACE_AMENDMENT_VERSIONS
    if is_v2_draft or is_v2_signed:
        design = _object(design, "required_search_space_design")
        if design.get("content_version") != "2":
            raise AutoSpaceValidationError("trusted v2 search-space content_version drift")
        slots = design.get("slots")
        if not isinstance(slots, list) or len(slots) != len(E2_V2_REQUIRED_SLOTS):
            raise AutoSpaceValidationError("trusted v2 search-space slot set drift")
        observed = {}
        for raw in slots:
            item = _object(raw, "trusted v2 slot")
            slot_id = _nonempty_string(item.get("slot_id"), "trusted v2 slot_id")
            if slot_id in observed:
                raise AutoSpaceValidationError("trusted v2 search-space contains duplicate slots")
            observed[slot_id] = item
        if set(observed) != set(E2_V2_REQUIRED_SLOTS):
            raise AutoSpaceValidationError("trusted v2 search-space slot set drift")
        for slot_id, expected in E2_V2_REQUIRED_SLOTS.items():
            item = observed[slot_id]
            if (
                item.get("composition_role") != expected["composition_role"]
                or item.get("conceptual_capability") != expected["conceptual_capability"]
                or tuple(item.get("legal_mutation_targets", ()))
                != expected["legal_mutation_targets"]
            ):
                raise AutoSpaceValidationError(
                    f"trusted v2 slot {slot_id!r} semantics drift"
                )
        if tuple(design.get("mutable_dimensions", ())) != E2_V2_REQUIRED_MUTABLE_DIMENSIONS:
            raise AutoSpaceValidationError("trusted v2 mutable dimensions drift")
        if design.get("max_primitives_per_candidate") != 3:
            raise AutoSpaceValidationError("trusted v2 max_primitives_per_candidate drift")
        if design.get("max_mutation_targets") != 2:
            raise AutoSpaceValidationError("trusted v2 max_mutation_targets drift")
        operators = design.get("operators")
        if not isinstance(operators, list) or len(operators) != 3:
            raise AutoSpaceValidationError("trusted v2 operator set drift")
        by_kind = {}
        for raw in operators:
            item = _object(raw, "trusted v2 operator")
            kind = _nonempty_string(item.get("kind"), "trusted v2 operator kind")
            if kind in by_kind:
                raise AutoSpaceValidationError("trusted v2 operator set contains duplicates")
            by_kind[kind] = item
        if set(by_kind) != set(E2_V2_REQUIRED_OPERATORS):
            raise AutoSpaceValidationError("trusted v2 operator set drift")
        for kind, expected in E2_V2_REQUIRED_OPERATORS.items():
            item = by_kind[kind]
            if (item.get("operator_id"), item.get("min_arity"), item.get("max_arity")) != expected:
                raise AutoSpaceValidationError(f"trusted v2 {kind} operator drift")
        custom = _object(design.get("custom_synthesis"), "trusted v2 custom_synthesis")
        if (
            custom.get("max_components") != 1
            or tuple(custom.get("allowed_slots", ())) != tuple(E2_V2_REQUIRED_SLOTS)
        ):
            raise AutoSpaceValidationError("trusted v2 custom synthesis drift")
        return design
    if design is not None:
        raise AutoSpaceValidationError("v1 AutoSpace contract must not declare the v2 required design")
    return None


def validate_trusted_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the model-independent E2 security/scientific ceiling."""

    value = _object(contract, "trusted Genesis contract")
    if value.get("schema") != "recclaw.external-comparison.e2-auto-space-genesis-contract.v1":
        raise AutoSpaceValidationError("unsupported trusted Genesis contract")
    if value.get("status") not in {DRAFT_GENERATION_STATUS, REAL_GENERATION_STATUS}:
        raise AutoSpaceValidationError("trusted Genesis contract has an unsupported status")
    if value.get("task_id") != "MOREC_AMAZON_CDS" or value.get("bootstrap_seed") != 94001:
        raise AutoSpaceValidationError("trusted Genesis task/bootstrap identity mismatch")

    mode = provider_output_mode(value)
    if mode != PROVIDER_OUTPUT_MODE_V161:
        raise AutoSpaceValidationError("MoRec requires the current six-axis v1.6.1 authoring method")
    research = _object(value.get("research_provenance"), "research_provenance")
    expected_research = {
        "comparison_status": "PROSPECTIVE_MOREC_DEVELOPMENT",
        "reused_method": "SASREC_V161_SIX_AXIS_AUTHORING_METHOD",
        "prior_candidate_or_result_supplied": False,
        "specific_winning_mechanisms_copied": False,
        "design_basis": ["MOREC_BASELINE_SOURCE", "SHARED_TITLE_ASSETS", "TRUSTED_MUTABLE_UPPER_BOUND"],
    }
    if research != expected_research:
        raise AutoSpaceValidationError("MoRec fresh research provenance drift")

    generation = _object(value.get("generation_policy"), "generation_policy")
    required_generation = {
        "generator_model": "gpt-5.6-terra",
        "generator_reasoning_effort": "medium",
        "critic_model": "gpt-5.6-terra",
        "critic_reasoning_effort": "medium",
        "max_critic_passes": 1,
        "model_fallback": False,
        "tools_enabled": False,
        "retrieval_enabled": False,
        "authorized_generation_calls": 1,
    }
    if any(generation.get(key) != expected for key, expected in required_generation.items()):
        raise AutoSpaceValidationError("trusted Genesis generation policy drift")
    provider_output_mode(value)
    pricing_digest = generation.get("pricing_manifest_sha256")
    if not isinstance(pricing_digest, str) or len(pricing_digest) != 64 or any(
        char not in "0123456789abcdef" for char in pricing_digest
    ):
        raise AutoSpaceValidationError("trusted Genesis pricing identity is invalid")

    repeat = _object(value.get("repeat_policy"), "repeat_policy")
    if repeat != {
        "repeat_count": 1,
        "search_seeds": [43001],
        "one_task_level_search_space": True,
        "regeneration_between_repeats": False,
    }:
        raise AutoSpaceValidationError("trusted AutoSpace Option-B repeat policy drift")

    protocol_identity = _object(value.get("protocol_identity"), "protocol_identity")
    amendment_version = protocol_identity.get("autospace_amendment_version")
    required_design = _validate_required_search_space_design(value, amendment_version)
    if provider_output_mode(value) == PROVIDER_OUTPUT_MODE_V161:
        contract_filename = "E2_AUTO_SPACE_GENESIS_CONTRACT_v161.json"
    elif provider_output_mode(value) == PROVIDER_OUTPUT_MODE_V16:
        contract_filename = "E2_AUTO_SPACE_GENESIS_CONTRACT_v16.json"
    elif amendment_version in {
        V13_DRAFT_AUTOSPACE_AMENDMENT_VERSION,
        V13_SIGNED_AUTOSPACE_AMENDMENT_VERSION,
        V14_SIGNED_AUTOSPACE_AMENDMENT_VERSION,
        V15_SIGNED_AUTOSPACE_AMENDMENT_VERSION,
    }:
        contract_filename = "E2_AUTO_SPACE_GENESIS_CONTRACT_v3.json"
    elif required_design is not None:
        contract_filename = "E2_AUTO_SPACE_GENESIS_CONTRACT_v2.json"
    else:
        contract_filename = "E2_AUTO_SPACE_GENESIS_CONTRACT_v1.json"

    input_policy = _object(value.get("genesis_input_policy"), "genesis_input_policy")
    expected_inputs = input_policy.get("expected_inputs")
    expected_identity = [
        {"role": "baseline_source", "path": "diagnosis_tools.py"},
        {"role": "baseline_source", "path": "main_code.py"},
        {"role": "baseline_source", "path": "models.py"},
        {"role": "baseline_source", "path": "morec_initialization.py"},
        {"role": "baseline_source", "path": "self_evolverec_interface.py"},
        {"role": "task_contract", "path": contract_filename},
    ]
    if input_policy.get("allowlist_mode") != "TRUSTED_CONTRACT_EXACT_SET" or expected_inputs != expected_identity:
        raise AutoSpaceValidationError("trusted Genesis exact input allowlist drift")

    baseline = _object(value.get("canonical_baseline_identity"), "canonical_baseline_identity")
    baseline_files = _object(baseline.get("files"), "canonical baseline files")
    if set(baseline_files) != {item["path"] for item in expected_identity[:-1]}:
        raise AutoSpaceValidationError("trusted canonical baseline file set drift")
    if any(
        not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
        for digest in baseline_files.values()
    ):
        raise AutoSpaceValidationError("trusted canonical baseline file digest is invalid")
    if baseline.get("aggregate_sha256") != canonical_sha256(dict(baseline_files)):
        raise AutoSpaceValidationError("trusted canonical baseline aggregate is stale")

    abi = _object(value.get("candidate_abi"), "candidate_abi")
    if (
        abi.get("required_files") != [
            "main_code.py", "models.py", "diagnosis_tools.py", "self_evolverec_interface.py", "morec_initialization.py"
        ]
        or abi.get("entrypoint") != "self_evolverec_interface:deepevolve_interface"
        or abi.get("type_ref") != "E2SASRecCandidate"
    ):
        raise AutoSpaceValidationError("trusted E2 candidate ABI drift")

    upper = _object(value.get("trusted_upper_bound"), "trusted_upper_bound")
    allowed_targets = set(_string_tuple(upper.get("allowed_mutation_targets"), "trusted allowed_mutation_targets", nonempty=True))
    frozen_targets = set(_string_tuple(upper.get("frozen_mutation_targets"), "trusted frozen_mutation_targets", nonempty=True))
    if allowed_targets.intersection(frozen_targets):
        raise AutoSpaceValidationError("trusted mutation ceiling both allows and freezes a target")
    allowed_roles = set(_string_tuple(upper.get("allowed_data_roles"), "trusted allowed_data_roles", nonempty=True))
    forbidden_roles = set(_string_tuple(upper.get("forbidden_data_roles"), "trusted forbidden_data_roles", nonempty=True))
    if allowed_roles.intersection(forbidden_roles):
        raise AutoSpaceValidationError("trusted data-role ceiling is contradictory")

    evaluation = _object(value.get("evaluation"), "evaluation")
    if evaluation.get("test_access") != "DENIED" or evaluation.get("evaluator_mutation") != "DENIED" or evaluation.get("split_mutation") != "DENIED":
        raise AutoSpaceValidationError("trusted evaluation/test denial drift")

    validate_protocol_identity(
        protocol_identity,
        require_signed=value.get("status") == REAL_GENERATION_STATUS,
    )
    return value

def trusted_contract_from_request(request: Mapping[str, Any]) -> dict[str, Any]:
    contracts = [item for item in request.get("inputs", []) if item.get("role") == "task_contract"]
    if len(contracts) != 1:
        raise AutoSpaceValidationError("Genesis request must contain exactly one task contract")
    try:
        content = contracts[0]["content"]
        if not isinstance(content, str):
            raise TypeError("task contract content must be text")
        contract = load_json_text(content, label="embedded Genesis task contract")
    except (KeyError, TypeError, AutoSpaceValidationError) as error:
        raise AutoSpaceValidationError("Genesis task contract is invalid") from error
    if not isinstance(contract, Mapping):
        raise AutoSpaceValidationError("Genesis task contract must be a JSON object")
    return validate_trusted_contract(contract)


def validate_embedded_genesis_request(request: Mapping[str, Any]) -> dict[str, Any]:
    validate_schema(request, "auto_space_genesis_request.schema.json", label="Genesis request")
    payload = dict(request)
    observed = payload.pop("request_digest", None)
    expected_digest = canonical_sha256(payload)
    if observed != expected_digest:
        raise AutoSpaceValidationError("Genesis request digest is stale")
    contract = trusted_contract_from_request(request)
    if contract.get("task_id") != request.get("task_id"):
        raise AutoSpaceValidationError("Genesis request and trusted contract task identities differ")
    expected_inputs = contract.get("genesis_input_policy", {}).get("expected_inputs")
    if not isinstance(expected_inputs, list) or not expected_inputs:
        raise AutoSpaceValidationError("trusted contract has no exact Genesis input set")
    actual_inputs = request.get("inputs")
    if not isinstance(actual_inputs, list):
        raise AutoSpaceValidationError("Genesis request inputs are invalid")
    expected_keys = [(item.get("role"), item.get("path")) for item in expected_inputs]
    actual_keys = [(item.get("role"), item.get("path")) for item in actual_inputs]
    if actual_keys != expected_keys:
        raise AutoSpaceValidationError("Genesis request inputs do not match the trusted ordered exact set")
    for item in actual_inputs:
        content = item.get("content")
        if not isinstance(content, str) or hashlib.sha256(content.encode("utf-8")).hexdigest() != item.get("sha256"):
            raise AutoSpaceValidationError(f"Genesis input digest is stale for {item.get('path')!r}")
    expected_baseline = contract.get("canonical_baseline_identity")
    if not isinstance(expected_baseline, Mapping):
        raise AutoSpaceValidationError("trusted contract has no canonical baseline identity")
    expected_files = expected_baseline.get("files")
    if not isinstance(expected_files, Mapping) or not expected_files:
        raise AutoSpaceValidationError("canonical baseline file identity is invalid")
    actual_files = {
        str(item["path"]): str(item["sha256"])
        for item in actual_inputs
        if item.get("role") == "baseline_source"
    }
    if actual_files != dict(expected_files):
        raise AutoSpaceValidationError("Genesis request does not bind the canonical unevolved E2 baseline")
    aggregate = canonical_sha256(actual_files)
    if aggregate != expected_baseline.get("aggregate_sha256"):
        raise AutoSpaceValidationError("canonical baseline aggregate identity mismatch")
    expected_policy = dict(contract.get("generation_policy", {}))
    expected_policy.pop("authorized_generation_calls", None)
    if request.get("generation_policy") != expected_policy:
        raise AutoSpaceValidationError("Genesis generation policy differs from the trusted contract")

    instructions = request.get("instructions")
    if (
        isinstance(instructions, str)
        and (
            GENESIS_PROVIDER_GUIDANCE_V2_MARKER in instructions
            or GENESIS_PROVIDER_GUIDANCE_V16_MARKER in instructions
            or GENESIS_PROVIDER_GUIDANCE_V161_MARKER in instructions
        )
    ):
        expected_instructions = genesis_provider_instructions(contract)
        if instructions != expected_instructions:
            raise AutoSpaceValidationError(
                "guided Genesis provider instructions differ from the canonical profile"
            )
        sources = {
            item["path"]: _symbol_index(item["content"], item["path"])
            for item in actual_inputs
            if item.get("role") == "baseline_source"
        }
        for entry in provider_design_guidance(contract).get(
            "conceptual_mutation_targets", []
        ):
            anchor = entry["required_real_baseline_anchor"]
            key = (anchor["symbol"], anchor["kind"])
            if anchor["path"] not in sources or key not in sources[anchor["path"]]:
                raise AutoSpaceValidationError(
                    "guided conceptual mutation target has no verified real baseline anchor"
                )

    audit = request.get("access_audit")
    if not isinstance(audit, Mapping) or audit.get("input_file_count") != len(actual_inputs):
        raise AutoSpaceValidationError("Genesis access audit input count mismatch")
    for field in (
        "literature_accessed", "test_artifacts_accessed", "prior_candidates_accessed",
        "prior_results_accessed", "historical_search_space_semantics_accessed",
        "tools_enabled", "retrieval_enabled",
    ):
        if audit.get(field) is not False:
            raise AutoSpaceValidationError(f"Genesis access audit failed: {field}")
    validate_protocol_identity(contract.get("protocol_identity", {}), require_signed=False)
    return contract


def _check_subset(actual: Sequence[str], ceiling: Sequence[str], label: str) -> None:
    excess = sorted(set(actual).difference(ceiling))
    if excess:
        raise AutoSpaceValidationError(f"{label} expands the trusted contract: {excess}")


def enforce_trusted_contract(
    family: Mapping[str, Any],
    primitives: Sequence[Mapping[str, Any]],
    grammar: Mapping[str, Any],
    policy: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    axes: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    boundary = _object(contract.get("trusted_upper_bound"), "trusted_upper_bound")
    allowed_targets = _string_tuple(
        boundary.get("allowed_mutation_targets"), "trusted allowed_mutation_targets", nonempty=True
    )
    frozen_targets = _string_tuple(
        boundary.get("frozen_mutation_targets"), "trusted frozen_mutation_targets", nonempty=True
    )
    allowed_dimensions = _string_tuple(
        boundary.get("allowed_mutable_dimensions"), "trusted allowed_mutable_dimensions", nonempty=True
    )
    frozen_dimensions = _string_tuple(
        boundary.get("frozen_protocol_dimensions"), "trusted frozen_protocol_dimensions", nonempty=True
    )
    allowed_roles = _string_tuple(
        boundary.get("allowed_data_roles"), "trusted allowed_data_roles", nonempty=True
    )
    forbidden_roles = _string_tuple(
        boundary.get("forbidden_data_roles"), "trusted forbidden_data_roles", nonempty=True
    )
    allowed_dependencies = _string_tuple(
        boundary.get("allowed_dependencies"), "trusted allowed_dependencies", nonempty=True
    )

    family_id = _nonempty_string(family.get("family_id"), "family.family_id")
    baseline_contract = _object(family.get("baseline_contract"), "family.baseline_contract")
    if baseline_contract.get("task_id") != contract.get("task_id"):
        raise AutoSpaceValidationError("family baseline contract targets the wrong task")
    if baseline_contract.get("baseline_model") != contract.get("baseline"):
        raise AutoSpaceValidationError("family baseline contract names the wrong baseline")
    canonical_baseline = _object(contract.get("canonical_baseline_identity"), "canonical_baseline_identity")
    canonical_paths = set(_object(canonical_baseline.get("files"), "canonical baseline files"))
    anchor_paths = set(_string_tuple(
        baseline_contract.get("required_anchor_paths"),
        "family baseline required_anchor_paths",
        nonempty=True,
    ))
    invalid_anchor_paths = []
    for path in anchor_paths:
        if path in canonical_paths:
            continue
        if path.count(":") != 1:
            invalid_anchor_paths.append(path)
            continue
        base_path, symbol_path = path.split(":", 1)
        if (
            base_path not in canonical_paths
            or not symbol_path
            or any(not part.isidentifier() for part in symbol_path.split("."))
        ):
            invalid_anchor_paths.append(path)
    invalid_anchor_paths = sorted(invalid_anchor_paths)
    if invalid_anchor_paths:
        raise AutoSpaceValidationError(
            f"family baseline anchors reference non-canonical baseline files: {invalid_anchor_paths}"
        )
    expected_output_type = contract.get("candidate_abi", {}).get("type_ref")
    if expected_output_type is not None and family.get("output_contract", {}).get("type_ref") != expected_output_type:
        raise AutoSpaceValidationError("family output contract targets the wrong candidate type")

    _check_subset(family["mutable_dimensions"], allowed_dimensions, "family mutable_dimensions")
    missing_frozen = sorted(set(frozen_dimensions).difference(family["frozen_protocol_dimensions"]))
    if missing_frozen:
        raise AutoSpaceValidationError(f"family omits trusted frozen dimensions: {missing_frozen}")
    _check_subset(family["allowed_data_roles"], allowed_roles, "family allowed_data_roles")
    if not set(forbidden_roles).issubset(family["forbidden_data_roles"]):
        raise AutoSpaceValidationError("family weakens trusted forbidden_data_roles")

    policy_families = set(_string_tuple(
        policy.get("legal_capability_families"), "policy legal_capability_families", nonempty=True
    ))
    if policy_families != {family_id}:
        raise AutoSpaceValidationError("generated capability policy must authorize exactly its generated family")
    policy_targets = set(_string_tuple(
        policy.get("allowed_mutation_targets"), "policy allowed_mutation_targets", nonempty=True
    ))
    policy_frozen = set(_string_tuple(
        policy.get("frozen_mutation_targets"), "policy frozen_mutation_targets", nonempty=True
    ))
    policy_roles = set(_string_tuple(
        policy.get("allowed_data_roles"), "policy allowed_data_roles", nonempty=True
    ))
    _check_subset(policy_targets, allowed_targets, "policy allowed_mutation_targets")
    if not set(frozen_targets).issubset(policy_frozen):
        raise AutoSpaceValidationError("policy weakens trusted frozen_mutation_targets")
    if policy_targets.intersection(policy_frozen):
        raise AutoSpaceValidationError("generated policy both allows and freezes the same mutation target")
    _check_subset(policy_roles, allowed_roles, "policy allowed_data_roles")
    if not set(forbidden_roles).issubset(policy["forbidden_data_roles"]):
        raise AutoSpaceValidationError("policy weakens trusted forbidden_data_roles")
    _check_subset(policy["allowed_dependencies"], allowed_dependencies, "policy allowed_dependencies")
    if policy.get("network_access") is not False:
        raise AutoSpaceValidationError("generated policy enables network access")
    if not set(family["allowed_data_roles"]).issubset(policy_roles):
        raise AutoSpaceValidationError("family allowed_data_roles exceed the generated capability policy")

    exact_primitive_anchors = uses_deterministic_provider_assembly(contract)
    for primitive in primitives:
        if primitive.get("family_id") != family_id:
            raise AutoSpaceValidationError(f"primitive {primitive.get('primitive_id')!r} crosses the generated family")
        primitive_anchors = {
            canonical_anchor_identity(anchor) if exact_primitive_anchors else anchor["path"]
            for anchor in primitive.get("baseline_anchors", ())
        }
        if not primitive_anchors.issubset(anchor_paths):
            raise AutoSpaceValidationError(
                f"primitive {primitive.get('primitive_id')!r} uses a baseline anchor outside the generated family contract"
            )
        primitive_targets = set(primitive["legal_mutation_targets"])
        primitive_roles = set(primitive["allowed_data_roles"])
        _check_subset(primitive_targets, policy_targets, "primitive mutation targets vs generated policy")
        _check_subset(primitive_targets, allowed_targets, "primitive mutation targets vs trusted contract")
        if primitive_targets.intersection(policy_frozen | set(frozen_targets)):
            raise AutoSpaceValidationError(
                f"primitive {primitive['primitive_id']!r} targets a frozen mutation boundary"
            )
        _check_subset(primitive_roles, policy_roles, "primitive allowed_data_roles vs generated policy")
        _check_subset(primitive_roles, family["allowed_data_roles"], "primitive allowed_data_roles vs family")
        _check_subset(primitive_roles, allowed_roles, "primitive allowed_data_roles vs trusted contract")
        if primitive_roles.intersection(forbidden_roles) or primitive_roles.intersection(policy["forbidden_data_roles"]):
            raise AutoSpaceValidationError(
                f"primitive {primitive['primitive_id']!r} reads a forbidden data role"
            )
        if primitive_targets.intersection(primitive.get("prohibited_mutation_targets", ())):
            raise AutoSpaceValidationError(
                f"primitive {primitive['primitive_id']!r} both allows and prohibits a mutation target"
            )

    custom = grammar["composition_policy"]["custom_synthesis"]
    operator_ids = {item["operator_id"] for item in grammar["operators"]}
    custom_operator_kinds = {item["operator_id"]: item["kind"] for item in grammar["operators"]}
    if custom["enabled"]:
        if custom["operator_id"] not in operator_ids:
            raise AutoSpaceValidationError("custom synthesis names an unknown operator")
        if custom_operator_kinds[custom["operator_id"]] != "CUSTOM_SYNTHESIS":
            raise AutoSpaceValidationError("custom synthesis policy must reference a CUSTOM_SYNTHESIS operator")
        if custom["max_components"] < 1:
            raise AutoSpaceValidationError("enabled custom synthesis requires max_components >= 1")
    else:
        if custom["operator_id"] is not None or custom["max_components"] != 0 or custom["allowed_slots"]:
            raise AutoSpaceValidationError("disabled custom synthesis must have an empty configuration")
        if "CUSTOM_SYNTHESIS" in custom_operator_kinds.values():
            raise AutoSpaceValidationError("disabled custom synthesis cannot expose a CUSTOM_SYNTHESIS operator")

    required_design = contract.get("required_search_space_design")
    if required_design is not None:
        if provider_output_mode(contract) == PROVIDER_OUTPUT_MODE_V161:
            if axes is None:
                raise AutoSpaceValidationError(
                    "trusted v1.6.1 design requires the conceptual axis registry"
                )
            from auto_space_v161 import REQUIRED_AXES, validate_materialized

            design = validate_materialized({"axes": list(axes)}, grammar, contract)
            del design
            required_design = _object(required_design, "required_search_space_design")
            if set(family["mutable_dimensions"]) != set(required_design["mutable_dimensions"]):
                raise AutoSpaceValidationError(
                    "v1.6.1 family mutable dimensions do not match the trusted design"
                )
            if (
                family.get("family_version") != "1.6.1"
                or grammar.get("grammar_version") != "1.6.1"
                or policy.get("policy_version") != "1.6.1"
                or any(item.get("version") != "1.6.1" for item in primitives)
            ):
                raise AutoSpaceValidationError("v1.6.1 content version drift")
            slot_contracts = {item["slot_id"]: item for item in required_design["slots"]}
            for primitive in primitives:
                slot = slot_contracts.get(primitive.get("slot_id"))
                if slot is None:
                    raise AutoSpaceValidationError("v1.6.1 primitive uses an untrusted axis")
                expected = REQUIRED_AXES[primitive["slot_id"]]
                if (
                    set(primitive["composition_roles"]) != {expected[1]}
                    or set(primitive["conceptual_capabilities"]) != {expected[2]}
                    or not set(primitive["legal_mutation_targets"]).issubset(slot["legal_mutation_targets"])
                ):
                    raise AutoSpaceValidationError(
                        f"v1.6.1 primitive {primitive.get('primitive_id')!r} violates its axis contract"
                    )
            target_union = {
                target for axis in axes for primitive in axis["primitives"] for target in primitive["legal_mutation_targets"]
            }
            if set(policy["allowed_mutation_targets"]) != target_union:
                raise AutoSpaceValidationError(
                    "v1.6.1 policy targets do not equal the accepted primitive target union"
                )
            return
        if provider_output_mode(contract) == PROVIDER_OUTPUT_MODE_V16:
            if axes is None:
                raise AutoSpaceValidationError(
                    "trusted v1.6 design requires the conceptual axis registry"
                )
            from auto_space_v16 import REQUIRED_AXES, validate_materialized

            design = validate_materialized({"axes": list(axes)}, grammar, contract)
            del design
            required_design = _object(
                required_design, "required_search_space_design"
            )
            if set(family["mutable_dimensions"]) != set(
                required_design["mutable_dimensions"]
            ):
                raise AutoSpaceValidationError(
                    "v1.6 family mutable dimensions do not match the trusted design"
                )
            if (
                family.get("family_version") != "1.6"
                or grammar.get("grammar_version") != "1.6"
                or policy.get("policy_version") != "1.6"
                or any(item.get("version") != "1.6" for item in primitives)
            ):
                raise AutoSpaceValidationError("v1.6 content version drift")
            slot_contracts = {
                item["slot_id"]: item for item in required_design["slots"]
            }
            for primitive in primitives:
                slot = slot_contracts.get(primitive.get("slot_id"))
                if slot is None:
                    raise AutoSpaceValidationError(
                        "v1.6 primitive uses an untrusted axis"
                    )
                expected = REQUIRED_AXES[primitive["slot_id"]]
                if (
                    set(primitive["composition_roles"]) != {expected[1]}
                    or set(primitive["conceptual_capabilities"]) != {expected[2]}
                    or not set(primitive["legal_mutation_targets"]).issubset(
                        slot["legal_mutation_targets"]
                    )
                ):
                    raise AutoSpaceValidationError(
                        f"v1.6 primitive {primitive.get('primitive_id')!r} violates its axis contract"
                    )
            target_union = {
                target
                for axis in axes
                for primitive in axis["primitives"]
                for target in primitive["legal_mutation_targets"]
            }
            if set(policy["allowed_mutation_targets"]) != target_union:
                raise AutoSpaceValidationError(
                    "v1.6 policy targets do not equal the accepted primitive target union"
                )
            return
        if axes is None:
            raise AutoSpaceValidationError("trusted v2 design requires the conceptual axis registry")
        required_design = _object(required_design, "required_search_space_design")
        slot_contracts = {item["slot_id"]: item for item in required_design["slots"]}
        observed_axes = {}
        for raw_axis in axes:
            axis = _object(raw_axis, "generated axis")
            slot_id = _nonempty_string(axis.get("slot_id"), "generated axis slot_id")
            if slot_id in observed_axes:
                raise AutoSpaceValidationError("generated axes contain duplicates")
            observed_axes[slot_id] = axis
        if set(observed_axes) != set(slot_contracts):
            raise AutoSpaceValidationError("generated axes do not match the trusted v2 slot set")
        if any(axis.get("allow_multiple") is not False for axis in observed_axes.values()):
            raise AutoSpaceValidationError("trusted v2 axes must set allow_multiple=false")
        content_version = required_design["content_version"]
        if (
            family.get("family_version") != content_version
            or grammar.get("grammar_version") != content_version
            or policy.get("policy_version") != content_version
            or any(primitive.get("version") != content_version for primitive in primitives)
        ):
            raise AutoSpaceValidationError("generated content versions do not match the trusted v2 design")
        for primitive in primitives:
            slot_id = primitive.get("slot_id")
            if slot_id not in slot_contracts:
                raise AutoSpaceValidationError("generated primitive uses a slot outside the trusted v2 design")
            slot = slot_contracts[slot_id]
            if set(primitive["legal_mutation_targets"]) != set(slot["legal_mutation_targets"]):
                raise AutoSpaceValidationError(
                    f"primitive {primitive.get('primitive_id')!r} violates its trusted slot target boundary"
                )
            if set(primitive["composition_roles"]) != {slot["composition_role"]}:
                raise AutoSpaceValidationError(
                    f"primitive {primitive.get('primitive_id')!r} violates its trusted slot role"
                )
            if set(primitive["conceptual_capabilities"]) != {slot["conceptual_capability"]}:
                raise AutoSpaceValidationError(
                    f"primitive {primitive.get('primitive_id')!r} violates its trusted slot capability"
                )
        if set(family["mutable_dimensions"]) != set(required_design["mutable_dimensions"]):
            raise AutoSpaceValidationError("family mutable dimensions do not match the trusted v2 design")
        composition = grammar["composition_policy"]
        if composition["max_primitives_per_candidate"] != required_design["max_primitives_per_candidate"]:
            raise AutoSpaceValidationError("grammar max_primitives_per_candidate does not match the trusted v2 design")
        if composition["max_mutation_targets"] != required_design["max_mutation_targets"]:
            raise AutoSpaceValidationError("grammar max_mutation_targets does not match the trusted v2 design")
        expected_operators = {item["kind"]: item for item in required_design["operators"]}
        observed_operators = {item["kind"]: item for item in grammar["operators"]}
        if set(observed_operators) != set(expected_operators):
            raise AutoSpaceValidationError("grammar operators do not match the trusted v2 design")
        allowed_roles = {item["composition_role"] for item in required_design["slots"]}
        for kind, expected in expected_operators.items():
            operator = observed_operators[kind]
            if (
                operator["operator_id"] != expected["operator_id"]
                or operator["min_arity"] != expected["min_arity"]
                or operator["max_arity"] != expected["max_arity"]
                or operator["requires_target"] is not False
                or operator["requires_replacement"] is not False
                or operator["required_capabilities"]
                or set(operator["allowed_composition_roles"]) != allowed_roles
                or operator["port_compatibility"] != "NOT_REQUIRED"
            ):
                raise AutoSpaceValidationError(f"{kind} does not match the trusted v2 design")
        expected_custom = required_design["custom_synthesis"]
        if (
            custom["enabled"] is not True
            or custom["operator_id"] != "custom_synthesis"
            or custom["max_components"] != expected_custom["max_components"]
            or set(custom["allowed_slots"]) != set(expected_custom["allowed_slots"])
            or composition["cross_family_composition"] != "DENY"
            or composition["incompatible_primitive_sets"]
            or composition["incompatible_composition_role_pairs"]
        ):
            raise AutoSpaceValidationError("custom composition does not match the trusted v2 design")
        conceptual_targets = {
            target
            for item in required_design["slots"]
            for target in item["legal_mutation_targets"]
        }
        if set(policy["allowed_mutation_targets"]) != conceptual_targets:
            raise AutoSpaceValidationError("v2 policy mutation targets do not match the conceptual slot union")


@dataclass(frozen=True)
class ConceptualPrimitiveV1:
    primitive_id: str
    family_id: str
    slot_id: str
    digest: str
    record: Mapping[str, Any]


@dataclass(frozen=True)
class ConceptualSearchCandidateBindingV1:
    proposal_ref: str
    proposal_digest: str
    search_space_id: str
    search_space_digest: str
    family_id: str
    family_digest: str
    operator_id: str
    target_primitive_id: str | None
    replacement_primitive_id: str | None
    primitive_ids: tuple[str, ...]
    primitive_digests: tuple[str, ...]
    custom_components: tuple[Mapping[str, Any], ...]
    custom_component_digests: tuple[str, ...]
    grammar_digest: str
    capability_policy_digest: str
    schema: str = "recclaw.auto-space.conceptual-candidate-binding.v1"

    @property
    def digest(self) -> str:
        return canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "proposal_ref": self.proposal_ref,
            "proposal_digest": self.proposal_digest,
            "search_space_id": self.search_space_id,
            "search_space_digest": self.search_space_digest,
            "family_id": self.family_id,
            "family_digest": self.family_digest,
            "operator_id": self.operator_id,
            "target_primitive_id": self.target_primitive_id,
            "replacement_primitive_id": self.replacement_primitive_id,
            "primitive_ids": list(self.primitive_ids),
            "primitive_digests": list(self.primitive_digests),
            "custom_components": [dict(item) for item in self.custom_components],
            "custom_component_digests": list(self.custom_component_digests),
            "grammar_digest": self.grammar_digest,
            "capability_policy_digest": self.capability_policy_digest,
            "directly_executable": False,
            "next_stage": "IMPLEMENTATION",
        }


@dataclass(frozen=True)
class NormalizedSearchSpaceV1:
    root: Path
    source: SearchSpaceSource
    manifest: Mapping[str, Any]
    family: Mapping[str, Any]
    primitive_registry: Mapping[str, Any]
    grammar: Mapping[str, Any]
    capability_policy: Mapping[str, Any]
    trusted_contract: Mapping[str, Any] | None
    genesis_request: Mapping[str, Any] | None
    search_space_id: str
    family_id: str
    digest: str
    family_digest: str
    grammar_digest: str
    capability_policy_digest: str
    primitives: Mapping[str, ConceptualPrimitiveV1]
    axes_allow_multiple: Mapping[str, bool]
    manual_contract: Mapping[str, Any] | None = None
    manual_provenance: Mapping[str, Any] | None = None
    frozen_manifest: Mapping[str, Any] | None = None

    def authorized_mutation_targets(
        self, binding: ConceptualSearchCandidateBindingV1
    ) -> tuple[str, ...]:
        """Derive conceptual target authority from this frozen search space."""

        if not isinstance(binding, ConceptualSearchCandidateBindingV1):
            raise AutoSpaceValidationError("conceptual binding has the wrong type")
        if (
            binding.search_space_id != self.search_space_id
            or binding.search_space_digest != self.digest
            or binding.family_id != self.family_id
            or binding.family_digest != self.family_digest
            or binding.grammar_digest != self.grammar_digest
            or binding.capability_policy_digest != self.capability_policy_digest
        ):
            raise AutoSpaceValidationError("conceptual binding identity drift")
        records = [self.primitives[item].record for item in binding.primitive_ids]
        records.extend(binding.custom_components)
        return tuple(
            sorted(
                {
                    target
                    for record in records
                    for target in record["legal_mutation_targets"]
                }
            )
        )

    def bind_proposal(
        self,
        *,
        proposal_ref: str,
        proposal_digest: str,
        primitive_ids: Sequence[str],
        operator_id: str | None = None,
        target_primitive_id: str | None = None,
        replacement_primitive_id: str | None = None,
        custom_components: Sequence[Mapping[str, Any]] = (),
    ) -> ConceptualSearchCandidateBindingV1:
        proposal_ref = _nonempty_string(proposal_ref, "proposal_ref")
        proposal_digest = _nonempty_string(proposal_digest, "proposal_digest")
        selected = tuple(primitive_ids)
        if len(set(selected)) != len(selected):
            raise AutoSpaceValidationError("conceptual primitive binding contains duplicates")
        missing = sorted(set(selected).difference(self.primitives))
        if missing:
            raise AutoSpaceValidationError(f"proposal references unknown conceptual primitives: {missing}")
        if self.manifest.get("search_space_version") in {"v1.6", "v1.6.1"}:
            if self.manifest.get("search_space_version") == "v1.6.1":
                from auto_space_v161 import canonicalize_selection
            else:
                from auto_space_v16 import canonicalize_selection

            selected = canonicalize_selection(selected, self.primitives)
        custom_records = tuple(_object(item, "custom component") for item in custom_components)
        custom_ids = [item.get("primitive_id") for item in custom_records]
        if any(not isinstance(item, str) or not item.strip() for item in custom_ids):
            raise AutoSpaceValidationError("custom component primitive_id must be a non-empty string")
        if len(set(custom_ids)) != len(custom_ids) or set(custom_ids).intersection(self.primitives):
            raise AutoSpaceValidationError("custom component IDs must be unique and distinct from registered primitives")
        if not selected and not custom_records:
            raise AutoSpaceValidationError("a proposal must bind a primitive or custom component")
        custom_policy = self.grammar["composition_policy"]["custom_synthesis"]
        if custom_records and not custom_policy["enabled"]:
            raise AutoSpaceValidationError("the frozen grammar does not authorize custom synthesis")
        if len(custom_records) > custom_policy["max_components"]:
            raise AutoSpaceValidationError("proposal exceeds the custom component limit")
        records = [self.primitives[item].record for item in selected]
        for custom in custom_records:
            validate_schema(
                {
                    "record_type": "search_space_primitive_registry",
                    "registry_id": "custom-binding-validation",
                    "registry_version": "1",
                    "axis_count": 1,
                    "primitive_count": 1,
                    "axes": [
                        {
                            "slot_id": custom.get("slot_id"),
                            "allow_multiple": False,
                            "primitives": [custom],
                        }
                    ],
                },
                "auto_space_primitive_registry.schema.json",
                label="custom component",
            )
            required = {
                "primitive_id", "version", "family_id", "slot_id", "implementation_status",
                "semantic_role", "baseline_anchors", "input_ports", "output_ports",
                "parameter_schema", "conceptual_capabilities", "allowed_data_roles",
                "composition_roles", "legal_mutation_targets", "prohibited_mutation_targets",
                "implementation_requirements", "required_invariants", "rationale",
                "family_boundary_justification",
            }
            if set(custom) != required:
                raise AutoSpaceValidationError("custom component has missing or unknown fields")
            if custom["family_id"] != self.family_id:
                raise AutoSpaceValidationError("custom component crosses the frozen family boundary")
            if custom["slot_id"] not in custom_policy["allowed_slots"]:
                raise AutoSpaceValidationError("custom component uses a disallowed slot")
            if custom["implementation_status"] != "CONCEPTUAL_UNIMPLEMENTED":
                raise AutoSpaceValidationError("custom component claims executable status")
            required_design = (
                self.trusted_contract.get("required_search_space_design")
                if self.trusted_contract is not None
                else None
            )
            if required_design is not None:
                slot_contracts = {
                    item["slot_id"]: item for item in required_design["slots"]
                }
                slot = slot_contracts.get(custom["slot_id"])
                if slot is None:
                    raise AutoSpaceValidationError("custom component uses a slot outside the trusted v2 design")
                if set(custom["composition_roles"]) != {slot["composition_role"]}:
                    raise AutoSpaceValidationError("custom component violates its trusted slot role")
                if set(custom["conceptual_capabilities"]) != {slot["conceptual_capability"]}:
                    raise AutoSpaceValidationError("custom component violates its trusted slot capability")
                if not set(custom["legal_mutation_targets"]).issubset(slot["legal_mutation_targets"]):
                    raise AutoSpaceValidationError("custom component exceeds its slot mutation-target boundary")
                registered_anchors = {
                    (anchor["path"], anchor["symbol"], anchor["kind"])
                    for primitive in self.primitives.values()
                    if primitive.slot_id == custom["slot_id"]
                    for anchor in primitive.record["baseline_anchors"]
                }
                custom_anchors = {
                    (anchor["path"], anchor["symbol"], anchor["kind"])
                    for anchor in custom["baseline_anchors"]
                }
                if not custom_anchors.issubset(registered_anchors):
                    raise AutoSpaceValidationError("custom component has an unverified structural baseline anchor")
            _assert_no_executable_claims(custom, "custom component")
            if self.genesis_request is None:
                raise AutoSpaceValidationError("custom synthesis requires an audited AutoSpace Genesis request")
            validate_baseline_anchors((custom,), self.genesis_request)
            records.append(custom)
        chosen_operator = operator_id or ("select" if len(records) == 1 else "compose")
        operators = {item["operator_id"]: item for item in self.grammar["operators"]}
        if chosen_operator not in operators:
            raise AutoSpaceValidationError(f"unknown grammar operator {chosen_operator!r}")
        operator = operators[chosen_operator]
        if custom_records and operator["kind"] != "CUSTOM_SYNTHESIS":
            raise AutoSpaceValidationError("custom components require the custom synthesis operator")
        if operator["kind"] == "CUSTOM_SYNTHESIS" and not custom_records:
            raise AutoSpaceValidationError("CUSTOM_SYNTHESIS requires at least one custom component")
        if not operator["min_arity"] <= len(records) <= operator["max_arity"]:
            raise AutoSpaceValidationError("operator arity is invalid")
        if operator["requires_target"] != (target_primitive_id is not None):
            raise AutoSpaceValidationError("operator target requirement is not satisfied")
        if operator["requires_replacement"] != (replacement_primitive_id is not None):
            raise AutoSpaceValidationError("operator replacement requirement is not satisfied")
        if target_primitive_id is not None and target_primitive_id not in selected:
            raise AutoSpaceValidationError("operator target is not selected")
        if replacement_primitive_id is not None and replacement_primitive_id not in selected:
            raise AutoSpaceValidationError("operator replacement is not selected")
        slots: dict[str, int] = {}
        for record in records:
            slots[record["slot_id"]] = slots.get(record["slot_id"], 0) + 1
        for slot_id, count in slots.items():
            if count > 1 and not self.axes_allow_multiple.get(slot_id, False):
                raise AutoSpaceValidationError(f"axis {slot_id!r} does not allow multiple selections")
        composition = self.grammar["composition_policy"]
        if len(records) > composition["max_primitives_per_candidate"]:
            raise AutoSpaceValidationError("proposal exceeds max_primitives_per_candidate")
        identifiers = {record["primitive_id"] for record in records}
        for incompatible in composition["incompatible_primitive_sets"]:
            if set(incompatible).issubset(identifiers):
                raise AutoSpaceValidationError("proposal selects an incompatible primitive set")
        roles = {role for record in records for role in record["composition_roles"]}
        if not roles.issubset(operator["allowed_composition_roles"]):
            raise AutoSpaceValidationError("proposal uses a composition role forbidden by the operator")
        for pair in composition["incompatible_composition_role_pairs"]:
            if set(pair).issubset(roles):
                raise AutoSpaceValidationError("proposal combines incompatible composition roles")
        capabilities = {item for record in records for item in record["conceptual_capabilities"]}
        if not set(operator["required_capabilities"]).issubset(capabilities):
            raise AutoSpaceValidationError("proposal lacks an operator-required capability")
        mutation_targets = {item for record in records for item in record["legal_mutation_targets"]}
        if len(mutation_targets) > composition["max_mutation_targets"]:
            raise AutoSpaceValidationError("proposal exceeds the candidate mutation-target limit")
        if operator["port_compatibility"] == "REQUIRED":
            for left, right in zip(records, records[1:]):
                provided = {item["type"] for item in left["output_ports"]}
                accepted = {item for port in right["input_ports"] for item in port["accepted_types"]}
                compatibility = {(item["provided"], item["accepted_as"]) for item in self.family["type_compatibility"]}
                if not provided.intersection(accepted) and not any(
                    source in provided and target in accepted for source, target in compatibility
                ):
                    raise AutoSpaceValidationError("proposal has incompatible primitive ports")
        if self.trusted_contract is not None:
            enforce_trusted_contract(
                self.family,
                records,
                self.grammar,
                self.capability_policy,
                self.trusted_contract,
                axes=self.primitive_registry["axes"],
            )
        return ConceptualSearchCandidateBindingV1(
            proposal_ref=proposal_ref,
            proposal_digest=proposal_digest,
            search_space_id=self.search_space_id,
            search_space_digest=self.digest,
            family_id=self.family_id,
            family_digest=self.family_digest,
            operator_id=chosen_operator,
            target_primitive_id=target_primitive_id,
            replacement_primitive_id=replacement_primitive_id,
            primitive_ids=selected,
            primitive_digests=tuple(self.primitives[item].digest for item in selected),
            custom_components=custom_records,
            custom_component_digests=tuple(canonical_sha256(item) for item in custom_records),
            grammar_digest=self.grammar_digest,
            capability_policy_digest=self.capability_policy_digest,
        )

    def provider_projection(self) -> dict[str, Any]:
        return {
            "source": self.source.value,
            "search_space_id": self.search_space_id,
            "search_space_digest": self.digest,
            "family": self.family,
            "primitive_registry": self.primitive_registry,
            "grammar": self.grammar,
            "capability_policy": self.capability_policy,
            "binding_rule": (
                "Construct a conceptual mechanism program using a frozen grammar operator. "
                "Registered primitives and any grammar-authorized custom component remain "
                "non-executable until downstream implementation and qualification."
            ),
        }


def _core_paths(root: Path) -> tuple[Path, ...]:
    auto = tuple(root / name for name in CORE_FILES)
    if all(path.is_file() for path in auto):
        return auto
    if any(path.is_file() for path in auto):
        missing = [path.name for path in auto if not path.is_file()]
        raise AutoSpaceValidationError(f"search-space core package is incomplete: {missing}")
    legacy = tuple(root / name for name in LEGACY_CORE_FILES)
    if all(path.is_file() for path in legacy):
        return legacy
    raise AutoSpaceValidationError("search-space core package is incomplete")


def _validate_core_structure(
    family: Mapping[str, Any],
    registry: Mapping[str, Any],
    grammar: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    family_id: str,
    primitives: Mapping[str, ConceptualPrimitiveV1],
    axes_allow_multiple: Mapping[str, bool],
    strict_policy_boundary: bool = False,
) -> None:
    if registry["axis_count"] != len(axes_allow_multiple):
        raise AutoSpaceValidationError("axis_count does not match registry contents")
    grammar_primitives = _string_tuple(
        grammar["primitive_ids"], "grammar.primitive_ids", nonempty=True
    )
    if set(grammar_primitives) != set(primitives):
        raise AutoSpaceValidationError(
            "search-space grammar has dangling or missing primitive references"
        )
    operator_ids = [item["operator_id"] for item in grammar["operators"]]
    if len(operator_ids) != len(set(operator_ids)):
        raise AutoSpaceValidationError("grammar contains duplicate operator IDs")
    for operator in grammar["operators"]:
        if operator["min_arity"] > operator["max_arity"]:
            raise AutoSpaceValidationError("grammar operator has inverted arity")
    for group in grammar["composition_policy"]["incompatible_primitive_sets"]:
        if not set(group).issubset(primitives):
            raise AutoSpaceValidationError(
                "grammar incompatibility references an unknown primitive"
            )
    custom_policy = grammar["composition_policy"]["custom_synthesis"]
    if not set(custom_policy["allowed_slots"]).issubset(axes_allow_multiple):
        raise AutoSpaceValidationError(
            "custom synthesis references an unknown conceptual axis"
        )
    if strict_policy_boundary:
        legal_families = set(policy["legal_capability_families"])
        if legal_families != {family_id}:
            raise AutoSpaceValidationError("capability policy family identity mismatch")
        allowed_targets = set(policy["allowed_mutation_targets"])
        frozen_targets = set(policy["frozen_mutation_targets"])
        allowed_roles = set(policy["allowed_data_roles"])
        forbidden_roles = set(policy["forbidden_data_roles"])
        if allowed_targets.intersection(frozen_targets):
            raise AutoSpaceValidationError("allowed and frozen mutation targets overlap")
        if allowed_roles.intersection(forbidden_roles):
            raise AutoSpaceValidationError("allowed and forbidden data roles overlap")
        if set(family["allowed_data_roles"]) != allowed_roles:
            raise AutoSpaceValidationError("family and policy allowed data roles differ")
        if set(family["forbidden_data_roles"]) != forbidden_roles:
            raise AutoSpaceValidationError("family and policy forbidden data roles differ")
        for primitive in primitives.values():
            record = primitive.record
            targets = set(record["legal_mutation_targets"])
            if not targets.issubset(allowed_targets):
                raise AutoSpaceValidationError(
                    f"primitive {primitive.primitive_id!r} exceeds allowed mutation targets"
                )
            if targets.intersection(frozen_targets):
                raise AutoSpaceValidationError(
                    f"primitive {primitive.primitive_id!r} targets a frozen protocol surface"
                )
            if not set(record["allowed_data_roles"]).issubset(allowed_roles):
                raise AutoSpaceValidationError(
                    f"primitive {primitive.primitive_id!r} exceeds allowed data roles"
                )
            if not set(record["prohibited_mutation_targets"]).issuperset(frozen_targets):
                raise AutoSpaceValidationError(
                    f"primitive {primitive.primitive_id!r} omits a frozen target prohibition"
                )


def _operator_contract(operator: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "operator_id": operator["operator_id"],
        "kind": operator["kind"],
        "min_arity": operator["min_arity"],
        "max_arity": operator["max_arity"],
    }


def _validate_manual_package(
    root: Path,
    manifest: Mapping[str, Any],
    family: Mapping[str, Any],
    registry: Mapping[str, Any],
    grammar: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    family_id: str,
    primitives: Mapping[str, ConceptualPrimitiveV1],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    stale = sorted(name for name in MANUAL_AUTO_EVIDENCE_FILES if (root / name).exists())
    if stale:
        raise AutoSpaceValidationError(
            f"ManualSpace package contains AutoSpace evidence: {stale}"
        )
    contract_path = root / MANUAL_CONTRACT_FILE
    provenance_path = root / MANUAL_PROVENANCE_FILE
    package_manifest_path = root / MANUAL_PACKAGE_MANIFEST_FILE
    contract = _object(load_json(contract_path), MANUAL_CONTRACT_FILE)
    provenance = _object(load_json(provenance_path), MANUAL_PROVENANCE_FILE)
    package_manifest = _object(
        load_json(package_manifest_path), MANUAL_PACKAGE_MANIFEST_FILE
    )
    validate_schema(contract, "manual_space_contract.schema.json", label=MANUAL_CONTRACT_FILE)
    validate_schema(
        provenance, "manual_space_provenance.schema.json", label=MANUAL_PROVENANCE_FILE
    )
    validate_schema(
        package_manifest,
        "manual_space_package_manifest.schema.json",
        label=MANUAL_PACKAGE_MANIFEST_FILE,
    )
    search_space_id = manifest["search_space_id"]
    if contract["manual_space_id"] != search_space_id:
        raise AutoSpaceValidationError("ManualSpace contract search-space identity mismatch")
    if contract["family_id"] != family_id:
        raise AutoSpaceValidationError("ManualSpace contract family identity mismatch")
    if contract["task_id"] != manifest["task_id"]:
        raise AutoSpaceValidationError("ManualSpace contract task identity mismatch")
    if provenance["manual_space_id"] != search_space_id:
        raise AutoSpaceValidationError("ManualSpace provenance identity mismatch")
    if package_manifest["manual_space_id"] != search_space_id:
        raise AutoSpaceValidationError("ManualSpace package manifest identity mismatch")

    design = contract["required_search_space_design"]
    slot_contracts = {item["slot_id"]: item for item in design["slots"]}
    if len(slot_contracts) != len(design["slots"]):
        raise AutoSpaceValidationError("ManualSpace contract contains duplicate slots")
    if set(slot_contracts) != {
        axis["slot_id"] for axis in registry["axes"]
    }:
        raise AutoSpaceValidationError("ManualSpace contract/core axis mismatch")
    contract_primitive_ids = {item["primitive_id"] for item in design["slots"]}
    if contract_primitive_ids != set(primitives):
        raise AutoSpaceValidationError("ManualSpace contract/core primitive mismatch")
    for primitive in primitives.values():
        slot = slot_contracts[primitive.slot_id]
        record = primitive.record
        if slot["primitive_id"] != primitive.primitive_id:
            raise AutoSpaceValidationError("ManualSpace slot primitive identity mismatch")
        if set(record["composition_roles"]) != {slot["composition_role"]}:
            raise AutoSpaceValidationError("ManualSpace primitive composition-role mismatch")
        if set(record["conceptual_capabilities"]) != {
            slot["conceptual_capability"]
        }:
            raise AutoSpaceValidationError("ManualSpace primitive capability mismatch")
        if set(record["legal_mutation_targets"]) != set(
            slot["legal_mutation_targets"]
        ):
            raise AutoSpaceValidationError("ManualSpace primitive target mismatch")
    if [_operator_contract(item) for item in grammar["operators"]] != design["operators"]:
        raise AutoSpaceValidationError("ManualSpace grammar/contract operator mismatch")
    composition = grammar["composition_policy"]
    if composition["max_primitives_per_candidate"] != design[
        "max_primitives_per_candidate"
    ]:
        raise AutoSpaceValidationError("ManualSpace primitive limit mismatch")
    if composition["max_mutation_targets"] != design["max_mutation_targets"]:
        raise AutoSpaceValidationError("ManualSpace mutation-target limit mismatch")
    if composition["custom_synthesis"] != design["custom_synthesis"]:
        raise AutoSpaceValidationError("ManualSpace custom-synthesis policy mismatch")
    if set(family["mutable_dimensions"]) != set(design["mutable_dimensions"]):
        raise AutoSpaceValidationError("ManualSpace family mutable-dimension mismatch")

    upper = contract["trusted_upper_bound"]
    exact_pairs = (
        (family["mutable_dimensions"], upper["allowed_mutable_dimensions"]),
        (policy["allowed_mutation_targets"], upper["allowed_mutation_targets"]),
        (policy["frozen_mutation_targets"], upper["frozen_mutation_targets"]),
        (policy["allowed_data_roles"], upper["allowed_data_roles"]),
        (policy["forbidden_data_roles"], upper["forbidden_data_roles"]),
        (policy["allowed_dependencies"], upper["allowed_dependencies"]),
        (family["frozen_protocol_dimensions"], upper["frozen_protocol_dimensions"]),
    )
    if any(set(left) != set(right) for left, right in exact_pairs):
        raise AutoSpaceValidationError("ManualSpace trusted boundary mismatch")
    anchor_allowlist = {
        (item["path"], item["symbol"], item["kind"])
        for item in contract["baseline_anchor_allowlist"]
    }
    for primitive in primitives.values():
        anchors = {
            (item["path"], item["symbol"], item["kind"])
            for item in primitive.record["baseline_anchors"]
        }
        if not anchors.issubset(anchor_allowlist):
            raise AutoSpaceValidationError(
                f"primitive {primitive.primitive_id!r} has an unauthorized baseline anchor"
            )
    return contract, provenance


def load_search_space_package(
    root: Path,
    *,
    expected_source: SearchSpaceSource | None = None,
    require_frozen: bool = False,
    expected_digest: str | None = None,
) -> NormalizedSearchSpaceV1:
    root = Path(root).resolve(strict=True)
    paths = _core_paths(root)
    documents = [_object(load_json(path), path.name) for path in paths]
    manifest, family, registry, grammar, policy = documents
    source_text = manifest.get("source", "MANUAL")
    try:
        source = SearchSpaceSource(source_text)
    except ValueError as error:
        raise AutoSpaceValidationError(f"unsupported search-space source {source_text!r}") from error
    if expected_source is not None and source is not expected_source:
        raise AutoSpaceValidationError(f"expected source {expected_source.value}, got {source.value}")
    legacy_manual = (
        source is SearchSpaceSource.MANUAL
        and paths[0].name == LEGACY_CORE_FILES[0]
    )
    trusted_contract: Mapping[str, Any] | None = None
    genesis_request: Mapping[str, Any] | None = None
    manual_contract: Mapping[str, Any] | None = None
    manual_provenance: Mapping[str, Any] | None = None
    frozen: Mapping[str, Any] | None = None
    if source is SearchSpaceSource.AUTO_GENERATED:
        core_schemas = dict(SCHEMA_BY_CORE_FILE)
        if manifest.get("search_space_version") in {"v1.6", "v1.6.1"}:
            suffix = "v161" if manifest.get("search_space_version") == "v1.6.1" else "v16"
            core_schemas.update(
                {
                    "auto_space_primitive_registry_v1.json": f"auto_space_{suffix}_primitive_registry.schema.json",
                    "auto_space_grammar_v1.json": f"auto_space_{suffix}_grammar.schema.json",
                }
            )
        for path in paths:
            validate_schema(load_json(path), core_schemas[path.name], label=path.name)
        request_path = root / "AUTO_SPACE_GENESIS_REQUEST.json"
        genesis_request = _object(load_json(request_path), request_path.name)
        trusted_contract = validate_embedded_genesis_request(genesis_request)
        contract_path = root / TRUSTED_CONTRACT_FILE
        copied_contract = _object(load_json(contract_path), TRUSTED_CONTRACT_FILE)
        if canonical_bytes(copied_contract) != canonical_bytes(trusted_contract):
            raise AutoSpaceValidationError("copied trusted contract differs from the audited Genesis request")
        if manifest.get("trusted_contract_sha256") != file_sha256(contract_path):
            raise AutoSpaceValidationError("manifest trusted contract identity mismatch")
        if manifest.get("genesis_request_digest") != genesis_request.get("request_digest"):
            raise AutoSpaceValidationError("manifest Genesis request identity mismatch")
        if manifest.get("task_id") != genesis_request.get("task_id"):
            raise AutoSpaceValidationError("manifest task identity differs from Genesis request")
    elif not legacy_manual:
        for path in paths:
            validate_schema(
                load_json(path),
                MANUAL_SCHEMA_BY_CORE_FILE[path.name],
                label=path.name,
            )
    search_space_id = _nonempty_string(manifest.get("search_space_id"), "manifest.search_space_id")
    family_id = _nonempty_string(family.get("family_id", manifest.get("family_id")), "family.family_id")
    if manifest.get("family_id") is not None and manifest["family_id"] != family_id:
        raise AutoSpaceValidationError("manifest and family family_id values differ")
    primitives: dict[str, ConceptualPrimitiveV1] = {}
    axes_allow_multiple: dict[str, bool] = {}
    for axis_index, raw_axis in enumerate(registry.get("axes", [])):
        axis = _object(raw_axis, f"registry.axes[{axis_index}]")
        slot_id = _nonempty_string(axis.get("slot_id"), "axis.slot_id")
        if slot_id in axes_allow_multiple:
            raise AutoSpaceValidationError(f"duplicate axis {slot_id!r}")
        axes_allow_multiple[slot_id] = bool(axis.get("allow_multiple"))
        records = axis.get("primitives")
        if not isinstance(records, list) or not records:
            raise AutoSpaceValidationError(f"axis {slot_id!r} has no primitives")
        for raw_record in records:
            record = _object(raw_record, f"primitive in {slot_id}")
            _assert_no_executable_claims(record, "conceptual primitive")
            primitive_id = _nonempty_string(record.get("primitive_id"), "primitive_id")
            if primitive_id in primitives:
                raise AutoSpaceValidationError(f"duplicate primitive_id {primitive_id!r}")
            record_slot = _nonempty_string(record.get("slot_id", slot_id), "primitive.slot_id")
            if record_slot != slot_id:
                raise AutoSpaceValidationError(f"primitive {primitive_id!r} is stored under the wrong axis")
            if record.get("family_id", family_id) != family_id:
                raise AutoSpaceValidationError(f"primitive {primitive_id!r} targets another family")
            primitives[primitive_id] = ConceptualPrimitiveV1(
                primitive_id=primitive_id,
                family_id=family_id,
                slot_id=slot_id,
                digest=canonical_sha256(record),
                record=record,
            )
    if registry.get("primitive_count") is not None and registry["primitive_count"] != len(primitives):
        raise AutoSpaceValidationError("primitive_count does not match registry contents")
    if not legacy_manual:
        _validate_core_structure(
            family,
            registry,
            grammar,
            policy,
            family_id=family_id,
            primitives=primitives,
            axes_allow_multiple=axes_allow_multiple,
            strict_policy_boundary=source is SearchSpaceSource.MANUAL,
        )
    if source is SearchSpaceSource.AUTO_GENERATED:
        primitive_records = [primitive.record for primitive in primitives.values()]
        assert genesis_request is not None and trusted_contract is not None
        validate_baseline_anchors(primitive_records, genesis_request)
        enforce_trusted_contract(
            family,
            primitive_records,
            grammar,
            policy,
            trusted_contract,
            axes=registry["axes"],
        )
        if manifest.get("search_space_version") in {"v1.6", "v1.6.1"}:
            if manifest.get("search_space_version") == "v1.6.1":
                from auto_space_v161 import (
                    QUALITY_REPORT_FILE, QUALITY_REPORT_SCHEMA, validate_recorded_quality_report
                )
            else:
                from auto_space_v16 import (
                    QUALITY_REPORT_FILE, QUALITY_REPORT_SCHEMA, validate_recorded_quality_report
                )

            quality_path = root / QUALITY_REPORT_FILE
            quality = _object(load_json(quality_path), QUALITY_REPORT_FILE)
            validate_schema(quality, QUALITY_REPORT_SCHEMA, label=QUALITY_REPORT_FILE)
            validate_recorded_quality_report(quality, registry, grammar, trusted_contract)
    elif not legacy_manual:
        manual_contract, manual_provenance = _validate_manual_package(
            root,
            manifest,
            family,
            registry,
            grammar,
            policy,
            family_id=family_id,
            primitives=primitives,
        )
    for label, document in (("manifest", manifest), ("family", family), ("grammar", grammar), ("capability policy", policy)):
        _assert_no_executable_claims(document, label)
    core = {
        "manifest": manifest,
        "family": family,
        "primitive_registry": registry,
        "grammar": grammar,
        "capability_policy": policy,
    }
    digest = canonical_sha256(core)
    if expected_digest is not None and digest != expected_digest:
        raise AutoSpaceValidationError("search-space digest does not match expected identity")
    if require_frozen:
        native_manual = source is SearchSpaceSource.MANUAL and not legacy_manual
        frozen_name = MANUAL_FROZEN_MANIFEST if native_manual else FROZEN_MANIFEST
        frozen_path = root / frozen_name
        frozen = _object(load_json(frozen_path), frozen_name)
        frozen_schema = (
            "manual_space_frozen_manifest.schema.json"
            if native_manual
            else "auto_space_frozen_manifest.schema.json"
        )
        validate_schema(frozen, frozen_schema, label=frozen_name)
        if frozen.get("status") != "FROZEN" or frozen.get("search_space_digest") != digest:
            raise AutoSpaceValidationError("frozen manifest does not bind this package")
        actual = {path.name: file_sha256(path) for path in paths}
        if frozen.get("core_file_sha256") != actual:
            raise AutoSpaceValidationError("frozen core file hashes do not match")
        if not native_manual:
            for field, name in (("provenance_sha256", "auto_space_provenance_v1.json"), ("conformance_report_sha256", CONFORMANCE_REPORT)):
                wrapper_path = root / name
                if not wrapper_path.is_file() or frozen.get(field) != file_sha256(wrapper_path):
                    raise AutoSpaceValidationError(f"frozen {name} identity does not match")
            if manifest.get("search_space_version") in {"v1.6", "v1.6.1"}:
                if manifest.get("search_space_version") == "v1.6.1":
                    from auto_space_v161 import QUALITY_REPORT_FILE
                    label = "v1.6.1"
                else:
                    from auto_space_v16 import QUALITY_REPORT_FILE
                    label = "v1.6"

                if frozen.get("quality_report_sha256") != file_sha256(root / QUALITY_REPORT_FILE):
                    raise AutoSpaceValidationError(
                        f"frozen {label} quality report identity does not match"
                    )
            generated = frozen.get("generated_output_sha256")
            actual_generated = {
                path.name: file_sha256(path)
                for path in sorted(root.iterdir())
                if path.is_file()
                and path.name not in {FROZEN_MANIFEST, POST_FREEZE_REPORT}
            }
            if generated != actual_generated:
                raise AutoSpaceValidationError("frozen generated output hashes do not match")
        else:
            assert manual_contract is not None and manual_provenance is not None
            expected_wrappers = {
                "contract_sha256": file_sha256(root / MANUAL_CONTRACT_FILE),
                "provenance_sha256": file_sha256(root / MANUAL_PROVENANCE_FILE),
                "package_manifest_sha256": file_sha256(
                    root / MANUAL_PACKAGE_MANIFEST_FILE
                ),
            }
            if any(frozen.get(field) != value for field, value in expected_wrappers.items()):
                raise AutoSpaceValidationError("frozen ManualSpace wrapper identity mismatch")
            if (
                frozen.get("source") != SearchSpaceSource.MANUAL.value
                or frozen.get("search_space_id") != search_space_id
                or frozen.get("family_id") != family_id
                or frozen.get("family_digest") != canonical_sha256(family)
                or frozen.get("axis_count") != len(axes_allow_multiple)
                or frozen.get("primitive_count") != len(primitives)
                or frozen.get("max_primitives_per_candidate")
                != grammar["composition_policy"]["max_primitives_per_candidate"]
                or frozen.get("max_mutation_targets")
                != grammar["composition_policy"]["max_mutation_targets"]
            ):
                raise AutoSpaceValidationError("frozen ManualSpace identity mismatch")
    return NormalizedSearchSpaceV1(
        root=root,
        source=source,
        manifest=manifest,
        family=family,
        primitive_registry=registry,
        grammar=grammar,
        capability_policy=policy,
        trusted_contract=trusted_contract,
        genesis_request=genesis_request,
        search_space_id=search_space_id,
        family_id=family_id,
        digest=digest,
        family_digest=canonical_sha256(family),
        grammar_digest=canonical_sha256(grammar),
        capability_policy_digest=canonical_sha256(policy),
        primitives=primitives,
        axes_allow_multiple=axes_allow_multiple,
        manual_contract=manual_contract,
        manual_provenance=manual_provenance,
        frozen_manifest=frozen,
    )


def reject_conceptual_execution(binding: Any) -> None:
    if isinstance(binding, ConceptualSearchCandidateBindingV1):
        raise AutoSpaceValidationError(
            "conceptual bindings cannot execute before implementation and qualification"
        )
