"""Deterministic quality gate and trusted assembly for AutoSpace v1.6.1."""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import re
from typing import Any, Mapping, Sequence


PROVIDER_OUTPUT_MODE = "SCIENTIFIC_PLAN_PRIMITIVES_TRUSTED_ASSEMBLY_V3"
PROVIDER_DRAFT_SCHEMA = "auto_space_v161_provider_draft.schema.json"
CRITIC_SCHEMA = "auto_space_v161_critic_merge.schema.json"
QUALITY_REPORT_SCHEMA = "auto_space_v161_quality_report.schema.json"
QUALITY_REPORT_FILE = "AUTO_SPACE_V161_QUALITY_REPORT.json"
ASSEMBLER_ID = "recclaw.auto-space.v161-trusted-assembler"
ASSEMBLER_VERSION = "v1"

REQUIRED_AXES = {
    "REPRESENTATION_TRANSFORM": (
        "representation",
        "representation",
        "representation_transform",
    ),
    "SCORING_TRANSFORM": (
        "candidate_scoring",
        "candidate_scoring",
        "candidate_scoring_transform",
    ),
    "TRAINING_OBJECTIVE_TRANSFORM": (
        "train_only_objective",
        "training_objective",
        "train_only_objective",
    ),
    "TRAIN_UPDATE_TRANSFORM": (
        "train_update_rule",
        "train_update",
        "train_update_transform",
    ),
    "MODEL_ARCHITECTURE_TRANSFORM": (
        "model_architecture",
        "model_architecture",
        "model_architecture_transform",
    ),
    "TRAIN_SAMPLING_TRANSFORM": (
        "train_only_sampling",
        "train_sampling",
        "train_only_sampling",
    ),
}
COMPOSITION_ORDER = tuple(REQUIRED_AXES)


# Trusted execution-route anchors for the frozen E2 SASRec candidate runtime.
# These are runtime structure facts, not scientific mechanism choices.
RUNTIME_ROUTE_ALLOWLIST = {
    "REPRESENTATION_TRANSFORM": {
        ("models.py", "Model", "class"),
        ("models.py", "Model.log2feats", "method"),
        ("models.py", "Model.forward", "method"),
    },
    "SCORING_TRANSFORM": {
        ("models.py", "Model", "class"),
        ("models.py", "Model.predict_score", "method"),
        ("models.py", "Model.forward", "method"),
    },
    "TRAINING_OBJECTIVE_TRANSFORM": {
        ("models.py", "Model.forward", "method"),
        ("main_code.py", "main", "function"),
    },
    "TRAIN_UPDATE_TRANSFORM": {
        ("main_code.py", "main", "function"),
    },
    "MODEL_ARCHITECTURE_TRANSFORM": {
        ("models.py", "Model", "class"),
        ("models.py", "Model.log2feats", "method"),
        ("models.py", "PointWiseFeedForward", "class"),
        ("models.py", "PointWiseFeedForward.forward", "method"),
    },
    "TRAIN_SAMPLING_TRANSFORM": {
        ("main_code.py", "main", "function"),
    },
}


def _source_symbol_index(source: str, path: str) -> set[tuple[str, str]]:
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as error:
        raise AutoSpaceV161Error(f"baseline source is not valid Python: {path}") from error
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


def _verify_static_reachability(
    primitive: Mapping[str, Any], baseline_sources: Mapping[str, str]
) -> None:
    primitive_id = str(primitive.get("primitive_id", "<unknown>"))
    slot_id = primitive.get("slot_id")
    if slot_id not in RUNTIME_ROUTE_ALLOWLIST:
        raise AutoSpaceV161Error(f"primitive {primitive_id!r} uses an unknown runtime axis")
    reachability = _object(primitive.get("runtime_reachability"), "runtime_reachability")
    if reachability.get("statically_reachable") is not True:
        raise AutoSpaceV161Error(f"primitive {primitive_id!r} does not claim a reachable path")
    anchor = _object(reachability.get("baseline_anchor"), "runtime_reachability.baseline_anchor")
    key = (anchor.get("path"), anchor.get("symbol"), anchor.get("kind"))
    if key not in RUNTIME_ROUTE_ALLOWLIST[slot_id]:
        raise AutoSpaceV161Error(
            f"primitive {primitive_id!r} anchor is not on a trusted execution route for {slot_id}"
        )
    path = anchor.get("path")
    if path not in baseline_sources:
        raise AutoSpaceV161Error(f"primitive {primitive_id!r} baseline source is missing: {path}")
    if (anchor.get("symbol"), anchor.get("kind")) not in _source_symbol_index(
        baseline_sources[path], str(path)
    ):
        raise AutoSpaceV161Error(
            f"primitive {primitive_id!r} baseline symbol is not present: {path}:{anchor.get('symbol')}"
        )
    target = reachability.get("mutation_target")
    if target not in primitive.get("legal_mutation_targets", []):
        raise AutoSpaceV161Error(f"primitive {primitive_id!r} reachability target is not legal")
    if target == "main_code.py:train_update_kernel" and key != ("main_code.py", "main", "function"):
        raise AutoSpaceV161Error(f"primitive {primitive_id!r} train-update target is disconnected from main")
    if target == "models.py" and path != "models.py":
        raise AutoSpaceV161Error(f"primitive {primitive_id!r} model target is disconnected from models.py")


class AutoSpaceV161Error(ValueError):
    """The proposed v1.6.1 space is not scientifically admissible."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AutoSpaceV161Error(f"{label} must be an object")
    return dict(value)


def _strings(value: Any, label: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise AutoSpaceV161Error(f"{label} must be an array of non-empty strings")
    result = tuple(item.strip() for item in value)
    if nonempty and not result:
        raise AutoSpaceV161Error(f"{label} must not be empty")
    if len(result) != len(set(result)):
        raise AutoSpaceV161Error(f"{label} contains duplicates")
    return result


def _normalized(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))



def verify_materialized_reachability(
    primitives: Sequence[Mapping[str, Any]], baseline_sources: Mapping[str, str]
) -> None:
    for primitive in primitives:
        _verify_static_reachability(primitive, baseline_sources)

def validate_contract_design(contract: Mapping[str, Any]) -> dict[str, Any]:
    generation = _object(contract.get("generation_policy"), "generation_policy")
    if generation.get("provider_output_mode") != PROVIDER_OUTPUT_MODE:
        raise AutoSpaceV161Error("v1.6.1 requires its explicit provider output mode")
    design = _object(
        contract.get("required_search_space_design"),
        "required_search_space_design",
    )
    if design.get("content_version") != "1.6.1":
        raise AutoSpaceV161Error("v1.6.1 content_version drift")
    slots = design.get("slots")
    if not isinstance(slots, list) or len(slots) != len(REQUIRED_AXES):
        raise AutoSpaceV161Error("v1.6.1 must declare exactly six required axes")
    by_id: dict[str, dict[str, Any]] = {}
    for raw in slots:
        slot = _object(raw, "v1.6.1 slot")
        slot_id = slot.get("slot_id")
        if not isinstance(slot_id, str) or slot_id in by_id:
            raise AutoSpaceV161Error("v1.6.1 slot IDs must be unique strings")
        by_id[slot_id] = slot
    if set(by_id) != set(REQUIRED_AXES):
        raise AutoSpaceV161Error("v1.6.1 required axis set drift")
    for slot_id, expected in REQUIRED_AXES.items():
        slot = by_id[slot_id]
        if (
            slot.get("mutable_dimension"),
            slot.get("composition_role"),
            slot.get("conceptual_capability"),
        ) != expected:
            raise AutoSpaceV161Error(f"v1.6.1 slot semantics drift: {slot_id}")
        _strings(
            slot.get("legal_mutation_targets"),
            f"{slot_id}.legal_mutation_targets",
            nonempty=True,
        )
    upper = _object(contract.get("trusted_upper_bound"), "trusted_upper_bound")
    allowed_targets = set(
        _strings(upper.get("allowed_mutation_targets"), "allowed_mutation_targets", nonempty=True)
    )
    frozen_targets = set(
        _strings(upper.get("frozen_mutation_targets"), "frozen_mutation_targets", nonempty=True)
    )
    for slot_id, slot in by_id.items():
        targets = set(slot["legal_mutation_targets"])
        if not targets.issubset(allowed_targets) or targets.intersection(frozen_targets):
            raise AutoSpaceV161Error(f"v1.6.1 slot {slot_id} exceeds the trusted mutation boundary")
    expected_dimensions = {item[0] for item in REQUIRED_AXES.values()}
    if set(_strings(design.get("mutable_dimensions"), "mutable_dimensions")) != expected_dimensions:
        raise AutoSpaceV161Error("v1.6.1 mutable dimensions drift")
    if not expected_dimensions.issubset(
        _strings(upper.get("allowed_mutable_dimensions"), "allowed_mutable_dimensions", nonempty=True)
    ):
        raise AutoSpaceV161Error("v1.6.1 mutable dimensions exceed the trusted upper bound")
    target = _object(design.get("target_primitive_count"), "target_primitive_count")
    if target != {"minimum": 12, "maximum": 16}:
        raise AutoSpaceV161Error("v1.6.1 final primitive-count bounds must be 12..16")
    if design.get("proposal_pool_primitive_count") != 18:
        raise AutoSpaceV161Error("v1.6.1 proposal pool must contain exactly 18 primitives")
    if design.get("proposal_mechanism_families_per_axis") != 3:
        raise AutoSpaceV161Error("v1.6.1 proposal pool must contain exactly three mechanism families per axis")
    if design.get("minimum_distinct_mechanism_families_per_axis") != 2:
        raise AutoSpaceV161Error("v1.6.1 final space requires two mechanism families per axis")
    if design.get("max_primitives_per_candidate") != 3:
        raise AutoSpaceV161Error("v1.6.1 max_primitives_per_candidate must be 3")
    if design.get("max_mutation_targets") != 2:
        raise AutoSpaceV161Error("v1.6.1 max_mutation_targets must be 2")
    raw_operators = design.get("operators")
    if not isinstance(raw_operators, list) or len(raw_operators) != 2:
        raise AutoSpaceV161Error("v1.6.1 must declare exactly two operators")
    operators = {
        item.get("kind"): item
        for item in raw_operators
        if isinstance(item, Mapping)
    }
    if set(operators) != {"SELECT", "COMPOSE"}:
        raise AutoSpaceV161Error("v1.6.1 permits only SELECT and COMPOSE")
    if (
        operators["SELECT"].get("operator_id"),
        operators["SELECT"].get("min_arity"),
        operators["SELECT"].get("max_arity"),
    ) != ("select", 1, 1):
        raise AutoSpaceV161Error("v1.6.1 SELECT contract drift")
    if (
        operators["COMPOSE"].get("operator_id"),
        operators["COMPOSE"].get("min_arity"),
        operators["COMPOSE"].get("max_arity"),
    ) != ("compose", 2, 3):
        raise AutoSpaceV161Error("v1.6.1 COMPOSE contract drift")
    custom = _object(design.get("custom_synthesis"), "custom_synthesis")
    if custom != {"enabled": False, "operator_id": None, "max_components": 0, "allowed_slots": []}:
        raise AutoSpaceV161Error("v1.6.1 custom synthesis must be disabled")
    guard = _object(design.get("effective_width_guard"), "effective_width_guard")
    minimum = guard.get("minimum_legal_candidate_count")
    ratio = guard.get("minimum_retention_ratio")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        raise AutoSpaceV161Error("v1.6.1 effective-width guard must be a positive integer")
    if not isinstance(ratio, (int, float)) or isinstance(ratio, bool) or not 0 < float(ratio) <= 1:
        raise AutoSpaceV161Error("v1.6.1 effective-width retention ratio must be in (0, 1]")
    return design


def _parameter_schema_is_const_only(schema: Mapping[str, Any]) -> bool:
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        return False
    properties = schema.get("properties")
    required = schema.get("required", [])
    if not isinstance(properties, Mapping) or not isinstance(required, list):
        return False
    if set(required) != set(properties):
        return False
    return all(isinstance(spec, Mapping) and "const" in spec for spec in properties.values())


def novelty_signature(primitive: Mapping[str, Any]) -> str:
    """Ignore names, prose and parameters when identifying semantic duplicates."""

    effect = _object(primitive.get("effect_path"), "effect_path")
    anchor = _object(effect.get("baseline_anchor"), "effect_path.baseline_anchor")
    identity = {
        "slot_id": primitive.get("slot_id"),
        "effect_phase": primitive.get("effect_phase"),
        "operation_kind": _normalized(str(effect.get("operation_kind", ""))),
        "changed_quantity": _normalized(str(effect.get("changed_quantity", ""))),
        "effect_anchor": {
            "path": anchor.get("path"),
            "symbol": anchor.get("symbol"),
            "kind": anchor.get("kind"),
        },
        "legal_mutation_targets": sorted(primitive.get("legal_mutation_targets", [])),
        "conceptual_capabilities": sorted(primitive.get("conceptual_capabilities", [])),
    }
    return _sha256(identity)


def _conflicts(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_compatible = set(left.get("compatibility_tags", []))
    right_compatible = set(right.get("compatibility_tags", []))
    if bool(
        set(left.get("incompatibility_tags", [])).intersection(right_compatible)
        or set(right.get("incompatibility_tags", [])).intersection(left_compatible)
    ):
        return True
    left_requirements = {_normalized(item) for item in left.get("implementation_requirements", [])}
    right_requirements = {_normalized(item) for item in right.get("implementation_requirements", [])}
    for requirements, other in (
        (left_requirements, right_requirements),
        (right_requirements, left_requirements),
    ):
        required = {item.removeprefix("requires ") for item in requirements if item.startswith("requires ")}
        forbidden = {item.removeprefix("forbids ") for item in other if item.startswith("forbids ")}
        if required.intersection(forbidden):
            return True
    return False


def _legal_composition(records: Sequence[Mapping[str, Any]]) -> bool:
    if not 1 <= len(records) <= 3:
        return False
    if len({item["slot_id"] for item in records}) != len(records):
        return False
    targets = {
        target for item in records for target in item["legal_mutation_targets"]
    }
    if len(targets) > 2:
        return False
    return not any(_conflicts(left, right) for left, right in itertools.combinations(records, 2))


UNRESOLVED_CHOICE_PATTERN = re.compile(
    r"\b(?:either|choose|choice\s+between|one\s+of|or\s+(?:a|an|the|smooth|hinge|equivalent))\b",
    re.IGNORECASE,
)


def _contains_unresolved_choice(primitive: Mapping[str, Any]) -> bool:
    fields = [
        str(primitive.get("semantic_role", "")),
        str(primitive.get("rationale", "")),
        *[str(item) for item in primitive.get("implementation_requirements", [])],
    ]
    return any(UNRESOLVED_CHOICE_PATTERN.search(text) for text in fields)


def _sampling_bucket_spec_is_incomplete(primitive: Mapping[str, Any]) -> bool:
    if primitive.get("slot_id") != "TRAIN_SAMPLING_TRANSFORM":
        return False
    text = _normalized(json.dumps({
        "effect_path": primitive.get("effect_path", {}),
        "requirements": primitive.get("implementation_requirements", []),
        "parameter_schema": primitive.get("parameter_schema", {}),
    }, ensure_ascii=False))
    if "bucket" not in text:
        return False
    properties = _object(primitive.get("parameter_schema"), "parameter_schema").get("properties", {})
    names = {_normalized(str(name)).replace(" ", "_") for name in properties}
    has_boundaries = any(any(token in name for token in ("boundary", "boundaries", "cutoff", "threshold")) for name in names)
    has_weights = any(any(token in name for token in ("prob", "probability", "weight", "mass")) for name in names)
    return not (has_boundaries and has_weights)


def _scalar_only_architecture(primitive: Mapping[str, Any]) -> bool:
    if primitive.get("slot_id") != "MODEL_ARCHITECTURE_TRANSFORM":
        return False
    effect = _object(primitive.get("effect_path"), "effect_path")
    text = _normalized(" ".join([
        str(effect.get("operation_kind", "")),
        str(effect.get("changed_quantity", "")),
        *[str(item) for item in primitive.get("implementation_requirements", [])],
    ]))
    props = _object(primitive.get("parameter_schema"), "parameter_schema").get("properties", {})
    scalar_signal = any(token in text.split() for token in ("scale", "scaling", "multiply", "multiplier", "gate", "gated"))
    residual_signal = "residual" in text.split()
    structural_signal = any(token in text.split() for token in ("branch", "adapter", "bottleneck", "head", "attention", "convolution", "mixing"))
    return len(props) <= 1 and scalar_signal and residual_signal and not structural_signal


def quality_report(
    primitives: Sequence[Mapping[str, Any]], design: Mapping[str, Any]
) -> dict[str, Any]:
    if not 12 <= len(primitives) <= 16:
        raise AutoSpaceV161Error("v1.6.1 requires 12..16 accepted primitives")
    grouped: dict[str, list[Mapping[str, Any]]] = {axis: [] for axis in REQUIRED_AXES}
    ids: set[str] = set()
    signatures: set[str] = set()
    for primitive in primitives:
        primitive_id = primitive.get("primitive_id")
        if not isinstance(primitive_id, str) or not primitive_id or primitive_id in ids:
            raise AutoSpaceV161Error("v1.6.1 primitive IDs must be unique non-empty strings")
        ids.add(primitive_id)
        slot_id = primitive.get("slot_id")
        if slot_id not in grouped:
            raise AutoSpaceV161Error(f"primitive {primitive_id!r} uses an unknown axis")
        if _contains_unresolved_choice(primitive):
            raise AutoSpaceV161Error(f"primitive {primitive_id!r} contains an unresolved implementation choice")
        if _sampling_bucket_spec_is_incomplete(primitive):
            raise AutoSpaceV161Error(f"primitive {primitive_id!r} has an incompletely specified bucketed sampling distribution")
        if _scalar_only_architecture(primitive):
            raise AutoSpaceV161Error(f"primitive {primitive_id!r} is only a scalar residual-strength architecture variant")
        if not _parameter_schema_is_const_only(_object(primitive.get("parameter_schema"), "parameter_schema")):
            raise AutoSpaceV161Error(f"primitive {primitive_id!r} must use const-only parameters")
        if primitive.get("runtime_reachability", {}).get("statically_reachable") is not True:
            raise AutoSpaceV161Error(f"primitive {primitive_id!r} lacks a reachable effect path")
        if primitive["slot_id"] in {
            "TRAINING_OBJECTIVE_TRANSFORM",
            "TRAIN_UPDATE_TRANSFORM",
            "TRAIN_SAMPLING_TRANSFORM",
        } and primitive.get("effect_phase") != "TRAIN_ONLY":
            raise AutoSpaceV161Error(
                f"primitive {primitive_id!r} violates its train-only axis phase"
            )
        if primitive.get("expected_effect_strength") not in {"LOW", "MEDIUM", "HIGH"}:
            raise AutoSpaceV161Error(f"primitive {primitive_id!r} has invalid effect strength")
        effect = _object(primitive.get("effect_path"), "effect_path")
        operation = _normalized(str(effect.get("operation_kind", "")))
        quantity = _normalized(str(effect.get("changed_quantity", "")))
        # "Identity of supervised positions" names what changes, not an
        # identity operation. Retain rejection of declared no-op operations
        # and unchanged/identity-map quantities without conflating that noun.
        no_op_quantity = (
            any(token in quantity.split() for token in ("noop", "unchanged"))
            or re.search(r"\bidentity(?:$| (?:map(?:ping)?|transform(?:ation)?|operator|operation|function)\b)", quantity)
        )
        if any(token in operation.split() for token in ("identity", "noop", "unchanged")) or no_op_quantity:
            raise AutoSpaceV161Error(f"primitive {primitive_id!r} is identity-equivalent")
        ranking_preserving = "RANKING_PRESERVING_INFERENCE" in primitive.get(
            "compatibility_tags", []
        )
        if (
            primitive["slot_id"] == "SCORING_TRANSFORM"
            and ranking_preserving
            and primitive.get("effect_phase") == "INFERENCE_ONLY"
            and primitive["low_impact_risk"] is not True
        ):
            raise AutoSpaceV161Error(
                f"primitive {primitive_id!r} understates inference-only ranking-preserving risk"
            )
        if set(primitive.get("compatibility_tags", [])).intersection(
            primitive.get("incompatibility_tags", [])
        ):
            raise AutoSpaceV161Error(f"primitive {primitive_id!r} contradicts its own compatibility tags")
        requirements = {
            _normalized(item)
            for item in primitive.get("implementation_requirements", [])
        }
        required = {
            item.removeprefix("requires ")
            for item in requirements
            if item.startswith("requires ")
        }
        forbidden = {
            item.removeprefix("forbids ")
            for item in requirements
            if item.startswith("forbids ")
        }
        if required.intersection(forbidden):
            raise AutoSpaceV161Error(
                f"primitive {primitive_id!r} has contradictory implementation requirements"
            )
        observed = novelty_signature(primitive)
        if primitive.get("novelty_signature") != observed:
            raise AutoSpaceV161Error(f"primitive {primitive_id!r} has a stale novelty signature")
        if observed in signatures:
            raise AutoSpaceV161Error("parameter-only, renamed, or equivalent primitives are not distinct")
        signatures.add(observed)
        grouped[slot_id].append(primitive)
    axis_reports = []
    for slot_id, records in grouped.items():
        families = {_normalized(str(item["mechanism_family"])) for item in records}
        if not 2 <= len(records) <= 3 or len(families) < 2:
            raise AutoSpaceV161Error(f"axis {slot_id} lacks two distinct mechanism families")
        meaningful = [
            item for item in records
            if not item["low_impact_risk"] and item["expected_effect_strength"] != "LOW"
        ]
        if not meaningful:
            raise AutoSpaceV161Error(f"axis {slot_id} contains only low-impact mechanisms")
        axis_reports.append(
            {
                "slot_id": slot_id,
                "primitive_count": len(records),
                "distinct_mechanism_family_count": len(families),
                "meaningful_effect_count": len(meaningful),
                "status": "PASS",
            }
        )
    singles = len(primitives)
    unconstrained_pairs = sum(
        len({item["slot_id"] for item in items}) == 2
        for items in itertools.combinations(primitives, 2)
    )
    unconstrained_triples = sum(
        len({item["slot_id"] for item in items}) == 3
        for items in itertools.combinations(primitives, 3)
    )
    unconstrained_total = singles + unconstrained_pairs + unconstrained_triples
    pairs = sum(_legal_composition(items) for items in itertools.combinations(primitives, 2))
    triples = sum(_legal_composition(items) for items in itertools.combinations(primitives, 3))
    total = singles + pairs + triples
    guard = design["effective_width_guard"]
    minimum = guard["minimum_legal_candidate_count"]
    minimum_ratio = float(guard["minimum_retention_ratio"])
    retention_ratio = total / unconstrained_total if unconstrained_total else 0.0
    if total < minimum or retention_ratio < minimum_ratio:
        raise AutoSpaceV161Error(
            "effective legal width collapsed below the trusted guard: "
            f"legal={total}, unconstrained={unconstrained_total}, "
            f"retention={retention_ratio:.6f}, minimum_count={minimum}, "
            f"minimum_retention={minimum_ratio:.6f}"
        )
    return {
        "schema": "recclaw.auto-space.v161-quality-report.v1",
        "status": "PASS",
        "search_space_version": "v1.6.1",
        "axis_reports": axis_reports,
        "proposal_pool_primitive_count": 18,
        "critic_accepted_primitive_count": len(primitives),
        "critic_rejected_primitive_count": 18 - len(primitives),
        "critic_pruning_applied": True,
        "primitive_count": len(primitives),
        "distinct_novelty_signature_count": len(signatures),
        "unconstrained_candidate_counts": {
            "singles": singles,
            "pairs": unconstrained_pairs,
            "triples": unconstrained_triples,
            "total": unconstrained_total,
        },
        "legal_candidate_counts": {
            "singles": singles,
            "pairs": pairs,
            "triples": triples,
            "total": total,
        },
        "effective_width_guard": {
            "minimum_legal_candidate_count": minimum,
            "minimum_retention_ratio": minimum_ratio,
            "unconstrained": unconstrained_total,
            "observed": total,
            "retention_ratio": retention_ratio,
            "status": "PASS",
        },
    }


def _critic_acceptance(
    generator: Mapping[str, Any], critic: Mapping[str, Any]
) -> list[dict[str, Any]]:
    if critic.get("status") != "PASS":
        raise AutoSpaceV161Error("v1.6.1 critic rejected the generated search space")
    audits = _object(critic.get("audits"), "critic audits")
    required = {
        "coverage",
        "redundancy",
        "runtime_reachability",
        "effect_strength",
        "composability",
    }
    if set(audits) != required:
        raise AutoSpaceV161Error("v1.6.1 critic must report all five audits")
    for name, audit in audits.items():
        if not isinstance(audit, Mapping) or audit.get("status") != "PASS":
            raise AutoSpaceV161Error(f"v1.6.1 critic audit failed: {name}")
        axis_findings = audit.get("axis_findings")
        if not isinstance(axis_findings, list):
            raise AutoSpaceV161Error(f"v1.6.1 critic audit lacks axis findings: {name}")
        axis_status = {
            item.get("slot_id"): item.get("status")
            for item in axis_findings
            if isinstance(item, Mapping)
        }
        if set(axis_status) != set(REQUIRED_AXES) or set(axis_status.values()) != {"PASS"}:
            raise AutoSpaceV161Error(f"v1.6.1 critic axis audit failed: {name}")
    primitives = generator.get("primitives")
    if not isinstance(primitives, list):
        raise AutoSpaceV161Error("generator primitives must be an array")
    by_id = {item.get("primitive_id"): item for item in primitives if isinstance(item, Mapping)}
    if len(by_id) != len(primitives) or None in by_id:
        raise AutoSpaceV161Error("generator primitive IDs must be unique")
    accepted = _strings(critic.get("accepted_primitive_ids"), "accepted_primitive_ids", nonempty=True)
    rejected_records = critic.get("rejected_primitives")
    if not isinstance(rejected_records, list):
        raise AutoSpaceV161Error("rejected_primitives must be an array")
    rejected = {
        item.get("primitive_id")
        for item in rejected_records
        if isinstance(item, Mapping)
    }
    if len(rejected) != len(rejected_records) or None in rejected:
        raise AutoSpaceV161Error("critic rejected primitive IDs must be unique")
    for name, audit in audits.items():
        audit_rejected = set(audit.get("rejected_primitive_ids", []))
        reasons = audit.get("rejection_reasons", [])
        if not audit_rejected.issubset(rejected) or (audit_rejected and not reasons):
            raise AutoSpaceV161Error(
                f"v1.6.1 critic rejection evidence is inconsistent: {name}"
            )
    if set(accepted).intersection(rejected) or set(accepted).union(rejected) != set(by_id):
        raise AutoSpaceV161Error("critic must classify every generator primitive exactly once")
    if len(by_id) != 18:
        raise AutoSpaceV161Error("v1.6.1 generator must submit exactly 18 proposal primitives")
    if not 12 <= len(accepted) <= 16:
        raise AutoSpaceV161Error("v1.6.1 critic must retain 12..16 final primitives")
    accepted_by_axis = {axis: 0 for axis in REQUIRED_AXES}
    proposed_by_axis = {axis: 0 for axis in REQUIRED_AXES}
    for item in by_id.values():
        slot_id = item.get("slot_id")
        if slot_id in proposed_by_axis:
            proposed_by_axis[slot_id] += 1
    for primitive_id in accepted:
        slot_id = by_id[primitive_id].get("slot_id")
        if slot_id in accepted_by_axis:
            accepted_by_axis[slot_id] += 1
    if set(proposed_by_axis.values()) != {3}:
        raise AutoSpaceV161Error("v1.6.1 generator must propose exactly three primitives per axis")
    if any(count < 2 or count > 3 for count in accepted_by_axis.values()):
        raise AutoSpaceV161Error("v1.6.1 critic pruning must retain two or three primitives per axis")
    return [dict(by_id[item]) for item in accepted]


def assemble(
    generator: Mapping[str, Any],
    critic: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    baseline_sources: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assemble trusted control-plane records without inventing scientific content."""

    design = validate_contract_design(contract)
    accepted = _critic_acceptance(generator, critic)
    upper = _object(contract.get("trusted_upper_bound"), "trusted_upper_bound")
    frozen_targets = list(upper["frozen_mutation_targets"])
    slots = {item["slot_id"]: item for item in design["slots"]}
    planner = _object(generator.get("space_plan"), "space_plan")
    planned_axes = planner.get("axes")
    if not isinstance(planned_axes, list):
        raise AutoSpaceV161Error("space_plan.axes must be an array")
    plan_text = json.dumps(planner, ensure_ascii=False)
    if "```" in plan_text or re.search(
        r"(?:^|\\n)\\s*(?:def|class|import|from)\\s+", plan_text
    ):
        raise AutoSpaceV161Error("Stage A planner must not emit executable code")
    plan_by_axis = {
        item.get("slot_id"): item
        for item in planned_axes
        if isinstance(item, Mapping)
    }
    if len(plan_by_axis) != len(planned_axes) or set(plan_by_axis) != set(REQUIRED_AXES):
        raise AutoSpaceV161Error("planner must cover exactly the six v1.6.1 axes")
    planned_families: dict[str, dict[str, Mapping[str, Any]]] = {}
    for slot_id, plan in plan_by_axis.items():
        families = plan.get("mechanism_families")
        if not isinstance(families, list) or len(families) != 3:
            raise AutoSpaceV161Error(f"planner axis {slot_id} must propose exactly three mechanism families")
        named = {
            _normalized(str(item.get("mechanism_family", ""))): item
            for item in families
            if isinstance(item, Mapping)
        }
        if "" in named or len(named) != len(families):
            raise AutoSpaceV161Error(f"planner axis {slot_id} contains duplicate families")
        allowed = set(slots[slot_id]["legal_mutation_targets"])
        if any(item.get("intended_legal_mutation_target") not in allowed for item in named.values()):
            raise AutoSpaceV161Error(f"planner axis {slot_id} exceeds its trusted target boundary")
        planned_families[slot_id] = named

    generator_pairs = [
        (item.get("slot_id"), _normalized(str(item.get("mechanism_family", ""))))
        for item in generator.get("primitives", [])
        if isinstance(item, Mapping)
    ]
    planned_pairs = {
        (slot_id, family)
        for slot_id, families in planned_families.items()
        for family in families
    }
    if len(generator_pairs) != len(set(generator_pairs)) or set(generator_pairs) != planned_pairs:
        raise AutoSpaceV161Error(
            "Stage B must produce exactly one primitive for every Stage A mechanism family"
        )

    family_draft = _object(generator.get("family"), "family")
    family_id = family_draft["family_id"]
    final_primitives: list[dict[str, Any]] = []
    trusted_fields = {
        "version",
        "family_id",
        "implementation_status",
        "conceptual_capabilities",
        "composition_roles",
        "prohibited_mutation_targets",
        "novelty_signature",
    }
    for draft in accepted:
        supplied_trusted = sorted(trusted_fields.intersection(draft))
        if supplied_trusted:
            raise AutoSpaceV161Error(
                f"provider primitive {draft.get('primitive_id')!r} authors trusted fields: {supplied_trusted}"
            )
        slot_id = draft["slot_id"]
        if slot_id not in slots:
            raise AutoSpaceV161Error(f"primitive {draft['primitive_id']!r} uses an unknown axis")
        family_key = _normalized(draft["mechanism_family"])
        if family_key not in planned_families[slot_id]:
            raise AutoSpaceV161Error(f"primitive {draft['primitive_id']!r} was not planned")
        plan = planned_families[slot_id][family_key]
        slot = slots[slot_id]
        targets = _strings(
            draft.get("legal_mutation_targets"),
            f"{draft['primitive_id']}.legal_mutation_targets",
            nonempty=True,
        )
        if not set(targets).issubset(slot["legal_mutation_targets"]):
            raise AutoSpaceV161Error(f"primitive {draft['primitive_id']!r} exceeds its trusted target boundary")
        if plan["intended_legal_mutation_target"] not in targets:
            raise AutoSpaceV161Error(
                f"primitive {draft['primitive_id']!r} does not implement its planned target"
            )
        if draft.get("effect_phase") != plan.get("effect_phase"):
            raise AutoSpaceV161Error(
                f"primitive {draft['primitive_id']!r} changes its planned effect phase"
            )
        reachability = _object(draft.get("runtime_reachability"), "runtime_reachability")
        effect_path = _object(draft.get("effect_path"), "effect_path")
        anchors = draft["baseline_anchors"]
        anchor_keys = {(item["path"], item["symbol"], item["kind"]) for item in anchors}
        reach_key = (
            reachability.get("baseline_anchor", {}).get("path"),
            reachability.get("baseline_anchor", {}).get("symbol"),
            reachability.get("baseline_anchor", {}).get("kind"),
        )
        effect_key = (
            effect_path.get("baseline_anchor", {}).get("path"),
            effect_path.get("baseline_anchor", {}).get("symbol"),
            effect_path.get("baseline_anchor", {}).get("kind"),
        )
        if reach_key not in anchor_keys or effect_key not in anchor_keys:
            raise AutoSpaceV161Error(f"primitive {draft['primitive_id']!r} has a disconnected baseline anchor")
        if reach_key != effect_key:
            raise AutoSpaceV161Error(
                f"primitive {draft['primitive_id']!r} effect and runtime anchors disagree"
            )
        planned_anchor = plan["intended_baseline_anchor"]
        planned_key = (
            planned_anchor.get("path"),
            planned_anchor.get("symbol"),
            planned_anchor.get("kind"),
        )
        if planned_key not in anchor_keys:
            raise AutoSpaceV161Error(
                f"primitive {draft['primitive_id']!r} does not implement its planned anchor"
            )
        if planned_key != reach_key:
            raise AutoSpaceV161Error(
                f"primitive {draft['primitive_id']!r} changes its planned runtime anchor"
            )
        if reachability.get("mutation_target") not in targets or effect_path.get("mutation_target") not in targets:
            raise AutoSpaceV161Error(f"primitive {draft['primitive_id']!r} has a disconnected mutation target")
        for target in targets:
            if target == "models.py" and not any(key[0] == "models.py" for key in anchor_keys):
                raise AutoSpaceV161Error(
                    f"primitive {draft['primitive_id']!r} does not anchor the executed model path"
                )
            if target == "main_code.py:train_update_kernel" and (
                "main_code.py", "main", "function"
            ) not in anchor_keys:
                raise AutoSpaceV161Error(
                    f"primitive {draft['primitive_id']!r} does not anchor the executed train-update path"
                )
        _verify_static_reachability(draft, baseline_sources)
        primitive = {
            **draft,
            "version": "1.6.1",
            "family_id": family_id,
            "implementation_status": "CONCEPTUAL_UNIMPLEMENTED",
            "conceptual_capabilities": [slot["conceptual_capability"]],
            "composition_roles": [slot["composition_role"]],
            "prohibited_mutation_targets": frozen_targets,
        }
        primitive["novelty_signature"] = novelty_signature(primitive)
        final_primitives.append(primitive)

    final_primitives.sort(
        key=lambda item: (COMPOSITION_ORDER.index(item["slot_id"]), item["primitive_id"])
    )
    report = quality_report(final_primitives, design)
    axes = [
        {
            "slot_id": slot_id,
            "allow_multiple": False,
            "primitives": [
                item for item in final_primitives if item["slot_id"] == slot_id
            ],
        }
        for slot_id in COMPOSITION_ORDER
    ]
    incompatible_sets = [
        [left["primitive_id"], right["primitive_id"]]
        for left, right in itertools.combinations(final_primitives, 2)
        if _conflicts(left, right)
    ]
    roles = [item[1] for item in REQUIRED_AXES.values()]
    operators = [
        {
            "operator_id": "select",
            "kind": "SELECT",
            "min_arity": 1,
            "max_arity": 1,
            "requires_target": False,
            "requires_replacement": False,
            "required_capabilities": [],
            "allowed_composition_roles": roles,
            "port_compatibility": "NOT_REQUIRED",
        },
        {
            "operator_id": "compose",
            "kind": "COMPOSE",
            "min_arity": 2,
            "max_arity": 3,
            "requires_target": False,
            "requires_replacement": False,
            "required_capabilities": [],
            "allowed_composition_roles": roles,
            "port_compatibility": "NOT_REQUIRED",
        },
    ]
    allowed_targets = sorted({target for item in final_primitives for target in item["legal_mutation_targets"]})
    family = {
        "record_type": "search_space_family",
        "family_id": family_id,
        "family_version": "1.6.1",
        "description": family_draft["description"],
        "scientific_boundary": family_draft["scientific_boundary"],
        "baseline_contract": {
            "task_id": contract["task_id"],
            "baseline_model": contract["baseline"],
            "required_anchor_paths": sorted(
                {
                    f"{anchor['path']}:{anchor['symbol']}"
                    for item in final_primitives
                    for anchor in item["baseline_anchors"]
                }
            ),
        },
        "mutable_dimensions": list(design["mutable_dimensions"]),
        "frozen_protocol_dimensions": list(upper["frozen_protocol_dimensions"]),
        "allowed_data_roles": list(upper["allowed_data_roles"]),
        "forbidden_data_roles": list(upper["forbidden_data_roles"]),
        "output_contract": {
            "type_ref": contract["candidate_abi"]["type_ref"],
            "candidate_universe_effect": "PRESERVE",
            "semantics": "Conceptual programs require downstream implementation and qualification.",
        },
        "type_compatibility": family_draft["type_compatibility"],
        "invariants": family_draft["invariants"],
    }
    response = {
        "schema": "recclaw.auto-space.generator-response.v1.6.1",
        "search_space_id": generator["search_space_id"],
        "analysis": generator["analysis"],
        "family": family,
        "primitive_registry": {
            "record_type": "search_space_primitive_registry",
            "registry_id": f"{family_id}.registry",
            "registry_version": "1.6.1",
            "axis_count": 6,
            "primitive_count": len(final_primitives),
            "axes": axes,
        },
        "grammar": {
            "record_type": "search_space_grammar",
            "grammar_id": f"{family_id}.grammar",
            "grammar_version": "1.6.1",
            "operators": operators,
            "primitive_ids": [item["primitive_id"] for item in final_primitives],
            "composition_policy": {
                "max_primitives_per_candidate": 3,
                "cross_family_composition": "DENY",
                "incompatible_primitive_sets": incompatible_sets,
                "incompatible_composition_role_pairs": [],
                "max_mutation_targets": 2,
                "custom_synthesis": {
                    "enabled": False,
                    "operator_id": None,
                    "max_components": 0,
                    "allowed_slots": [],
                },
                "composition_order": list(COMPOSITION_ORDER),
                "effective_width_guard": report["effective_width_guard"],
            },
        },
        "capability_policy": {
            "record_type": "search_space_capability_policy",
            "policy_id": f"{family_id}.policy",
            "policy_version": "1.6.1",
            "legal_capability_families": [family_id],
            "allowed_mutation_targets": allowed_targets,
            "frozen_mutation_targets": list(upper["frozen_mutation_targets"]),
            "allowed_data_roles": list(upper["allowed_data_roles"]),
            "forbidden_data_roles": list(upper["forbidden_data_roles"]),
            "allowed_dependencies": list(upper["allowed_dependencies"]),
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
        },
    }
    return response, report


def validate_materialized(
    registry: Mapping[str, Any], grammar: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    design = validate_contract_design(contract)
    primitives = [
        item
        for axis in registry.get("axes", [])
        for item in axis.get("primitives", [])
    ]
    if grammar.get("composition_policy", {}).get("composition_order") != list(COMPOSITION_ORDER):
        raise AutoSpaceV161Error("v1.6.1 composition order drift")
    return quality_report(primitives, design)


def validate_recorded_quality_report(
    recorded: Mapping[str, Any],
    registry: Mapping[str, Any],
    grammar: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    expected = validate_materialized(registry, grammar, contract)
    if _canonical_bytes(recorded) != _canonical_bytes(expected):
        raise AutoSpaceV161Error("v1.6.1 quality report is stale")
    return expected


def canonicalize_selection(
    primitive_ids: Sequence[str], primitives: Mapping[str, Any]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            primitive_ids,
            key=lambda primitive_id: (
                COMPOSITION_ORDER.index(primitives[primitive_id].slot_id),
                primitive_id,
            ),
        )
    )
