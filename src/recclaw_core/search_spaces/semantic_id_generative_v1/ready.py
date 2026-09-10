"""Build the verified official LIGER/TIGER exact-parent package."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import shutil
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
)
from recclaw_core.mechanism_space import CompileStatus
from recclaw_core.mechanism_space.canonical import deep_thaw
from recclaw_core.mechanism_space.catalog import resolve_provider
from recclaw_core.research_line.frozen_family_profile import (
    build_frozen_family_profile,
    verify_frozen_family_assets,
)
from recclaw_core.research_line.single_parent_profile import (
    ACTUAL_TRAINING_SEED,
)
from recclaw_core.research_line.single_parent_search import (
    SEMANTIC_ID_SINGLE_PARENT_SPEC,
    focused_mechanism_language,
)


EXPECTED_PARENT_RUNNER_SHA256 = (
    "c6dac8f952fcc71c6f9df85ffd1f6744c1657c31d097dae96906d07a3d91337f"
)
EXPECTED_PARENT_RESULT_SHA256 = (
    "8342fb14a3bb9c6447fa3850cf3c750a0bd7797cbda0cbd26b92d1dcb78fd801"
)
EXPECTED_CONTENT_ASSET_SHA256 = (
    "dfa12623e026474a5ee5e4fe62a77136bcaf9acc2359bbcf4fba1ccb423a8703"
)
EXPECTED_STRUCTURED_MANIFEST_SHA256 = (
    "74c79a63af26002b66b0db3979f28ab97788a6b10bc3259f38321b492cc21fdd"
)
EXPECTED_CATALOG_MAPPING_SHA256 = (
    "fd167c8e3b505a4feea02c0dba229b845f3ef6f5c06d667a7e2e2554287ac9dd"
)
EXPECTED_SID_MAPPING_SHA256 = (
    "eef6757cfe116b6c228b2d03c1bcf1f50f05874a7ef8ad8cbef37f640e18745a"
)
EXPECTED_PARENT_NDCG_AT_10 = 0.06440683664232674
EXPECTED_DENSE_DIAGNOSTIC_NDCG_AT_10 = 0.07024223064474094
OFFICIAL_REVISION = "b6ccc37af5ee623ddc1d1ead3490c31aaeaf4524"
EXPECTED_SOURCE_HASHES = {
    "configs/dataset/amazon.yaml": (
        "369033aaea88543f7a40e08906c00c831c6b2d860a9282cf54323ab71d50e054"
    ),
    "configs/method/setting.yaml": (
        "04f59e1c2c24e05de31dac16e4d5628bf6becc710e48a23fbe6f5cc0dd5f03a0"
    ),
    "src/evaluation.py": (
        "cbe9df83443a350801f9a007cd4c56e4647dde94b64668583f128415c246e48a"
    ),
    "src/tiger.py": (
        "be78150fe8b2c90b49f59992765a587046e932fafba306be9d6a93ae9354d8e3"
    ),
}
_PARENT_PRIMITIVES = {
    "feature.frozen_text_encoder",
    "tokenizer.frozen_letter_collision_suffix",
    "context.sid_history_with_frozen_content",
    "generator.t5_encoder_decoder",
    "objective.token_cross_entropy",
    "objective.train_seen_full_catalog_dense_cross_entropy",
    "decode.autoregressive_beam_then_invalid_drop",
    "resolution.invalid_sid_drop_lookup",
    "retrieval.generated_legal_dense_rerank_to_score",
}
_FOCUSED_REQUIRED_PRIMITIVES = _PARENT_PRIMITIVES
_ASSET_ROOT = Path(__file__).resolve().parent / "assets"


class SemanticIdGenerativeReadyError(ValueError):
    """Raised when supplied files do not describe the frozen LIGER parent."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, role: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SemanticIdGenerativeReadyError(
            f"{role} is not valid UTF-8 JSON: {path}"
        ) from error
    if not isinstance(value, dict):
        raise SemanticIdGenerativeReadyError(f"{role} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = canonical_json_bytes(canonical_value(dict(value))) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _verify_exact_input(path: Path, expected: str, *, role: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise SemanticIdGenerativeReadyError(f"{role} is missing: {resolved}")
    if _sha256(resolved) != expected:
        raise SemanticIdGenerativeReadyError(
            f"{role} sha256 differs from the frozen P6 asset"
        )
    return resolved


def _verify_official_source_root(path: Path) -> Path:
    root = path.expanduser().resolve()
    if not root.is_dir():
        raise SemanticIdGenerativeReadyError(
            f"official LIGER source root is missing: {root}"
        )
    for relative, expected in EXPECTED_SOURCE_HASHES.items():
        _verify_exact_input(
            root / relative,
            expected,
            role=f"official LIGER source {relative}",
        )
    return root


def _official_evaluation_helpers(path: Path) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = []
    for name in ("model_forward", "get_target_embed"):
        node = next(
            (
                item
                for item in tree.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name == name
            ),
            None,
        )
        if node is None:
            raise SemanticIdGenerativeReadyError(
                f"official LIGER evaluation source lacks {name}"
            )
        segment = ast.get_source_segment(source, node)
        if not isinstance(segment, str) or not segment:
            raise SemanticIdGenerativeReadyError(
                f"official LIGER evaluation source cannot extract {name}"
            )
        if name == "model_forward":
            from .semantic_decode_scaffold import preserve_liger_context_input_gradients

            preserve_liger_context_input_gradients(node)
            segment = ast.unparse(node)
        selected.append(segment)
    return "\n\n\n".join(selected) + "\n"


def _copy_search_partition(
    source: Path,
    destination: Path,
    *,
    expected_suffix: str,
) -> str:
    source = source.expanduser().resolve()
    if not source.is_file() or not source.name.endswith(expected_suffix):
        raise SemanticIdGenerativeReadyError(
            f"search partition must be an existing {expected_suffix} file: {source}"
        )
    try:
        with source.open("rb") as stream:
            header = stream.readline().decode("utf-8", errors="strict").strip().split("\t")
    except UnicodeDecodeError as error:
        raise SemanticIdGenerativeReadyError(
            f"search partition header is not UTF-8: {source}"
        ) from error
    required = {
        "user_id:token",
        "item_id:token",
        "chrono_order:float",
        "item_id_list:token_seq",
        "chrono_order_list:float_seq",
    }
    if not required <= set(header):
        missing = ", ".join(sorted(required - set(header)))
        raise SemanticIdGenerativeReadyError(
            f"semantic-ID search partition lacks frozen prefix fields: {missing}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return _sha256(destination)


def _materialize_catalog_item_feature(
    source: Path,
    destination: Path,
    *,
    expected_items: int,
) -> str:
    rows = []
    with source.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            token = value.get("item_id") if isinstance(value, Mapping) else None
            if (
                not isinstance(token, str)
                or not token
                or any(character.isspace() for character in token)
            ):
                raise SemanticIdGenerativeReadyError(
                    "P6 catalog item_id is not a RecBole token"
                )
            rows.append(token)
    if len(rows) != expected_items or len(rows) != len(set(rows)):
        raise SemanticIdGenerativeReadyError(
            "P6 catalog item feature differs from the frozen item universe"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "item_id:token\n" + "".join(f"{token}\n" for token in rows),
        encoding="utf-8",
        newline="",
    )
    return _sha256(destination)


def _expanded_parent_sids(path: Path, *, expected_items: int) -> list[list[int]]:
    raw = _read_json(path, role="LIGER SID mapping")
    try:
        indices = {int(key) for key in raw}
    except (TypeError, ValueError) as error:
        raise SemanticIdGenerativeReadyError(
            "LIGER SID mapping keys must be integer item offsets"
        ) from error
    if indices != set(range(expected_items)):
        raise SemanticIdGenerativeReadyError(
            "LIGER SID mapping does not cover the frozen item universe"
        )
    collision_counts: defaultdict[tuple[int, int, int], int] = defaultdict(int)
    expanded: list[list[int]] = []
    for index in range(expected_items):
        value = raw[str(index)]
        if (
            not isinstance(value, list)
            or len(value) < 3
            or any(isinstance(code, bool) or not isinstance(code, int) for code in value[:3])
            or any(code < 0 or code >= 256 for code in value[:3])
        ):
            raise SemanticIdGenerativeReadyError(
                "LIGER SID mapping must expose three base-256 prefix codes per item"
            )
        prefix = tuple(value[:3])
        collision = collision_counts[prefix]
        collision_counts[prefix] += 1
        expanded.append(
            [
                prefix[0] + 1,
                prefix[1] + 257,
                prefix[2] + 513,
                collision + 769,
            ]
        )
    if len({tuple(row) for row in expanded}) != expected_items:
        raise SemanticIdGenerativeReadyError(
            "official LIGER collision extension did not produce unique SIDs"
        )
    return expanded


def _verify_parent_result(path: Path) -> tuple[dict[str, Any], float]:
    result = _read_json(path, role="LIGER parent result")
    best = result.get("best_valid_result")
    protocol = result.get("protocol")
    qualification = result.get("qualification")
    embedding = result.get("embedding_cache")
    input_hashes = result.get("input_hashes")
    training = result.get("training")
    if not all(
        isinstance(value, Mapping)
        for value in (best, protocol, qualification, embedding, input_hashes, training)
    ):
        raise SemanticIdGenerativeReadyError(
            "LIGER parent result lacks metric, protocol, or executable identity"
        )
    hybrid = best.get("hybrid_gen20_dense")
    dense = best.get("dense")
    if not isinstance(hybrid, Mapping) or not isinstance(dense, Mapping):
        raise SemanticIdGenerativeReadyError(
            "LIGER parent result lacks the hybrid Gen20+dense selection metric"
        )
    hybrid_metric = hybrid.get("ndcg@10")
    dense_metric = dense.get("ndcg@10")
    if (
        result.get("schema") != "recclaw.profile-baseline.liger-beauty.v1"
        or result.get("model") != "LIGER"
        or result.get("source_revision") != OFFICIAL_REVISION
        or canonical_value(result.get("source_hashes"))
        != canonical_value(EXPECTED_SOURCE_HASHES)
        or protocol.get("seed") != ACTUAL_TRAINING_SEED
        or protocol.get("selection_partition")
        != "validation hybrid_gen20_dense NDCG@10"
        or protocol.get("test_read_after_selection") is not False
        or result.get("test_result") is not None
        or embedding.get("sha256") != EXPECTED_CONTENT_ASSET_SHA256
        or input_hashes.get("items") != EXPECTED_CATALOG_MAPPING_SHA256
        or input_hashes.get("sids") != EXPECTED_SID_MAPPING_SHA256
        or training.get("num_beams") != 20
        or qualification.get("dense_and_generative_paths_both_active") is not True
        or qualification.get("invalid_generated_sids_filtered_instead_of_mapping_to_item_zero")
        is not True
        or qualification.get("official_liger_model_and_dual_loss") is not True
        or qualification.get("static_scores_or_hidden_fallback") is not False
    ):
        raise SemanticIdGenerativeReadyError(
            "LIGER parent must be the exact seed-54201 validation-only hybrid run"
        )
    for name, value, expected in (
        ("hybrid Gen20+dense", hybrid_metric, EXPECTED_PARENT_NDCG_AT_10),
        ("dense diagnostic", dense_metric, EXPECTED_DENSE_DIAGNOSTIC_NDCG_AT_10),
        ("selected", result.get("best_valid_score"), EXPECTED_PARENT_NDCG_AT_10),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-15)
        ):
            raise SemanticIdGenerativeReadyError(
                f"LIGER {name} development NDCG@10 differs from the frozen result"
            )
    return result, float(hybrid_metric)


def _parent_program(provider: Any, profile_ref: Mapping[str, Any]) -> dict[str, Any]:
    identity = provider.identity()
    components = [
        {
            "component_id": "liger_content_encoder",
            "slot_id": "ITEM_FEATURE_ENCODER",
            "primitive_id": "feature.frozen_text_encoder",
            "inputs": [
                {
                    "port": "content",
                    "source": {
                        "kind": "DATA",
                        "data_role": "FROZEN_ITEM_CONTENT_FEATURE",
                    },
                }
            ],
            "parameters": {"projection_dimension": 128, "normalize": True},
        },
        {
            "component_id": "liger_frozen_letter_sid",
            "slot_id": "SEMANTIC_TOKENIZER",
            "primitive_id": "tokenizer.frozen_letter_collision_suffix",
            "inputs": [
                {
                    "port": "sid",
                    "source": {
                        "kind": "DATA",
                        "data_role": "FROZEN_LETTER_ITEM_SID",
                    },
                }
            ],
            "parameters": {
                "semantic_prefix_levels": 3,
                "codes_per_level": 256,
                "collision_suffix": "ORDERED_PREFIX_COLLISION_INDEX",
                "position_offset_encoding": True,
            },
        },
        {
            "component_id": "liger_history_context",
            "slot_id": "USER_CONTEXT_ENCODER",
            "primitive_id": "context.sid_history_with_frozen_content",
            "inputs": [
                {
                    "port": "history",
                    "source": {
                        "kind": "DATA",
                        "data_role": "TRAIN_ITEM_SEQUENCE",
                    },
                },
                {
                    "port": "sid",
                    "source": {
                        "kind": "COMPONENT",
                        "component_id": "liger_frozen_letter_sid",
                        "output_port": "sid",
                    },
                },
                {
                    "port": "item_latent",
                    "source": {
                        "kind": "COMPONENT",
                        "component_id": "liger_content_encoder",
                        "output_port": "latent",
                    },
                },
            ],
            "parameters": {
                "max_items_per_sequence": 20,
                "sid_tokens_per_item": 4,
                "appended_padding_tokens": 22,
                "dimension": 128,
            },
        },
        {
            "component_id": "liger_generator",
            "slot_id": "GENERATIVE_BACKBONE",
            "primitive_id": "generator.t5_encoder_decoder",
            "inputs": [
                {
                    "port": "context",
                    "source": {
                        "kind": "COMPONENT",
                        "component_id": "liger_history_context",
                        "output_port": "context",
                    },
                },
                {
                    "port": "sid",
                    "source": {
                        "kind": "COMPONENT",
                        "component_id": "liger_frozen_letter_sid",
                        "output_port": "sid",
                    },
                },
            ],
            "parameters": {
                "encoder_layers": 6,
                "decoder_layers": 6,
                "dimension": 128,
                "feed_forward_dimension": 1024,
                "attention_heads": 6,
                "key_value_dimension": 64,
                "dropout_rate": 0.2,
            },
        },
        {
            "component_id": "liger_beam20",
            "slot_id": "DECODING_STRATEGY",
            "primitive_id": "decode.autoregressive_beam_then_invalid_drop",
            "inputs": [
                {
                    "port": "token_state",
                    "source": {
                        "kind": "COMPONENT",
                        "component_id": "liger_generator",
                        "output_port": "token_state",
                    },
                },
                {
                    "port": "sid",
                    "source": {
                        "kind": "COMPONENT",
                        "component_id": "liger_frozen_letter_sid",
                        "output_port": "sid",
                    },
                },
            ],
            "parameters": {
                "beam_width": 20,
                "length_normalization": 0.0,
                "invalid_sentinel": -1,
            },
        },
        {
            "component_id": "liger_item_resolution",
            "slot_id": "ITEM_RESOLUTION",
            "primitive_id": "resolution.invalid_sid_drop_lookup",
            "inputs": [
                {
                    "port": "hypothesis",
                    "source": {
                        "kind": "COMPONENT",
                        "component_id": "liger_beam20",
                        "output_port": "hypothesis",
                    },
                },
                {
                    "port": "catalog",
                    "source": {
                        "kind": "DATA",
                        "data_role": "TRAIN_CATALOG_MAPPING",
                    },
                },
            ],
            "parameters": {
                "invalid_sentinel": -1,
                "invalid_policy": "DROP",
                "deduplication": "ORDER_PRESERVING",
            },
        },
        {
            "component_id": "liger_dense_rerank",
            "slot_id": "DENSE_RETRIEVAL_CORRECTION",
            "primitive_id": "retrieval.generated_legal_dense_rerank_to_score",
            "inputs": [
                {
                    "port": "hypothesis",
                    "source": {
                        "kind": "COMPONENT",
                        "component_id": "liger_item_resolution",
                        "output_port": "hypothesis",
                    },
                },
                {
                    "port": "context",
                    "source": {
                        "kind": "COMPONENT",
                        "component_id": "liger_history_context",
                        "output_port": "context",
                    },
                },
                {
                    "port": "item_latent",
                    "source": {
                        "kind": "COMPONENT",
                        "component_id": "liger_content_encoder",
                        "output_port": "latent",
                    },
                },
            ],
            "parameters": {
                "generator_topk": 20,
                "rerank_depth": 20,
                "candidate_membership": (
                    "GENERATOR_ONLY_AFTER_LEGAL_SID_MAPPING_AND_DEDUPLICATION"
                ),
                "ranking_score": "DENSE_DOT_PRODUCT_ONLY",
                "generator_score_fusion": "NONE",
            },
        },
        {
            "component_id": "liger_token_objective",
            "slot_id": "GENERATIVE_OBJECTIVE",
            "primitive_id": "objective.token_cross_entropy",
            "inputs": [
                {
                    "port": "token_state",
                    "source": {
                        "kind": "COMPONENT",
                        "component_id": "liger_generator",
                        "output_port": "token_state",
                    },
                },
                {
                    "port": "targets",
                    "source": {
                        "kind": "DATA",
                        "data_role": "TRAINED_ITEM_SID",
                    },
                },
            ],
            "parameters": {
                "position_weights": "UNIFORM",
                "label_smoothing": 0.0,
                "loss_weight": 1.0,
            },
        },
        {
            "component_id": "liger_dense_objective",
            "slot_id": "GENERATIVE_OBJECTIVE",
            "primitive_id": "objective.train_seen_full_catalog_dense_cross_entropy",
            "inputs": [
                {
                    "port": "context",
                    "source": {
                        "kind": "COMPONENT",
                        "component_id": "liger_history_context",
                        "output_port": "context",
                    },
                },
                {
                    "port": "item_latent",
                    "source": {
                        "kind": "COMPONENT",
                        "component_id": "liger_content_encoder",
                        "output_port": "latent",
                    },
                },
                {
                    "port": "targets",
                    "source": {"kind": "DATA", "data_role": "ITEM_ID"},
                },
            ],
            "parameters": {
                "negative_catalog_mask": "TRAIN_SEEN_ONLY",
                "normalize_logits": True,
                "temperature": 0.07,
                "loss_weight": 1.0,
            },
        },
    ]
    return {
        "record_type": "MECHANISM_PROGRAM_ENVELOPE",
        "kernel_schema_version": "recclaw.mechanism-space.kernel.v1",
        "search_space_id": identity.search_space_id,
        "search_space_digest": identity.search_space_digest,
        "family_id": identity.family_id,
        "family_version": identity.family_version,
        "profile_ref": canonical_value(dict(profile_ref)),
        "program_payload": {
            "schema_version": (
                f"recclaw.{identity.family_id.lower().replace('_', '-')}.mechanism-program.v1"
            ),
            "family_contract_id": identity.family_id,
            "construction_mode": "COMPOSITION",
            "parent_refs": [],
            "research_question": (
                "Can a faithful semantic-ID intervention beat frozen LIGER?"
            ),
            "core_hypothesis": (
                "The verified official LIGER/TIGER T5 exact parent provides the "
                "ordinary beam20, explicit invalid-SID drop, and dense-only "
                "reranking path for one causally local semantic-ID intervention."
            ),
            "declared_data_roles": [
                "FROZEN_ITEM_CONTENT_FEATURE",
                "FROZEN_LETTER_ITEM_SID",
                "ITEM_ID",
                "TRAIN_CATALOG_MAPPING",
                "TRAIN_ITEM_SEQUENCE",
                "TRAINED_ITEM_SID",
            ],
            "components": components,
            "architecture_operators": [
                {
                    "operator_id": "add_component",
                    "targets": [],
                    "replacements": [],
                    "parameters": {},
                    "rationale": (
                        "Bind the official Gen20 T5 parent before applying the local delta."
                    ),
                }
            ],
            "changed_slots": [
                {
                    "slot_id": "DENSE_RETRIEVAL_CORRECTION",
                    "change_role": "CORE",
                }
            ],
            "removed_slots": [],
            "custom_components": [],
            "mechanism_explanation": (
                "Encode observed item histories, generate 20 raw semantic-ID beams, "
                "drop codes absent from the catalog without mapping them to item zero, "
                "deduplicate legal items, and rank only those items "
                "with dense dot products without generator-score fusion."
            ),
            "expected_effects": {
                "relevance": (
                "Exercise the official LIGER hybrid path against frozen development quality."
                ),
                "efficiency": "Charge content, generation, and dense costs separately.",
                "robustness": "Drop invalid SIDs instead of mapping to item zero or a fallback.",
                "coverage": "Resolve generated SIDs against the full frozen catalog.",
            },
            "matched_control": {
                "control_ref": "frozen_liger_parent",
                "rationale": (
                    "Use identical train/dev bytes, assets, seed, evaluator, and budget."
                ),
            },
            "ablation_plan": [
                {
                    "ablation_id": "dense_rerank_off",
                    "remove_component_ids": ["liger_dense_rerank"],
                    "expected_observation": (
                        "Removing legal generation leaves only the separately reported "
                        "dense diagnostic and cannot claim the LIGER hybrid mechanism."
                    ),
                }
            ],
            "discriminating_predictions": [
                {
                    "metric_or_probe": "same-protocol data/dev NDCG@10",
                    "if_supported": "The candidate exceeds frozen hybrid LIGER.",
                    "if_refuted": "Frozen hybrid LIGER remains at least as strong.",
                }
            ],
            "failure_interpretation": {
                "mechanism_failure": "The semantic-ID revision fails against LIGER.",
                "optimization_failure": "Token or dense optimization is unstable.",
                "protocol_failure": "Frozen data, asset, or evaluator identity differs.",
                "resource_failure": "The charged staged model cannot execute.",
            },
            "implementation_plan": [
                "Clone the official T5-compatible LIGER parent before a local causal delta.",
                "Keep raw beam generation, invalid drop, legal membership, and dense ordering separately observable.",
                "Preserve official T5 generation, dual loss, and candidate-local dev evaluation.",
            ],
            "resource_contract": {
                "relative_training_compute": "HIGH",
                "relative_memory": "HIGH",
                "precompute_required": True,
                "separate_budget_stages": [
                    "frozen_content_embedding",
                    "semantic_id_construction",
                    "generator_training",
                    "legal_generation",
                    "dense_reranking",
                ],
            },
            "claim_ceiling": "DEVELOPMENT_ONLY_SINGLE_PROTOCOL_NO_GENERAL_CLAIM",
            "protocol_impact": {"status": "UNCHANGED", "requested_changes": []},
        },
    }


def _render_candidate_source(
    *,
    program: Mapping[str, Any],
    report: Any,
    official_source_root: Path,
) -> str:
    components = tuple(program["program_payload"]["components"])
    component_specs = {
        str(item["component_id"]): canonical_value(dict(item))
        for item in components
    }
    replacements = {
        "__RECCLAW_CANDIDATE_ID__": repr(report.candidate_id),
        "__RECCLAW_PROGRAM_DIGEST__": repr(report.mechanism_program_digest),
        "__RECCLAW_SEMANTICS_DIGEST__": repr(report.mechanism_semantics_digest),
        "__RECCLAW_COMPONENT_IDS__": repr(sorted(component_specs)),
        "__RECCLAW_COMPONENT_SPECS__": repr(component_specs),
        "__RECCLAW_PRIMITIVE_IDS__": repr(
            sorted({str(item["primitive_id"]) for item in components})
        ),
        "__RECCLAW_OFFICIAL_TIGER_SOURCE__": (
            official_source_root / "src/tiger.py"
        ).read_text(encoding="utf-8"),
        "__RECCLAW_OFFICIAL_EVALUATION_HELPERS__": (
            _official_evaluation_helpers(official_source_root / "src/evaluation.py")
        ),
    }
    source = (_ASSET_ROOT / "liger_parent_candidate.py.tmpl").read_text(
        encoding="utf-8"
    )
    for marker, value in replacements.items():
        source = source.replace(marker, value)
    if "__RECCLAW_" in source:
        raise SemanticIdGenerativeReadyError(
            "LIGER parent source template contains unresolved markers"
        )
    compile(source, "recclaw_ext/candidate.py", "exec")
    return source


def _source_bundle(
    *,
    program: Mapping[str, Any],
    report: Any,
    official_source_root: Path,
    capability_ref: str,
) -> Mapping[str, Any]:
    candidate = _render_candidate_source(
        program=program,
        report=report,
        official_source_root=official_source_root,
    )
    trainer = (_ASSET_ROOT / "liger_parent_trainer.py.tmpl").read_text(
        encoding="utf-8"
    )
    compile(trainer, "recclaw_ext/trainer.py", "exec")
    package = (
        "from .candidate import FreshCandidateModel\n"
        "from .trainer import FreshCandidateTrainer\n"
    )
    files = []
    for path, content in (
        ("recclaw_ext/__init__.py", package),
        ("recclaw_ext/candidate.py", candidate),
        ("recclaw_ext/trainer.py", trainer),
    ):
        files.append(
            {
                "path": path,
                "content": content,
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )
    rows = [
        {
            "path": item["path"],
            "sha256": item["sha256"],
            "size_bytes": len(item["content"].encode("utf-8")),
        }
        for item in files
    ]
    return canonical_value(
        {
            "candidate_id": report.candidate_id,
            "program_digest": report.mechanism_program_digest,
            "capability_ref": capability_ref,
            "instruction": "CLONE_EXACT_PARENT_AND_LOCAL_PATCH",
            "source_tree_digest": sha256_digest(
                {"files": sorted(rows, key=lambda row: row["path"])}
            ),
            "files": files,
        }
    )


def build_semantic_id_generative_ready_launch(
    *,
    template_path: Path,
    parent_source_root_path: Path,
    parent_runner_path: Path,
    parent_result_path: Path,
    train_path: Path,
    dev_path: Path,
    content_asset_path: Path,
    structured_manifest_path: Path,
    catalog_mapping_path: Path,
    sid_mapping_path: Path,
    output_root: Path,
) -> Path:
    """Freeze the real LIGER parent into a development-only READY_NO_RUN package."""

    template_path = template_path.expanduser().resolve()
    payload = _read_json(template_path, role="P6 launch template")
    if payload.get("profile_key") != "semantic_id_generative":
        raise SemanticIdGenerativeReadyError(
            "launch template is not semantic_id_generative"
        )
    output_root = output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SemanticIdGenerativeReadyError("output_root must be absent or empty")

    runner = _verify_exact_input(
        parent_runner_path,
        EXPECTED_PARENT_RUNNER_SHA256,
        role="LIGER parent runner",
    )
    official_source_root = _verify_official_source_root(parent_source_root_path)
    result_path = _verify_exact_input(
        parent_result_path,
        EXPECTED_PARENT_RESULT_SHA256,
        role="LIGER parent result",
    )
    content = _verify_exact_input(
        content_asset_path,
        EXPECTED_CONTENT_ASSET_SHA256,
        role="LIGER content embedding",
    )
    structured = _verify_exact_input(
        structured_manifest_path,
        EXPECTED_STRUCTURED_MANIFEST_SHA256,
        role="P6 structured-field manifest",
    )
    catalog = _verify_exact_input(
        catalog_mapping_path,
        EXPECTED_CATALOG_MAPPING_SHA256,
        role="P6 catalog mapping",
    )
    sid_mapping = _verify_exact_input(
        sid_mapping_path,
        EXPECTED_SID_MAPPING_SHA256,
        role="LIGER SID mapping",
    )
    parent_result, metric_value = _verify_parent_result(result_path)

    frozen = payload["frozen_profile"]["frozen_fields"]
    if (
        frozen["content_asset_identity"]["sha256"]
        != EXPECTED_CONTENT_ASSET_SHA256
        or frozen["structured_field_asset_identity"]["sha256"]
        != EXPECTED_STRUCTURED_MANIFEST_SHA256
        or frozen["item_id_mapping_identity"]["catalog_items_sha256"]
        != EXPECTED_CATALOG_MAPPING_SHA256
    ):
        raise SemanticIdGenerativeReadyError(
            "P6 template asset identities differ from the frozen LIGER inputs"
        )
    expected_items = int(frozen["dataset_snapshot"]["items"])
    _expanded_parent_sids(
        sid_mapping,
        expected_items=expected_items,
    )

    spec = SEMANTIC_ID_SINGLE_PARENT_SPEC
    provider = resolve_provider(spec.mechanism_space_id)
    required = set(
        focused_mechanism_language(spec)["parent_contract"][
            "required_parent_foundation_primitives"
        ]
    )
    if required != _FOCUSED_REQUIRED_PRIMITIVES:
        raise SemanticIdGenerativeReadyError(
            "focused mechanism language differs from the executable LIGER parent"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    evidence_root = output_root / "evidence"
    evidence_root.mkdir(parents=True, exist_ok=True)
    copied_runner = evidence_root / "run_liger_beauty_protocol.py"
    copied_result = evidence_root / "p6_liger_seed54201_devonly.json"
    copied_sid = evidence_root / "letter_sids_aligned_raw.json"
    shutil.copyfile(runner, copied_runner)
    shutil.copyfile(result_path, copied_result)
    shutil.copyfile(sid_mapping, copied_sid)
    official_evidence_root = evidence_root / "official_liger_source"
    for relative in EXPECTED_SOURCE_HASHES:
        destination = official_evidence_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(official_source_root / relative, destination)

    dataset = str(frozen["dataset"])
    search_root = output_root / "search_data"
    dataset_root = search_root / dataset
    train_destination = dataset_root / f"{dataset}.train.inter"
    dev_destination = dataset_root / f"{dataset}.dev.inter"
    item_destination = dataset_root / f"{dataset}.item"
    train_sha256 = _copy_search_partition(
        train_path,
        train_destination,
        expected_suffix=".train.inter",
    )
    dev_sha256 = _copy_search_partition(
        dev_path,
        dev_destination,
        expected_suffix=".dev.inter",
    )
    item_sha256 = _materialize_catalog_item_feature(
        catalog,
        item_destination,
        expected_items=expected_items,
    )
    asset_root = search_root / "assets"
    asset_root.mkdir(parents=True, exist_ok=True)
    copied_content = asset_root / "liger_sentence_t5_embeddings.pt"
    copied_structured = asset_root / "resid_beauty_manifest.json"
    copied_catalog = asset_root / "beauty_items.jsonl"
    shutil.copyfile(content, copied_content)
    shutil.copyfile(structured, copied_structured)
    shutil.copyfile(catalog, copied_catalog)
    overrides = payload["execution_contract"]["config_overrides"]
    overrides["recclaw_content_asset"] = str(copied_content.resolve())
    overrides["recclaw_structured_field_manifest"] = str(
        copied_structured.resolve()
    )
    overrides["recclaw_catalog_mapping"] = str(copied_catalog.resolve())
    overrides["recclaw_sid_mapping"] = str(copied_sid.resolve())

    manifest = {
        "schema": "recclaw.search-only-dataset.v1",
        "dataset": dataset,
        "parent_interaction_sha256": frozen["dataset_snapshot"][
            "interaction_sha256"
        ],
        "parent_split": frozen["split"],
        "partition_files": {
            "train": {
                "path": f"{dataset}/{dataset}.train.inter",
                "sha256": train_sha256,
            },
            "development": {
                "path": f"{dataset}/{dataset}.dev.inter",
                "sha256": dev_sha256,
            },
        },
        "item_feature": {
            "path": f"{dataset}/{dataset}.item",
            "sha256": item_sha256,
            "items": expected_items,
        },
        "abi_partition_roles": {
            "train": "TRAIN",
            "valid": "DEVELOPMENT_VALIDATION",
            "test": "DEVELOPMENT_VALIDATION",
        },
        "heldout_partition_present": False,
    }
    manifest_path = search_root / "search-data-manifest.json"
    manifest_sha256 = _write_json(manifest_path, manifest)
    payload["execution_contract"]["search_data"] = {
        "root": str(search_root.resolve()),
        "manifest_ref": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
    }

    profile_ref, _ = build_frozen_family_profile(
        provider,
        profile_id=payload["frozen_profile"]["profile_id"],
        frozen_fields=frozen,
    )
    program = _parent_program(provider, profile_ref)
    report = provider.compile(deep_thaw(program))
    if (
        report.status is not CompileStatus.VALID_NEEDS_IMPLEMENTATION
        or report.candidate_id is None
        or report.mechanism_program_digest is None
        or report.mechanism_semantics_digest is None
    ):
        raise SemanticIdGenerativeReadyError(
            "executable LIGER parent program does not compile: " + report.to_json()
        )
    capability_ref = (
        "parent:semantic-id-generative:liger:b6ccc37a:seed54201:hybrid-gen20-dense"
    )
    source_bundle = _source_bundle(
        program=program,
        report=report,
        official_source_root=official_source_root,
        capability_ref=capability_ref,
    )
    binding = {
        "candidate_id": report.candidate_id,
        "program_digest": report.mechanism_program_digest,
    }
    payload["baseline_context"]["parent_anchor"].update(
        {
            "binding": binding,
            "mechanism_program": program,
            "source_equivalence": (
                "OFFICIAL_LIGER_T5_EXACT_PARENT_WITH_RECBole_BATCH_ADAPTER"
            ),
            "source_bundle": source_bundle,
        }
    )
    payload["baseline_context"]["parent_anchor"]["paired_metric"][
        "value"
    ] = metric_value

    verified_assets = verify_frozen_family_assets(
        frozen_fields=frozen,
        config_overrides=overrides,
        search_data=payload["execution_contract"]["search_data"],
    )
    parent_binding = {
        **binding,
        "source_tree_digest": source_bundle["source_tree_digest"],
    }
    comparator_digest = sha256_digest(
        {
            "schema": "recclaw.single-parent-comparator-executable.v1",
            "parent_binding": parent_binding,
            "protocol_digest": profile_ref["profile_digest"],
            "asset_manifest_digest": verified_assets["digest"],
        }
    )

    hybrid = parent_result["best_valid_result"]["hybrid_gen20_dense"]
    worker_result = {
        "schema": "recclaw.single-parent-worker-result-projection.v1",
        "model": "LIGER",
        "best_valid_result": {
            "ndcg@10": metric_value,
            "recall@10": float(hybrid["recall@10"]),
            "valid_generated_candidates": parent_result["best_valid_result"][
                "valid_generated_candidates"
            ],
            "invalid_generated_candidates": parent_result["best_valid_result"][
                "invalid_generated_candidates"
            ],
        },
        "best_valid_score": metric_value,
        "exit_status": "SUCCESS",
        "metric_source": "BEST_VALID_RESULT",
        "online_partition_role": "DEVELOPMENT_VALIDATION",
        "seed": ACTUAL_TRAINING_SEED,
        "split": "data/dev",
        "test_result": None,
        "source_result_ref": str(copied_result.resolve()),
        "source_result_sha256": EXPECTED_PARENT_RESULT_SHA256,
        "source_result_schema": parent_result["schema"],
    }
    worker_path = output_root / "observations" / "liger-parent-worker-result.json"
    worker_digest = _write_json(worker_path, worker_result)
    receipt = {
        "schema": "recclaw.single-parent-parent-observation.v1",
        "profile_key": "semantic_id_generative",
        "parent_binding": parent_binding,
        "protocol_digest": profile_ref["profile_digest"],
        "asset_manifest_digest": verified_assets["digest"],
        "seed": ACTUAL_TRAINING_SEED,
        "split": "data/dev",
        "metric": {
            "name": "NDCG@10",
            "source": "BEST_VALID_RESULT",
            "partition_role": "DEVELOPMENT_VALIDATION",
            "value": metric_value,
        },
        "comparator_ref": capability_ref,
        "comparator_digest": comparator_digest,
        "worker_result_ref": str(worker_path.resolve()),
        "worker_result_digest": worker_digest,
    }
    receipt_path = output_root / "observations" / "liger-parent-observation.json"
    receipt_digest = _write_json(receipt_path, receipt)
    payload["parent_source"] = {
        "source_ref": str(receipt_path.resolve()),
        "source_sha256": receipt_digest,
        "comparator_ref": capability_ref,
        "comparator_digest": comparator_digest,
        "frozen_ndcg_at_10": metric_value,
    }
    payload["evidence"] = {
        "known_parent_result_seed": ACTUAL_TRAINING_SEED,
        "known_parent_development_ndcg_at_10": metric_value,
        "known_result_ref": str(copied_result.resolve()),
        "known_result_sha256": EXPECTED_PARENT_RESULT_SHA256,
        "known_runner_ref": str(copied_runner.resolve()),
        "known_runner_sha256": EXPECTED_PARENT_RUNNER_SHA256,
        "known_sid_mapping_ref": str(copied_sid.resolve()),
        "known_sid_mapping_sha256": EXPECTED_SID_MAPPING_SHA256,
        "official_source_ref": str(official_evidence_root.resolve()),
        "official_source_revision": OFFICIAL_REVISION,
        "official_source_hashes": EXPECTED_SOURCE_HASHES,
        "exact_parent_runtime_blocker": None,
        "launch_asset_status": "READY_NO_RUN",
        "formal_launch_rule": (
            "Clone the verified official TIGER/T5 parent and make one local causal "
            "mechanism change while preserving seed, train/dev bytes, "
            "candidate-local evaluator, heldout prohibition, and the parent optimizer "
            "unless compiled OPTIMIZATION_STAGING explicitly owns its change."
        ),
    }

    ready_path = (
        output_root / "semantic_id_generative_liger_single_parent_v1.ready.json"
    )
    _write_json(ready_path, payload)
    return ready_path


__all__ = [
    "SemanticIdGenerativeReadyError",
    "build_semantic_id_generative_ready_launch",
]
