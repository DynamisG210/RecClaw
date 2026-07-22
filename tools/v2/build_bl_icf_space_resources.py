#!/usr/bin/env python3
"""Build or check the package-owned BL-ICF v1 resource closure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import rfc8785

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
GENERIC_RESOURCES = SRC / "recclaw_core" / "mechanism_space" / "resources"
BL_RESOURCES = SRC / "recclaw_core" / "search_spaces" / "bl_icf_v1" / "resources"
PROVIDER_SOURCE = SRC / "recclaw_core" / "search_spaces" / "bl_icf_v1" / "provider.py"
KERNEL_SOURCES = (
    SRC / "recclaw_core" / "mechanism_space" / "canonical.py",
    SRC / "recclaw_core" / "mechanism_space" / "contracts.py",
    SRC / "recclaw_core" / "mechanism_space" / "kernel.py",
)
RUNTIME_DEPENDENCIES = {
    "jsonschema": "4.26.0",
    "rfc8785": "0.1.4",
}

SPACE_ID = "BL_ICF_MECHANISM_SPACE_V1"
SPACE_VERSION = "1.0.0"
FAMILY_ID = "BL_ICF_V1"
FAMILY_VERSION = "1.0.0"
PROVIDER_ID = "recclaw.search-space-provider.bl-icf.v1"
PROGRAM_SCHEMA_VERSION = "recclaw.bl-icf.mechanism-program.v1"
CLAIM_CEILING = "DEVELOPMENT_ONLY_SINGLE_PROTOCOL_NO_GENERAL_CLAIM"

CAPABILITIES = [
    "CONFIG_MUTATION",
    "LOSS_MUTATION",
    "SAMPLER_MUTATION",
    "RELATION_MUTATION",
    "GRAPH_OPERATOR_MUTATION",
    "AUXILIARY_OBJECTIVE_MUTATION",
    "TRAINING_ALGORITHM_MUTATION",
    "CUSTOM_MODEL_IMPLEMENTATION",
    "ANALYTIC_REWRITE",
    "POSTHOC_RERANK",
    "EFFICIENCY_REWRITE",
]

SLOTS = [
    "RELATION_VIEW",
    "EMBEDDING",
    "ENCODER",
    "MESSAGE",
    "PROPAGATION_AGGREGATION",
    "RELATION_DECOMPOSITION",
    "FUSION_ROUTING",
    "SCORE_HEAD",
    "PRIMARY_OBJECTIVE",
    "NEGATIVE_SAMPLER",
    "SELF_SUPERVISION",
    "GEOMETRY_REGULARIZATION",
    "DENOISING_LONG_TAIL",
    "TRAINING_PROCEDURE",
    "EFFICIENCY_APPROXIMATION",
    "POSTHOC_RERANK",
]

DATA_ROLES = {
    "USER_ID": "core/user_id",
    "ITEM_ID": "core/item_id",
    "TRAIN_INTERACTIONS": "bl_icf/train_interactions",
    "TRAIN_USER_ITEM_GRAPH": "bl_icf/user_item_relation",
    "TRAIN_ITEM_ITEM_GRAPH": "bl_icf/item_item_relation",
    "TRAIN_USER_USER_GRAPH": "bl_icf/user_user_relation",
    "TRAIN_DERIVED_SPECTRAL_VIEW": "bl_icf/spectral_relation",
    "TRAIN_DERIVED_PROTOTYPES": "bl_icf/prototype_relation",
    "TRAIN_DERIVED_STATISTICS": "bl_icf/train_statistics",
}

FROZEN_PROTOCOL_FIELDS = [
    "dataset",
    "dataset_snapshot",
    "split",
    "preprocessing",
    "evaluation_mode",
    "candidate_universe",
    "seen_repeat_policy",
    "metric_semantics",
    "comparator_semantics",
    "seed_policy",
]

ARCHITECTURE_OPERATORS = [
    "add_component",
    "remove_component",
    "replace_component",
    "split_component",
    "fuse_components",
    "share_parameters",
    "decouple_relations",
    "route_by_user_or_item",
    "gate_component",
    "reweight_relation",
    "normalize_signal",
    "precompute_operator",
    "sparsify_relation",
    "approximate_fixed_point",
    "compile_propagation_to_constraint",
    "derive_closed_form",
    "distill_teacher",
    "prune_path",
    "freeze_path",
    "alternate_optimization",
    "synthesize_custom_model",
]

_AXIS_PRIMITIVES: dict[str, list[str]] = {
    "RELATION_VIEW": [
        "relation.user_item_bipartite",
        "relation.item_item_cooccurrence",
        "relation.item_item_cooccurrence_topk",
        "relation.user_user_collaborative",
        "relation.k_hop",
        "relation.hypergraph_incidence",
        "relation.svd_global",
        "relation.prototype_cluster",
        "relation.degree_popularity",
        "relation.denoised_reweighted",
        "relation.multi_relation",
    ],
    "EMBEDDING": [
        "embedding.independent_user_item",
        "embedding.shared_subspace",
        "embedding.factorized",
        "embedding.multi_table",
        "embedding.normalized",
        "embedding.hyperspherical",
        "embedding.degree_conditioned_initialization",
        "embedding.low_rank_parameterization",
        "embedding.prototype_residual",
        "embedding.hash_compressed",
        "embedding.partially_frozen",
    ],
    "ENCODER": [
        "encoder.none_mf",
        "encoder.explicit_message_passing",
        "encoder.precomputed_graph_filter",
        "encoder.closed_form_filter",
        "encoder.fixed_point_constraint",
        "encoder.relation_constraint",
        "encoder.hybrid_multi_branch",
    ],
    "MESSAGE": [
        "message.identity",
        "message.linear_transform",
        "message.elementwise_interaction",
        "message.bilinear_interaction",
        "message.normalized_neighbor",
        "message.degree_aware",
        "message.popularity_aware",
        "message.attention_weighted",
        "message.confidence_weighted",
        "message.signed_positive_negative",
        "message.relation_specific",
    ],
    "PROPAGATION_AGGREGATION": [
        "propagation.sum",
        "propagation.mean",
        "propagation.symmetric_normalization",
        "propagation.random_walk_normalization",
        "propagation.learned_normalization",
        "propagation.residual_skip_ego",
        "propagation.weighted_layer_sum",
        "propagation.concatenation",
        "propagation.attention_aggregation",
        "propagation.layer_gate",
        "propagation.user_item_specific_depth",
        "propagation.adaptive_stopping",
        "propagation.multi_hop_decoupling",
        "propagation.order_wise_aggregation",
        "propagation.graph_polynomial_filter",
        "propagation.spectral_low_pass_filter",
        "propagation.spectral_high_pass_filter",
        "propagation.spectral_band_pass_filter",
        "propagation.dropout",
        "propagation.none",
    ],
    "RELATION_DECOMPOSITION": [
        "relation_decomposition.user_item",
        "relation_decomposition.item_item",
        "relation_decomposition.user_user",
        "relation_decomposition.local_global",
        "relation_decomposition.structural_semantic",
        "relation_decomposition.independent_weight",
        "relation_decomposition.independent_loss",
        "relation_decomposition.independent_sampler",
        "relation_decomposition.independent_sparsification",
        "relation_decomposition.independent_update_schedule",
        "relation_decomposition.relation_gate",
    ],
    "FUSION_ROUTING": [
        "fusion.layer_weighted_sum",
        "fusion.residual",
        "fusion.gated",
        "fusion.attention",
        "fusion.mixture_of_experts",
        "fusion.user_specific_route",
        "fusion.item_specific_route",
        "fusion.confidence_route",
        "fusion.sparse_route",
        "fusion.dual_path",
        "fusion.base_augmented_branch",
        "fusion.early",
        "fusion.late",
    ],
    "SCORE_HEAD": [
        "score.dot_product",
        "score.cosine_similarity",
        "score.normalized_dot_product",
        "score.bilinear",
        "score.additive_bias",
        "score.distance_based",
        "score.shallow_mlp",
        "score.multi_head",
        "score.calibrated",
        "score.ensemble",
    ],
    "PRIMARY_OBJECTIVE": [
        "objective.bpr",
        "objective.margin_bpr",
        "objective.weighted_bpr",
        "objective.adaptive_margin_pairwise",
        "objective.pointwise_bce",
        "objective.sampled_softmax",
        "objective.listwise_surrogate",
        "objective.partial_auc_surrogate",
        "objective.cosine_contrastive",
        "objective.alignment_uniformity",
        "objective.graph_constraint",
        "objective.user_item_structure_constraint",
        "objective.item_item_relation_constraint",
        "objective.multi_relevance",
    ],
    "NEGATIVE_SAMPLER": [
        "sampler.uniform",
        "sampler.sampled_unobserved",
        "sampler.popularity",
        "sampler.inverse_popularity",
        "sampler.dynamic_hard",
        "sampler.softmax_adversarial",
        "sampler.in_batch",
        "sampler.mixed",
        "sampler.graph_neighborhood_aware",
        "sampler.multi_hop",
        "sampler.embedding_space_mixup",
        "sampler.area_wise_mixup",
        "sampler.dimension_wise_mixup",
        "sampler.false_negative_aware",
        "sampler.uncertainty_aware",
        "sampler.curriculum_hardness",
        "sampler.adaptive_negative_count",
        "sampler.per_user_item",
        "sampler.ensemble",
    ],
    "SELF_SUPERVISION": [
        "ssl.view.edge_dropout",
        "ssl.view.node_dropout",
        "ssl.view.random_walk",
        "ssl.view.embedding_dropout",
        "ssl.view.gaussian_perturbation",
        "ssl.view.uniform_perturbation",
        "ssl.view.adversarial_perturbation",
        "ssl.view.svd_global",
        "ssl.view.structural_neighbor",
        "ssl.view.semantic_prototype",
        "ssl.view.trainable_denoising",
        "ssl.view.generative",
        "ssl.view.diffusion",
        "ssl.view.twin_ema_encoder",
        "ssl.view.cross_layer",
        "ssl.view.consecutive_state",
        "ssl.objective.info_nce",
        "ssl.objective.supervised_contrastive",
        "ssl.objective.prototype_contrastive",
        "ssl.objective.neighbor_contrastive",
        "ssl.objective.bootstrapping",
        "ssl.objective.alignment_uniformity",
        "ssl.objective.decorrelation",
        "ssl.objective.redundancy_reduction",
        "ssl.objective.mutual_prediction",
        "ssl.objective.unified_supervised_contrastive",
    ],
    "GEOMETRY_REGULARIZATION": [
        "regularizer.l2",
        "regularizer.user_item_specific_l2",
        "regularizer.max_norm",
        "regularizer.norm_matching",
        "regularizer.residual_norm",
        "regularizer.alignment",
        "regularizer.uniformity",
        "regularizer.decorrelation",
        "regularizer.orthogonality",
        "regularizer.hypersphere",
        "regularizer.graph_smoothness",
        "regularizer.layer_consistency",
        "regularizer.layer_diversity",
        "regularizer.prototype_compactness",
        "regularizer.confidence",
        "regularizer.adversarial_robustness",
        "regularizer.spectral",
        "regularizer.popularity_sensitive",
    ],
    "DENOISING_LONG_TAIL": [
        "robustness.edge_confidence",
        "robustness.interaction_denoising",
        "robustness.suspicious_edge_downweighting",
        "robustness.popularity_aware_loss",
        "robustness.inverse_propensity_like_heuristic",
        "robustness.head_tail_balanced_sampler",
        "robustness.tail_aware_regularization",
        "robustness.degree_normalization",
        "robustness.degree_adaptive_propagation",
        "robustness.false_negative_suppression",
        "robustness.robust_loss",
        "robustness.teacher_agreement",
        "robustness.uncertainty_filtering",
        "robustness.head_torso_tail_routing",
    ],
    "TRAINING_PROCEDURE": [
        "training.sgd",
        "training.adam",
        "training.adamw",
        "training.warmup_schedule",
        "training.cosine_schedule",
        "training.plateau_schedule",
        "training.alternating_optimization",
        "training.block_coordinate",
        "training.two_stage",
        "training.multi_stage",
        "training.curriculum",
        "training.hard_negative_curriculum",
        "training.progressive_depth",
        "training.progressive_augmentation",
        "training.objective_weight_schedule",
        "training.view_model_alternating_update",
        "training.teacher_student_update",
        "training.freeze_unfreeze",
        "training.early_stopping",
        "training.restart",
        "training.candidate_specific_checkpoint",
        "training.gradient_clipping",
        "training.mixed_precision",
        "training.adaptive_batch_negative_budget",
    ],
    "EFFICIENCY_APPROXIMATION": [
        "efficiency.remove_message_passing",
        "efficiency.cache_precompute_propagation",
        "efficiency.decouple_train_inference_propagation",
        "efficiency.sparse_matrix_kernel",
        "efficiency.graph_filter_precomputation",
        "efficiency.low_rank_approximation",
        "efficiency.fixed_point_approximation",
        "efficiency.relation_sparsification",
        "efficiency.topk_relation_selection",
        "efficiency.sampled_relation_constraint",
        "efficiency.parameter_sharing",
        "efficiency.embedding_factorization",
        "efficiency.pruning",
        "efficiency.quantization",
        "efficiency.distillation",
        "efficiency.early_exit",
        "efficiency.adaptive_depth",
        "efficiency.closed_form_solver",
        "efficiency.incremental_update",
        "efficiency.memory_aware_batching",
    ],
    "POSTHOC_RERANK": [
        "posthoc.score_normalization",
        "posthoc.calibration",
        "posthoc.popularity_penalty",
        "posthoc.coverage_aware_rerank",
        "posthoc.long_tail_rerank",
        "posthoc.diversity_constraint",
        "posthoc.candidate_confidence_filter",
        "posthoc.ensemble",
        "posthoc.pareto_rerank",
    ],
}

_SLOT_CAPABILITIES = {
    "RELATION_VIEW": ["RELATION_MUTATION"],
    "EMBEDDING": ["CONFIG_MUTATION"],
    "ENCODER": ["GRAPH_OPERATOR_MUTATION"],
    "MESSAGE": ["GRAPH_OPERATOR_MUTATION"],
    "PROPAGATION_AGGREGATION": ["GRAPH_OPERATOR_MUTATION"],
    "RELATION_DECOMPOSITION": ["RELATION_MUTATION"],
    "FUSION_ROUTING": ["GRAPH_OPERATOR_MUTATION"],
    "SCORE_HEAD": ["CONFIG_MUTATION"],
    "PRIMARY_OBJECTIVE": ["LOSS_MUTATION"],
    "NEGATIVE_SAMPLER": ["SAMPLER_MUTATION"],
    "SELF_SUPERVISION": ["AUXILIARY_OBJECTIVE_MUTATION"],
    "GEOMETRY_REGULARIZATION": ["AUXILIARY_OBJECTIVE_MUTATION"],
    "DENOISING_LONG_TAIL": ["LOSS_MUTATION"],
    "TRAINING_PROCEDURE": ["TRAINING_ALGORITHM_MUTATION"],
    "EFFICIENCY_APPROXIMATION": ["EFFICIENCY_REWRITE"],
    "POSTHOC_RERANK": ["POSTHOC_RERANK"],
}

_BUILTIN_PRIMITIVES = {
    "relation.user_item_bipartite",
    "embedding.independent_user_item",
    "encoder.none_mf",
    "encoder.explicit_message_passing",
    "message.identity",
    "propagation.symmetric_normalization",
    "propagation.weighted_layer_sum",
    "fusion.layer_weighted_sum",
    "score.dot_product",
    "objective.bpr",
    "sampler.uniform",
    "training.adam",
}


def _strict_object(required: list[str], properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def _envelope_schema() -> dict[str, Any]:
    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://recclaw.dev/schema/mechanism-program-envelope-v1",
        **_strict_object(
            [
                "record_type",
                "kernel_schema_version",
                "search_space_id",
                "search_space_digest",
                "family_id",
                "family_version",
                "profile_ref",
                "program_payload",
            ],
            {
                "record_type": {"const": "MECHANISM_PROGRAM_ENVELOPE"},
                "kernel_schema_version": {"const": "recclaw.mechanism-space.kernel.v1"},
                "search_space_id": {"type": "string", "minLength": 1},
                "search_space_digest": digest,
                "family_id": {"type": "string", "minLength": 1},
                "family_version": {"type": "string", "minLength": 1},
                "profile_ref": _strict_object(
                    ["profile_id", "profile_digest", "profile_kind"],
                    {
                        "profile_id": {"type": "string", "minLength": 1},
                        "profile_digest": digest,
                        "profile_kind": {"type": "string", "minLength": 1},
                    },
                ),
                "program_payload": {"type": "object"},
            },
        ),
    }


def _binding_schema() -> dict[str, Any]:
    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    ref = _strict_object(
        ["id", "digest"],
        {"id": {"type": "string", "minLength": 1}, "digest": digest},
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://recclaw.dev/schema/candidate-execution-binding-v1",
        **_strict_object(
            [
                "record_type",
                "binding_version",
                "authority",
                "evidence_class",
                "formal_acceptance",
                "run_id",
                "candidate_id",
                "search_space_id",
                "search_space_digest",
                "mechanism_program_digest",
                "mechanism_semantics_digest",
                "profile_ref",
                "budget_ref",
                "implementation_ref",
                "capability_envelope",
            ],
            {
                "record_type": {"const": "CANDIDATE_EXECUTION_BINDING"},
                "binding_version": {"const": "recclaw.candidate-execution-binding.v1"},
                "authority": {"const": "NONE"},
                "evidence_class": {"const": "DEVELOPMENT_ONLY"},
                "formal_acceptance": {"const": False},
                "run_id": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"},
                "candidate_id": {
                    "type": "string",
                    "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
                },
                "search_space_id": {"type": "string", "minLength": 1},
                "search_space_digest": digest,
                "mechanism_program_digest": digest,
                "mechanism_semantics_digest": digest,
                "profile_ref": _strict_object(
                    ["profile_id", "profile_digest", "profile_kind"],
                    {
                        "profile_id": {"type": "string", "minLength": 1},
                        "profile_digest": digest,
                        "profile_kind": {"type": "string", "minLength": 1},
                    },
                ),
                "budget_ref": ref,
                "implementation_ref": ref,
                "capability_envelope": _strict_object(
                    ["envelope_id", "envelope_digest", "granted_capabilities", "write_roots"],
                    {
                        "envelope_id": {"type": "string", "minLength": 1},
                        "envelope_digest": digest,
                        "granted_capabilities": {
                            "type": "array",
                            "uniqueItems": True,
                            "items": {
                                "type": "string",
                                "pattern": "^[A-Z][A-Z0-9_]{0,127}$",
                            },
                        },
                        "write_roots": {
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "string", "minLength": 1},
                        },
                    },
                ),
            },
        ),
    }


def _program_schema() -> dict[str, Any]:
    component_source = {
        "oneOf": [
            _strict_object(
                ["kind", "data_role"],
                {"kind": {"const": "DATA"}, "data_role": {"enum": sorted(DATA_ROLES)}},
            ),
            _strict_object(
                ["kind", "component_id", "output_port"],
                {
                    "kind": {"const": "COMPONENT"},
                    "component_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
                    "output_port": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
                },
            ),
        ]
    }
    port = _strict_object(
        ["port", "source"],
        {
            "port": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
            "source": component_source,
        },
    )
    component = {
        **_strict_object(
            ["component_id", "slot_id", "inputs", "parameters"],
            {
                "component_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
                "slot_id": {"enum": SLOTS},
                "primitive_id": {"type": "string", "pattern": "^[a-z][a-z0-9_.-]{2,127}$"},
                "custom_component_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
                "inputs": {"type": "array", "items": port},
                "parameters": {"type": "object"},
            },
        ),
        "oneOf": [
            {"required": ["primitive_id"], "not": {"required": ["custom_component_id"]}},
            {"required": ["custom_component_id"], "not": {"required": ["primitive_id"]}},
        ],
    }
    custom_port = _strict_object(
        ["port", "types", "minimum"],
        {
            "port": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
            "types": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "pattern": "^(core|bl_icf)/[a-z0-9_]+$"},
            },
            "minimum": {"type": "integer", "minimum": 0, "maximum": 8},
        },
    )
    custom_output = _strict_object(
        ["port", "type"],
        {
            "port": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
            "type": {"type": "string", "pattern": "^(core|bl_icf)/[a-z0-9_]+$"},
        },
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://recclaw.dev/schema/bl-icf-mechanism-program-v1",
        **_strict_object(
            [
                "schema_version",
                "family_contract_id",
                "construction_mode",
                "parent_refs",
                "research_question",
                "core_hypothesis",
                "input_semantics",
                "components",
                "architecture_operators",
                "changed_slots",
                "removed_slots",
                "custom_components",
                "mechanism_explanation",
                "expected_effects",
                "matched_control",
                "ablation_plan",
                "failure_modes",
                "implementation_plan",
                "estimated_cost",
                "claim_ceiling",
                "protocol_impact",
            ],
            {
                "schema_version": {"const": PROGRAM_SCHEMA_VERSION},
                "family_contract_id": {"const": FAMILY_ID},
                "construction_mode": {"enum": ["COMPOSITION", "ARCHITECTURE_REWRITE", "CUSTOM_MODEL"]},
                "parent_refs": {
                    "type": "array",
                    "uniqueItems": True,
                    "items": _strict_object(
                        ["candidate_id", "program_digest"],
                        {
                            "candidate_id": {"type": "string", "minLength": 1},
                            "program_digest": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                        },
                    ),
                },
                "research_question": {"type": "string", "minLength": 12, "maxLength": 2000},
                "core_hypothesis": {"type": "string", "minLength": 12, "maxLength": 4000},
                "input_semantics": _strict_object(
                    ["user_id", "item_id", "train_interactions", "external_features"],
                    {
                        "user_id": {"const": True},
                        "item_id": {"const": True},
                        "train_interactions": {"const": True},
                        "external_features": {"const": False},
                    },
                ),
                "components": {"type": "array", "minItems": 4, "maxItems": 96, "items": component},
                "architecture_operators": {
                    "type": "array",
                    "items": _strict_object(
                        ["operator_id", "targets", "replacements", "parameters", "rationale"],
                        {
                            "operator_id": {"enum": ARCHITECTURE_OPERATORS},
                            "targets": {
                                "type": "array",
                                "uniqueItems": True,
                                "items": {"type": "string", "pattern": "^(slot:[A-Z][A-Z0-9_]{0,63}|[a-z][a-z0-9_]{0,63})$"},
                            },
                            "replacements": {
                                "type": "array",
                                "uniqueItems": True,
                                "items": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
                            },
                            "parameters": {"type": "object"},
                            "rationale": {"type": "string", "minLength": 8},
                        },
                    ),
                },
                "changed_slots": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": _strict_object(
                        ["slot_id", "change_role"],
                        {"slot_id": {"enum": SLOTS}, "change_role": {"enum": ["CORE", "SUPPORT"]}},
                    ),
                },
                "removed_slots": {"type": "array", "uniqueItems": True, "items": {"enum": SLOTS}},
                "custom_components": {
                    "type": "array",
                    "maxItems": 24,
                    "items": _strict_object(
                        [
                            "custom_component_id",
                            "slot_id",
                            "mathematical_definition",
                            "algorithm_definition",
                            "input_ports",
                            "output_ports",
                            "allowed_read_roles",
                            "family_boundary_justification",
                            "minimal_implementation",
                            "matched_control_rationale",
                            "ablation",
                            "failure_modes",
                            "estimated_cost",
                        ],
                        {
                            "custom_component_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
                            "slot_id": {"enum": SLOTS},
                            "mathematical_definition": {"type": "string", "minLength": 20},
                            "algorithm_definition": {"type": "string", "minLength": 20},
                            "input_ports": {"type": "array", "items": custom_port},
                            "output_ports": {"type": "array", "minItems": 1, "items": custom_output},
                            "allowed_read_roles": {
                                "type": "array", "uniqueItems": True, "items": {"enum": sorted(DATA_ROLES)}
                            },
                            "family_boundary_justification": {"type": "string", "minLength": 20},
                            "minimal_implementation": _strict_object(
                                ["entrypoint_role", "file_roles", "steps"],
                                {
                                    "entrypoint_role": {"enum": ["MODEL", "RELATION_BUILDER", "SAMPLER", "TRAINER", "RERANKER"]},
                                    "file_roles": {
                                        "type": "array",
                                        "minItems": 1,
                                        "uniqueItems": True,
                                        "items": {"enum": ["MODEL", "RELATION_BUILDER", "LOSS", "SAMPLER", "TRAINER", "RERANKER", "CONFIG", "TEST"]},
                                    },
                                    "steps": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 5}},
                                },
                            ),
                            "matched_control_rationale": {"type": "string", "minLength": 12},
                            "ablation": {"type": "string", "minLength": 12},
                            "failure_modes": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 8}},
                            "estimated_cost": {"enum": ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]},
                        },
                    ),
                },
                "mechanism_explanation": {"type": "string", "minLength": 20},
                "expected_effects": _strict_object(
                    ["relevance", "efficiency", "robustness", "coverage"],
                    {key: {"type": "string", "minLength": 3} for key in ["relevance", "efficiency", "robustness", "coverage"]},
                ),
                "matched_control": _strict_object(
                    ["control_ref", "rationale"],
                    {
                        "control_ref": {"type": "string", "minLength": 1},
                        "rationale": {"type": "string", "minLength": 12},
                    },
                ),
                "ablation_plan": {
                    "type": "array",
                    "minItems": 1,
                    "items": _strict_object(
                        ["ablation_id", "remove_component_ids", "expected_observation"],
                        {
                            "ablation_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
                            "remove_component_ids": {
                                "type": "array",
                                "minItems": 1,
                                "uniqueItems": True,
                                "items": {"type": "string", "pattern": "^[a-z][a-z0-9_]{0,63}$"},
                            },
                            "expected_observation": {"type": "string", "minLength": 8},
                        },
                    ),
                },
                "failure_modes": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 8}},
                "implementation_plan": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 5}},
                "estimated_cost": _strict_object(
                    ["relative_training_compute", "relative_memory", "precompute_required"],
                    {
                        "relative_training_compute": {"enum": ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]},
                        "relative_memory": {"enum": ["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]},
                        "precompute_required": {"type": "boolean"},
                    },
                ),
                "claim_ceiling": {"const": CLAIM_CEILING},
                "protocol_impact": _strict_object(
                    ["status", "requested_changes"],
                    {
                        "status": {"enum": ["UNCHANGED", "BRANCH_REQUIRED"]},
                        "requested_changes": {
                            "type": "array", "uniqueItems": True, "items": {"enum": FROZEN_PROTOCOL_FIELDS}
                        },
                    },
                ),
            },
        ),
    }


def _family_contract() -> dict[str, Any]:
    return {
        "record_type": "MECHANISM_FAMILY_CONTRACT",
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "scientific_object": "static_offline_implicit_feedback_id_only_collaborative_topn_ranking",
        "supported_profile_kinds": ["OFFLINE_TOPN"],
        "allowed_data_roles": [
            {"role_id": role, "type_ref": type_ref, "derivation_scope": "TRAIN_ONLY"}
            for role, type_ref in sorted(DATA_ROLES.items())
        ],
        "forbidden_data_roles": [
            "SEQUENCE_ORDER",
            "ITEM_METADATA",
            "TEXT",
            "IMAGE",
            "KNOWLEDGE_GRAPH",
            "VALIDATION_LABELS_AS_MODEL_INPUT",
            "TEST_STATISTICS",
            "EXTERNAL_USER_DATA",
            "NETWORK",
        ],
        "output_contract": {
            "type_ref": "core/user_item_relevance_score",
            "candidate_universe_effect": "PRESERVE",
        },
        "frozen_protocol_fields": FROZEN_PROTOCOL_FIELDS,
        "mechanism_mutable_fields": [
            "training_negative_sampling",
            "ranking_objective",
            "optimizer",
            "training_schedule",
            "representation",
            "relation_construction",
            "posthoc_ordering",
        ],
        "excluded_primary_families": [
            "ADMMSLIM",
            "SLIM",
            "EASE",
            "SEQUENTIAL_RECOMMENDATION",
            "SESSION_RECOMMENDATION",
            "MULTIMODAL_RECOMMENDATION",
            "KNOWLEDGE_GRAPH_RECOMMENDATION",
            "COLD_START_PROTOCOL",
            "ONLINE_BANDIT_OPE",
            "LLM_AS_RECOMMENDER",
        ],
        "sparse_linear_boundary": {
            "forbidden": "learned_free_sparse_item_item_reconstruction_as_primary_scorer",
            "allowed": "train_derived_item_item_auxiliary_relation_or_closed_form_graph_filter",
        },
        "type_compatibility": [
            {"provided": "bl_icf/embedding", "accepted_as": "bl_icf/representation"},
            {"provided": "bl_icf/user_item_relation", "accepted_as": "bl_icf/relation"},
            {"provided": "bl_icf/item_item_relation", "accepted_as": "bl_icf/relation"},
            {"provided": "bl_icf/user_user_relation", "accepted_as": "bl_icf/relation"},
            {"provided": "bl_icf/spectral_relation", "accepted_as": "bl_icf/relation"},
            {"provided": "bl_icf/prototype_relation", "accepted_as": "bl_icf/relation"},
        ],
        "claim_ceiling": CLAIM_CEILING,
    }


def _port_contract(
    slot: str, primitive_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    port = lambda name, types, minimum=0: {"port": name, "accepted_types": types, "minimum": minimum}
    if slot == "RELATION_VIEW":
        return [port("source", ["bl_icf/train_interactions", "bl_icf/relation", "bl_icf/train_statistics"], 1)], [{"port": "relation", "type": "bl_icf/relation"}]
    if slot == "EMBEDDING":
        return [port("identity", ["core/user_id", "core/item_id", "bl_icf/train_statistics"], 0)], [{"port": "embedding", "type": "bl_icf/embedding"}]
    if slot == "ENCODER":
        return [port("representation", ["bl_icf/embedding", "bl_icf/representation"], 1), port("relation", ["bl_icf/relation"], 0)], [{"port": "representation", "type": "bl_icf/representation"}]
    if slot == "MESSAGE":
        return [port("representation", ["bl_icf/embedding", "bl_icf/representation"], 1), port("relation", ["bl_icf/relation"], 1)], [{"port": "message", "type": "bl_icf/message"}]
    if slot == "PROPAGATION_AGGREGATION":
        return [port("signal", ["bl_icf/message", "bl_icf/embedding", "bl_icf/representation"], 1), port("relation", ["bl_icf/relation"], 0)], [{"port": "representation", "type": "bl_icf/representation"}]
    if slot == "RELATION_DECOMPOSITION":
        return [port("relation", ["bl_icf/relation"], 1)], [{"port": "relation", "type": "bl_icf/relation"}]
    if slot == "FUSION_ROUTING":
        return [port("branch", ["bl_icf/embedding", "bl_icf/representation"], 1)], [{"port": "representation", "type": "bl_icf/representation"}]
    if slot == "SCORE_HEAD":
        return [port("representation", ["bl_icf/embedding", "bl_icf/representation"], 1)], [{"port": "score", "type": "core/user_item_relevance_score"}]
    if slot == "PRIMARY_OBJECTIVE":
        return [port("score", ["core/user_item_relevance_score"], 1), port("supervision", ["bl_icf/train_interactions"], 1), port("negative_samples", ["bl_icf/negative_samples"], 0)], [{"port": "objective", "type": "bl_icf/objective"}]
    if slot == "NEGATIVE_SAMPLER":
        return [port("interactions", ["bl_icf/train_interactions"], 1)], [{"port": "negative_samples", "type": "bl_icf/negative_samples"}]
    if slot == "SELF_SUPERVISION":
        if primitive_id.startswith("ssl.view."):
            return [
                port(
                    "signal",
                    ["bl_icf/relation", "bl_icf/embedding", "bl_icf/representation"],
                    1,
                )
            ], [{"port": "view", "type": "bl_icf/ssl_view"}]
        return [
            port(
                "views",
                ["bl_icf/embedding", "bl_icf/representation", "bl_icf/ssl_view"],
                2,
            )
        ], [{"port": "auxiliary_objective", "type": "bl_icf/auxiliary_objective"}]
    if slot == "GEOMETRY_REGULARIZATION":
        return [port("representation", ["bl_icf/embedding", "bl_icf/representation"], 1)], [{"port": "regularization", "type": "bl_icf/regularization_term"}]
    if slot == "DENOISING_LONG_TAIL":
        return [port("signal", ["bl_icf/relation", "core/user_item_relevance_score", "bl_icf/objective"], 1)], [{"port": "robustness_term", "type": "bl_icf/robustness_term"}]
    if slot == "TRAINING_PROCEDURE":
        return [port("objective", ["bl_icf/objective", "bl_icf/auxiliary_objective", "bl_icf/regularization_term", "bl_icf/robustness_term"], 1)], [{"port": "training_plan", "type": "bl_icf/training_plan"}]
    if slot == "EFFICIENCY_APPROXIMATION":
        return [port("target", ["bl_icf/relation", "bl_icf/embedding", "bl_icf/representation", "bl_icf/training_plan"], 1)], [{"port": "optimized", "type": "bl_icf/optimized_artifact"}]
    if slot == "POSTHOC_RERANK":
        return [port("score", ["core/user_item_relevance_score"], 1)], [{"port": "score", "type": "core/user_item_relevance_score"}]
    raise ValueError(slot)


def _parameter_schema(slot: str, primitive_id: str) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    if slot == "EMBEDDING":
        properties["dimension"] = {"type": "integer", "minimum": 4, "maximum": 4096}
    if slot == "PROPAGATION_AGGREGATION":
        properties["depth"] = {"type": "integer", "minimum": 0, "maximum": 64}
    if primitive_id == "message.linear_transform":
        properties["activation"] = {"enum": ["NONE", "RELU", "LEAKY_RELU"]}
    if "dropout" in primitive_id:
        properties["rate"] = {"type": "number", "minimum": 0.0, "maximum": 0.95}
    if "topk" in primitive_id or "top_k" in primitive_id:
        properties["top_k"] = {"type": "integer", "minimum": 1}
    if slot in {"PRIMARY_OBJECTIVE", "SELF_SUPERVISION", "GEOMETRY_REGULARIZATION", "DENOISING_LONG_TAIL"}:
        properties["weight"] = {"type": "number", "exclusiveMinimum": 0.0}
    if any(token in primitive_id for token in ("contrastive", "uniformity", "softmax", "info_nce")):
        properties["temperature"] = {"type": "number", "exclusiveMinimum": 0.0}
    if "margin" in primitive_id:
        properties["margin"] = {"type": "number", "minimum": 0.0}
    if slot == "NEGATIVE_SAMPLER":
        properties.update(
            {
                "negative_count": {"type": "integer", "minimum": 1},
                "replacement": {"type": "boolean"},
                "hardness": {"enum": ["NONE", "STATIC", "DYNAMIC", "ADVERSARIAL", "CURRICULUM"]},
                "refresh_frequency": {"enum": ["BATCH", "EPOCH", "STAGE", "FIXED"]},
                "false_negative_policy": {"enum": ["ALLOW_UNKNOWN", "SUPPRESS_KNOWN", "UNCERTAINTY_FILTER"]},
                "purpose": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": True,
                    "items": {"enum": ["RANKING_SIGNAL", "PREVENT_CONSTRAINT_COLLAPSE", "HARDNESS_APPROXIMATION"]},
                },
            }
        )
        required = ["negative_count", "replacement", "hardness", "refresh_frequency", "false_negative_policy", "purpose"]
    if slot == "TRAINING_PROCEDURE":
        properties["learning_rate"] = {"type": "number", "exclusiveMinimum": 0.0}
    return {"type": "object", "additionalProperties": False, "required": required, "properties": properties}


def _primitive_registry() -> dict[str, Any]:
    axes: list[dict[str, Any]] = []
    for slot in SLOTS:
        entries: list[dict[str, Any]] = []
        for primitive_id in _AXIS_PRIMITIVES[slot]:
            inputs, outputs = _port_contract(slot, primitive_id)
            capabilities = list(_SLOT_CAPABILITIES[slot])
            if primitive_id.startswith("encoder.") and any(token in primitive_id for token in ("filter", "constraint")):
                capabilities.append("ANALYTIC_REWRITE")
            if primitive_id in {
                "efficiency.graph_filter_precomputation",
                "efficiency.fixed_point_approximation",
                "efficiency.closed_form_solver",
                "efficiency.remove_message_passing",
            }:
                capabilities.append("ANALYTIC_REWRITE")
            tags: list[str] = []
            if primitive_id.startswith("ssl.view."):
                tags.append("VIEW_GENERATOR")
            if primitive_id.startswith("ssl.objective."):
                tags.append("SSL_OBJECTIVE")
            if primitive_id == "encoder.explicit_message_passing":
                tags.append("EXPLICIT_MESSAGE_PASSING")
            if primitive_id == "encoder.none_mf":
                tags.append("NO_MESSAGE_PASSING")
            if primitive_id == "encoder.closed_form_filter":
                tags.append("CLOSED_FORM_FILTER")
            if primitive_id == "encoder.fixed_point_constraint":
                tags.extend(["FIXED_POINT_CONSTRAINT", "NO_MESSAGE_PASSING"])
            entries.append(
                {
                    "primitive_id": primitive_id,
                    "slot_id": slot,
                    "version": "1",
                    "input_ports": inputs,
                    "output_ports": outputs,
                    "parameter_schema": _parameter_schema(slot, primitive_id),
                    "capabilities": sorted(set(capabilities)),
                    "implementation_status": "BUILTIN_RUNTIME" if primitive_id in _BUILTIN_PRIMITIVES else "LOCAL_RENDERER_REQUIRED",
                    "allowed_read_roles": sorted(DATA_ROLES) if slot in {"RELATION_VIEW", "NEGATIVE_SAMPLER"} else [],
                    "tags": sorted(tags),
                }
            )
        axes.append(
            {
                "slot_id": slot,
                "allow_multiple": slot not in {"ENCODER", "SCORE_HEAD", "NEGATIVE_SAMPLER"},
                "primitives": entries,
            }
        )
    return {
        "record_type": "KNOWN_MECHANISM_PRIMITIVE_REGISTRY",
        "registry_id": "recclaw.bl-icf.known-primitives.v1",
        "registry_version": "1.0.0",
        "axis_count": len(axes),
        "primitive_count": sum(len(axis["primitives"]) for axis in axes),
        "axes": axes,
    }


def _grammar() -> dict[str, Any]:
    analytic = {"approximate_fixed_point", "compile_propagation_to_constraint", "derive_closed_form"}
    efficiency = {"precompute_operator", "sparsify_relation", "distill_teacher", "prune_path", "freeze_path"}
    operators = []
    for operator_id in ARCHITECTURE_OPERATORS:
        capabilities: list[str] = []
        if operator_id in analytic:
            capabilities.append("ANALYTIC_REWRITE")
        if operator_id in efficiency:
            capabilities.append("EFFICIENCY_REWRITE")
        if operator_id == "synthesize_custom_model":
            capabilities.append("CUSTOM_MODEL_IMPLEMENTATION")
        if operator_id in {"decouple_relations", "reweight_relation", "sparsify_relation"}:
            capabilities.append("RELATION_MUTATION")
        if operator_id in {"alternate_optimization"}:
            capabilities.append("TRAINING_ALGORITHM_MUTATION")
        if not capabilities:
            capabilities.append("GRAPH_OPERATOR_MUTATION")
        operators.append(
            {
                "operator_id": operator_id,
                "capabilities": sorted(set(capabilities)),
                "requires_target": operator_id not in {"add_component", "synthesize_custom_model"},
                "requires_replacement": operator_id in {"replace_component", "compile_propagation_to_constraint"},
            }
        )
    return {
        "record_type": "COMPOSITIONAL_MECHANISM_GRAMMAR",
        "grammar_id": "recclaw.bl-icf.mechanism-grammar.v1",
        "grammar_version": "1.0.0",
        "operators": operators,
        "ordinary_candidate_limits": {"core_changes": 1, "support_changes": 1},
        "cross_family_composition": "DENY_UNLESS_REGISTERED_BRIDGE",
    }


def _capability_policy() -> dict[str, Any]:
    return {
        "record_type": "DEVELOPMENT_CAPABILITY_POLICY",
        "policy_id": "recclaw.bl-icf.capability-policy.v1",
        "policy_version": "1.0.0",
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
        "capability_families": CAPABILITIES,
        "candidate_root_template": "recclaw_ext/generated/{candidate_id}/",
        "artifact_root_template": "artifacts/{run_id}/",
        "file_roles": ["MODEL", "RELATION_BUILDER", "LOSS", "SAMPLER", "TRAINER", "RERANKER", "CONFIG", "TEST"],
        "forbidden_changes": [
            "RECBOLE_CORE_EDIT",
            "DATA_SPLIT_CHANGE",
            "EVALUATOR_CHANGE",
            "CANDIDATE_UNIVERSE_CHANGE",
            "BASELINE_RESULT_CHANGE",
            "FORMAL_CLAIM_CHANGE",
            "FORMAL_REGISTRY_CHANGE",
            "ONLINE_DEPENDENCY_INSTALL",
            "EXTERNAL_DATA_ACCESS",
            "OTHER_CANDIDATE_WRITE",
        ],
    }


def _manifest() -> dict[str, Any]:
    refs = [
        ("recclaw_core.mechanism_space.resources", "mechanism_program_envelope_v1.schema.json", "MECHANISM_PROGRAM_ENVELOPE_SCHEMA"),
        ("recclaw_core.mechanism_space.resources", "candidate_execution_binding_v1.schema.json", "CANDIDATE_EXECUTION_BINDING_SCHEMA"),
        ("recclaw_core.search_spaces.bl_icf_v1.resources", "family_contract_v1.json", "FAMILY_CONTRACT"),
        ("recclaw_core.search_spaces.bl_icf_v1.resources", "primitive_registry_v1.json", "PRIMITIVE_REGISTRY"),
        ("recclaw_core.search_spaces.bl_icf_v1.resources", "mechanism_grammar_v1.json", "MECHANISM_GRAMMAR"),
        ("recclaw_core.search_spaces.bl_icf_v1.resources", "capability_policy_v1.json", "CAPABILITY_POLICY"),
        ("recclaw_core.search_spaces.bl_icf_v1.resources", "mechanism_program_v1.schema.json", "FAMILY_PROGRAM_SCHEMA"),
    ]
    return {
        "record_type": "SEARCH_SPACE_MANIFEST",
        "search_space_id": SPACE_ID,
        "search_space_version": SPACE_VERSION,
        "family_id": FAMILY_ID,
        "family_version": FAMILY_VERSION,
        "provider_id": PROVIDER_ID,
        "kernel_schema_version": "recclaw.mechanism-space.kernel.v1",
        "supported_profile_kinds": ["OFFLINE_TOPN"],
        "bridge_policy": "DENY_UNLESS_REGISTERED_BRIDGE",
        "runner_adapter_id": "recclaw.bl-icf.runner-adapter.v1",
        "prompt_projection_version": "recclaw.bl-icf.prompt-projection.v1",
        "runtime_dependencies": RUNTIME_DEPENDENCIES,
        "resource_refs": [
            {"package": package, "name": name, "role": role}
            for package, name, role in refs
        ],
    }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _domain_sha256(domain: str, value: Any) -> str:
    return hashlib.sha256(domain.encode("utf-8") + b"\x00" + rfc8785.dumps(value)).hexdigest()


def _documents() -> dict[Path, bytes]:
    return {
        GENERIC_RESOURCES / "mechanism_program_envelope_v1.schema.json": _json_bytes(_envelope_schema()),
        GENERIC_RESOURCES / "candidate_execution_binding_v1.schema.json": _json_bytes(_binding_schema()),
        BL_RESOURCES / "search_space_manifest_v1.json": _json_bytes(_manifest()),
        BL_RESOURCES / "family_contract_v1.json": _json_bytes(_family_contract()),
        BL_RESOURCES / "primitive_registry_v1.json": _json_bytes(_primitive_registry()),
        BL_RESOURCES / "mechanism_grammar_v1.json": _json_bytes(_grammar()),
        BL_RESOURCES / "capability_policy_v1.json": _json_bytes(_capability_policy()),
        BL_RESOURCES / "mechanism_program_v1.schema.json": _json_bytes(_program_schema()),
    }


def _closure(documents: dict[Path, bytes]) -> bytes:
    if not PROVIDER_SOURCE.exists():
        raise FileNotFoundError(PROVIDER_SOURCE)
    resource_hashes = {
        path.relative_to(SRC).as_posix(): hashlib.sha256(raw).hexdigest()
        for path, raw in sorted(documents.items(), key=lambda item: item[0].as_posix())
    }
    provider_sha256 = hashlib.sha256(PROVIDER_SOURCE.read_bytes()).hexdigest()
    kernel_source_hashes = {
        path.relative_to(SRC).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in KERNEL_SOURCES
    }
    identity_payload = {
        "kernel_source_sha256": kernel_source_hashes,
        "provider_id": PROVIDER_ID,
        "provider_source_sha256": provider_sha256,
        "resource_sha256": resource_hashes,
        "runtime_dependencies": RUNTIME_DEPENDENCIES,
        "search_space_id": SPACE_ID,
        "search_space_version": SPACE_VERSION,
    }
    return _json_bytes(
        {
            "record_type": "SEARCH_SPACE_RESOURCE_CLOSURE",
            **identity_payload,
            "search_space_digest": _domain_sha256("recclaw.search-space.resource-closure.v1", identity_payload),
        }
    )


def build(*, check: bool) -> int:
    documents = _documents()
    documents[BL_RESOURCES / "closure_v1.json"] = _closure(documents)
    mismatches: list[str] = []
    for path, expected in sorted(documents.items(), key=lambda item: item[0].as_posix()):
        if check:
            actual = path.read_bytes() if path.exists() else None
            if actual != expected:
                mismatches.append(path.relative_to(ROOT).as_posix())
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(expected)
    if mismatches:
        for path in mismatches:
            print(f"DRIFT {path}")
        return 2
    if check:
        print(f"PASS resources={len(documents)}")
    else:
        print(f"WROTE resources={len(documents)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return build(check=args.check)


if __name__ == "__main__":
    raise SystemExit(main())
