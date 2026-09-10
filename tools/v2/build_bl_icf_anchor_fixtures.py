#!/usr/bin/env python3
"""Build test-only BL-ICF anchor expressibility fixtures.

These recipes are deliberately stored under ``tests/fixtures`` and are never
loaded by the production provider or proposal prompt projection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.mechanism_space import space_identity  # noqa: E402

OUTPUT = ROOT / "tests" / "fixtures" / "bl_icf_anchor_programs_v1.json"
PROFILE_REF = {
    "profile_id": "recclaw.dev-profile.offline-topn.ml1m.fixture.v1",
    "profile_digest": "1" * 64,
    "profile_kind": "OFFLINE_TOPN",
}
CLAIM_CEILING = "DEVELOPMENT_ONLY_SINGLE_PROTOCOL_NO_GENERAL_CLAIM"


def data(role: str) -> dict[str, str]:
    return {"kind": "DATA", "data_role": role}


def source(component_id: str, output_port: str) -> dict[str, str]:
    return {"kind": "COMPONENT", "component_id": component_id, "output_port": output_port}


def component(
    component_id: str,
    slot_id: str,
    primitive_id: str,
    inputs: list[tuple[str, dict[str, str]]],
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "component_id": component_id,
        "slot_id": slot_id,
        "primitive_id": primitive_id,
        "inputs": [{"port": port, "source": value} for port, value in inputs],
        "parameters": parameters or {},
    }


def relation(component_id: str, primitive_id: str, *, role: str = "TRAIN_INTERACTIONS", **params: Any) -> dict[str, Any]:
    return component(component_id, "RELATION_VIEW", primitive_id, [("source", data(role))], params)


def embedding(
    component_id: str = "embedding",
    primitive_id: str = "embedding.independent_user_item",
    **params: Any,
) -> dict[str, Any]:
    return component(
        component_id,
        "EMBEDDING",
        primitive_id,
        [("identity", data("USER_ID")), ("identity", data("ITEM_ID"))],
        {"dimension": 64, **params},
    )


def encoder(
    primitive_id: str,
    *,
    component_id: str = "encoder",
    representation: str = "embedding",
    relations: tuple[str, ...] = (),
) -> dict[str, Any]:
    inputs = [("representation", source(representation, "embedding" if representation == "embedding" else "representation"))]
    inputs.extend(("relation", source(item, "relation")) for item in relations)
    return component(component_id, "ENCODER", primitive_id, inputs)


def message(
    component_id: str,
    primitive_id: str,
    *,
    representation: str,
    relation_id: str,
    **params: Any,
) -> dict[str, Any]:
    return component(
        component_id,
        "MESSAGE",
        primitive_id,
        [
            ("representation", source(representation, "representation" if representation != "embedding" else "embedding")),
            ("relation", source(relation_id, "relation")),
        ],
        params,
    )


def propagation(
    component_id: str,
    primitive_id: str,
    *,
    signal_id: str,
    signal_port: str = "message",
    additional_signals: tuple[tuple[str, str], ...] = (),
    relation_id: str | None = None,
    **params: Any,
) -> dict[str, Any]:
    inputs = [("signal", source(signal_id, signal_port))]
    inputs.extend(("signal", source(item, port)) for item, port in additional_signals)
    if relation_id:
        inputs.append(("relation", source(relation_id, "relation")))
    return component(component_id, "PROPAGATION_AGGREGATION", primitive_id, inputs, params)


def fusion(
    component_id: str,
    primitive_id: str,
    branches: list[tuple[str, str]],
    **params: Any,
) -> dict[str, Any]:
    return component(
        component_id,
        "FUSION_ROUTING",
        primitive_id,
        [("branch", source(item, port)) for item, port in branches],
        params,
    )


def score(
    component_id: str,
    primitive_id: str,
    representation: str,
    port: str = "representation",
    **params: Any,
) -> dict[str, Any]:
    return component(
        component_id,
        "SCORE_HEAD",
        primitive_id,
        [("representation", source(representation, port))],
        params,
    )


def sampler(
    primitive_id: str = "sampler.uniform",
    *,
    negative_count: int = 1,
    purpose: list[str] | None = None,
) -> dict[str, Any]:
    return component(
        "sampler",
        "NEGATIVE_SAMPLER",
        primitive_id,
        [("interactions", data("TRAIN_INTERACTIONS"))],
        {
            "negative_count": negative_count,
            "replacement": True,
            "hardness": "NONE" if "hard" not in primitive_id else "DYNAMIC",
            "refresh_frequency": "BATCH",
            "false_negative_policy": "ALLOW_UNKNOWN",
            "purpose": purpose or ["RANKING_SIGNAL"],
        },
    )


def objective(
    component_id: str,
    primitive_id: str,
    *,
    score_id: str = "score",
    sampler_id: str | None = "sampler",
    **params: Any,
) -> dict[str, Any]:
    inputs = [
        ("score", source(score_id, "score")),
        ("supervision", data("TRAIN_INTERACTIONS")),
    ]
    if sampler_id:
        inputs.append(("negative_samples", source(sampler_id, "negative_samples")))
    return component(component_id, "PRIMARY_OBJECTIVE", primitive_id, inputs, params)


def ssl_view(
    component_id: str,
    primitive_id: str,
    signal_id: str,
    signal_port: str,
    **params: Any,
) -> dict[str, Any]:
    return component(
        component_id,
        "SELF_SUPERVISION",
        primitive_id,
        [("signal", source(signal_id, signal_port))],
        params,
    )


def ssl_objective(
    component_id: str,
    primitive_id: str,
    views: list[tuple[str, str]],
    **params: Any,
) -> dict[str, Any]:
    return component(
        component_id,
        "SELF_SUPERVISION",
        primitive_id,
        [("views", source(item, port)) for item, port in views],
        {"weight": 0.1, "temperature": 0.2, **params},
    )


def regularizer(
    component_id: str,
    primitive_id: str,
    representation: str,
    port: str,
    **params: Any,
) -> dict[str, Any]:
    return component(
        component_id,
        "GEOMETRY_REGULARIZATION",
        primitive_id,
        [("representation", source(representation, port))],
        {"weight": 0.1, **params},
    )


def training(objective_refs: list[tuple[str, str]]) -> dict[str, Any]:
    return component(
        "training",
        "TRAINING_PROCEDURE",
        "training.adam",
        [("objective", source(item, port)) for item, port in objective_refs],
        {"learning_rate": 0.001},
    )


def efficiency(component_id: str, primitive_id: str, target_id: str, target_port: str, **params: Any) -> dict[str, Any]:
    return component(
        component_id,
        "EFFICIENCY_APPROXIMATION",
        primitive_id,
        [("target", source(target_id, target_port))],
        params,
    )


def envelope(
    *,
    components: list[dict[str, Any]],
    construction_mode: str,
    changed_slots: list[tuple[str, str]],
    core_hypothesis: str,
    ablation_component: str,
    operators: list[dict[str, Any]] | None = None,
    removed_slots: list[str] | None = None,
    custom_components: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    identity = space_identity("BL_ICF_MECHANISM_SPACE_V1")
    return {
        "record_type": "MECHANISM_PROGRAM_ENVELOPE",
        "kernel_schema_version": "recclaw.mechanism-space.kernel.v1",
        "search_space_id": identity.search_space_id,
        "search_space_digest": identity.search_space_digest,
        "family_id": identity.family_id,
        "family_version": identity.family_version,
        "profile_ref": PROFILE_REF,
        "program_payload": {
            "schema_version": "recclaw.bl-icf.mechanism-program.v1",
            "family_contract_id": "BL_ICF_V1",
            "construction_mode": construction_mode,
            "parent_refs": [],
            "research_question": "Can this mechanism program reproduce the intended collaborative-filtering anchor semantics?",
            "core_hypothesis": core_hypothesis,
            "input_semantics": {
                "user_id": True,
                "item_id": True,
                "train_interactions": True,
                "external_features": False,
            },
            "components": components,
            "architecture_operators": operators or [],
            "changed_slots": [
                {"slot_id": slot_id, "change_role": role}
                for slot_id, role in changed_slots
            ],
            "removed_slots": removed_slots or [],
            "custom_components": custom_components or [],
            "mechanism_explanation": "A test-only executable mechanism graph used to verify search-space expressibility without exposing an answer recipe to proposal generation.",
            "expected_effects": {
                "relevance": "fixture only",
                "efficiency": "fixture only",
                "robustness": "fixture only",
                "coverage": "fixture only",
            },
            "matched_control": {
                "control_ref": "TEST_ONLY_ANCHOR_CONTROL",
                "rationale": "The fixture checks semantic expression against an independently declared anchor signature.",
            },
            "ablation_plan": [
                {
                    "ablation_id": "remove_core_mechanism",
                    "remove_component_ids": [ablation_component],
                    "expected_observation": "The expected anchor signature is no longer present after removing the core mechanism.",
                }
            ],
            "failure_modes": ["The grammar may accept the document while failing to preserve the intended mechanism signature."],
            "implementation_plan": ["Compile the fixture into resolved mechanism IR and compare its structural signature."],
            "estimated_cost": {
                "relative_training_compute": "LOW",
                "relative_memory": "LOW",
                "precompute_required": any(item["slot_id"] == "EFFICIENCY_APPROXIMATION" for item in components),
            },
            "claim_ceiling": CLAIM_CEILING,
            "protocol_impact": {"status": "UNCHANGED", "requested_changes": []},
        },
    }


def lightgcn_core() -> list[dict[str, Any]]:
    return [
        relation("ui_graph", "relation.user_item_bipartite"),
        embedding(),
        encoder("encoder.explicit_message_passing", relations=("ui_graph",)),
        message("message", "message.identity", representation="encoder", relation_id="ui_graph"),
        propagation("propagation", "propagation.symmetric_normalization", signal_id="message", relation_id="ui_graph", depth=3),
        fusion("fusion", "fusion.layer_weighted_sum", [("encoder", "representation"), ("propagation", "representation")]),
        score("score", "score.dot_product", "fusion"),
        sampler(),
        objective("objective", "objective.bpr"),
        training([("objective", "objective")]),
    ]


def fixtures() -> list[dict[str, Any]]:
    bpr = [
        embedding(),
        encoder("encoder.none_mf"),
        score("score", "score.dot_product", "encoder"),
        sampler(),
        objective("objective", "objective.bpr"),
        training([("objective", "objective")]),
    ]

    ngcf = [
        relation("ui_graph", "relation.user_item_bipartite"),
        embedding(),
        encoder("encoder.explicit_message_passing", relations=("ui_graph",)),
        message(
            "linear_message",
            "message.linear_transform",
            representation="encoder",
            relation_id="ui_graph",
            activation="LEAKY_RELU",
        ),
        message("interaction_message", "message.elementwise_interaction", representation="encoder", relation_id="ui_graph"),
        propagation(
            "propagation",
            "propagation.attention_aggregation",
            signal_id="linear_message",
            additional_signals=(("interaction_message", "message"),),
            relation_id="ui_graph",
            depth=3,
        ),
        score("score", "score.dot_product", "propagation"),
        sampler(),
        objective("objective", "objective.bpr"),
        training([("objective", "objective")]),
    ]

    sgl = lightgcn_core()
    sgl.insert(7, ssl_view("edge_view", "ssl.view.edge_dropout", "ui_graph", "relation"))
    sgl.insert(
        8,
        ssl_objective(
            "ssl_objective",
            "ssl.objective.info_nce",
            [("fusion", "representation"), ("edge_view", "view")],
        ),
    )
    sgl[-1] = training([("objective", "objective"), ("ssl_objective", "auxiliary_objective")])

    # RecSys-2025 SGCL (DavidZWZ/SGCL), not the acronym-colliding
    # TOIS-2025 Symmetric Graph Contrastive Learning model.  SGCL replaces
    # the decoupled BPR + SSL objectives with one augmentation-free supervised
    # graph-contrastive primary loss and does not use an explicit sampler.
    sgcl_recsys2025 = lightgcn_core()[:7]
    sgcl_recsys2025.extend(
        [
            component(
                "objective",
                "PRIMARY_OBJECTIVE",
                "objective.unified_supervised_graph_contrastive",
                [
                    ("representation", source("fusion", "representation")),
                    ("supervision", data("TRAIN_INTERACTIONS")),
                ],
                {
                    "weight": 1.0,
                    "temperature": 0.2,
                    "positive_pair_source": "TRAIN_INTERACTIONS",
                    "representation_normalization": "L2",
                    "denominator_scope": "IN_BATCH_SAME_TYPE_USER_PLUS_ITEM",
                    "separate_recommendation_loss": False,
                    "graph_augmentation": False,
                },
            ),
            training([("objective", "objective")]),
        ]
    )

    simgcl = lightgcn_core()
    simgcl.insert(
        7,
        ssl_view(
            "noise_view",
            "ssl.view.uniform_perturbation",
            "fusion",
            "representation",
            epsilon=0.1,
        ),
    )
    simgcl.insert(
        8,
        ssl_objective(
            "ssl_objective",
            "ssl.objective.info_nce",
            [("fusion", "representation"), ("noise_view", "view")],
        ),
    )
    simgcl[-1] = training([("objective", "objective"), ("ssl_objective", "auxiliary_objective")])

    xsimgcl = lightgcn_core()
    xsimgcl.insert(
        7,
        ssl_view(
            "signed_noise_view",
            "ssl.view.signed_uniform_perturbation",
            "fusion",
            "representation",
            epsilon=0.1,
        ),
    )
    xsimgcl.insert(
        8,
        ssl_view(
            "cross_layer_view",
            "ssl.view.cross_layer",
            "propagation",
            "representation",
        ),
    )
    xsimgcl.insert(
        9,
        ssl_objective(
            "cross_layer_objective",
            "ssl.objective.cross_layer_info_nce",
            [
                ("signed_noise_view", "view"),
                ("cross_layer_view", "view"),
            ],
            source_layer=1,
            target_layer=3,
        ),
    )
    xsimgcl[-1] = training(
        [("objective", "objective"), ("cross_layer_objective", "auxiliary_objective")]
    )

    dgcf = [
        relation(
            "intent_graph",
            "relation.intent_aware_bipartite",
            intent_count=4,
        ),
        embedding("embedding", "embedding.intent_chunks", intent_count=4),
        encoder("encoder.explicit_message_passing", relations=("intent_graph",)),
        component(
            "latent_intent_graphs",
            "RELATION_DECOMPOSITION",
            "relation_decomposition.latent_intent_graphs",
            [("relation", source("intent_graph", "relation"))],
            {"intent_count": 4},
        ),
        component(
            "intent_refinement",
            "RELATION_DECOMPOSITION",
            "relation_decomposition.iterative_intent_refinement",
            [("relation", source("latent_intent_graphs", "relation"))],
            {"intent_count": 4, "routing_iterations": 2},
        ),
        message(
            "intent_message",
            "message.intent_routed",
            representation="encoder",
            relation_id="intent_refinement",
            intent_count=4,
            routing_iterations=2,
        ),
        propagation(
            "intent_propagation",
            "propagation.symmetric_normalization",
            signal_id="intent_message",
            relation_id="intent_refinement",
            depth=3,
        ),
        fusion(
            "fusion",
            "fusion.gated",
            [("encoder", "representation"), ("intent_propagation", "representation")],
        ),
        score("score", "score.dot_product", "fusion"),
        sampler(),
        objective("objective", "objective.bpr"),
        regularizer(
            "intent_independence",
            "regularizer.distance_correlation_intent_independence",
            "fusion",
            "representation",
        ),
        training(
            [("objective", "objective"), ("intent_independence", "regularization")]
        ),
    ]

    ncl = lightgcn_core()
    ncl.insert(1, relation("prototype_relation", "relation.prototype_cluster"))
    ncl.insert(8, ssl_view("structural_view", "ssl.view.structural_neighbor", "ui_graph", "relation"))
    ncl.insert(9, ssl_view("semantic_view", "ssl.view.semantic_prototype", "prototype_relation", "relation"))
    ncl.insert(
        10,
        ssl_objective(
            "prototype_objective",
            "ssl.objective.prototype_contrastive",
            [
                ("fusion", "representation"),
                ("structural_view", "view"),
                ("semantic_view", "view"),
            ],
        ),
    )
    ncl[-1] = training([("objective", "objective"), ("prototype_objective", "auxiliary_objective")])

    lightgcl = lightgcn_core()
    lightgcl.insert(1, relation("svd_view", "relation.svd_global"))
    lightgcl.insert(
        8,
        ssl_view(
            "directional_support",
            "ssl.view.directional_edge_dropout",
            "ui_graph",
            "relation",
            user_to_item_rate=0.1,
            item_to_user_rate=0.1,
            independent_directions=True,
        ),
    )
    lightgcl.insert(9, ssl_view("global_view", "ssl.view.svd_global", "svd_view", "relation"))
    lightgcl.insert(
        10,
        ssl_objective(
            "local_global_objective",
            "ssl.objective.info_nce",
            [
                ("fusion", "representation"),
                ("directional_support", "view"),
                ("global_view", "view"),
            ],
            weight=0.0001,
            temperature=2.0,
        ),
    )
    lightgcl[-1] = training([("objective", "objective"), ("local_global_objective", "auxiliary_objective")])

    directau = [
        embedding(),
        encoder("encoder.none_mf"),
        score("score", "score.dot_product", "encoder"),
        sampler(),
        objective("objective", "objective.alignment_uniformity", weight=1.0, temperature=0.2),
        regularizer("alignment", "regularizer.alignment", "encoder", "representation"),
        regularizer("uniformity", "regularizer.uniformity", "encoder", "representation"),
        training([("objective", "objective"), ("alignment", "regularization"), ("uniformity", "regularization")]),
    ]

    simplex = [
        embedding(),
        encoder("encoder.none_mf"),
        score("score", "score.cosine_similarity", "encoder"),
        sampler(negative_count=100),
        objective("objective", "objective.cosine_contrastive", weight=1.0, temperature=0.2),
        training([("objective", "objective")]),
    ]

    gfcf = [
        relation("ui_graph", "relation.user_item_bipartite"),
        relation("spectral_view", "relation.svd_global"),
        embedding(),
        encoder("encoder.closed_form_filter", relations=("ui_graph", "spectral_view")),
        efficiency("precompute", "efficiency.graph_filter_precomputation", "spectral_view", "relation"),
        score("score", "score.dot_product", "encoder"),
    ]

    ultra = [
        relation("ui_graph", "relation.user_item_bipartite"),
        relation("ii_topk", "relation.item_item_cooccurrence_topk", top_k=10),
        embedding(),
        encoder("encoder.fixed_point_constraint", relations=("ui_graph", "ii_topk")),
        score("score", "score.dot_product", "encoder"),
        sampler(
            "sampler.sampled_unobserved",
            negative_count=1,
            purpose=["RANKING_SIGNAL", "PREVENT_CONSTRAINT_COLLAPSE"],
        ),
        objective("pointwise", "objective.pointwise_bce"),
        objective("ui_constraint", "objective.user_item_structure_constraint"),
        objective("ii_constraint", "objective.item_item_relation_constraint"),
        efficiency("remove_propagation", "efficiency.remove_message_passing", "encoder", "representation"),
        efficiency("precompute_relations", "efficiency.topk_relation_selection", "ii_topk", "relation", top_k=10),
        training([("pointwise", "objective"), ("ui_constraint", "objective"), ("ii_constraint", "objective")]),
    ]
    ultra_operators = [
        {
            "operator_id": "remove_component",
            "targets": ["slot:PROPAGATION_AGGREGATION"],
            "replacements": [],
            "parameters": {},
            "rationale": "Remove iterative message passing from the final mechanism graph.",
        },
        {
            "operator_id": "compile_propagation_to_constraint",
            "targets": ["slot:PROPAGATION_AGGREGATION"],
            "replacements": ["encoder", "ui_constraint", "ii_constraint"],
            "parameters": {},
            "rationale": "Compile the intended structural effect into train-derived relation constraints.",
        },
    ]

    lightgode = [
        relation("ui_graph", "relation.user_item_bipartite"),
        embedding(),
        encoder(
            "encoder.post_training_graph_ode",
            relations=("ui_graph",),
        ),
        efficiency(
            "one_step_ode",
            "efficiency.one_step_euler_ode",
            "encoder",
            "representation",
            integration_steps=1,
            step_size=1.0,
        ),
        score("score", "score.dot_product", "one_step_ode"),
        sampler(),
        objective("objective", "objective.bpr"),
        component(
            "training",
            "TRAINING_PROCEDURE",
            "training.pretrain_then_post_graph_inference",
            [("objective", source("objective", "objective"))],
            {"learning_rate": 0.001},
        ),
    ]

    lightgcnpp = [
        relation("ui_graph", "relation.user_item_bipartite"),
        embedding(),
        encoder("encoder.explicit_message_passing", relations=("ui_graph",)),
        message("message", "message.identity", representation="encoder", relation_id="ui_graph"),
        propagation(
            "learned_normalization",
            "propagation.learned_normalization",
            signal_id="message",
            relation_id="ui_graph",
            depth=3,
            left_degree_exponent=-0.5,
            right_degree_exponent=-0.5,
            learnable=True,
        ),
        fusion(
            "residual_coefficients",
            "fusion.layer_weighted_sum",
            [("encoder", "representation"), ("learned_normalization", "representation")],
            learnable_weights=True,
        ),
        score("score", "score.dot_product", "residual_coefficients"),
        sampler(),
        objective("objective", "objective.bpr"),
        training([("objective", "objective")]),
    ]

    lightcscf = [
        relation("ui_graph", "relation.user_item_bipartite"),
        embedding(),
        encoder("encoder.none_mf"),
        ssl_view(
            "parallel_filter_view",
            "ssl.view.parallel_graph_filter",
            "ui_graph",
            "relation",
            filter_order=1,
            normalization="GENERALIZED_GRAM",
            weight=1.0,
        ),
        score(
            "score",
            "score.margin_constrained_cosine",
            "encoder",
            margin=0.2,
        ),
        sampler(),
        objective(
            "objective",
            "objective.cosine_contrastive",
            weight=1.0,
            temperature=0.3,
        ),
        ssl_objective(
            "parallel_filter_objective",
            "ssl.objective.info_nce",
            [("encoder", "representation"), ("parallel_filter_view", "view")],
            weight=1.0,
            temperature=0.3,
        ),
        training(
            [
                ("objective", "objective"),
                ("parallel_filter_objective", "auxiliary_objective"),
            ]
        ),
    ]

    lightccf = lightgcn_core()
    lightccf.insert(
        7,
        ssl_view(
            "neighborhood_view",
            "ssl.view.structural_neighbor",
            "ui_graph",
            "relation",
        ),
    )
    lightccf.insert(
        8,
        ssl_objective(
            "neighborhood_objective",
            "ssl.objective.neighborhood_aggregate",
            [("fusion", "representation"), ("neighborhood_view", "view")],
            weight=0.1,
            temperature=0.2,
        ),
    )
    lightccf[-1] = training(
        [("objective", "objective"), ("neighborhood_objective", "auxiliary_objective")]
    )

    def egcf_program(training_primitive: str) -> list[dict[str, Any]]:
        values = [
            relation("ui_graph", "relation.user_item_bipartite"),
            component(
                "id_signal",
                "EMBEDDING",
                "embedding.embeddingless_id_signal",
                [("identity", data("USER_ID")), ("identity", data("ITEM_ID"))],
                {"signal_mode": "ONE_HOT_IDENTITY"},
            ),
            encoder(
                "encoder.embeddingless_propagation",
                representation="id_signal",
                relations=("ui_graph",),
            ),
            message(
                "message",
                "message.identity",
                representation="encoder",
                relation_id="ui_graph",
            ),
            propagation(
                "propagation",
                "propagation.symmetric_normalization",
                signal_id="message",
                relation_id="ui_graph",
                depth=3,
            ),
            score("score", "score.dot_product", "propagation"),
            sampler(),
            objective("objective", "objective.bpr"),
            ssl_objective(
                "layer_objective",
                "ssl.objective.augmentation_free_layer_contrastive",
                [("encoder", "representation"), ("propagation", "representation")],
                weight=0.1,
                temperature=0.2,
            ),
        ]
        training_parameters: dict[str, Any] = {"learning_rate": 0.001}
        if training_primitive == "training.parallel_joint_update":
            training_parameters["update_mode"] = "PARALLEL_JOINT"
        values.append(
            component(
                "training",
                "TRAINING_PROCEDURE",
                training_primitive,
                [
                    ("objective", source("objective", "objective")),
                    ("objective", source("layer_objective", "auxiliary_objective")),
                ],
                training_parameters,
            )
        )
        return values

    egcf_parallel = egcf_program("training.parallel_joint_update")
    egcf_alternating = egcf_program("training.view_model_alternating_update")

    hgformer = [
        relation("ui_graph", "relation.user_item_bipartite"),
        embedding("embedding", "embedding.hyperbolic_manifold", curvature=1.0),
        encoder(
            "encoder.hyperbolic_local_global",
            relations=("ui_graph",),
        ),
        message(
            "hyperbolic_centroid",
            "message.hyperbolic_centroid",
            representation="encoder",
            relation_id="ui_graph",
            curvature=1.0,
        ),
        propagation(
            "centroid_propagation",
            "propagation.mean",
            signal_id="hyperbolic_centroid",
            relation_id="ui_graph",
            depth=2,
        ),
        fusion(
            "cross_attention",
            "fusion.hyperbolic_cross_attention",
            [("encoder", "representation"), ("centroid_propagation", "representation")],
            curvature=1.0,
        ),
        efficiency(
            "linear_attention",
            "efficiency.linear_cross_attention",
            "cross_attention",
            "representation",
            feature_map="ELU_PLUS_ONE",
            projection_dimension=64,
        ),
        score("score", "score.distance_based", "linear_attention"),
        sampler(),
        objective("objective", "objective.bpr"),
        regularizer(
            "distortion",
            "regularizer.hyperbolic_distortion",
            "linear_attention",
            "representation",
            curvature=1.0,
        ),
        training([("objective", "objective"), ("distortion", "regularization")]),
    ]

    ddc = lightgcn_core()
    ddc.insert(
        -1,
        component(
            "directional_correction",
            "DENOISING_LONG_TAIL",
            "robustness.asymmetric_directional_correction",
            [("signal", source("objective", "objective"))],
            {
                "popularity_direction_source": "PRETRAINED_ITEM_EMBEDDING",
                "preference_direction_source": "USER_HISTORY",
                "positive_correction": "PREFERENCE_ALIGNED",
                "negative_correction": "POPULARITY_OPPOSING",
                "freeze_base_representation": True,
                "history_fraction": 0.2,
                "weight": 1.0,
            },
        ),
    )
    ddc[-1] = training(
        [("objective", "objective"), ("directional_correction", "robustness_term")]
    )

    recipes = [
        ("BPR_MF", bpr, "COMPOSITION", [("PRIMARY_OBJECTIVE", "CORE")], "A no-propagation latent-factor program with dot scoring and pairwise BPR objective.", "objective", [], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("LIGHTGCN", lightgcn_core(), "COMPOSITION", [("PROPAGATION_AGGREGATION", "CORE")], "Linear symmetric graph propagation with layer fusion reproduces the LightGCN mechanism signature.", "propagation", [], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("NGCF", ngcf, "ARCHITECTURE_REWRITE", [("MESSAGE", "CORE"), ("PROPAGATION_AGGREGATION", "SUPPORT")], "Transformed and interaction messages express explicit nonlinear graph collaborative filtering.", "interaction_message", [], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("SGL", sgl, "COMPOSITION", [("SELF_SUPERVISION", "CORE"), ("RELATION_VIEW", "SUPPORT")], "Graph augmentation and a contrastive auxiliary objective express self-supervised graph CF.", "ssl_objective", [], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("SGCL_RECSYS2025", sgcl_recsys2025, "ARCHITECTURE_REWRITE", [("PRIMARY_OBJECTIVE", "CORE")], "A single augmentation-free supervised graph-contrastive objective unifies interaction supervision and representation contrast without a separate BPR or SSL branch.", "objective", [], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("SIMGCL", simgcl, "COMPOSITION", [("SELF_SUPERVISION", "CORE")], "Embedding perturbation plus same-depth contrastive learning expresses the simple noise-based graph SSL mechanism.", "noise_view", [], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("XSIMGCL", xsimgcl, "ARCHITECTURE_REWRITE", [("SELF_SUPERVISION", "CORE"), ("FUSION_ROUTING", "SUPPORT")], "Signed uniform perturbation and cross-layer contrast express a noise-based graph SSL mechanism with depth-separated views.", "cross_layer_objective", [], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("DGCF", dgcf, "ARCHITECTURE_REWRITE", [("RELATION_DECOMPOSITION", "CORE"), ("GEOMETRY_REGULARIZATION", "SUPPORT")], "Intent-aware graph routing and distance-correlation regularization express disentangled collaborative propagation.", "intent_refinement", [], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("NCL", ncl, "ARCHITECTURE_REWRITE", [("SELF_SUPERVISION", "CORE"), ("RELATION_VIEW", "SUPPORT")], "Structural and prototype-semantic neighbors support a prototype contrastive objective.", "prototype_objective", [], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("LIGHTGCL", lightgcl, "ARCHITECTURE_REWRITE", [("RELATION_VIEW", "CORE"), ("SELF_SUPERVISION", "SUPPORT")], "A train-derived SVD global relation enables local-global contrastive learning.", "svd_view", [], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("DIRECTAU", directau, "ARCHITECTURE_REWRITE", [("PRIMARY_OBJECTIVE", "CORE"), ("GEOMETRY_REGULARIZATION", "SUPPORT")], "Direct alignment and uniformity optimization expresses a geometry-first CF objective.", "objective", [], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("SIMPLEX", simplex, "COMPOSITION", [("PRIMARY_OBJECTIVE", "CORE"), ("NEGATIVE_SAMPLER", "SUPPORT")], "Cosine contrastive training with a large negative ratio expresses sampling-centric CF.", "objective", [], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("GF_CF", gfcf, "ARCHITECTURE_REWRITE", [("ENCODER", "CORE"), ("EFFICIENCY_APPROXIMATION", "SUPPORT")], "A train-derived spectral relation and precomputed closed-form filter express graph-filter CF.", "precompute", [{"operator_id": "derive_closed_form", "targets": ["encoder"], "replacements": ["precompute"], "parameters": {}, "rationale": "Derive the scoring representation through a precomputed graph filter."}], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("ULTRAGCN", ultra, "ARCHITECTURE_REWRITE", [("ENCODER", "CORE"), ("RELATION_DECOMPOSITION", "SUPPORT")], "Fixed-point user-item and item-item constraints replace explicit propagation while preserving collaborative structure.", "encoder", ultra_operators, ["PROPAGATION_AGGREGATION"], "VALID_NEEDS_IMPLEMENTATION"),
        ("LIGHTGODE", lightgode, "ARCHITECTURE_REWRITE", [("ENCODER", "CORE"), ("EFFICIENCY_APPROXIMATION", "SUPPORT")], "MF-style training followed by one-step graph-ODE inference separates representation learning from graph smoothing.", "one_step_ode", [], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("LIGHTGCNPP", lightgcnpp, "ARCHITECTURE_REWRITE", [("PROPAGATION_AGGREGATION", "CORE"), ("FUSION_ROUTING", "SUPPORT")], "Learnable propagation normalization and residual layer coefficients expose a controlled LightGCN propagation replacement.", "learned_normalization", [], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("LIGHTCSCF", lightcscf, "ARCHITECTURE_REWRITE", [("SCORE_HEAD", "CORE"), ("SELF_SUPERVISION", "SUPPORT")], "Margin-constrained cosine scoring and a parallel train-derived graph-filter view express the audited cosine-filter mechanism without changing the frozen candidate universe.", "score", [], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("LIGHTCCF", lightccf, "ARCHITECTURE_REWRITE", [("SELF_SUPERVISION", "CORE")], "Neighborhood-aggregate contrast over a graph-derived support view expresses the audited local-neighborhood contrastive mechanism.", "neighborhood_objective", [], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("EGCF_PARALLEL", egcf_parallel, "ARCHITECTURE_REWRITE", [("EMBEDDING", "SUPPORT"), ("ENCODER", "CORE"), ("SELF_SUPERVISION", "SUPPORT"), ("TRAINING_PROCEDURE", "SUPPORT")], "A deterministic identity signal, embeddingless graph propagation, augmentation-free layer contrast, and parallel joint updates express the published embeddingless regime without a hidden embedding table.", "encoder", [], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("EGCF_ALTERNATING", egcf_alternating, "ARCHITECTURE_REWRITE", [("EMBEDDING", "SUPPORT"), ("ENCODER", "CORE"), ("SELF_SUPERVISION", "SUPPORT"), ("TRAINING_PROCEDURE", "SUPPORT")], "The same embeddingless augmentation-free mechanism with explicit view/model alternating updates keeps the published optimization regimes semantically distinct.", "training", [], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("HGFORMER", hgformer, "ARCHITECTURE_REWRITE", [("EMBEDDING", "SUPPORT"), ("ENCODER", "CORE"), ("FUSION_ROUTING", "SUPPORT"), ("EFFICIENCY_APPROXIMATION", "SUPPORT"), ("GEOMETRY_REGULARIZATION", "SUPPORT")], "Hyperbolic local-global encoding, centroid aggregation, cross-attention, distortion control, and a causal linear-attention approximation express the audited hyperbolic mechanism and its stability boundary.", "cross_attention", [], [], "VALID_NEEDS_IMPLEMENTATION"),
        ("DDC", ddc, "ARCHITECTURE_REWRITE", [("DENOISING_LONG_TAIL", "CORE"), ("TRAINING_PROCEDURE", "SUPPORT")], "A train-derived global popularity direction and user-specific preference direction support asymmetric positive and negative corrections while the pretrained collaborative representation stays frozen.", "directional_correction", [], [], "VALID_NEEDS_IMPLEMENTATION"),
    ]
    result = []
    for name, values, mode, changes, hypothesis, ablation, operators, removed, expected_status in recipes:
        result.append(
            {
                "anchor_name": name,
                "expected_status": expected_status,
                "expected_primitives": sorted(item["primitive_id"] for item in values),
                "program": envelope(
                    components=values,
                    construction_mode=mode,
                    changed_slots=changes,
                    core_hypothesis=hypothesis,
                    ablation_component=ablation,
                    operators=operators,
                    removed_slots=removed,
                ),
            }
        )
    return result


def fixture_document() -> dict[str, Any]:
    return {
        "record_type": "BL_ICF_ANCHOR_EXPRESSIBILITY_FIXTURES",
        "fixture_version": "1.1.0",
        "test_only": True,
        "proposal_prompt_exposure": "FORBIDDEN",
        "fixtures": fixtures(),
    }


def encoded() -> bytes:
    return (
        json.dumps(fixture_document(), ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = encoded()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_bytes() != expected:
            print(f"DRIFT {OUTPUT.relative_to(ROOT).as_posix()}")
            return 2
        print(f"PASS fixtures={len(fixtures())}")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_bytes(expected)
    print(f"WROTE fixtures={len(fixtures())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
