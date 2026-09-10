"""Semantic-ID generative recommendation domain language for V2R4."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from recclaw_core.mechanism_space import CompileDiagnostic, CompileStatus
from recclaw_core.mechanism_space.canonical import deep_thaw
from recclaw_core.mechanism_space.declarative_provider import (
    DeclarativeFamilySpec,
    DeclarativeMechanismSpaceProvider,
    axis,
    object_parameters,
    primitive,
    standard_operators,
)
from recclaw_core.search_spaces.semantic_id_generative_v1.primitive_bindings import (
    EXECUTABLE_PRIMITIVE_BINDINGS,
    SUSPENDED_PRIMITIVES,
    component_binding_projection,
    executable_axes,
)

SPACE_ID = "SEMANTIC_ID_GENERATIVE_MECHANISM_SPACE_V1"
FAMILY_ID = "SEMANTIC_ID_GENERATIVE_V1"
PROVIDER_ID = "recclaw.search-space-provider.semantic-id-generative.v1"

FEATURE = "semantic_id/item_feature"
STRUCTURED_FIELDS = "semantic_id/item_structured_fields"
LATENT = "semantic_id/item_latent"
CODEBOOK = "semantic_id/codebook"
SID = "semantic_id/item_code_sequence"
CONTEXT = "semantic_id/user_context"
TOKEN_STATE = "semantic_id/token_state"
HYPOTHESIS = "semantic_id/item_hypothesis"
SCORE = "core/user_item_relevance_score"
_LIGER_PROJECTION_DIMENSION = 128
_LIGER_PRODUCT_QUANTIZATION_SUBSPACES = (1, 2, 4, 8, 16, 32, 64)
_PRODUCT_QUANTIZER_PRIMITIVES = {
    "tokenizer.product_quantization",
    "tokenizer.orthogonally_preconditioned_product_quantization",
}
_FROZEN_LETTER_TOKENIZER_PRIMITIVE = "tokenizer.frozen_letter_collision_suffix"
_PARENT_LOCAL_SEAM_SLOTS = frozenset(
    {"USER_CONTEXT_ENCODER", "GENERATIVE_OBJECTIVE", "DECODING_STRATEGY"}
)
_PARENT_LOCAL_DECODING_PRIMITIVES = frozenset(
    {
        "decode.autoregressive_beam_then_invalid_drop",
        "decode.constrained_beam_trie",
    }
)
_CANONICAL_PARENT_DECLARED_SLOTS = frozenset({"DENSE_RETRIEVAL_CORRECTION"})


def _p(properties: dict, *required: str) -> dict:
    return object_parameters(properties, required=required)


AXES = (
    axis(
        "ITEM_FEATURE_ENCODER",
        research_role="Map allowed item content and/or train-only collaborative evidence into tokenizer input latents.",
        interaction_guidance="The ordinary LIGER path has no live feature-encoder seam; any change here requires CUSTOM_MODEL full-source ownership.",
        allow_multiple=False,
        primitives=(
            primitive("feature.frozen_text_encoder", "ITEM_FEATURE_ENCODER", inputs=(("content", (FEATURE,), 1),), outputs=(("latent", LATENT),), parameters=_p({"projection_dimension": {"type": "integer", "minimum": 16, "maximum": 4096}, "normalize": {"type": "boolean"}}, "projection_dimension", "normalize"), capabilities=("FEATURE_ENCODER_MUTATION",), causal_effect="Project frozen item text features into a quantization space.", failure_signal="Semantic proximity ignores collaborative substitutability.", resource_effect="One cached content-encoding stage plus a projection."),
            primitive("feature.collaborative_spectral_encoder", "ITEM_FEATURE_ENCODER", inputs=(("interactions", ("semantic_id/train_interactions",), 1),), outputs=(("latent", LATENT),), parameters=_p({"dimension": {"type": "integer", "minimum": 16, "maximum": 2048}, "normalization": {"enum": ["NONE", "DEGREE", "WHITEN"]}}, "dimension", "normalization"), capabilities=("FEATURE_ENCODER_MUTATION",), causal_effect="Derive tokenizer latents from train-only collaborative geometry.", failure_signal="Collaborative codes memorize popularity and lose semantic transfer.", resource_effect="Separately measured train-only spectral/factor precompute."),
            primitive("feature.semantic_collaborative_fusion", "ITEM_FEATURE_ENCODER", inputs=(("content", (FEATURE,), 1), ("collaborative", ("semantic_id/collaborative_feature",), 1)), outputs=(("latent", LATENT),), parameters=_p({"fusion": {"enum": ["CONCAT_PROJECT", "GATED_SUM", "ORTHOGONAL_SUBSPACES"]}, "dimension": {"type": "integer", "minimum": 16, "maximum": 4096}}, "fusion", "dimension"), capabilities=("FEATURE_ENCODER_MUTATION", "ALIGNMENT_MUTATION"), causal_effect="Combine semantic and collaborative item evidence before tokenization.", failure_signal="One modality dominates or fusion improves only through increased dimension.", resource_effect="Adds a trainable fusion projection."),
            primitive("feature.recommendation_native_structured_field_autoencoder", "ITEM_FEATURE_ENCODER", inputs=(("fields", (STRUCTURED_FIELDS,), 1), ("interactions", ("semantic_id/train_interactions",), 1)), outputs=(("latent", LATENT),), parameters=_p({"field_fusion": {"enum": ["CONCAT_PROJECT", "HIERARCHICAL_FIELD_ATTENTION", "MASKED_FIELD_AUTOENCODER"]}, "dimension": {"type": "integer", "minimum": 16, "maximum": 4096}, "field_mask_probability": {"type": "number", "minimum": 0.0, "exclusiveMaximum": 1.0}, "reconstruction_weight": {"type": "number", "minimum": 0.0}, "recommendation_weight": {"type": "number", "minimum": 0.0}, "missing_field_policy": {"const": "EXPLICIT_UNKNOWN"}}, "field_fusion", "dimension", "field_mask_probability", "reconstruction_weight", "recommendation_weight", "missing_field_policy"), capabilities=("FEATURE_ENCODER_MUTATION", "ALIGNMENT_MUTATION", "STRUCTURED_FIELD_REPRESENTATION"), causal_effect="Learn one item representation from frozen structured fields while jointly preserving field reconstruction and train-only recommendation geometry; missing brand/category or other declared fields remain explicit unknown values rather than filtered items.", failure_signal="Field missingness becomes an item shortcut, shallow category paths dominate, recommendation loss erases field structure, or reconstruction improves without item-ranking transfer.", resource_effect="Separately budgeted structured-field autoencoding plus train-only recommendation supervision; report per-field coverage, unknown rate, reconstruction cost and recommendation-branch gradients."),
        ),
    ),
    axis(
        "SEMANTIC_TOKENIZER",
        research_role="Discretize item latents into reusable token identities.",
        interaction_guidance="The frozen LETTER tokenizer is parent-owned in ordinary search; any changed tokenizer requires CUSTOM_MODEL full-source ownership.",
        allow_multiple=False,
        primitives=(
            primitive(
                "tokenizer.frozen_letter_collision_suffix",
                "SEMANTIC_TOKENIZER",
                inputs=(("sid", (SID,), 1),),
                outputs=(("sid", SID),),
                parameters=_p(
                    {
                        "semantic_prefix_levels": {"const": 3},
                        "codes_per_level": {"const": 256},
                        "collision_suffix": {
                            "const": "ORDERED_PREFIX_COLLISION_INDEX"
                        },
                        "position_offset_encoding": {"const": True},
                    },
                    "semantic_prefix_levels",
                    "codes_per_level",
                    "collision_suffix",
                    "position_offset_encoding",
                ),
                capabilities=("TOKENIZER_MUTATION",),
                causal_effect="Use the frozen first three LETTER codes and append the official per-prefix collision index before position-specific vocabulary offsets.",
                failure_signal="The frozen LETTER mapping is recomputed, a fourth RVQ level is invented, or the collision suffix is not catalog-unique.",
                resource_effect="No runtime tokenizer training; load and identity-check one frozen SID mapping.",
            ),
            primitive("tokenizer.residual_vector_quantization", "SEMANTIC_TOKENIZER", inputs=(("latent", (LATENT,), 1),), outputs=(("codebook", CODEBOOK), ("sid", SID)), parameters=_p({"levels": {"type": "integer", "minimum": 1, "maximum": 32}, "codes_per_level": {"type": "integer", "minimum": 8, "maximum": 65536}, "commitment_weight": {"type": "number", "minimum": 0.0}}, "levels", "codes_per_level", "commitment_weight"), capabilities=("TOKENIZER_MUTATION",), causal_effect="Quantize successive residuals to form a hierarchical discrete item identity.", failure_signal="Code collapse, high collision, or quantization error prevents faithful item resolution.", resource_effect="Separate RQ-style tokenizer pretraining and codebook storage."),
            primitive("tokenizer.product_quantization", "SEMANTIC_TOKENIZER", inputs=(("latent", (LATENT,), 1),), outputs=(("codebook", CODEBOOK), ("sid", SID)), parameters=_p({"subspaces": {"type": "integer", "minimum": 1, "maximum": 64}, "codes_per_subspace": {"type": "integer", "minimum": 8, "maximum": 65536}}, "subspaces", "codes_per_subspace"), capabilities=("TOKENIZER_MUTATION",), causal_effect="Quantize independent latent subspaces into a compositional semantic ID.", failure_signal="Subspace independence destroys cross-feature semantics or creates invalid combinations.", resource_effect="Parallel codebooks and compact lookup."),
            primitive("tokenizer.orthogonally_preconditioned_product_quantization", "SEMANTIC_TOKENIZER", inputs=(("latent", (LATENT,), 1), ("interactions", ("semantic_id/train_interactions",), 1)), outputs=(("codebook", CODEBOOK), ("sid", SID)), parameters=_p({"subspaces": {"type": "integer", "minimum": 1, "maximum": 64}, "codes_per_subspace": {"type": "integer", "minimum": 8, "maximum": 65536}, "rotation_scope": {"enum": ["GLOBAL", "BLOCK_DIAGONAL"]}, "rotation_schedule": {"enum": ["PRETRAIN_THEN_FREEZE", "ALTERNATING_WITH_CODEBOOK"]}, "quantization_weight": {"type": "number", "minimum": 0.0}, "recommendation_weight": {"type": "number", "minimum": 0.0}}, "subspaces", "codes_per_subspace", "rotation_scope", "rotation_schedule", "quantization_weight", "recommendation_weight"), capabilities=("TOKENIZER_MUTATION", "ALIGNMENT_MUTATION", "ORTHOGONAL_PRODUCT_QUANTIZATION"), causal_effect="Learn an orthogonal pre-rotation that redistributes latent variance before product subspace quantization, optionally using train-only recommendation geometry, so compact independent codes retain more item structure.", failure_signal="The rotation departs from orthogonality, merely relocates quantization error, concentrates information in a few subspaces, or improves reconstruction without ranking transfer; compare against the same product quantizer with rotation disabled.", resource_effect="Separately measured rotation fitting plus product-codebook construction; report rotation orthogonality, per-subspace error/utilization, code memory and rotation-off ablation."),
            primitive("tokenizer.hierarchical_balanced_clustering", "SEMANTIC_TOKENIZER", inputs=(("latent", (LATENT,), 1),), outputs=(("codebook", CODEBOOK), ("sid", SID)), parameters=_p({"depth": {"type": "integer", "minimum": 1, "maximum": 32}, "branching": {"type": "integer", "minimum": 2, "maximum": 4096}, "balance_weight": {"type": "number", "minimum": 0.0}}, "depth", "branching", "balance_weight"), capabilities=("TOKENIZER_MUTATION",), causal_effect="Build a balanced semantic hierarchy with bounded prefix fan-out.", failure_signal="Balancing separates naturally dense semantic groups or encodes arbitrary cluster boundaries.", resource_effect="Hierarchical clustering/pretraining stage."),
            primitive("tokenizer.recommendation_aware_quantization", "SEMANTIC_TOKENIZER", inputs=(("latent", (LATENT,), 1), ("interactions", ("semantic_id/train_interactions",), 1)), outputs=(("codebook", CODEBOOK), ("sid", SID)), parameters=_p({"reconstruction_weight": {"type": "number", "minimum": 0.0}, "recommendation_weight": {"type": "number", "minimum": 0.0}, "levels": {"type": "integer", "minimum": 1, "maximum": 32}}, "reconstruction_weight", "recommendation_weight", "levels"), capabilities=("TOKENIZER_MUTATION", "ALIGNMENT_MUTATION"), causal_effect="Shape codes jointly by item reconstruction and train-only recommendation utility.", failure_signal="Tokenizer overfits interaction popularity or joint training destabilizes code assignments.", resource_effect="Tokenizer stage includes an interaction-derived objective and must be budgeted separately."),
            primitive("tokenizer.end_to_end_recommendation_quantization", "SEMANTIC_TOKENIZER", inputs=(("latent", (LATENT,), 1), ("interactions", ("semantic_id/train_interactions",), 1)), outputs=(("codebook", CODEBOOK), ("sid", SID)), parameters=_p({"levels": {"type": "integer", "minimum": 1, "maximum": 32}, "codes_per_level": {"type": "integer", "minimum": 8, "maximum": 65536}, "recommendation_weight": {"type": "number", "minimum": 0.0}, "reconstruction_weight": {"type": "number", "minimum": 0.0}}, "levels", "codes_per_level", "recommendation_weight", "reconstruction_weight"), capabilities=("TOKENIZER_MUTATION", "ALIGNMENT_MUTATION", "END_TO_END_SID_GRADIENT"), causal_effect="Let the train-only recommendation objective update item features, quantizer assignments, codebooks, and the generative recommender through one declared gradient path.", failure_signal="The recommendation path is detached, overwhelms semantic reconstruction, or collapses the codebook despite nominal joint training.", resource_effect="Joint tokenizer/recommender backward pass with separately measured codebook and encoder gradient norms."),
        ),
    ),
    axis(
        "CODEBOOK_GEOMETRY",
        research_role="Control code utilization, hierarchy, and separation independently of decoder architecture.",
        interaction_guidance="The ordinary LIGER path has no live codebook-update seam; any codebook change requires CUSTOM_MODEL full-source ownership.",
        allow_multiple=False,
        primitives=(
            primitive("codebook.euclidean_residual", "CODEBOOK_GEOMETRY", inputs=(("codebook", (CODEBOOK,), 1), ("sid", (SID,), 1)), outputs=(("sid", SID),), parameters=_p({"normalize_levels": {"type": "boolean"}}, "normalize_levels"), capabilities=("CODEBOOK_MUTATION",), causal_effect="Use Euclidean residual geometry at each code level.", failure_signal="Later levels become unused or encode only quantization noise.", resource_effect="Standard nearest-code lookup."),
            primitive("codebook.spherical", "CODEBOOK_GEOMETRY", inputs=(("codebook", (CODEBOOK,), 1), ("sid", (SID,), 1)), outputs=(("sid", SID),), parameters=_p({"temperature": {"type": "number", "exclusiveMinimum": 0.0}}, "temperature"), capabilities=("CODEBOOK_MUTATION",), causal_effect="Quantize directions while removing latent-norm dominance.", failure_signal="Norm carried useful popularity/confidence information.", resource_effect="Normalized nearest-code lookup."),
            primitive("codebook.orthogonal_levels", "CODEBOOK_GEOMETRY", inputs=(("codebook", (CODEBOOK,), 1), ("sid", (SID,), 1)), outputs=(("sid", SID),), parameters=_p({"orthogonality_weight": {"type": "number", "minimum": 0.0}}, "orthogonality_weight"), capabilities=("CODEBOOK_MUTATION",), causal_effect="Encourage code levels to carry nonredundant factors.", failure_signal="Orthogonality conflicts with hierarchical refinement.", resource_effect="Additional codebook regularization."),
            primitive("codebook.globally_aligned_orthogonal_quantization", "CODEBOOK_GEOMETRY", inputs=(("codebook", (CODEBOOK,), 1), ("sid", (SID,), 1), ("latent", (LATENT,), 1)), outputs=(("sid", SID),), parameters=_p({"alignment_target": {"enum": ["ITEM_LATENT", "PAIRWISE_GEOMETRY", "BOTH"]}, "alignment_scope": {"const": "FULL_TRAIN_CATALOG"}, "global_alignment_weight": {"type": "number", "minimum": 0.0}, "orthogonality_weight": {"type": "number", "minimum": 0.0}, "usage_balance_weight": {"type": "number", "minimum": 0.0}}, "alignment_target", "alignment_scope", "global_alignment_weight", "orthogonality_weight", "usage_balance_weight"), capabilities=("CODEBOOK_MUTATION", "ALIGNMENT_MUTATION", "GLOBAL_ORTHOGONAL_QUANTIZATION"), causal_effect="Align semantic-ID code geometry to the full train-catalog item representation while orthogonalizing code dimensions or levels and balancing assignment usage.", failure_signal="Global alignment washes out local item distinctions, orthogonality destroys hierarchical refinement, or usage balancing forces unnatural uniformity; each term requires a matched ablation.", resource_effect="Full-train-catalog alignment statistics, orthogonality penalty and assignment-usage accounting; report alignment quality, orthogonality error, utilization and catalog-scale memory."),
            primitive("codebook.usage_balanced", "CODEBOOK_GEOMETRY", inputs=(("codebook", (CODEBOOK,), 1), ("sid", (SID,), 1)), outputs=(("sid", SID),), parameters=_p({"entropy_weight": {"type": "number", "minimum": 0.0}, "minimum_usage": {"type": "number", "minimum": 0.0, "maximum": 1.0}}, "entropy_weight", "minimum_usage"), capabilities=("CODEBOOK_MUTATION",), causal_effect="Prevent dead codes and excessive prefix concentration.", failure_signal="Uniform usage harms natural item-frequency or semantic structure.", resource_effect="Adds batch/global usage statistics during tokenizer training."),
            primitive("codebook.hard_forward_soft_backward_assignment", "CODEBOOK_GEOMETRY", inputs=(("codebook", (CODEBOOK,), 1), ("sid", (SID,), 1)), outputs=(("sid", SID),), parameters=_p({"hard_assignment": {"enum": ["ARGMAX", "SINKHORN_BALANCED"]}, "backward_distribution": {"enum": ["GUMBEL_SOFTMAX", "TEMPERED_SOFTMAX"]}, "temperature": {"type": "number", "exclusiveMinimum": 0.0}, "sinkhorn_epsilon": {"type": "number", "minimum": 0.0}, "sinkhorn_iterations": {"type": "integer", "minimum": 1, "maximum": 1000}, "uncertainty_decay": {"enum": ["NONE", "GLOBAL_STD", "FREQUENCY_CONDITIONAL", "COMBINED"]}}, "hard_assignment", "backward_distribution", "temperature", "sinkhorn_epsilon", "sinkhorn_iterations", "uncertainty_decay"), capabilities=("CODEBOOK_MUTATION", "DIFFERENTIABLE_ASSIGNMENT_MUTATION"), causal_effect="Use discrete catalog-valid code choices in the forward pass while routing recommendation gradients through a soft assignment, optionally balancing hard choices and decaying Gumbel exploration by uncertainty or code frequency.", failure_signal="The hard/soft surrogate disagrees, gradients detach before the tokenizer, or exploration decay causes code collapse or train-inference mismatch.", resource_effect="Soft all-code probabilities, optional Sinkhorn iterations, code-frequency tracking, and explicit gradient-flow probes."),
            primitive("codebook.soft_to_hard_annealed_assignment", "CODEBOOK_GEOMETRY", inputs=(("codebook", (CODEBOOK,), 1), ("sid", (SID,), 1)), outputs=(("sid", SID),), parameters=_p({"initial_temperature": {"type": "number", "exclusiveMinimum": 0.0}, "final_temperature": {"type": "number", "exclusiveMinimum": 0.0}, "annealing_schedule": {"enum": ["LINEAR", "COSINE", "EXPONENTIAL"]}, "hardening_start_fraction": {"type": "number", "minimum": 0.0, "maximum": 1.0}, "inference_assignment": {"const": "HARD_ARGMAX"}}, "initial_temperature", "final_temperature", "annealing_schedule", "hardening_start_fraction", "inference_assignment"), capabilities=("CODEBOOK_MUTATION", "SOFT_ID_ANNEALING"), causal_effect="Train with continuous code mixtures early, progressively align them to hard catalog identifiers, and finish with a declared hard assignment at inference.", failure_signal="Soft identifiers never harden, annealing collapses too early, or the soft-trained representation fails under hard inference; measure the soft-hard assignment and ranking gap throughout training.", resource_effect="Stores soft code distributions during the declared annealing window and separately reports soft/hard validation cost."),
        ),
    ),
    axis(
        "SID_STRUCTURE",
        research_role="Define the length, hierarchy, ordering, and uniqueness policy of item token identities.",
        interaction_guidance="The four-level parent SID is frozen in ordinary search; any structural change requires CUSTOM_MODEL full-source ownership.",
        allow_multiple=False,
        primitives=(
            primitive("sid.fixed_hierarchical_sequence", "SID_STRUCTURE", inputs=(("sid", (SID,), 1),), outputs=(("sid", SID),), parameters=_p({"length": {"type": "integer", "minimum": 1, "maximum": 64}, "append_unique_token": {"type": "boolean"}}, "length", "append_unique_token"), capabilities=("SID_STRUCTURE_MUTATION",), causal_effect="Represent each item as an ordered coarse-to-fine code sequence.", failure_signal="Long IDs compound decoding errors or unique suffixes erase semantic sharing.", resource_effect="Autoregressive steps scale with fixed SID length."),
            primitive("sid.variable_depth_tree", "SID_STRUCTURE", inputs=(("sid", (SID,), 1),), outputs=(("sid", SID),), parameters=_p({"minimum_depth": {"type": "integer", "minimum": 1}, "maximum_depth": {"type": "integer", "minimum": 1, "maximum": 64}, "stop_condition": {"enum": ["LEARNED_STOP", "CATALOG_SINGLETON_PREFIX", "AMBIGUITY_THRESHOLD"]}, "serialization_after_stop": {"enum": ["STOP_TOKEN", "CONSTANT_PADDING_TOKEN"]}, "stop_token": {"const": True}}, "minimum_depth", "maximum_depth", "stop_condition", "serialization_after_stop", "stop_token"), capabilities=("SID_STRUCTURE_MUTATION", "PREFIX_UNIQUENESS_MUTATION"), causal_effect="Allocate more code depth only while an item prefix remains ambiguous, optionally stopping as soon as the train-catalog prefix is unique while retaining a fixed constant padding token for tensor serialization.", failure_signal="Early stopping creates prefix collisions, learned stopping favors short IDs, or a constant suffix is misread as code collapse despite complete unique-prefix catalog resolution; report uniqueness and collisions at every depth.", resource_effect="Variable effective decoding length plus prefix-trie metadata; separately report serialized length, effective nonconstant length and redundant-suffix decoding cost."),
            primitive("sid.unordered_set", "SID_STRUCTURE", inputs=(("sid", (SID,), 1),), outputs=(("sid", SID),), parameters=_p({"length": {"type": "integer", "minimum": 1, "maximum": 64}, "position_invariant_loss": {"const": True}}, "length", "position_invariant_loss"), capabilities=("SID_STRUCTURE_MUTATION",), causal_effect="Treat item codes as an unordered semantic set to enable parallel prediction.", failure_signal="Permutation invariance loses hierarchical meaning or creates ambiguous item mappings.", resource_effect="Supports parallel code prediction with set-matching cost."),
            primitive("sid.prefix_constrained_unique", "SID_STRUCTURE", inputs=(("sid", (SID,), 1),), outputs=(("sid", SID),), parameters=_p({"shared_prefix_levels": {"type": "integer", "minimum": 0}, "unique_suffix_levels": {"type": "integer", "minimum": 1}}, "shared_prefix_levels", "unique_suffix_levels"), capabilities=("SID_STRUCTURE_MUTATION",), causal_effect="Share semantic prefixes while guaranteeing item-unique suffix resolution.", failure_signal="Unique suffix dominates likelihood and reduces semantic generalization.", resource_effect="Prefix trie plus guaranteed unique suffix tokens."),
        ),
    ),
    axis(
        "SEMANTIC_COLLABORATIVE_ALIGNMENT",
        research_role="Align discrete semantic neighborhoods with recommendation relations without changing heldout access.",
        interaction_guidance="The ordinary LIGER path has no live alignment-training seam; any alignment change requires CUSTOM_MODEL full-source ownership.",
        allow_multiple=True,
        primitives=(
            primitive("alignment.item_pair_contrastive", "SEMANTIC_COLLABORATIVE_ALIGNMENT", inputs=(("sid", (SID,), 1), ("interactions", ("semantic_id/train_interactions",), 1)), outputs=(("sid", SID),), parameters=_p({"weight": {"type": "number", "minimum": 0.0}, "temperature": {"type": "number", "exclusiveMinimum": 0.0}}, "weight", "temperature"), capabilities=("ALIGNMENT_MUTATION",), causal_effect="Pull co-preferred items together in code space while retaining content geometry.", failure_signal="Alignment collapses diverse co-consumption or increases code collisions.", resource_effect="Train-only pair sampling and contrastive loss."),
            primitive("alignment.sequence_item", "SEMANTIC_COLLABORATIVE_ALIGNMENT", inputs=(("sid", (SID,), 1), ("history", ("semantic_id/train_item_sequence",), 1)), outputs=(("sid", SID),), parameters=_p({"weight": {"type": "number", "minimum": 0.0}, "context_window": {"type": "integer", "minimum": 1}}, "weight", "context_window"), capabilities=("ALIGNMENT_MUTATION",), causal_effect="Align item codes with the contexts in which items are selected.", failure_signal="Sequence context overfits local popularity and harms content transfer.", resource_effect="Adds history encoding or sampled context loss to tokenizer stage."),
            primitive("alignment.preference_semantic_dual", "SEMANTIC_COLLABORATIVE_ALIGNMENT", inputs=(("sid", (SID,), 1), ("latent", (LATENT,), 1), ("interactions", ("semantic_id/train_interactions",), 1)), outputs=(("sid", SID),), parameters=_p({"semantic_weight": {"type": "number", "minimum": 0.0}, "preference_weight": {"type": "number", "minimum": 0.0}}, "semantic_weight", "preference_weight"), capabilities=("ALIGNMENT_MUTATION",), causal_effect="Maintain separate semantic fidelity and collaborative preference constraints.", failure_signal="The two objectives conflict and code assignments oscillate.", resource_effect="Two measured tokenizer objectives."),
            primitive("alignment.rank_guided_code_order", "SEMANTIC_COLLABORATIVE_ALIGNMENT", inputs=(("sid", (SID,), 1), ("interactions", ("semantic_id/train_interactions",), 1)), outputs=(("sid", SID),), parameters=_p({"weight": {"type": "number", "minimum": 0.0}, "margin": {"type": "number", "minimum": 0.0}}, "weight", "margin"), capabilities=("ALIGNMENT_MUTATION",), causal_effect="Make code likelihood order reflect train-only item preference ranking.", failure_signal="Rank guidance encodes popularity rather than user-conditional relevance.", resource_effect="Additional sampled ranking constraints."),
            primitive("alignment.codeword_representation_uniformity", "SEMANTIC_COLLABORATIVE_ALIGNMENT", inputs=(("codebook", (CODEBOOK,), 1), ("sid", (SID,), 1)), outputs=(("sid", SID),), parameters=_p({"scope": {"enum": ["PER_LEVEL", "ALL_CODEWORDS"]}, "geometry": {"enum": ["HYPERSPHERICAL", "PAIRWISE_REPULSION"]}, "weight": {"type": "number", "minimum": 0.0}, "temperature": {"type": "number", "exclusiveMinimum": 0.0}}, "scope", "geometry", "weight", "temperature"), capabilities=("ALIGNMENT_MUTATION", "CODEWORD_UNIFORMITY"), causal_effect="Spread learned codeword representations geometrically without forcing uniform assignment frequency, preserving capacity for distinct semantic directions.", failure_signal="Codewords become uniformly spaced but semantically arbitrary, or representation repulsion harms natural density while usage remains imbalanced; separate geometry from assignment-frequency effects.", resource_effect="Pairwise or hyperspherical codeword statistics with per-level uniformity and utilization reporting."),
            primitive("alignment.dual_collaborative_distillation", "SEMANTIC_COLLABORATIVE_ALIGNMENT", inputs=(("sid", (SID,), 1), ("latent", (LATENT,), 1), ("teacher", ("semantic_id/collaborative_feature",), 1), ("interactions", ("semantic_id/train_interactions",), 1)), outputs=(("sid", SID),), parameters=_p({"distillation_targets": {"enum": ["ITEM_REPRESENTATION_AND_PAIRWISE_RELATION", "ITEM_REPRESENTATION_AND_RANK_DISTRIBUTION", "PAIRWISE_RELATION_AND_RANK_DISTRIBUTION"]}, "teacher_temperature": {"type": "number", "exclusiveMinimum": 0.0}, "first_branch_weight": {"type": "number", "minimum": 0.0}, "second_branch_weight": {"type": "number", "minimum": 0.0}}, "distillation_targets", "teacher_temperature", "first_branch_weight", "second_branch_weight"), capabilities=("ALIGNMENT_MUTATION", "DUAL_COLLABORATIVE_DISTILLATION"), causal_effect="Distill two separately measurable train-only collaborative teacher targets into semantic-ID geometry so item representations and relational or ranking structure can transfer without sharing heldout labels.", failure_signal="One teacher branch is inactive, the two targets duplicate each other, or student gains disappear when teacher capacity and branch weights are controlled; ablate each branch independently.", resource_effect="Requires a frozen, identity-bound collaborative teacher cache plus two logged distillation losses and branch-specific gradients."),
        ),
    ),
    axis(
        "USER_CONTEXT_ENCODER",
        research_role="Encode a user's training-prefix interaction context into a generative condition.",
        interaction_guidance="Keep chronological/random-order semantics fixed by the Profile; this axis cannot reinterpret the data split.",
        allow_multiple=False,
        primitives=(
            primitive(
                "context.sid_history_with_frozen_content",
                "USER_CONTEXT_ENCODER",
                inputs=(
                    ("history", ("semantic_id/train_item_sequence",), 1),
                    ("sid", (SID,), 1),
                    ("item_latent", (LATENT,), 1),
                ),
                outputs=(("context", CONTEXT),),
                parameters=_p(
                    {
                        "max_items_per_sequence": {"const": 20},
                        "sid_tokens_per_item": {"const": 4},
                        "appended_padding_tokens": {"const": 22},
                        "dimension": {"const": 128},
                    },
                    "max_items_per_sequence",
                    "sid_tokens_per_item",
                    "appended_padding_tokens",
                    "dimension",
                ),
                capabilities=("CONTEXT_ENCODER_MUTATION",),
                causal_effect="Condition on the flattened four-token SID history while adding the aligned frozen item-content embeddings used by official LIGER.",
                failure_signal="The generator sees item IDs instead of SID history, content embeddings are trainable or misaligned, or padding changes the 102-position contract.",
                resource_effect="Expand twenty history items to eighty SID tokens plus twenty-two reserved padding positions and load frozen content embeddings.",
            ),
            primitive("context.sid_autoregressive_history", "USER_CONTEXT_ENCODER", inputs=(("history", ("semantic_id/train_item_sequence",), 1), ("sid", (SID,), 1)), outputs=(("context", CONTEXT),), parameters=_p({"layers": {"type": "integer", "minimum": 1, "maximum": 64}, "dimension": {"type": "integer", "minimum": 16, "maximum": 4096}}, "layers", "dimension"), capabilities=("CONTEXT_ENCODER_MUTATION",), causal_effect="Represent interaction history directly as semantic-ID token sequences.", failure_signal="Long expanded token histories make optimization or context length infeasible.", resource_effect="Token count multiplies by SID length."),
            primitive("context.item_then_sid_condition", "USER_CONTEXT_ENCODER", inputs=(("history", ("semantic_id/train_item_sequence",), 1),), outputs=(("context", CONTEXT),), parameters=_p({"layers": {"type": "integer", "minimum": 1, "maximum": 64}, "dimension": {"type": "integer", "minimum": 16, "maximum": 4096}}, "layers", "dimension"), capabilities=("CONTEXT_ENCODER_MUTATION",), causal_effect="Encode item history compactly and generate only the target SID.", failure_signal="Item-level context fails to exploit compositional code sharing.", resource_effect="Shorter context than fully expanded SID history."),
            primitive("context.semantic_collaborative_dual_path", "USER_CONTEXT_ENCODER", inputs=(("history", ("semantic_id/train_item_sequence",), 1), ("sid", (SID,), 1)), outputs=(("context", CONTEXT),), parameters=_p({"fusion": {"enum": ["GATED", "CROSS_ATTENTION", "LATE_SUM"]}, "dimension": {"type": "integer", "minimum": 16, "maximum": 4096}}, "fusion", "dimension"), capabilities=("CONTEXT_ENCODER_MUTATION",), causal_effect="Preserve item-identity and semantic-code history paths separately before fusion.", failure_signal="One path dominates or gains are only added parameters.", resource_effect="Two history paths and fusion."),
        ),
    ),
    axis(
        "GENERATIVE_BACKBONE",
        research_role="Keep the frozen LIGER/T5 encoder-decoder active for every parent-preserving candidate; a different generator is a whole-model comparison.",
        interaction_guidance="COMPOSITION and ARCHITECTURE_REWRITE retain the exact frozen autoregressive T5 backbone. Context, additive-objective and T5-native decoding use parent seams; training-stage capabilities use full-source model and trainer ownership. A generator replacement requires explicit CUSTOM_MODEL synthesis.",
        allow_multiple=False,
        primitives=(
            primitive(
                "generator.t5_encoder_decoder",
                "GENERATIVE_BACKBONE",
                inputs=(("context", (CONTEXT,), 1), ("sid", (SID,), 1)),
                outputs=(("token_state", TOKEN_STATE),),
                parameters=_p(
                    {
                        "encoder_layers": {"type": "integer", "minimum": 1},
                        "decoder_layers": {"type": "integer", "minimum": 1},
                        "dimension": {"type": "integer", "minimum": 16},
                        "feed_forward_dimension": {
                            "type": "integer",
                            "minimum": 16,
                        },
                        "attention_heads": {"type": "integer", "minimum": 1},
                        "key_value_dimension": {
                            "type": "integer",
                            "minimum": 1,
                        },
                        "dropout_rate": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                    "encoder_layers",
                    "decoder_layers",
                    "dimension",
                    "feed_forward_dimension",
                    "attention_heads",
                    "key_value_dimension",
                    "dropout_rate",
                ),
                capabilities=("GENERATOR_MUTATION",),
                causal_effect="Run the T5 encoder-decoder used by the official semantic-ID generator over SID-history and frozen-content conditioning.",
                failure_signal="A custom Transformer, GRU, or catalog scorer is substituted for the declared T5 generator.",
                resource_effect="Autoregressive T5 training and beam decoding with the declared encoder and decoder depths.",
            ),
            primitive("generator.encoder_decoder_transformer", "GENERATIVE_BACKBONE", inputs=(("context", (CONTEXT,), 1), ("sid", (SID,), 1)), outputs=(("token_state", TOKEN_STATE),), parameters=_p({"encoder_layers": {"type": "integer", "minimum": 1, "maximum": 64}, "decoder_layers": {"type": "integer", "minimum": 1, "maximum": 64}, "dimension": {"type": "integer", "minimum": 16, "maximum": 4096}}, "encoder_layers", "decoder_layers", "dimension"), capabilities=("GENERATOR_MUTATION",), causal_effect="Autoregressively condition each SID token on user context and prior code tokens.", failure_signal="Exposure error compounds across SID positions or compute is dominated by serial decoding.", resource_effect="Autoregressive token training and decoding."),
            primitive("generator.decoder_only_prefix_lm", "GENERATIVE_BACKBONE", inputs=(("context", (CONTEXT,), 1), ("sid", (SID,), 1)), outputs=(("token_state", TOKEN_STATE),), parameters=_p({"layers": {"type": "integer", "minimum": 1, "maximum": 96}, "dimension": {"type": "integer", "minimum": 16, "maximum": 8192}}, "layers", "dimension"), capabilities=("GENERATOR_MUTATION",), causal_effect="Use a unified prefix language model for context and target semantic IDs.", failure_signal="Expanded token sequence consumes context budget without better item ranking.", resource_effect="Single causal stack; token count includes context plus target codes."),
            primitive("generator.parallel_code_predictor", "GENERATIVE_BACKBONE", inputs=(("context", (CONTEXT,), 1), ("sid", (SID,), 1)), outputs=(("token_state", TOKEN_STATE),), parameters=_p({"layers": {"type": "integer", "minimum": 1, "maximum": 64}, "code_positions": {"type": "integer", "minimum": 1, "maximum": 64}, "dependency_refinement_steps": {"type": "integer", "minimum": 0, "maximum": 16}}, "layers", "code_positions", "dependency_refinement_steps"), capabilities=("GENERATOR_MUTATION", "PARALLEL_DECODING"), causal_effect="Predict SID positions in parallel, optionally refining cross-code dependencies.", failure_signal="Independent positions produce invalid combinations or lose hierarchy.", resource_effect="Constant/iterative parallel decoding rather than SID-length serial steps."),
            primitive("generator.hybrid_coarse_ar_fine_parallel", "GENERATIVE_BACKBONE", inputs=(("context", (CONTEXT,), 1), ("sid", (SID,), 1)), outputs=(("token_state", TOKEN_STATE),), parameters=_p({"autoregressive_levels": {"type": "integer", "minimum": 1}, "parallel_levels": {"type": "integer", "minimum": 1}, "layers": {"type": "integer", "minimum": 1, "maximum": 64}}, "autoregressive_levels", "parallel_levels", "layers"), capabilities=("GENERATOR_MUTATION", "PARALLEL_DECODING"), causal_effect="Decode coarse semantic routing serially and fine identity codes in parallel.", failure_signal="Boundary between coarse/fine levels is arbitrary or errors still compound in coarse routing.", resource_effect="Reduced serial steps with parallel fine-code heads."),
        ),
    ),
    axis(
        "DECODING_STRATEGY",
        research_role="Turn conditional code distributions into valid candidate item hypotheses.",
        interaction_guidance="Ordinary changes must be expressible as T5 generation kwargs or postprocessing of the actual T5 output; parallel logits and full-catalog factorized scoring require CUSTOM_MODEL.",
        allow_multiple=False,
        primitives=(
            primitive("decode.autoregressive_beam_then_invalid_drop", "DECODING_STRATEGY", inputs=(("token_state", (TOKEN_STATE,), 1), ("sid", (SID,), 1)), outputs=(("hypothesis", HYPOTHESIS),), parameters=_p({"beam_width": {"type": "integer", "minimum": 1, "maximum": 4096}, "length_normalization": {"type": "number", "minimum": 0.0}, "invalid_sentinel": {"const": -1}}, "beam_width", "length_normalization", "invalid_sentinel"), capabilities=("DECODING_MUTATION",), causal_effect="Run ordinary autoregressive beam search, retain the declared beam width, then mark complete codes absent from the catalog with a negative sentinel for explicit dropping before ranking.", failure_signal="Invalid codes map to item zero, survive reranking, or ordinary beam is silently replaced by catalog-prefix constraints.", resource_effect="Unconstrained serial beam cost plus explicit complete-code catalog lookup."),
            primitive("decode.constrained_beam_trie", "DECODING_STRATEGY", inputs=(("token_state", (TOKEN_STATE,), 1), ("sid", (SID,), 1)), outputs=(("hypothesis", HYPOTHESIS),), parameters=_p({"beam_width": {"type": "integer", "minimum": 1, "maximum": 4096}, "length_normalization": {"type": "number", "minimum": 0.0}}, "beam_width", "length_normalization"), capabilities=("DECODING_MUTATION",), causal_effect="Restrict beam expansion to prefixes of known catalog items.", failure_signal="Beam pruning misses relevant items or gains vanish with exact candidate scoring.", resource_effect="Trie-constrained serial beam cost."),
            primitive("decode.parallel_valid_assignment", "DECODING_STRATEGY", inputs=(("token_state", (TOKEN_STATE,), 1), ("sid", (SID,), 1)), outputs=(("hypothesis", HYPOTHESIS),), parameters=_p({"assignment": {"enum": ["NEAREST_VALID_CODE", "BIPARTITE_MATCH", "GRAPH_GUIDED"]}, "top_codes_per_position": {"type": "integer", "minimum": 1, "maximum": 1024}}, "assignment", "top_codes_per_position"), capabilities=("DECODING_MUTATION", "PARALLEL_DECODING"), causal_effect="Resolve parallel code predictions to valid catalog IDs under an explicit assignment rule.", failure_signal="Assignment heuristic, not learned recommendation, drives ranking.", resource_effect="Parallel code logits plus catalog-validity resolution."),
            primitive("decode.catalog_score_factorization", "DECODING_STRATEGY", inputs=(("token_state", (TOKEN_STATE,), 1), ("sid", (SID,), 1)), outputs=(("hypothesis", HYPOTHESIS),), parameters=_p({"aggregation": {"enum": ["SUM_LOG_PROB", "MIN_TOKEN", "CALIBRATED_SUM"]}, "chunk_size": {"type": "integer", "minimum": 1}}, "aggregation", "chunk_size"), capabilities=("DECODING_MUTATION",), causal_effect="Score every catalog item by factorized code likelihood while preserving full ranking.", failure_signal="Factorized score ignores code dependencies or is computationally infeasible.", resource_effect="Exact catalog scoring in chunks; no candidate approximation."),
        ),
    ),
    axis(
        "DENSE_RETRIEVAL_CORRECTION",
        research_role="Combine semantic-ID generation with an independently ablatable dense catalog path without changing the full-ranking evaluator.",
        interaction_guidance="The exact parent dense reranker and metric path are frozen in ordinary search; any change here requires CUSTOM_MODEL full-source ownership.",
        allow_multiple=False,
        primitives=(
            primitive(
                "retrieval.generated_legal_dense_rerank_to_score",
                "DENSE_RETRIEVAL_CORRECTION",
                inputs=(
                    ("hypothesis", (HYPOTHESIS,), 1),
                    ("context", (CONTEXT,), 1),
                    ("item_latent", (LATENT,), 1),
                ),
                outputs=(("score", SCORE),),
                parameters=_p(
                    {
                        "generator_topk": {"type": "integer", "minimum": 1},
                        "rerank_depth": {"type": "integer", "minimum": 1},
                        "candidate_membership": {
                            "const": "GENERATOR_ONLY_AFTER_LEGAL_SID_MAPPING_AND_DEDUPLICATION"
                        },
                        "ranking_score": {"const": "DENSE_DOT_PRODUCT_ONLY"},
                        "generator_score_fusion": {"const": "NONE"},
                    },
                    "generator_topk",
                    "rerank_depth",
                    "candidate_membership",
                    "ranking_score",
                    "generator_score_fusion",
                ),
                capabilities=("DENSE_RETRIEVAL_CORRECTION", "RERANK_MUTATION"),
                causal_effect="Rank only already resolved legal generated items with the dense content head and emit the metric-facing score tensor.",
                failure_signal="Dense retrieval expands membership, generator scores leak into ordering, or invalid and non-generated items receive rankable scores.",
                resource_effect="Dense dot products only for the deduplicated legal generated candidates.",
            ),
            primitive("retrieval.generative_candidate_dense_rerank", "DENSE_RETRIEVAL_CORRECTION", inputs=(("hypothesis", (HYPOTHESIS,), 1), ("catalog", ("semantic_id/catalog_mapping",), 1), ("context", (CONTEXT,), 1), ("item_latent", (LATENT,), 1)), outputs=(("hypothesis", HYPOTHESIS),), parameters=_p({"generator_topk": {"type": "integer", "minimum": 1}, "rerank_depth": {"type": "integer", "minimum": 1}, "candidate_membership": {"const": "GENERATOR_ONLY_AFTER_LEGAL_SID_MAPPING_AND_DEDUPLICATION"}, "ranking_score": {"const": "DENSE_DOT_PRODUCT_ONLY"}, "generator_score_fusion": {"const": "NONE"}}, "generator_topk", "rerank_depth", "candidate_membership", "ranking_score", "generator_score_fusion"), capabilities=("DENSE_RETRIEVAL_CORRECTION", "RERANK_MUTATION"), causal_effect="Let the generator alone determine candidate membership, map legal SIDs to catalog items and deduplicate them, then rank only those generated candidates by dense user-item logits without fusing generator scores.", failure_signal="The dense branch changes candidate membership, an invalid SID survives mapping, generator scores leak into the final ordering, or the generator contributes no legal candidates.", resource_effect="Charges generation, legal SID mapping/deduplication, and dense scoring over the declared generated-candidate depth separately."),
            primitive("retrieval.generative_dense_union", "DENSE_RETRIEVAL_CORRECTION", inputs=(("hypothesis", (HYPOTHESIS,), 1), ("context", (CONTEXT,), 1), ("item_latent", (LATENT,), 1)), outputs=(("hypothesis", HYPOTHESIS),), parameters=_p({"dense_topk": {"type": "integer", "minimum": 1}, "generator_topk": {"type": "integer", "minimum": 1}, "deduplication": {"enum": ["MAX_SCORE", "LOGSUMEXP", "SOURCE_AWARE"]}}, "dense_topk", "generator_topk", "deduplication"), capabilities=("DENSE_RETRIEVAL_CORRECTION",), causal_effect="Union generative semantic-ID hypotheses with dense item retrieval so each path can recover the other's misses.", failure_signal="The dense path supplies nearly all recall or the union only changes a hidden candidate cap.", resource_effect="Adds an exact or indexed dense retrieval pass whose recall and latency are separately measured."),
            primitive("retrieval.dense_residual_rerank", "DENSE_RETRIEVAL_CORRECTION", inputs=(("hypothesis", (HYPOTHESIS,), 1), ("context", (CONTEXT,), 1), ("item_latent", (LATENT,), 1)), outputs=(("hypothesis", HYPOTHESIS),), parameters=_p({"residual_weight": {"type": "number", "minimum": 0.0}, "calibration": {"enum": ["FIXED", "TRAIN_ONLY_TEMPERATURE", "CONTEXT_GATE"]}, "rerank_depth": {"type": "integer", "minimum": 1}}, "residual_weight", "calibration", "rerank_depth"), capabilities=("DENSE_RETRIEVAL_CORRECTION", "RERANK_MUTATION"), causal_effect="Use dense relevance as a calibrated residual over valid generative candidates instead of replacing semantic-ID likelihood.", failure_signal="Quality vanishes when the dense residual is removed, showing generator noncontribution or calibration leakage.", resource_effect="Adds dense scoring for a declared rerank depth plus calibration probes."),
        ),
    ),
    axis(
        "ITEM_RESOLUTION",
        research_role="Map valid semantic hypotheses back to unique catalog item scores under the frozen evaluator.",
        interaction_guidance="The exact parent SID resolver and invalid-drop path are frozen in ordinary search; any change here requires CUSTOM_MODEL full-source ownership.",
        allow_multiple=False,
        primitives=(
            primitive(
                "resolution.invalid_sid_drop_lookup",
                "ITEM_RESOLUTION",
                inputs=(
                    ("hypothesis", (HYPOTHESIS,), 1),
                    ("catalog", ("semantic_id/catalog_mapping",), 1),
                ),
                outputs=(("hypothesis", HYPOTHESIS),),
                parameters=_p(
                    {
                        "invalid_sentinel": {"const": -1},
                        "invalid_policy": {"const": "DROP"},
                        "deduplication": {"const": "ORDER_PRESERVING"},
                    },
                    "invalid_sentinel",
                    "invalid_policy",
                    "deduplication",
                ),
                capabilities=("ITEM_RESOLUTION_MUTATION",),
                causal_effect="Resolve complete catalog SIDs, mark invalid beams with -1, drop them, and deduplicate legal item IDs before scoring.",
                failure_signal="An invalid SID becomes padding item zero or any static fallback, or duplicate items survive into ranking.",
                resource_effect="Hash lookup over every generated beam followed by an explicit validity mask and stable deduplication.",
            ),
            primitive("resolution.unique_sid_lookup", "ITEM_RESOLUTION", inputs=(("hypothesis", (HYPOTHESIS,), 1), ("catalog", ("semantic_id/catalog_mapping",), 1)), outputs=(("score", SCORE),), parameters=_p({"invalid_policy": {"enum": ["NEGATIVE_INFINITY", "DROP_AND_RENORMALIZE"]}}, "invalid_policy"), capabilities=("ITEM_RESOLUTION_MUTATION",), causal_effect="Resolve one valid SID to exactly one item.", failure_signal="Tokenizer collisions violate the uniqueness assumption.", resource_effect="Trie/hash lookup and full score scatter."),
            primitive("resolution.collision_rerank", "ITEM_RESOLUTION", inputs=(("hypothesis", (HYPOTHESIS,), 1), ("catalog", ("semantic_id/catalog_mapping",), 1), ("context", (CONTEXT,), 1)), outputs=(("score", SCORE),), parameters=_p({"reranker": {"enum": ["CONTEXT_DOT", "COLLABORATIVE_PRIOR", "SEMANTIC_DISTANCE"]}, "maximum_collision_group": {"type": "integer", "minimum": 1}}, "reranker", "maximum_collision_group"), capabilities=("ITEM_RESOLUTION_MUTATION", "RERANK_MUTATION"), causal_effect="Disambiguate items sharing a semantic code with an explicit user-conditional rule.", failure_signal="Reranker carries most of the recommendation quality and makes the generator noncausal.", resource_effect="Local scoring within collision groups."),
            primitive("resolution.prefix_candidate_expansion", "ITEM_RESOLUTION", inputs=(("hypothesis", (HYPOTHESIS,), 1), ("catalog", ("semantic_id/catalog_mapping",), 1)), outputs=(("score", SCORE),), parameters=_p({"prefix_depth": {"type": "integer", "minimum": 1}, "maximum_candidates": {"type": "integer", "minimum": 1}}, "prefix_depth", "maximum_candidates"), capabilities=("ITEM_RESOLUTION_MUTATION",), causal_effect="Expand an uncertain semantic prefix to all catalog-compatible items before scoring.", failure_signal="Candidate cap changes recall or becomes an unreported sampled-evaluation protocol.", resource_effect="Variable candidate expansion; full-universe equivalence must be qualified."),
        ),
    ),
    axis(
        "GENERATIVE_OBJECTIVE",
        research_role="Weight token, item, alignment, and post-training signals while preserving stage identities.",
        interaction_guidance="Additive objectives retain the parent T5 token loss and train-seen dense cross-entropy. A POST_TRAINING_MUTATION objective instead owns a separate phase through full-source model and trainer implementation; pair it with an explicit OPTIMIZATION_STAGING component, including phase update budgets and reference policy, within the fixed total training budget.",
        allow_multiple=True,
        primitives=(
            primitive("objective.token_cross_entropy", "GENERATIVE_OBJECTIVE", inputs=(("token_state", (TOKEN_STATE,), 1), ("targets", (SID,), 1)), outputs=(("objective", "semantic_id/objective"),), parameters=_p({"position_weights": {"enum": ["UNIFORM", "COARSE_HEAVY", "FINE_HEAVY", "LEARNED_NORMALIZED"]}, "label_smoothing": {"type": "number", "minimum": 0.0, "maximum": 0.5}, "loss_weight": {"type": "number", "minimum": 0.0}}, "position_weights", "label_smoothing"), capabilities=("GENERATIVE_OBJECTIVE_MUTATION",), causal_effect="Optimize target SID likelihood with explicit position weighting.", failure_signal="Coarse/fine weighting improves token accuracy but not item ranking.", resource_effect="Standard token-level supervised updates."),
            primitive(
                "objective.train_seen_full_catalog_dense_cross_entropy",
                "GENERATIVE_OBJECTIVE",
                inputs=(
                    ("context", (CONTEXT,), 1),
                    ("item_latent", (LATENT,), 1),
                    ("targets", ("core/item_id",), 1),
                ),
                outputs=(("objective", "semantic_id/objective"),),
                parameters=_p(
                    {
                        "negative_catalog_mask": {"const": "TRAIN_SEEN_ONLY"},
                        "normalize_logits": {"const": True},
                        "temperature": {"const": 0.07},
                        "loss_weight": {"const": 1.0},
                    },
                    "negative_catalog_mask",
                    "normalize_logits",
                    "temperature",
                    "loss_weight",
                ),
                capabilities=("GENERATIVE_OBJECTIVE_MUTATION",),
                causal_effect="Apply normalized full-catalog dense cross entropy at temperature 0.07 while excluding validation/test-only items from the training negative set.",
                failure_signal="Heldout-only items enter the negative catalog, normalization or temperature is omitted, or the dense loss is detached from the generator context.",
                resource_effect="One normalized dense score over the train-seen catalog per training example.",
            ),
            primitive("objective.item_likelihood_margin", "GENERATIVE_OBJECTIVE", inputs=(("hypothesis", (HYPOTHESIS,), 1), ("targets", (SID,), 1)), outputs=(("objective", "semantic_id/objective"),), parameters=_p({"margin": {"type": "number", "minimum": 0.0}, "negative_items": {"type": "integer", "minimum": 1}}, "margin", "negative_items"), capabilities=("GENERATIVE_OBJECTIVE_MUTATION",), causal_effect="Contrast complete target item codes against negative item codes using each code's own autoregressive prefix: log p(code_t | history, code_<t). The hypothesis port identifies complete codes; reuse history encoder states, not positive-target decoder logits, to score negatives.", failure_signal="Negative sampling rather than semantic generation explains gains; gathering negative tokens from positive-prefix logits is a different surrogate, not complete-code likelihood.", resource_effect="Additional batched teacher-forced decoder scoring for every declared negative code, reusing the live history encoder and attention mask."),
            primitive("objective.sequence_level_preference", "GENERATIVE_OBJECTIVE", inputs=(("hypothesis", (HYPOTHESIS,), 1), ("interactions", ("semantic_id/train_interactions",), 1)), outputs=(("objective", "semantic_id/objective"),), parameters=_p({"method": {"enum": ["PAIRWISE_PREFERENCE", "POLICY_GRADIENT", "GROUP_RELATIVE"]}, "weight": {"type": "number", "minimum": 0.0}}, "method", "weight"), capabilities=("GENERATIVE_OBJECTIVE_MUTATION", "POST_TRAINING_MUTATION"), causal_effect="Post-train complete SID generation toward recommendation preference.", failure_signal="High-variance post-training overfits reward proxies or breaks validity.", resource_effect="Separately budgeted sampled generations and reward evaluations."),
            primitive("objective.validity_regularization", "GENERATIVE_OBJECTIVE", inputs=(("token_state", (TOKEN_STATE,), 1), ("catalog", ("semantic_id/catalog_mapping",), 1)), outputs=(("objective", "semantic_id/objective"),), parameters=_p({"weight": {"type": "number", "minimum": 0.0}, "level": {"enum": ["PREFIX", "COMPLETE_CODE", "BOTH"]}}, "weight", "level"), capabilities=("GENERATIVE_OBJECTIVE_MUTATION",), causal_effect="Penalize probability mass assigned to catalog-invalid code paths.", failure_signal="Validity rises by collapsing diversity or copying catalog frequency.", resource_effect="Trie/catalog constraint evaluation during training."),
        ),
    ),
    axis(
        "OPTIMIZATION_STAGING",
        research_role="Define causal ordering and freezing between tokenizer, recommender, alignment, and post-training stages.",
        interaction_guidance="Typed staging is available in COMPOSITION and ARCHITECTURE_REWRITE through complete stage-model and trainer-method ownership while the exact parent framework and unchanged generation remain bound. Specify the phase ordering, update budgets, freezing and reference policy explicitly; preserve the parent supervised mechanism and fixed total experiment budget.",
        allow_multiple=False,
        primitives=(
            primitive("staging.freeze_tokenizer_then_sft", "OPTIMIZATION_STAGING", inputs=(("objective", ("semantic_id/objective",), 1),), outputs=(("objective", "semantic_id/objective"),), parameters=_p({"tokenizer_updates": {"type": "integer", "minimum": 1}, "recommender_updates": {"type": "integer", "minimum": 1}}, "tokenizer_updates", "recommender_updates"), capabilities=("TRAINING_STAGE_MUTATION",), causal_effect="Freeze item codes before supervised recommender training.", failure_signal="Fixed codes misalign with recommendation and cannot adapt.", resource_effect="Two non-overlapping measured stages."),
            primitive("staging.alternating_tokenizer_recommender", "OPTIMIZATION_STAGING", inputs=(("objective", ("semantic_id/objective",), 1),), outputs=(("objective", "semantic_id/objective"),), parameters=_p({"cycles": {"type": "integer", "minimum": 1, "maximum": 128}, "tokenizer_updates_per_cycle": {"type": "integer", "minimum": 1}, "recommender_updates_per_cycle": {"type": "integer", "minimum": 1}, "scheduler": {"enum": ["CONSTANT", "COSINE_WITH_WARMUP"]}, "warmup_updates": {"type": "integer", "minimum": 0}, "final_recommender_finetune_updates": {"type": "integer", "minimum": 0}}, "cycles", "tokenizer_updates_per_cycle", "recommender_updates_per_cycle", "scheduler", "warmup_updates", "final_recommender_finetune_updates"), capabilities=("TRAINING_STAGE_MUTATION",), causal_effect="Alternate code adaptation and recommender fitting under an explicit schedule, optionally followed by a separately selected recommender-only finetune stage while preserving tokenizer identity.", failure_signal="Code identity churn invalidates cached targets, optimization oscillates, defining alignment branches remain inactive, or omitting the declared warmup/final finetune changes the claimed parent identity.", resource_effect="Separately logged tokenizer/recommender cycles, scheduler warmup and recommender-only finetune updates."),
            primitive("staging.sft_then_preference_posttrain", "OPTIMIZATION_STAGING", inputs=(("objective", ("semantic_id/objective",), 1),), outputs=(("objective", "semantic_id/objective"),), parameters=_p({"sft_updates": {"type": "integer", "minimum": 1}, "posttrain_updates": {"type": "integer", "minimum": 1}, "reference_policy": {"enum": ["FROZEN_SFT", "EMA"]}}, "sft_updates", "posttrain_updates", "reference_policy"), capabilities=("TRAINING_STAGE_MUTATION", "POST_TRAINING_MUTATION"), causal_effect="Add a recommendation-aware preference stage after stable supervised generation.", failure_signal="Post-training reward hacking reduces exact ranking or code validity.", resource_effect="Separate SFT and preference-generation budgets."),
            primitive("staging.joint_tokenizer_recommender", "OPTIMIZATION_STAGING", inputs=(("objective", ("semantic_id/objective",), 1),), outputs=(("objective", "semantic_id/objective"),), parameters=_p({"joint_updates": {"type": "integer", "minimum": 1}, "tokenizer_warmstart": {"type": "boolean"}, "gradient_balance": {"enum": ["FIXED_WEIGHTS", "GRADIENT_NORM", "UNCERTAINTY_WEIGHTED"]}}, "joint_updates", "tokenizer_warmstart", "gradient_balance"), capabilities=("TRAINING_STAGE_MUTATION", "END_TO_END_SID_GRADIENT"), causal_effect="Optimize tokenizer assignments and next-SID recommendation concurrently so recommendation gradients directly shape the discrete item identity.", failure_signal="One objective dominates, gradients do not reach the tokenizer, or code identities churn faster than the recommender can track.", resource_effect="One joint stage with tokenizer and recommender gradient, utilization, and assignment-stability receipts."),
        ),
    ),
)


_EXECUTABLE_AXES = executable_axes(AXES)
_EXECUTABLE_CAPABILITY_FAMILIES = tuple(
    sorted(
        {
            str(capability)
            for axis_spec in _EXECUTABLE_AXES
            for primitive_spec in axis_spec["primitives"]
            for capability in primitive_spec["capabilities"]
        }
        | {"CUSTOM_MODEL_IMPLEMENTATION"}
    )
)


def _executable_episode_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    suspended_outcomes = {
        "dual_distillation_branch_deltas",
        "field_reconstruction_quality",
    }
    suspended_failures = {
        "DUAL_COLLABORATIVE_DISTILLATION_FAILURE",
        "STRUCTURED_FIELD_REPRESENTATION_FAILURE",
    }
    suspended_update_keys = {
        "distillation_branch_balance",
        "dual_collaborative_distillation",
        "missing_field_handling",
        "structured_field_representation",
    }
    outcome_classes = [
        item for item in value["outcome_classes"] if item not in suspended_failures
    ]
    return {
        **dict(value),
        "required_outcomes": [
            item
            for item in value["required_outcomes"]
            if item not in suspended_outcomes
        ],
        "failure_classes": [
            item
            for item in value["failure_classes"]
            if item not in suspended_failures
        ],
        "outcome_classes": outcome_classes,
        "outcome_memory_lanes": {
            item: value["outcome_memory_lanes"][item] for item in outcome_classes
        },
        "negative_update_keys": [
            item
            for item in value["negative_update_keys"]
            if item not in suspended_update_keys
        ],
        "primitive_update_keys": {
            primitive_id: [
                item for item in update_keys if item not in suspended_update_keys
            ]
            for primitive_id, update_keys in value["primitive_update_keys"].items()
            if primitive_id not in SUSPENDED_PRIMITIVES
        },
        "slot_update_keys": {
            slot_id: [
                item for item in update_keys if item not in suspended_update_keys
            ]
            for slot_id, update_keys in value["slot_update_keys"].items()
        },
    }


def _executable_qualification_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    suspended_behavior_probes = {
        "dual_distillation_each_branch_ablation",
        "dual_distillation_teacher_identity_and_branch_activation",
        "missing_structured_fields_use_explicit_unknown",
        "structured_field_asset_identity_when_declared",
        "structured_field_reconstruction_and_recommendation_branch_ablation_when_declared",
    }
    suspended_resource_probes = {
        "collaborative_teacher_cache_and_inference_cost",
        "field_coverage_and_unknown_rate",
        "structured_field_encoder_updates_and_cost",
    }
    return {
        **dict(value),
        "behavior_probes": [
            item
            for item in value["behavior_probes"]
            if item not in suspended_behavior_probes
        ],
        "resource_probes": [
            item
            for item in value["resource_probes"]
            if item not in suspended_resource_probes
        ],
    }

SPEC = DeclarativeFamilySpec(
    search_space_id=SPACE_ID,
    search_space_version="1.6.2-executable-bound",
    family_id=FAMILY_ID,
    family_version="1.6.2",
    provider_id=PROVIDER_ID,
    candidate_prefix="sid1",
    scientific_object="catalog_recommendation_by_learned_discrete_semantic_item_identifiers_and_valid_generative_decoding",
    supported_profile_kinds=("SEMANTIC_ID_GENERATIVE_FULL_RANKING",),
    allowed_data_roles={
        "ITEM_ID": "core/item_id",
        "USER_ID": "core/user_id",
        "TRAIN_INTERACTIONS": "semantic_id/train_interactions",
        "TRAIN_ITEM_SEQUENCE": "semantic_id/train_item_sequence",
        "FROZEN_ITEM_CONTENT_FEATURE": FEATURE,
        "TRAINED_ITEM_CODEBOOK": CODEBOOK,
        "TRAINED_ITEM_SID": SID,
        "FROZEN_LETTER_ITEM_SID": SID,
        "TRAIN_CATALOG_MAPPING": "semantic_id/catalog_mapping",
    },
    forbidden_data_roles=("VALIDATION_LABELS_AS_TOKENIZER_INPUT", "TEST_INTERACTIONS", "EXTERNAL_GENERATION_NETWORK", "UNMAPPED_FREE_TEXT_ITEM_OUTPUT"),
    frozen_protocol_fields=("dataset", "dataset_snapshot", "content_asset_identity", "structured_field_asset_identity", "item_id_mapping_identity", "split", "history_protocol", "candidate_universe", "seen_repeat_policy", "metric_semantics", "evaluator", "seed_policy", "tokenizer_update_budget", "recommender_update_budget", "posttraining_budget", "decoding_budget"),
    output_type=SCORE,
    output_slots=("ITEM_RESOLUTION", "DENSE_RETRIEVAL_CORRECTION"),
    required_slots=("SEMANTIC_TOKENIZER", "USER_CONTEXT_ENCODER", "GENERATIVE_BACKBONE", "DECODING_STRATEGY", "ITEM_RESOLUTION"),
    type_compatibility=(),
    axes=_EXECUTABLE_AXES,
    operators=standard_operators(structure_capability="GENERATOR_MUTATION", efficiency_capability="DECODING_MUTATION", training_capability="TRAINING_STAGE_MUTATION"),
    capability_families=_EXECUTABLE_CAPABILITY_FAMILIES,
    qualification_contract=_executable_qualification_contract({
        "api_probe": "history_to_catalog_item_scores_with_exact_item_mapping",
        "behavior_probes": ["tokenizer_train_only_provenance", "structured_field_asset_identity_when_declared", "missing_structured_fields_use_explicit_unknown", "no_item_or_interaction_filtering_after_profile_freeze", "padding_only_item_id_shift_preserves_targets", "structured_field_reconstruction_and_recommendation_branch_ablation_when_declared", "orthogonal_pre_rotation_is_learned_when_declared", "rotation_matrix_orthogonality_when_declared", "rotation_off_product_quantization_ablation_when_declared", "product_subspace_error_and_utilization_when_declared", "global_alignment_is_full_train_catalog_when_declared", "global_alignment_orthogonality_and_balance_ablation_when_declared", "soft_to_hard_temperature_and_hardening_match_declaration", "soft_hard_validation_gap_when_declared", "codeword_uniformity_distinct_from_usage_balance", "dual_distillation_teacher_identity_and_branch_activation", "dual_distillation_each_branch_ablation", "defining_alignment_weights_and_gradients_when_parent_claimed", "semantic_dependency_identity_when_declared", "alternating_scheduler_warmup_and_final_finetune_match_declaration", "prefix_uniqueness_and_collision_by_depth", "constant_suffix_is_redundancy_not_collapse_when_prefix_unique", "variable_depth_stop_and_serialization_match_declaration", "train_only_dense_negative_catalog_mask", "recommendation_gradient_reaches_tokenizer_when_joint", "detached_gradient_path_absent_when_joint", "hard_forward_soft_backward_consistency_when_declared", "code_utilization_and_collision", "code_usage_trajectory_when_trainable", "sid_assignment_churn_when_trainable", "sid_to_item_resolution_totality", "post_warmup_legal_sid_generation", "invalid_decode_rate", "full_candidate_coverage", "legal_sid_to_item_mapping_before_metric", "exclude_history_applied_after_item_resolution", "validation_loop_does_not_access_test", "single_gated_test_after_best_validation", "effective_batch_size_matches_frozen_contract", "evaluation_chunking_score_invariance", "generator_dense_path_ablation_when_present", "heldout_asset_noninterference"],
        "resource_probes": ["tokenizer_updates_and_cost", "structured_field_encoder_updates_and_cost", "field_coverage_and_unknown_rate", "orthogonal_rotation_training_cost", "product_codebook_memory", "soft_id_distribution_memory_and_cost", "collaborative_teacher_cache_and_inference_cost", "recommender_updates_and_cost", "final_recommender_finetune_updates_and_cost", "physical_microbatch_and_gradient_accumulation", "evaluation_chunk_size", "joint_tokenizer_recommender_backward_cost", "global_alignment_catalog_cost", "codebook_orthogonality_error", "warmup_updates_completed", "posttraining_generations", "decode_latency", "dense_retrieval_latency_when_present", "peak_device_memory", "sid_tokens_per_item"],
        "rank_admission_gate": {
            "warmup_complete": True,
            "minimum_legal_sid_predictions": 1,
            "hybrid_validation_complete": True,
            "qualification_metric_rankable": False,
            "validation_test_isolation": True,
            "legal_sid_to_item_mapping_complete": True,
            "exclude_history_at_item_level": True,
            "single_gated_test_after_best_validation": True,
        },
        "metric_episode": "qualification_metrics_never_ranked_then_profile_dev_selection_and_test_once",
        "hidden_fallback": "FORBIDDEN",
        "adapter_boundary": "RECBole3_PROFILE_ADAPTER_WITH_TOKENIZER_DECODER_IDENTITIES",
    }),
    episode_contract=_executable_episode_contract({
        "required_outcomes": ["ndcg_at_10", "recall_at_10", "valid_decode_rate", "catalog_coverage", "code_utilization", "collision_rate", "soft_hard_assignment_gap", "codeword_uniformity", "dual_distillation_branch_deltas", "prefix_unique_catalog_coverage", "mean_effective_sid_length", "representation_alignment", "rotation_orthogonality_error", "product_subspace_utilization", "codebook_orthogonality_error", "field_reconstruction_quality", "decode_latency", "stage_costs"],
        "failure_classes": ["MECHANISM_REFUTED", "STRUCTURED_FIELD_REPRESENTATION_FAILURE", "ORTHOGONAL_PRODUCT_QUANTIZATION_FAILURE", "GLOBAL_ALIGNMENT_OR_ORTHOGONALITY_FAILURE", "SOFT_ID_ANNEALING_FAILURE", "CODEWORD_UNIFORMITY_FAILURE", "DUAL_COLLABORATIVE_DISTILLATION_FAILURE", "PREFIX_UNIQUENESS_OR_SUFFIX_FAILURE", "TOKENIZER_COLLAPSE", "GENERATOR_UNDERFIT", "INVALID_OR_COLLIDING_DECODES", "RESOURCE_INFEASIBLE", "ASSET_OR_PROTOCOL_INVALID"],
        "outcome_classes": ["SUPPORTED", "MECHANISM_REFUTED", "STRUCTURED_FIELD_REPRESENTATION_FAILURE", "ORTHOGONAL_PRODUCT_QUANTIZATION_FAILURE", "GLOBAL_ALIGNMENT_OR_ORTHOGONALITY_FAILURE", "SOFT_ID_ANNEALING_FAILURE", "CODEWORD_UNIFORMITY_FAILURE", "DUAL_COLLABORATIVE_DISTILLATION_FAILURE", "PREFIX_UNIQUENESS_OR_SUFFIX_FAILURE", "TOKENIZER_COLLAPSE", "GENERATOR_UNDERFIT", "INVALID_OR_COLLIDING_DECODES", "RESOURCE_INFEASIBLE", "ASSET_OR_PROTOCOL_INVALID", "INCONCLUSIVE"],
        "outcome_memory_lanes": {
            "SUPPORTED": "MECHANISM_POSITIVE",
            "MECHANISM_REFUTED": "MECHANISM_NEGATIVE",
            "STRUCTURED_FIELD_REPRESENTATION_FAILURE": "MECHANISM_NEGATIVE",
            "ORTHOGONAL_PRODUCT_QUANTIZATION_FAILURE": "MECHANISM_NEGATIVE",
            "GLOBAL_ALIGNMENT_OR_ORTHOGONALITY_FAILURE": "MECHANISM_NEGATIVE",
            "SOFT_ID_ANNEALING_FAILURE": "MECHANISM_NEGATIVE",
            "CODEWORD_UNIFORMITY_FAILURE": "MECHANISM_NEGATIVE",
            "DUAL_COLLABORATIVE_DISTILLATION_FAILURE": "MECHANISM_NEGATIVE",
            "PREFIX_UNIQUENESS_OR_SUFFIX_FAILURE": "MECHANISM_NEGATIVE",
            "TOKENIZER_COLLAPSE": "MECHANISM_NEGATIVE",
            "GENERATOR_UNDERFIT": "OPTIMIZATION_DIAGNOSTIC",
            "INVALID_OR_COLLIDING_DECODES": "MECHANISM_NEGATIVE",
            "RESOURCE_INFEASIBLE": "RESOURCE_DIAGNOSTIC",
            "ASSET_OR_PROTOCOL_INVALID": "PROTOCOL_DIAGNOSTIC",
            "INCONCLUSIVE": "INCONCLUSIVE",
        },
        "negative_update_keys": ["feature_geometry", "structured_field_representation", "missing_field_handling", "tokenizer_geometry", "codebook_utilization", "orthogonal_preconditioning", "product_subspace_partition", "global_code_alignment", "codebook_orthogonality", "soft_hard_annealing", "codeword_uniformity", "dual_collaborative_distillation", "distillation_branch_balance", "alignment_branch_activation", "semantic_dependency_identity", "final_recommender_finetune", "end_to_end_gradient_alignment", "hard_soft_assignment_mismatch", "exploration_decay", "sid_length_and_structure", "prefix_stop_geometry", "redundant_suffix_decoding", "semantic_collaborative_alignment", "context_conditioning", "serial_error_compounding", "parallel_code_dependence", "validity_constraint", "dense_generator_complementarity", "stage_budget", "reranker_dominance"],
        "primitive_update_keys": {
            "feature.recommendation_native_structured_field_autoencoder": ["structured_field_representation", "missing_field_handling"],
            "tokenizer.orthogonally_preconditioned_product_quantization": ["orthogonal_preconditioning", "product_subspace_partition"],
            "codebook.globally_aligned_orthogonal_quantization": ["global_code_alignment", "codebook_orthogonality"],
            "codebook.soft_to_hard_annealed_assignment": ["soft_hard_annealing"],
            "alignment.codeword_representation_uniformity": ["codeword_uniformity"],
            "alignment.dual_collaborative_distillation": ["dual_collaborative_distillation", "distillation_branch_balance", "semantic_dependency_identity"],
            "sid.variable_depth_tree": ["prefix_stop_geometry", "redundant_suffix_decoding"],
            "staging.alternating_tokenizer_recommender": ["alignment_branch_activation", "final_recommender_finetune", "stage_budget"],
        },
        "slot_update_keys": {
            "ITEM_FEATURE_ENCODER": ["feature_geometry", "semantic_collaborative_alignment"],
            "SEMANTIC_TOKENIZER": ["tokenizer_geometry", "codebook_utilization", "end_to_end_gradient_alignment"],
            "CODEBOOK_GEOMETRY": ["codebook_utilization", "tokenizer_geometry", "hard_soft_assignment_mismatch", "exploration_decay"],
            "SID_STRUCTURE": ["sid_length_and_structure"],
            "SEMANTIC_COLLABORATIVE_ALIGNMENT": ["semantic_collaborative_alignment", "end_to_end_gradient_alignment"],
            "USER_CONTEXT_ENCODER": ["context_conditioning"],
            "GENERATIVE_BACKBONE": ["serial_error_compounding", "parallel_code_dependence"],
            "DECODING_STRATEGY": ["validity_constraint", "serial_error_compounding", "parallel_code_dependence"],
            "DENSE_RETRIEVAL_CORRECTION": ["dense_generator_complementarity", "reranker_dominance"],
            "ITEM_RESOLUTION": ["validity_constraint", "reranker_dominance"],
            "GENERATIVE_OBJECTIVE": ["semantic_collaborative_alignment", "validity_constraint", "end_to_end_gradient_alignment"],
            "OPTIMIZATION_STAGING": ["stage_budget", "end_to_end_gradient_alignment", "exploration_decay"],
        },
    }),
)


class SemanticIdGenerativeProviderV1(DeclarativeMechanismSpaceProvider):
    def __init__(self) -> None:
        super().__init__(SPEC)

    def compile(self, envelope: Mapping[str, Any]):
        report = super().compile(envelope)
        if report.resolved_ir is None:
            return report
        resolved_ir = deep_thaw(report.resolved_ir)
        component_bindings = component_binding_projection(resolved_ir["components"])
        payload = envelope.get("program_payload")
        construction_mode = (
            payload.get("construction_mode") if isinstance(payload, Mapping) else None
        )
        changed_slots = {
            str(item.get("slot_id"))
            for item in (
                payload.get("changed_slots", ())
                if isinstance(payload, Mapping)
                else ()
            )
            if isinstance(item, Mapping) and isinstance(item.get("slot_id"), str)
        }
        removed_slots = {
            str(slot_id)
            for slot_id in (
                payload.get("removed_slots", ())
                if isinstance(payload, Mapping)
                else ()
            )
            if isinstance(slot_id, str)
        }
        generator_components = [
            item
            for item in resolved_ir["components"]
            if isinstance(item, Mapping)
            and item.get("slot_id") == "GENERATIVE_BACKBONE"
        ]
        generator_is_exact_parent = (
            len(generator_components) == 1
            and component_bindings.get(
                str(generator_components[0].get("component_id", ""))
            )
            == "EXACT_LIGER_PARENT"
        )
        is_exact_parent_program = bool(component_bindings) and set(
            component_bindings.values()
        ) == {"EXACT_LIGER_PARENT"}
        is_canonical_parent_program = (
            is_exact_parent_program
            and changed_slots == _CANONICAL_PARENT_DECLARED_SLOTS
            and not removed_slots
        )
        if construction_mode != "CUSTOM_MODEL" and (
            "GENERATIVE_BACKBONE" in changed_slots or not generator_is_exact_parent
        ):
            diagnostic = CompileDiagnostic(
                "PARENT_GENERATIVE_BACKBONE_REPLACEMENT_REQUIRES_CUSTOM_MODEL",
                "parent-preserving semantic-ID programs must keep the exact frozen "
                "LIGER/T5 autoregressive generator active; a generator replacement "
                "must use CUSTOM_MODEL with explicit whole-model synthesis",
                path=("/program_payload/components/GENERATIVE_BACKBONE",),
                expected={
                    "construction_mode": "CUSTOM_MODEL for replacement",
                    "parent_preserving_primitive": "generator.t5_encoder_decoder",
                },
                actual={
                    "construction_mode": construction_mode,
                    "changed_slot_declared": "GENERATIVE_BACKBONE" in changed_slots,
                    "exact_parent_binding": generator_is_exact_parent,
                },
            )
            return replace(
                report,
                status=CompileStatus.INVALID,
                diagnostics=(*report.diagnostics, diagnostic),
                resolved_ir=None,
            )
        actual_non_parent_slots = {
            str(component.get("slot_id"))
            for component in resolved_ir["components"]
            if isinstance(component, Mapping)
            and component_bindings.get(str(component.get("component_id", "")))
            != "EXACT_LIGER_PARENT"
        }
        declared_effective_slots = changed_slots | removed_slots
        actual_effective_slots = actual_non_parent_slots | removed_slots
        if (
            construction_mode != "CUSTOM_MODEL"
            and not is_canonical_parent_program
            and actual_effective_slots != declared_effective_slots
        ):
            diagnostic = CompileDiagnostic(
                "PARENT_LOCAL_CHANGE_DECLARATION_MISMATCH",
                "ordinary semantic-ID programs must declare exactly the slots whose "
                "components differ from the frozen LIGER parent or are removed",
                path=("/program_payload/changed_slots",),
                expected=sorted(actual_effective_slots),
                actual=sorted(declared_effective_slots),
            )
            return replace(
                report,
                status=CompileStatus.INVALID,
                diagnostics=(*report.diagnostics, diagnostic),
                resolved_ir=None,
            )
        unsupported_parent_local_slots = (
            actual_effective_slots - _PARENT_LOCAL_SEAM_SLOTS - {"OPTIMIZATION_STAGING"}
        )
        if (
            construction_mode != "CUSTOM_MODEL"
            and not is_canonical_parent_program
            and unsupported_parent_local_slots
        ):
            diagnostic = CompileDiagnostic(
                "PARENT_LOCAL_SLOT_REQUIRES_CUSTOM_MODEL",
                "parent-preserving semantic-ID programs may change only slots with "
                "an active machine-consumed LIGER seam or explicit training staging; "
                "tokenizer, codebook, SID, alignment, resolution, dense-retrieval, and generator "
                "replacement require CUSTOM_MODEL with explicit whole-model synthesis",
                path=("/program_payload/changed_slots",),
                expected=sorted(_PARENT_LOCAL_SEAM_SLOTS | {"OPTIMIZATION_STAGING"}),
                actual=sorted(unsupported_parent_local_slots),
            )
            return replace(
                report,
                status=CompileStatus.INVALID,
                diagnostics=(*report.diagnostics, diagnostic),
                resolved_ir=None,
            )
        changed_decoding_primitives = {
            str(component.get("primitive_id"))
            for component in resolved_ir["components"]
            if isinstance(component, Mapping)
            and component.get("slot_id") == "DECODING_STRATEGY"
        }
        unsupported_parent_local_decoding = (
            changed_decoding_primitives - _PARENT_LOCAL_DECODING_PRIMITIVES
            if "DECODING_STRATEGY" in actual_effective_slots
            else set()
        )
        if (
            construction_mode != "CUSTOM_MODEL"
            and not is_canonical_parent_program
            and unsupported_parent_local_decoding
        ):
            diagnostic = CompileDiagnostic(
                "PARENT_LOCAL_DECODING_REQUIRES_CUSTOM_MODEL",
                "parent-preserving semantic-ID programs may change decoding only "
                "through the active T5 autoregressive generation seam; parallel "
                "decoding and catalog-score factorization require CUSTOM_MODEL "
                "with explicit whole-model synthesis",
                path=("/program_payload/components/DECODING_STRATEGY",),
                expected=sorted(_PARENT_LOCAL_DECODING_PRIMITIVES),
                actual=sorted(unsupported_parent_local_decoding),
            )
            return replace(
                report,
                status=CompileStatus.INVALID,
                diagnostics=(*report.diagnostics, diagnostic),
                resolved_ir=None,
            )
        resolved_ir["component_runtime_bindings"] = component_bindings
        resolved_ir["source_ownership"] = (
            "EXACT_LIGER_PARENT"
            if is_exact_parent_program
            else "FULL_SOURCE_REQUIRED"
        )
        requirements = []
        diagnostics = []
        for component in resolved_ir["components"]:
            primitive_id = component.get("primitive_id")
            if primitive_id == _FROZEN_LETTER_TOKENIZER_PRIMITIVE:
                parameters = component["parameters"]
                codes_per_level = int(parameters["codes_per_level"])
                semantic_prefix_levels = int(parameters["semantic_prefix_levels"])
                offsets = [
                    1 + position * codes_per_level
                    for position in range(semantic_prefix_levels + 1)
                ]
                requirements.append(
                    f"{component['component_id']}: frozen LETTER SID codec maps "
                    "raw [c0,c1,c2,collision] to absolute expanded IDs "
                    f"[c0+{offsets[0]},c1+{offsets[1]},c2+{offsets[2]},"
                    f"collision+{offsets[3]}]; the official shared-vocabulary "
                    "T5 path consumes those absolute IDs unchanged; any "
                    "position-local or parallel cross-entropy head must use "
                    f"offsets={offsets} and target_j=expanded_sid[:,j]-"
                    "offsets[j], giving prefix targets 0..255 and collision "
                    "targets 0..max_collision_code from the loaded frozen "
                    "mapping; assignment emits one complete position-local "
                    "tuple and the compiler-owned frozen LETTER codec adds "
                    "the same offsets exactly once at the generated/resolver "
                    "boundary before absolute catalog SID lookup; already "
                    "absolute official-parent tuples pass through unchanged; "
                    "offset 257 belongs only to position 1 and must not be "
                    "reused for positions 2 or 3"
                )
            if primitive_id not in _PRODUCT_QUANTIZER_PRIMITIVES:
                continue
            subspaces = int(component["parameters"]["subspaces"])
            if _LIGER_PROJECTION_DIMENSION % subspaces:
                diagnostics.append(
                    CompileDiagnostic(
                        "PRIMITIVE_PARAMETERS_INVALID",
                        "subspaces must divide the frozen LIGER projection dimension",
                        path=(
                            "/program_payload/components/"
                            f"{component['component_id']}/parameters/subspaces"
                        ),
                        expected=list(_LIGER_PRODUCT_QUANTIZATION_SUBSPACES),
                        actual=subspaces,
                    )
                )
                continue
            subspace_dimension = _LIGER_PROJECTION_DIMENSION // subspaces
            requirement = (
                f"{component['component_id']}: projection_dimension=128; "
                f"subspaces={subspaces}; subdim=128//{subspaces}="
                f"{subspace_dimension}; partition the final latent dimension as "
                f"latent.reshape(*latent.shape[:-1], {subspaces}, "
                f"{subspace_dimension}); every codebook vector has dimension "
                f"{subspace_dimension}"
            )
            if (
                component["primitive_id"]
                == "tokenizer.orthogonally_preconditioned_product_quantization"
            ):
                if component["parameters"]["rotation_scope"] == "GLOBAL":
                    requirement += (
                        "; GLOBAL rotation shape=[128,128] and identity construction="
                        "torch.eye(128, device=latent.device, dtype=latent.dtype)"
                    )
                else:
                    requirement += (
                        f"; BLOCK_DIAGONAL rotation shape=[{subspaces},"
                        f"{subspace_dimension},{subspace_dimension}] and identity "
                        f"construction=torch.eye({subspace_dimension}, "
                        "device=latent.device, dtype=latent.dtype)"
                        f".repeat({subspaces}, 1, 1); never reshape torch.eye(128) "
                        "into per-subspace blocks"
                    )
            requirements.append(requirement)
        if diagnostics:
            return replace(
                report,
                status=CompileStatus.INVALID,
                diagnostics=(*report.diagnostics, *diagnostics),
                resolved_ir=None,
            )
        if not requirements:
            return report
        resolved_ir["implementation_requirement"] = (
            "CANDIDATE_LOCAL_IMPLEMENTATION_AND_QUALIFICATION_REQUIRED; "
            + "; ".join(requirements)
        )
        return replace(report, resolved_ir=resolved_ir)


PROVIDER = SemanticIdGenerativeProviderV1()

__all__ = ["PROVIDER", "SPACE_ID", "FAMILY_ID", "SemanticIdGenerativeProviderV1"]
