"""Chronological sequential-recommendation domain language for V2R5."""

from __future__ import annotations

from recclaw_core.mechanism_space.declarative_provider import (
    DeclarativeFamilySpec,
    DeclarativeMechanismSpaceProvider,
    axis,
    object_parameters,
    primitive,
    standard_operators,
)

SPACE_ID = "SEQUENTIAL_SCALING_MECHANISM_SPACE_V1"
FAMILY_ID = "SEQUENTIAL_SCALING_V1"
PROVIDER_ID = "recclaw.search-space-provider.sequential-scaling.v1"

SEQ = "sequential/item_sequence"
TIME = "sequential/time_sequence"
QUERY_TIME = "sequential/query_timestamp"
REP = "sequential/sequence_representation"
REGISTER = "sequential/sequence_register"
STATE = "sequential/sequence_state"
SCORE = "core/user_item_relevance_score"


def _p(properties: dict, *required: str) -> dict:
    return object_parameters(properties, required=required)


def _s5_spectrum_stability() -> dict:
    """Require every continuous-S5 program to declare its real-spectrum policy."""

    constrained = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "mode": {
                "enum": ["NEGATIVE_SOFTPLUS", "NEGATIVE_REAL_CLIP"]
            },
            "margin": {"type": "number", "exclusiveMinimum": 0.0},
        },
        "required": ["mode", "margin"],
    }
    return {
        "oneOf": [
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {"mode": {"const": "UNCONSTRAINED"}},
                "required": ["mode"],
            },
            constrained,
        ]
    }


AXES = (
    axis(
        "ITEM_REPRESENTATION",
        research_role="Embed chronological item identities before any temporal mixing.",
        interaction_guidance="Keep embedding size fixed for backbone comparisons; representation expansion is a support change.",
        allow_multiple=False,
        primitives=(
            primitive("representation.shared_item_embedding", "ITEM_REPRESENTATION", inputs=(("items", (SEQ,), 1),), outputs=(("representation", REP),), parameters=_p({"dimension": {"type": "integer", "minimum": 16, "maximum": 2048}, "tie_output": {"type": "boolean"}}, "dimension", "tie_output"), capabilities=("REPRESENTATION_MUTATION",), causal_effect="Share one item space across history encoding and candidate scoring.", failure_signal="Capacity rather than temporal mechanism explains gains.", resource_effect="Embedding memory scales with item count and dimension."),
            primitive("representation.factorized_item_embedding", "ITEM_REPRESENTATION", inputs=(("items", (SEQ,), 1),), outputs=(("representation", REP),), parameters=_p({"model_dimension": {"type": "integer", "minimum": 16, "maximum": 2048}, "factor_dimension": {"type": "integer", "minimum": 8, "maximum": 1024}}, "model_dimension", "factor_dimension"), capabilities=("REPRESENTATION_MUTATION", "SEQUENCE_EFFICIENCY_REWRITE"), causal_effect="Decouple item-table capacity from sequence-state width.", failure_signal="Compression harms rare-item identity before the backbone is tested.", resource_effect="Reduces item-table memory with projection overhead."),
            primitive("representation.frequency_adaptive", "ITEM_REPRESENTATION", inputs=(("items", (SEQ,), 1), ("statistics", ("sequential/train_statistics",), 1)), outputs=(("representation", REP),), parameters=_p({"head_dimension": {"type": "integer", "minimum": 16}, "tail_dimension": {"type": "integer", "minimum": 4}, "frequency_buckets": {"type": "integer", "minimum": 2, "maximum": 16}}, "head_dimension", "tail_dimension", "frequency_buckets"), capabilities=("REPRESENTATION_MUTATION",), causal_effect="Allocate representation width by train-only item frequency.", failure_signal="Adaptive width encodes popularity and weakens tail ranking.", resource_effect="Reduces table memory but adds bucket projections."),
        ),
    ),
    axis(
        "TEMPORAL_POSITION_ENCODING",
        research_role="Expose order and elapsed-time structure without accessing future events.",
        interaction_guidance="Use at most one primary encoding plus one support bias; compare with identical backbone and history length.",
        allow_multiple=True,
        primitives=(
            primitive("temporal.absolute_position", "TEMPORAL_POSITION_ENCODING", inputs=(("representation", (REP,), 1),), outputs=(("representation", REP),), parameters=_p({"maximum_length": {"type": "integer", "minimum": 2, "maximum": 65536}, "mode": {"enum": ["LEARNED", "SINUSOIDAL"]}}, "maximum_length", "mode"), capabilities=("TEMPORAL_ENCODING_MUTATION",), causal_effect="Encode absolute position inside the truncated history window.", failure_signal="Position table overfits window length and does not transfer to longer contexts.", resource_effect="Linear position injection."),
            primitive("temporal.relative_position_bias", "TEMPORAL_POSITION_ENCODING", inputs=(("representation", (REP,), 1),), outputs=(("representation", REP),), parameters=_p({"buckets": {"type": "integer", "minimum": 4, "maximum": 512}, "maximum_distance": {"type": "integer", "minimum": 4, "maximum": 65536}}, "buckets", "maximum_distance"), capabilities=("TEMPORAL_ENCODING_MUTATION",), causal_effect="Bias interactions by relative event distance.", failure_signal="Relative distance adds no information beyond causal order.", resource_effect="Small bias table; pairwise application for attention."),
            primitive("temporal.interval_bucket", "TEMPORAL_POSITION_ENCODING", inputs=(("representation", (REP,), 1), ("time", (TIME,), 1)), outputs=(("representation", REP),), parameters=_p({"buckets": {"type": "integer", "minimum": 4, "maximum": 512}, "scale": {"enum": ["LINEAR", "LOG", "QUANTILE_TRAIN_ONLY"]}}, "buckets", "scale"), capabilities=("TEMPORAL_ENCODING_MUTATION",), causal_effect="Encode elapsed time between historical events.", failure_signal="Timestamp granularity or dataset artifacts dominate rather than user dynamics.", resource_effect="One train-only interval discretization and embedding lookup."),
            primitive("temporal.continuous_decay", "TEMPORAL_POSITION_ENCODING", inputs=(("representation", (REP,), 1), ("time", (TIME,), 1)), outputs=(("representation", REP),), parameters=_p({"decay_family": {"enum": ["EXPONENTIAL", "POWER", "LEARNED_MONOTONE"]}, "minimum_half_life": {"type": "number", "exclusiveMinimum": 0.0}}, "decay_family", "minimum_half_life"), capabilities=("TEMPORAL_ENCODING_MUTATION",), causal_effect="Apply continuous recency bias with a declared monotonic form.", failure_signal="Aggressive decay discards durable long-term preference.", resource_effect="Linear elementwise temporal modulation."),
        ),
    ),
    axis(
        "TRAIN_INPUT_VIEW",
        research_role="Construct train-only views of the observed chronological prefix without changing the next-item target or evaluation input.",
        interaction_guidance="Input corruption is a support mechanism, not masked-item reconstruction. Evaluation must use the clean frozen prefix, and the view requires a no-corruption matched control.",
        allow_multiple=False,
        primitives=(
            primitive("view.observed_history_masking", "TRAIN_INPUT_VIEW", inputs=(("representation", (REP,), 1),), outputs=(("representation", REP),), parameters=_p({"mask_probability": {"type": "number", "minimum": 0.0, "exclusiveMaximum": 1.0}, "replacement": {"enum": ["LEARNED_MASK_TOKEN", "ZERO_VECTOR", "ITEM_DROPOUT"]}}, "mask_probability", "replacement"), capabilities=("TRAIN_INPUT_VIEW_MUTATION",), causal_effect="Replace a declared fraction of observed-history representations during training while preserving the chronological next-item target; evaluation always consumes the uncorrupted prefix.", failure_signal="The learned mask identity becomes a shortcut, train/evaluation view shift harms ranking, or improvement survives neither a clean-input control nor a mask-view ablation.", resource_effect="No extra sequence pass; one train-only mask sample and optional learned replacement vector per prefix."),
        ),
    ),
    axis(
        "SEQUENCE_REGISTER_GEOMETRY",
        research_role="Pack variable-length chronological prefixes while preserving an explicit per-sequence computation boundary.",
        interaction_guidance="A sequence register is non-embedded segment metadata, not a learned token. Use it only with a register-aware backbone and require relabeling invariance, cross-sequence isolation, and a padded-equivalence control.",
        allow_multiple=False,
        primitives=(
            primitive("packing.padded_prefix_batch", "SEQUENCE_REGISTER_GEOMETRY", inputs=(("representation", (REP,), 1),), outputs=(("representation", REP),), parameters=_p({"maximum_length": {"type": "integer", "minimum": 2, "maximum": 65536}}, "maximum_length"), capabilities=("SEQUENCE_EFFICIENCY_REWRITE",), causal_effect="Keep each chronological prefix in its own padded batch row and expose its exact valid-prefix length, so recurrent state cannot cross user boundaries.", failure_signal="Padding changes the gathered valid-prefix state or dominates compute relative to real tokens.", resource_effect="Simple native RecBole batching with compute proportional to padded rather than packed tokens."),
            primitive("packing.segment_registered_variable_length", "SEQUENCE_REGISTER_GEOMETRY", inputs=(("representation", (REP,), 1),), outputs=(("representation", REP), ("register", REGISTER)), parameters=_p({}), capabilities=("SEQUENCE_REGISTER_GEOMETRY_MUTATION", "SEQUENCE_EFFICIENCY_REWRITE"), causal_effect="Concatenate variable-length chronological prefixes and attach one non-embedded batch-local segment label to every token so a compatible sequence mixer remains block-isolated by prefix without padding or additional truncation.", failure_signal="Segment relabeling changes scores, state crosses a sequence boundary, packed and padded controls disagree, or the apparent gain is only a larger effective history.", resource_effect="Compute follows real tokens rather than padded tokens, with one integer segment label per token and explicit packed-token accounting."),
        ),
    ),
    axis(
        "SEQUENCE_BACKBONE",
        research_role="Keep the exact frozen BiSSD transition as the active construction anchor for every ordinary parent-preserving candidate.",
        interaction_guidance="Do not declare SEQUENCE_BACKBONE as changed in COMPOSITION or ARCHITECTURE_REWRITE. Ordinary research must augment the active BiSSD through supported temporal, state-gating, or interest-routing mechanisms. A scientifically necessary full backbone replacement must use CUSTOM_MODEL with explicit synthesis and an honest whole-architecture comparison.",
        allow_multiple=False,
        primitives=(
            primitive("backbone.bidirectional_prefix_reversal_state_space_duality", "SEQUENCE_BACKBONE", inputs=(("representation", (REP,), 1),), outputs=(("state", STATE),), parameters=_p({"layers": {"type": "integer", "minimum": 1, "maximum": 64}, "state_dimension": {"type": "integer", "minimum": 8, "maximum": 2048}, "head_dimension": {"type": "integer", "minimum": 8, "maximum": 512}, "expansion": {"type": "integer", "minimum": 1, "maximum": 16}, "local_convolution": {"type": "integer", "minimum": 1, "maximum": 64}, "backward_weight": {"type": "number", "minimum": 0.0, "maximum": 4.0}, "direction_parameter_sharing": {"enum": ["SHARED", "INDEPENDENT"]}, "reverse_output_alignment": {"enum": ["AS_EMITTED_REVERSE_ORDER", "CHRONOLOGICAL_REALIGNED"]}, "post_mixer": {"enum": ["RESIDUAL_FFN", "IDENTITY"]}, "ffn_multiplier": {"type": "number", "minimum": 1.0, "maximum": 16.0}}, "layers", "state_dimension", "head_dimension", "expansion", "local_convolution", "backward_weight", "direction_parameter_sharing", "reverse_output_alignment", "post_mixer", "ffn_multiplier"), capabilities=("SEQUENCE_BACKBONE_MUTATION", "SEQUENCE_EFFICIENCY_REWRITE", "BIDIRECTIONAL_PREFIX_MIXING"), causal_effect="Run one shared state-space-duality mixer over the observed prefix and its valid-prefix reversal, preserve the declared reverse-output alignment, fuse both streams with the item residual, and optionally apply a residual feed-forward mixer.", failure_signal="A direction is inert, reverse alignment differs from the declaration, padding affects the gathered state, or the feed-forward path rather than bidirectional SSD explains the result.", resource_effect="Two near-linear padded-prefix SSD passes plus residual fusion and an optional feed-forward mixer; report padded-token overhead and both directional contributions."),
        ),
    ),
    axis(
        "STATE_UPDATE_GATING",
        research_role="Control how new events overwrite, retain, or reset sequence state.",
        interaction_guidance="A gating proposal must expose gate saturation and retention probes in the Episode.",
        allow_multiple=False,
        primitives=(
            primitive("state.input_selective_gate", "STATE_UPDATE_GATING", inputs=(("state", (STATE,), 1), ("representation", (REP,), 1)), outputs=(("state", STATE),), parameters=_p({"gate_rank": {"type": "integer", "minimum": 1, "maximum": 256}, "bias": {"type": "number"}}, "gate_rank", "bias"), capabilities=("STATE_UPDATE_MUTATION",), causal_effect="Make retention depend on the current item representation.", failure_signal="Gate saturates and behaves as identity or full overwrite.", resource_effect="One low-rank gate per token."),
            primitive("state.time_aware_forgetting", "STATE_UPDATE_GATING", inputs=(("state", (STATE,), 1), ("time", (TIME,), 1)), outputs=(("state", STATE),), parameters=_p({"minimum_retention": {"type": "number", "minimum": 0.0, "maximum": 1.0}, "decay_family": {"enum": ["EXPONENTIAL", "LEARNED_MONOTONE"]}, "time_scale_seconds": {"const": 86400.0}}, "minimum_retention", "decay_family", "time_scale_seconds"), capabilities=("STATE_UPDATE_MUTATION", "TEMPORAL_ENCODING_MUTATION"), causal_effect="Forget stale state as a function of elapsed days obtained by dividing nonnegative raw-second gaps by the compiled time scale.", failure_signal="An omitted or reinterpreted time scale saturates the retention gate and removes periodic or durable preference.", resource_effect="Linear state modulation."),
            primitive("state.surprise_reset", "STATE_UPDATE_GATING", inputs=(("state", (STATE,), 1), ("representation", (REP,), 1)), outputs=(("state", STATE),), parameters=_p({"threshold": {"type": "number", "minimum": 0.0}, "reset_floor": {"type": "number", "minimum": 0.0, "maximum": 1.0}}, "threshold", "reset_floor"), capabilities=("STATE_UPDATE_MUTATION",), causal_effect="Reset part of state when the incoming event is incompatible with the current interest trajectory.", failure_signal="Reset treats exploration/noise as regime shifts and harms continuity.", resource_effect="One compatibility score and gated reset per token."),
        ),
    ),
    axis(
        "INTEREST_ROUTING",
        research_role="Separate short-lived intent from persistent preference before final prediction.",
        interaction_guidance="Route from a shared backbone state unless the split itself is the declared architecture rewrite.",
        allow_multiple=False,
        primitives=(
            primitive("interest.short_long_dual_path", "INTEREST_ROUTING", inputs=(("state", (STATE,), 1),), outputs=(("state", STATE),), parameters=_p({"short_window": {"type": "integer", "minimum": 1, "maximum": 512}, "fusion": {"enum": ["GATED_SUM", "CONCAT_PROJECT", "CROSS_ATTEND"]}}, "short_window", "fusion"), capabilities=("INTEREST_ROUTING_MUTATION",), causal_effect="Preserve a local intent path alongside the long-context state.", failure_signal="One path dominates or the short window acts as another tuned context length.", resource_effect="Adds bounded local aggregation and fusion."),
            primitive("interest.convolutional_gru_short_path", "INTEREST_ROUTING", inputs=(("state", (STATE,), 1), ("representation", (REP,), 1)), outputs=(("state", STATE),), parameters=_p({"convolution_kernel": {"type": "integer", "minimum": 1, "maximum": 64}, "gru_layers": {"type": "integer", "minimum": 1, "maximum": 8}, "hidden_dimension": {"type": "integer", "minimum": 8, "maximum": 2048}, "fusion": {"enum": ["LEARNED_WEIGHTED_SUM", "GATED_SUM", "CONCAT_PROJECT"]}}, "convolution_kernel", "gru_layers", "hidden_dimension", "fusion"), capabilities=("INTEREST_ROUTING_MUTATION", "STATE_UPDATE_MUTATION"), causal_effect="Extract local item features with a one-dimensional convolution, model them through a GRU path, and fuse that short-term state with the primary long-context state.", failure_signal="The recurrent path dominates, only adds capacity, or improves aggregate quality without helping short-history users.", resource_effect="One convolutional feature pass and recurrent scan in parallel with the primary backbone."),
            primitive("interest.multiscale_pyramid", "INTEREST_ROUTING", inputs=(("state", (STATE,), 1),), outputs=(("state", STATE),), parameters=_p({"scales": {"type": "array", "minItems": 2, "uniqueItems": True, "items": {"type": "integer", "minimum": 1}}, "aggregation": {"enum": ["MEAN", "ATTENTION", "STATE_POOL"]}}, "scales", "aggregation"), capabilities=("INTEREST_ROUTING_MUTATION",), causal_effect="Represent preference at several explicit temporal resolutions.", failure_signal="Multiscale branches are redundant and only increase capacity.", resource_effect="Adds pooled branch states proportional to number of scales."),
            primitive("interest.user_conditioned_router", "INTEREST_ROUTING", inputs=(("state", (STATE,), 1), ("user", ("core/user_id",), 1)), outputs=(("state", STATE),), parameters=_p({"routes": {"type": "integer", "minimum": 2, "maximum": 16}, "router_temperature": {"type": "number", "exclusiveMinimum": 0.0}}, "routes", "router_temperature"), capabilities=("INTEREST_ROUTING_MUTATION",), causal_effect="Allocate users among shared dynamic routes without external features.", failure_signal="Router memorizes user frequency or collapses routes.", resource_effect="Small routing network and route-state storage."),
        ),
    ),
    axis(
        "PREDICTION_HEAD",
        research_role="Project the final chronological state to scores over the frozen full candidate universe.",
        interaction_guidance="Do not change candidate protocol here; decoding approximations require a protocol branch.",
        allow_multiple=False,
        primitives=(
            primitive("prediction.tied_dot_product", "PREDICTION_HEAD", inputs=(("state", (STATE,), 1), ("candidate_items", ("core/item_id",), 1)), outputs=(("score", SCORE),), parameters=_p({"temperature": {"type": "number", "exclusiveMinimum": 0.0}}, "temperature"), capabilities=("PREDICTION_MUTATION",), causal_effect="Score all items in the same representation space as history embeddings.", failure_signal="Temperature rescales loss but does not improve ordering.", resource_effect="Full item-matrix scoring."),
            primitive("prediction.normalized_cosine", "PREDICTION_HEAD", inputs=(("state", (STATE,), 1), ("candidate_items", ("core/item_id",), 1)), outputs=(("score", SCORE),), parameters=_p({"learned_scale": {"type": "boolean"}}, "learned_scale"), capabilities=("PREDICTION_MUTATION",), causal_effect="Separate angular preference from state/item norm.", failure_signal="Norm removal harms popularity calibration or only changes optimization.", resource_effect="Full normalized item scoring."),
            primitive("prediction.multi_interest_max", "PREDICTION_HEAD", inputs=(("state", (STATE,), 2), ("candidate_items", ("core/item_id",), 1)), outputs=(("score", SCORE),), parameters=_p({"aggregation": {"enum": ["MAX", "LOGSUMEXP", "GATED"]}}, "aggregation"), capabilities=("PREDICTION_MUTATION", "INTEREST_ROUTING_MUTATION"), causal_effect="Let separate interest states compete for each candidate.", failure_signal="Multiple heads collapse or improve only through added parameters.", resource_effect="Scoring cost scales with number of interest states."),
        ),
    ),
    axis(
        "SEQUENTIAL_OBJECTIVE",
        research_role="Define train-only supervision over chronological prefixes.",
        interaction_guidance="Objective changes are core unless used solely to support a new backbone; never expose heldout suffixes.",
        allow_multiple=True,
        primitives=(
            primitive("objective.next_item_softmax", "SEQUENTIAL_OBJECTIVE", inputs=(("score", (SCORE,), 1), ("targets", ("sequential/next_item_targets",), 1)), outputs=(("objective", "sequential/objective"),), parameters=_p({"label_smoothing": {"type": "number", "minimum": 0.0, "maximum": 0.5}}, "label_smoothing"), capabilities=("SEQUENTIAL_OBJECTIVE_MUTATION",), causal_effect="Optimize exact next-item likelihood over chronological prefixes.", failure_signal="Softmax favors head items or becomes the dominant compute bottleneck.", resource_effect="Full-vocabulary or exact equivalent loss required by profile."),
            primitive("objective.masked_item_reconstruction", "SEQUENTIAL_OBJECTIVE", inputs=(("score", (SCORE,), 1), ("targets", ("sequential/masked_item_targets",), 1)), outputs=(("objective", "sequential/objective"),), parameters=_p({"mask_probability": {"type": "number", "exclusiveMinimum": 0.0, "exclusiveMaximum": 1.0}, "mask_strategy": {"enum": ["RANDOM", "SPAN"]}}, "mask_probability", "mask_strategy"), capabilities=("SEQUENTIAL_OBJECTIVE_MUTATION",), causal_effect="Learn bidirectional train-prefix context through masked reconstruction.", failure_signal="Pretraining task fails to transfer to chronological next-item ordering.", resource_effect="Additional masked targets per sequence."),
            primitive("objective.generative_transduction", "SEQUENTIAL_OBJECTIVE", inputs=(("score", (SCORE,), 1), ("targets", ("sequential/next_item_targets",), 1)), outputs=(("objective", "sequential/objective"),), parameters=_p({"multiple_targets": {"type": "integer", "minimum": 1, "maximum": 64}, "target_weighting": {"enum": ["UNIFORM", "RECENCY", "POSITION"]}}, "multiple_targets", "target_weighting"), capabilities=("SEQUENTIAL_OBJECTIVE_MUTATION", "GENERATIVE_SEQUENCE_TRAINING"), causal_effect="Train several future-item transduction targets from each prefix without changing evaluation chronology.", failure_signal="Longer-horizon supervision dilutes immediate next-item relevance.", resource_effect="More supervised targets per input token; budget separately by predicted targets."),
            primitive("objective.state_consistency_auxiliary", "SEQUENTIAL_OBJECTIVE", inputs=(("state", (STATE,), 2),), outputs=(("objective", "sequential/objective"),), parameters=_p({"weight": {"type": "number", "minimum": 0.0}, "distance": {"enum": ["COSINE", "MSE", "CONTRASTIVE"]}}, "weight", "distance"), capabilities=("SEQUENTIAL_OBJECTIVE_MUTATION",), causal_effect="Regularize state consistency across declared train-only views or truncations.", failure_signal="Auxiliary dominates ranking or enforces oversmoothed state.", resource_effect="Additional view encoding and loss."),
        ),
    ),
)

SPEC = DeclarativeFamilySpec(
    search_space_id=SPACE_ID,
    search_space_version="1.8.1-v2r6",
    family_id=FAMILY_ID,
    family_version="1.8.1",
    provider_id=PROVIDER_ID,
    candidate_prefix="sq1",
    scientific_object="chronological_next_item_recommendation_under_frozen_full_ranking_and_history_protocol",
    supported_profile_kinds=("CHRONOLOGICAL_NEXT_ITEM_FULL_RANKING",),
    allowed_data_roles={
        "USER_ID": "core/user_id",
        "ITEM_ID": "core/item_id",
        "TRAIN_PREFIX_ITEM_SEQUENCE": SEQ,
        "TRAIN_PREFIX_TIMESTAMP_SEQUENCE": TIME,
        "QUERY_TIMESTAMP": QUERY_TIME,
        "TRAIN_NEXT_ITEM_TARGETS": "sequential/next_item_targets",
        "TRAIN_MASKED_ITEM_TARGETS": "sequential/masked_item_targets",
        "TRAIN_SEQUENCE_STATISTICS": "sequential/train_statistics",
    },
    forbidden_data_roles=("FUTURE_SUFFIX_AS_INPUT", "VALIDATION_TARGETS_AS_MODEL_INPUT", "TEST_TARGETS", "RANDOM_ORDER_INTERACTIONS", "NETWORK"),
    frozen_protocol_fields=("dataset", "dataset_snapshot", "timestamp_order", "ordering_time_field", "interval_time_field", "interval_time_unit", "timestamp_interval_construction", "chronological_split", "history_construction", "maximum_history_length", "evaluation_mode", "candidate_universe", "exclude_seen_policy", "metric_semantics", "evaluator", "seed_policy", "token_update_budget"),
    output_type=SCORE,
    output_slots=("PREDICTION_HEAD",),
    required_slots=("ITEM_REPRESENTATION", "SEQUENCE_BACKBONE", "PREDICTION_HEAD"),
    type_compatibility=((REP, STATE),),
    axes=AXES,
    operators=standard_operators(structure_capability="SEQUENCE_BACKBONE_MUTATION", efficiency_capability="SEQUENCE_EFFICIENCY_REWRITE", training_capability="SEQUENTIAL_OBJECTIVE_MUTATION"),
    capability_families=("REPRESENTATION_MUTATION", "TEMPORAL_ENCODING_MUTATION", "TRAIN_INPUT_VIEW_MUTATION", "SEQUENCE_REGISTER_GEOMETRY_MUTATION", "SEQUENCE_BACKBONE_MUTATION", "STATE_UPDATE_MUTATION", "INTEREST_ROUTING_MUTATION", "PREDICTION_MUTATION", "SEQUENTIAL_OBJECTIVE_MUTATION", "SEQUENCE_EFFICIENCY_REWRITE", "GENERATIVE_SEQUENCE_TRAINING", "BIDIRECTIONAL_PREFIX_MIXING", "CUSTOM_MODEL_IMPLEMENTATION"),
    qualification_contract={
        "api_probe": "prefix_batch_to_full_item_scores",
        "behavior_probes": ["future_suffix_perturbation_invariance", "strict_prefix_causality", "timestamp_order_preserved", "ordering_field_isolated_from_interval_value_field", "raw_interval_values_do_not_change_split_identity", "adjacent_and_query_gap_construction_when_declared", "timestamp_unit_and_monotonicity_audit_when_declared", "continuous_time_step_uses_intervals_when_declared", "s5_width_matches_representation_when_declared", "s5_branch_ablation_when_present", "relation_selective_s6_branch_ablation_when_present", "serial_s5_s6_residual_topology_when_declared", "s5_real_spectrum_matches_declared_stability_policy", "zoh_real_exponent_and_transition_finiteness", "matched_raw_gap_unconstrained_control_for_stability_candidate", "per_batch_finite_forward_loss_gradients_parameters", "nonfinite_boundary_reproduces_from_last_finite_checkpoint_when_observed", "time_interval_and_dt_activation_range_when_declared", "history_length_exact", "train_history_mask_absent_at_evaluation_when_declared", "learned_mask_token_receives_gradient_when_declared", "history_mask_ablation_when_present", "segment_register_is_non_embedded_when_declared", "segment_register_relabeling_invariance_when_declared", "cross_sequence_state_isolation_when_register_present", "packed_vs_padded_prefix_equivalence_when_declared", "bidirectional_ssd_uses_prefix_only_when_declared", "reverse_output_alignment_matches_declaration", "direction_contribution_ablation_when_present", "post_ssd_ffn_ablation_when_present", "low_dimension_ssd_stability_when_declared", "partial_flip_uses_prefix_only_when_declared", "retained_recent_suffix_order_when_declared", "direction_gate_balance_when_present", "short_recurrent_path_ablation_when_present", "evaluator_owned_seen_mask"],
        "resource_probes": ["tokens_per_second", "peak_device_memory", "effective_context_length", "parameters", "predicted_targets_per_update", "continuous_s5_parameters", "selective_s6_parameters", "branch_gradient_activity", "activation_norm_trajectory", "gradient_norm_trajectory", "parameter_finiteness", "maximum_raw_time_gap", "maximum_zoh_real_exponent", "nonfinite_transition_count", "real_to_padded_token_ratio", "sequence_register_bytes"],
        "metric_episode": "chronological_dev_ndcg_at_10_validation_only",
        "hidden_fallback": "FORBIDDEN",
        "numerical_stability_gate": {
            "per_batch_forward_finite": True,
            "per_batch_loss_finite": True,
            "all_gradients_finite": True,
            "all_parameters_finite": True,
            "spectrum_stability_policy_declared": True,
            "interval_transform_declared": True,
            "zoh_transition_finite": True,
            "silent_eigenvalue_clipping_or_timestamp_rescale_forbidden": True,
            "matched_raw_gap_unconstrained_control_for_stability_candidate": True,
            "resume_reproduction_required_after_nonfinite": True,
            "metric_rankable_before_gate": False,
            "test_read_before_gate": False,
        },
        "reference_profile": {
            "protocol": "native_recbole_to_ls_valid_only",
            "maximum_history_length": 50,
            "evaluation_mode": "FULL_RANKING",
            "repeatable": True,
            "exclude_seen": False,
            "chronological_split": "PER_USER_LAST_VALID_NO_TEST",
            "ordering_time_field": "chrono_order",
            "interval_time_field": "raw_timestamp",
            "interval_time_unit": "UNIX_SECONDS",
            "timestamp_interval_construction": "ADJACENT_PREFIX_GAPS_PLUS_QUERY_GAP",
            "selection_metric": "NDCG_AT_10",
            "budget_freeze_status": "FINAL_CAP_200_VALIDATION_ONLY_BOUNDARY_AUDIT",
            "budget_revision_policy": "FROZEN_NO_FURTHER_CAP_CHANGE",
            "epochs_cap": 200,
            "eval_step": 1,
            "stopping_step": 20,
            "early_stopping_rule": "RECBole_EXACT_IMPROVEMENT",
            "checkpoint_selection": "BEST_VALIDATION_NDCG_AT_10",
            "test_read_policy": "NO_HELDOUT_READ_DURING_SEARCH",
        },
    },
    episode_contract={
        "required_outcomes": ["ndcg_at_10", "recall_at_10", "length_bucket_ndcg", "tail_item_recall", "tokens_per_second", "peak_memory"],
        "failure_classes": ["MECHANISM_REFUTED", "OPTIMIZATION_UNDERFIT", "NUMERICAL_INSTABILITY", "LONG_CONTEXT_DEGRADATION", "RESOURCE_INFEASIBLE", "CHRONOLOGY_OR_EVALUATOR_INVALID"],
        "outcome_classes": ["SUPPORTED", "MECHANISM_REFUTED", "OPTIMIZATION_UNDERFIT", "NUMERICAL_INSTABILITY", "LONG_CONTEXT_DEGRADATION", "RESOURCE_INFEASIBLE", "CHRONOLOGY_OR_EVALUATOR_INVALID", "INCONCLUSIVE"],
        "outcome_memory_lanes": {
            "SUPPORTED": "MECHANISM_POSITIVE",
            "MECHANISM_REFUTED": "MECHANISM_NEGATIVE",
            "OPTIMIZATION_UNDERFIT": "OPTIMIZATION_DIAGNOSTIC",
            "NUMERICAL_INSTABILITY": "OPTIMIZATION_DIAGNOSTIC",
            "LONG_CONTEXT_DEGRADATION": "MECHANISM_NEGATIVE",
            "RESOURCE_INFEASIBLE": "RESOURCE_DIAGNOSTIC",
            "CHRONOLOGY_OR_EVALUATOR_INVALID": "PROTOCOL_DIAGNOSTIC",
            "INCONCLUSIVE": "INCONCLUSIVE",
        },
        "negative_update_keys": ["backbone_family", "continuous_time_discretization", "time_interval_signal", "timestamp_field_semantics", "continuous_time_spectrum_stability", "zoh_transition_stability", "s5_s6_composition", "relation_selective_transition", "directional_context_mixing", "direction_gate_collapse", "bidirectional_ssd_fusion", "bidirectional_ssd_alignment", "post_ssd_ffn_effect", "train_input_view", "segmented_packing_geometry", "cross_sequence_state_isolation", "low_dimension_ssd_stability", "state_retention", "temporal_encoding", "short_long_routing", "short_recurrent_path_dominance", "compression_loss", "scale_regime", "objective_mismatch"],
        "primitive_update_keys": {},
        "slot_update_keys": {
            "ITEM_REPRESENTATION": ["scale_regime"],
            "TEMPORAL_POSITION_ENCODING": ["temporal_encoding", "time_interval_signal", "timestamp_field_semantics"],
            "TRAIN_INPUT_VIEW": ["train_input_view", "objective_mismatch"],
            "SEQUENCE_REGISTER_GEOMETRY": ["segmented_packing_geometry", "cross_sequence_state_isolation", "scale_regime"],
            "SEQUENCE_BACKBONE": ["backbone_family", "continuous_time_discretization", "time_interval_signal", "timestamp_field_semantics", "s5_s6_composition", "relation_selective_transition", "directional_context_mixing", "direction_gate_collapse", "bidirectional_ssd_fusion", "bidirectional_ssd_alignment", "post_ssd_ffn_effect", "cross_sequence_state_isolation", "low_dimension_ssd_stability", "scale_regime"],
            "STATE_UPDATE_GATING": ["state_retention"],
            "INTEREST_ROUTING": ["short_long_routing", "short_recurrent_path_dominance"],
            "PREDICTION_HEAD": ["objective_mismatch"],
            "SEQUENTIAL_OBJECTIVE": ["objective_mismatch"],
        },
    },
)


class SequentialScalingProviderV1(DeclarativeMechanismSpaceProvider):
    def __init__(self) -> None:
        super().__init__(SPEC)


PROVIDER = SequentialScalingProviderV1()

__all__ = ["PROVIDER", "SPACE_ID", "FAMILY_ID", "SequentialScalingProviderV1"]
