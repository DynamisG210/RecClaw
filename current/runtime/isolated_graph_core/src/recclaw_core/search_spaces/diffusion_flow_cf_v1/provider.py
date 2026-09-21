"""Diffusion-Flow collaborative-filtering domain language for V2R4."""

from __future__ import annotations

from recclaw_core.mechanism_space.declarative_provider import (
    DeclarativeFamilySpec,
    DeclarativeMechanismSpaceProvider,
    axis,
    object_parameters,
    primitive,
    standard_operators,
)

SPACE_ID = "DIFFUSION_FLOW_CF_MECHANISM_SPACE_V1"
FAMILY_ID = "DIFFUSION_FLOW_CF_V1"
PROVIDER_ID = "recclaw.search-space-provider.diffusion-flow-cf.v1"

SIGNAL = "diffusion_flow/interaction_signal"
STATE = "diffusion_flow/generative_state"
TIME = "diffusion_flow/time_state"
CONDITION = "diffusion_flow/condition"
FIELD = "diffusion_flow/vector_field"
RECOVERED = "diffusion_flow/recovered_signal"
SCORE = "core/user_item_relevance_score"


def _p(properties: dict, *required: str) -> dict:
    return object_parameters(properties, required=required)


AXES = (
    axis(
        "STATE_REPRESENTATION",
        research_role="Choose the space in which user interaction signal is corrupted, transported, or denoised.",
        interaction_guidance="Latent compression is a core/support mechanism with its own reconstruction probe; it cannot silently change candidate coverage.",
        allow_multiple=False,
        primitives=(
            primitive("state.full_interaction_vector", "STATE_REPRESENTATION", inputs=(("history", (SIGNAL,), 1),), outputs=(("state", STATE),), parameters=_p({"value_encoding": {"enum": ["BINARY", "CONFIDENCE", "CENTERED"]}}, "value_encoding"), capabilities=("STATE_REPRESENTATION_MUTATION",), causal_effect="Run the generative process directly over the full item interaction signal.", failure_signal="High-dimensional noise overwhelms sparse personal history.", resource_effect="State width equals frozen item universe."),
            primitive("state.learned_latent_autoencoder", "STATE_REPRESENTATION", inputs=(("history", (SIGNAL,), 1),), outputs=(("state", STATE),), parameters=_p({"latent_dimension": {"type": "integer", "minimum": 8, "maximum": 4096}, "layers": {"type": "integer", "minimum": 1, "maximum": 16}, "reconstruction_weight": {"type": "number", "minimum": 0.0}}, "latent_dimension", "layers", "reconstruction_weight"), capabilities=("STATE_REPRESENTATION_MUTATION", "LATENT_PROCESS_MUTATION"), causal_effect="Encode history into latent_dimension coordinates; perform iterative generation in latent space, then decode back to all items. Train jointly with reconstruction_weight times mean squared clean-history reconstruction error, added once outside diffusion timestep weighting.", failure_signal="Autoencoder reconstruction, not diffusion/flow, sets the performance ceiling.", resource_effect="Separate encoder/decoder parameters and latent-stage training."),
            primitive("state.spectral_graph_coordinates", "STATE_REPRESENTATION", inputs=(("history", (SIGNAL,), 1), ("basis", ("diffusion_flow/spectral_basis",), 1)), outputs=(("state", STATE),), parameters=_p({"rank": {"type": "integer", "minimum": 4, "maximum": 4096}, "spectral_index_ranges": {"type": "array", "minItems": 1, "items": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0}}}}, "rank", "spectral_index_ranges"), capabilities=("STATE_REPRESENTATION_MUTATION", "GRAPH_CONDITIONING_MUTATION"), causal_effect="Project into explicit zero-based half-open index ranges of the descending positive binary-train Gram spectrum; ascending disjoint ranges select exactly rank columns; adjacent intervals must be merged in the draft, with [0,rank) selecting the leading subspace.", failure_signal="Selected spectral coordinates remove personalized evidence.", resource_effect="One top-K spectral precompute, K=max range end, then rank selected columns; deeper selections cost more even at fixed rank."),
            primitive("state.multi_hop_graph_features", "STATE_REPRESENTATION", inputs=(("history", (SIGNAL,), 1), ("graph", ("diffusion_flow/item_graph",), 1)), outputs=(("state", STATE),), parameters=_p({"hops": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "integer", "minimum": 1, "maximum": 8}}, "normalization": {"enum": ["SYMMETRIC", "RANDOM_WALK"]}}, "hops", "normalization"), capabilities=("STATE_REPRESENTATION_MUTATION", "GRAPH_CONDITIONING_MUTATION"), causal_effect="Expose explicit multi-hop collaborative context as state channels.", failure_signal="Higher-hop signal oversmooths or leaks popularity while adding little personal information.", resource_effect="Sparse graph propagation for declared hop set."),
        ),
    ),
    axis(
        "FORWARD_PATH",
        research_role="Preserve the compiler-owned Gaussian-VP path paired with DiffRec's clean-state objective and posterior solver.",
        interaction_guidance="This slot is a frozen parent mechanic, not an ordinary search axis: keep the exact Gaussian-VP primitive and parameters and do not declare FORWARD_PATH changed. A different corruption or prior-to-data path is legal only in an explicit CUSTOM_MODEL that co-designs and owns the matching target, inverse, and recovery solver.",
        allow_multiple=False,
        primitives=(
            primitive("forward.gaussian_variance_preserving", "FORWARD_PATH", inputs=(("state", (STATE,), 1),), outputs=(("state", STATE), ("time", TIME)), parameters=_p({"steps": {"const": 5}, "noise_scale": {"const": 0.001}, "beta_fixed": {"const": True}, "fixed_first_beta": {"const": 0.00001}}, "steps", "noise_scale", "beta_fixed", "fixed_first_beta"), capabilities=(), causal_effect="Apply the exact frozen DiffRec variance-preserving Gaussian path paired with its clean-state target and posterior solver.", failure_signal="Any independent path or parameter change invalidates the parent objective/solver pairing and requires a full matched mathematical rewrite.", resource_effect="One sampled corruption timestep and dense full-state noise per training update."),
        ),
    ),
    axis(
        "TIME_SCHEDULE",
        research_role="Preserve the linear schedule used by the frozen Gaussian-VP parent path.",
        interaction_guidance="This exact schedule is compiler-owned in ordinary parent-relative candidates. A different time process belongs to a full matched CUSTOM_MODEL trajectory rewrite.",
        allow_multiple=False,
        primitives=(
            primitive("schedule.linear", "TIME_SCHEDULE", inputs=(("time", (TIME,), 1),), outputs=(("time", TIME),), parameters=_p({"start": {"const": 0.0005}, "end": {"const": 0.005}}, "start", "end"), capabilities=(), causal_effect="Apply the exact frozen DiffRec linear VP schedule.", failure_signal="A schedule change would invalidate the frozen forward/objective/solver bundle.", resource_effect="No extra model cost."),
        ),
    ),
    axis(
        "CONDITIONING",
        research_role="Provide train-derived collaborative structure to the generative field without replacing personal state.",
        interaction_guidance="Conditioning is ablated by removing only the declared condition path; log condition/state dominance.",
        allow_multiple=True,
        primitives=(
            primitive("condition.user_history_skip", "CONDITIONING", inputs=(("history", (SIGNAL,), 1),), outputs=(("condition", CONDITION),), parameters=_p({"projection": {"enum": ["IDENTITY", "LINEAR", "MLP"]}}, "projection"), capabilities=("CONDITIONING_MUTATION", "PERSONALIZATION_MUTATION"), causal_effect="Expose an uncorrupted personal-history skip condition.", failure_signal="Model copies seen history or ignores the generative state.", resource_effect="One condition projection."),
            primitive("condition.masked_spectral_history_classifier_free", "CONDITIONING", inputs=(("history", (SIGNAL,), 1), ("basis", ("diffusion_flow/spectral_basis",), 1)), outputs=(("condition", CONDITION),), parameters=_p({"rank": {"type": "integer", "minimum": 4, "maximum": 4096}, "history_mask_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0}, "unconditional_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0}}, "rank", "history_mask_probability", "unconditional_probability"), capabilities=("CONDITIONING_MUTATION", "PERSONALIZATION_MUTATION", "GRAPH_CONDITIONING_MUTATION"), causal_effect="Project a randomly masked train-history condition into the graph spectrum and retain an explicit unconditional training path.", failure_signal="Condition dropout removes too much personal signal or the conditional branch dominates the unconditional control.", resource_effect="One train-only history mask, spectral projection, and conditional/unconditional branch accounting."),
            primitive("condition.multi_hop_graph", "CONDITIONING", inputs=(("history", (SIGNAL,), 1), ("graph", ("diffusion_flow/item_graph",), 1)), outputs=(("condition", CONDITION),), parameters=_p({"hops": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "integer", "minimum": 1, "maximum": 8}}, "fusion": {"enum": ["CONCAT", "GATED", "CROSS_ATTENTION"]}}, "hops", "fusion"), capabilities=("CONDITIONING_MUTATION", "GRAPH_CONDITIONING_MUTATION"), causal_effect="Condition recovery on explicit two/three-or-more-hop collaborative neighborhoods.", failure_signal="Graph condition oversmooths or carries the entire prediction independent of denoising.", resource_effect="Declared sparse propagation and condition fusion."),
            primitive("condition.spectral_band", "CONDITIONING", inputs=(("history", (SIGNAL,), 1), ("basis", ("diffusion_flow/spectral_basis",), 1)), outputs=(("condition", CONDITION),), parameters=_p({"bands": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"enum": ["LOW", "MID", "HIGH"]}}, "rank": {"type": "integer", "minimum": 4, "maximum": 4096}}, "bands", "rank"), capabilities=("CONDITIONING_MUTATION", "GRAPH_CONDITIONING_MUTATION"), causal_effect="Expose selected graph-frequency components to the denoiser/field.", failure_signal="Spectral condition duplicates state representation or emphasizes noise.", resource_effect="Spectral precompute and low-rank condition."),
            primitive("condition.popularity_calibrated", "CONDITIONING", inputs=(("statistics", ("diffusion_flow/train_statistics",), 1),), outputs=(("condition", CONDITION),), parameters=_p({"power": {"type": "number", "minimum": -2.0, "maximum": 2.0}, "center": {"type": "boolean"}}, "power", "center"), capabilities=("CONDITIONING_MUTATION", "POPULARITY_RECOVERY_MUTATION"), causal_effect="Make train-only popularity bias explicit so recovery can subtract or calibrate it.", failure_signal="Condition reinforces rather than corrects head-item bias.", resource_effect="Small static condition vector."),
        ),
    ),
    axis(
        "GENERATIVE_DYNAMICS",
        research_role="Predict denoising target, score, or flow velocity that transports state toward recommendation signal.",
        interaction_guidance="This is normally the core mechanism. Keep the exact Gaussian-VP path, schedule, solver, and score. Use the clean-state objective for x0 fields; behavior-guided VELOCITY must be paired with objective.flow_matching_velocity.",
        allow_multiple=False,
        primitives=(
            primitive("dynamics.time_conditioned_mlp", "GENERATIVE_DYNAMICS", inputs=(("state", (STATE,), 1), ("time", (TIME,), 1), ("condition", (CONDITION,), 0)), outputs=(("field", FIELD),), parameters=_p({"layers": {"type": "integer", "minimum": 1, "maximum": 32}, "hidden_dimension": {"type": "integer", "minimum": 16, "maximum": 8192}, "time_embedding_dimension": {"type": "integer", "minimum": 4, "maximum": 1024}, "time_fusion": {"enum": ["CONCAT", "ADD"]}, "activation": {"enum": ["TANH", "RELU", "GELU", "SILU"]}, "dropout": {"type": "number", "minimum": 0.0, "maximum": 1.0}, "normalize_input": {"type": "boolean"}}, "layers", "hidden_dimension", "time_embedding_dimension", "time_fusion", "activation", "dropout", "normalize_input"), capabilities=("DYNAMICS_MUTATION",), causal_effect="Predict a time-conditioned dense denoising field over the interaction state with explicit time fusion, activation, dropout, and input normalization.", failure_signal="Dense network memorizes popularity or scales poorly with item count.", resource_effect="Dense state-width computation per sampled/inference step."),
            primitive("dynamics.film_conditional_spectral_mlp", "GENERATIVE_DYNAMICS", inputs=(("state", (STATE,), 1), ("time", (TIME,), 1), ("condition", (CONDITION,), 1)), outputs=(("field", FIELD),), parameters=_p({"film_hidden_dimension": {"type": "integer", "minimum": 1, "maximum": 1024}, "time_embedding_dimension": {"type": "integer", "minimum": 4, "maximum": 1024}, "hidden_dimension": {"type": "integer", "minimum": 16, "maximum": 8192}, "layers": {"type": "integer", "minimum": 1, "maximum": 32}, "elementwise_modulation": {"const": True}}, "film_hidden_dimension", "time_embedding_dimension", "hidden_dimension", "layers", "elementwise_modulation"), capabilities=("DYNAMICS_MUTATION", "GRAPH_CONDITIONING_MUTATION", "PERSONALIZATION_MUTATION"), causal_effect="Use elementwise feature-wise affine modulation to fuse spectral history condition into noisy graph-frequency state before time-conditioned denoising.", failure_signal="FiLM collapses to a condition-only shortcut or architecture-width assumptions dominate the anisotropic process.", resource_effect="Per-frequency affine modulation plus a declared spectral-state MLP."),
            primitive("dynamics.cross_attention_multihop_autoencoder", "GENERATIVE_DYNAMICS", inputs=(("state", (STATE,), 1), ("time", (TIME,), 1), ("condition", (CONDITION,), 1)), outputs=(("field", FIELD),), parameters=_p({"layers": {"type": "integer", "minimum": 1, "maximum": 32}, "heads": {"type": "integer", "minimum": 1, "maximum": 64}, "latent_dimension": {"type": "integer", "minimum": 8, "maximum": 4096}}, "layers", "heads", "latent_dimension"), capabilities=("DYNAMICS_MUTATION", "GRAPH_CONDITIONING_MUTATION"), causal_effect="Cross-attend noisy personal state to explicit multi-hop collaborative features during recovery.", failure_signal="Condition dominates personal state or cross-attention adds capacity without hop-specific value.", resource_effect="Latent autoencoding plus cross-attention per step."),
            primitive("dynamics.graph_heat_reverse", "GENERATIVE_DYNAMICS", inputs=(("state", (STATE,), 1), ("time", (TIME,), 1), ("graph", ("diffusion_flow/item_graph",), 1), ("condition", (CONDITION,), 0)), outputs=(("field", FIELD),), parameters=_p({"sharpen_strength": {"type": "number", "minimum": 0.0, "maximum": 8.0}, "stability_clip": {"type": "number", "exclusiveMinimum": 0.0}}, "sharpen_strength", "stability_clip"), capabilities=("DYNAMICS_MUTATION", "GRAPH_CONDITIONING_MUTATION"), causal_effect="Reverse deterministic graph blur with controlled sharpening conditioned on history.", failure_signal="Reverse heat dynamics amplify noise or are equivalent to a fixed graph filter.", resource_effect="Sparse graph-vector operations per reverse step."),
            primitive("dynamics.behavior_guided_velocity", "GENERATIVE_DYNAMICS", inputs=(("state", (STATE,), 1), ("time", (TIME,), 1), ("condition", (CONDITION,), 0)), outputs=(("field", FIELD),), parameters=_p({"layers": {"type": "integer", "minimum": 1, "maximum": 32}, "field_parameterization": {"const": "VELOCITY"}}, "layers", "field_parameterization"), capabilities=("DYNAMICS_MUTATION", "FLOW_FIELD_MUTATION"), causal_effect="Learn the velocity parameterization exactly paired with the frozen Gaussian-VP path and compiler-owned VP velocity target/inverse.", failure_signal="The velocity field underfits or fails to improve the parent despite the matched VP parameterization.", resource_effect="One learned full-state VP velocity evaluation per parent recovery step."),
            primitive("dynamics.latent_score_network", "GENERATIVE_DYNAMICS", inputs=(("state", (STATE,), 1), ("time", (TIME,), 1), ("condition", (CONDITION,), 0)), outputs=(("field", FIELD),), parameters=_p({"residual_blocks": {"type": "integer", "minimum": 1, "maximum": 64}, "hidden_dimension": {"type": "integer", "minimum": 16, "maximum": 4096}}, "residual_blocks", "hidden_dimension"), capabilities=("DYNAMICS_MUTATION", "LATENT_PROCESS_MUTATION"), causal_effect="Estimate a score/noise field in a compressed user latent space.", failure_signal="Latent decoder bottleneck or prior mismatch masks the diffusion mechanism.", resource_effect="Lower-dimensional iterative field plus final decoder."),
        ),
    ),
    axis(
        "PERSONAL_INFORMATION_PRESERVATION",
        research_role="Reserve explicit preservation mechanics for full trajectory rewrites.",
        interaction_guidance="No standalone preservation hook is compiler-supported in ordinary parent-relative candidates. Express train-derived personalization through a typed CONDITIONING plus changed field, or use a full matched CUSTOM_MODEL.",
        allow_multiple=False,
        primitives=(),
    ),
    axis(
        "DENOISING_FLOW_OBJECTIVE",
        research_role="Select one compiler-supported target that is mathematically paired with the frozen Gaussian-VP path.",
        interaction_guidance="Keep the exact parent clean-state objective for x0 fields. objective.flow_matching_velocity is legal only with dynamics.behavior_guided_velocity; target and inverse are compiler-owned as a pair.",
        allow_multiple=False,
        primitives=(
            primitive("objective.predict_clean_state", "DENOISING_FLOW_OBJECTIVE", inputs=(("field", (FIELD,), 1), ("clean", (STATE,), 1), ("time", (TIME,), 1)), outputs=(("objective", "diffusion_flow/objective"),), parameters=_p({"loss": {"const": "WEIGHTED_MSE"}, "time_weighting": {"const": "SNR"}, "timestep_sampling": {"const": "LOSS_SECOND_MOMENT_AFTER_WARMUP"}, "history_num_per_term": {"const": 10}, "uniform_mixture_probability": {"const": 0.001}}, "loss", "time_weighting", "timestep_sampling", "history_num_per_term", "uniform_mixture_probability"), capabilities=(), causal_effect="Predict the clean interaction state under the exact frozen parent x0 objective, sampler, and SNR weighting.", failure_signal="A changed target or weighting is not paired with the frozen parent recovery math.", resource_effect="One sampled-time clean-state target plus bounded parent loss-history statistics."),
            primitive("objective.flow_matching_velocity", "DENOISING_FLOW_OBJECTIVE", inputs=(("field", (FIELD,), 1), ("time", (TIME,), 1)), outputs=(("objective", "diffusion_flow/objective"),), parameters=_p({"weighting": {"enum": ["UNIFORM", "ENDPOINT_HEAVY", "PATH_SPEED"]}, "metric": {"enum": ["L2", "HUBER"]}}, "weighting", "metric"), capabilities=("OBJECTIVE_MUTATION", "FLOW_FIELD_MUTATION"), causal_effect="Match the exact compiler-owned VP velocity target paired with dynamics.behavior_guided_velocity and its inverse to x0.", failure_signal="The paired velocity objective underfits or its ranking recovery does not improve the parent.", resource_effect="One VP velocity target per update."),
        ),
    ),
    axis(
        "PRIOR",
        research_role="Reserve an explicit prior slot for full mathematical rewrites; the parent-x0 profile has no independent prior input.",
        interaction_guidance="No ordinary parent-relative prior primitive is compiler-supported. Introduce a prior only in an explicit CUSTOM_MODEL that also owns a compatible forward path, training target, inverse, and recovery solver.",
        allow_multiple=False,
        primitives=(),
    ),
    axis(
        "RECOVERY_SOLVER",
        research_role="Preserve DiffRec's deterministic posterior solver paired with the frozen Gaussian-VP path.",
        interaction_guidance="This exact solver and start state are compiler-owned for ordinary candidates. A different solver requires a full CUSTOM_MODEL with matched forward and objective math.",
        allow_multiple=False,
        primitives=(
            primitive("solver.deterministic_reduced_step", "RECOVERY_SOLVER", inputs=(("field", (FIELD,), 1), ("state", (STATE,), 1), ("time", (TIME,), 1)), outputs=(("recovered", RECOVERED),), parameters=_p({"steps": {"const": 5}, "spacing": {"const": "UNIFORM"}, "sampling_steps": {"const": 0}, "sampling_noise": {"const": False}, "start_state": {"const": "OBSERVED_INTERACTION_STATE"}}, "steps", "spacing", "sampling_steps", "sampling_noise", "start_state"), capabilities=(), causal_effect="Apply the exact frozen DiffRec deterministic Gaussian posterior transitions from the observed interaction state.", failure_signal="Any independent solver change invalidates the parent path/objective pairing.", resource_effect="Exactly five dense field evaluations per recovery."),
        ),
    ),
    axis(
        "GUIDANCE_CALIBRATION",
        research_role="Reserve post-recovery guidance for full mathematical rewrites.",
        interaction_guidance="No standalone guidance hook is compiler-supported in ordinary parent-relative candidates. Put train-derived information into a typed condition on a changed field, or use a full matched CUSTOM_MODEL.",
        allow_multiple=False,
        primitives=(),
    ),
    axis(
        "SCORE_HEAD",
        research_role="Preserve the recovered-state logits used by the frozen DiffRec parent.",
        interaction_guidance="The exact unit-temperature score head is compiler-owned in ordinary candidates; seen masking and candidate filtering remain evaluator-owned.",
        allow_multiple=False,
        primitives=(
            primitive("score.recovered_logits", "SCORE_HEAD", inputs=(("recovered", (RECOVERED,), 1),), outputs=(("score", SCORE),), parameters=_p({"temperature": {"const": 1.0}}, "temperature"), capabilities=(), causal_effect="Use the recovered item signal directly as full-ranking logits at the exact parent temperature.", failure_signal="A score-head change could hide invalid recovery rather than improve the generative mechanism.", resource_effect="No additional model cost."),
        ),
    ),
)

SPEC = DeclarativeFamilySpec(
    search_space_id=SPACE_ID,
    search_space_version="1.3.0-v2r4",
    family_id=FAMILY_ID,
    family_version="1.3.0",
    provider_id=PROVIDER_ID,
    candidate_prefix="df1",
    scientific_object="static_implicit_collaborative_ranking_by_diffusion_graph_process_or_flow_based_signal_recovery",
    supported_profile_kinds=("OFFLINE_TOPN_DIFFUSION_FLOW",),
    allowed_data_roles={
        "USER_ID": "core/user_id",
        "ITEM_ID": "core/item_id",
        "TRAIN_INTERACTIONS": "diffusion_flow/train_interactions",
        "TRAIN_USER_INTERACTION_SIGNAL": SIGNAL,
        "TRAIN_ITEM_GRAPH": "diffusion_flow/item_graph",
        "TRAIN_SPECTRAL_BASIS": "diffusion_flow/spectral_basis",
        "TRAIN_STATISTICS": "diffusion_flow/train_statistics",
    },
    forbidden_data_roles=("VALIDATION_LABELS_AS_CONDITION", "TEST_STATISTICS", "TEST_INTERACTIONS", "NETWORK", "EXTERNAL_USER_FEATURES"),
    frozen_protocol_fields=("dataset", "dataset_snapshot", "item_id_mapping_identity", "split", "preprocessing", "evaluation_mode", "candidate_universe", "seen_repeat_policy", "metric_semantics", "evaluator", "seed_policy", "training_update_budget", "inference_function_evaluation_budget"),
    output_type=SCORE,
    output_slots=("SCORE_HEAD",),
    required_slots=("STATE_REPRESENTATION", "FORWARD_PATH", "GENERATIVE_DYNAMICS", "RECOVERY_SOLVER", "SCORE_HEAD"),
    type_compatibility=((STATE, RECOVERED),),
    axes=AXES,
    operators=standard_operators(structure_capability="DYNAMICS_MUTATION", efficiency_capability="SAMPLING_EFFICIENCY_REWRITE", training_capability="OBJECTIVE_MUTATION"),
    capability_families=("STATE_REPRESENTATION_MUTATION", "LATENT_PROCESS_MUTATION", "FORWARD_PROCESS_MUTATION", "SCHEDULE_MUTATION", "CONDITIONING_MUTATION", "GRAPH_CONDITIONING_MUTATION", "DYNAMICS_MUTATION", "FLOW_FIELD_MUTATION", "PERSONALIZATION_MUTATION", "POPULARITY_RECOVERY_MUTATION", "OBJECTIVE_MUTATION", "PRIOR_MUTATION", "SOLVER_MUTATION", "SAMPLING_EFFICIENCY_REWRITE", "GUIDANCE_MUTATION", "SCORE_MUTATION", "CUSTOM_MODEL_IMPLEMENTATION"),
    qualification_contract={
        "api_probe": "user_interaction_signal_to_full_item_scores",
        "behavior_probes": ["train_only_condition_derivation", "train_only_spectral_basis_derivation", "frequency_noise_bounds", "state_condition_ablation", "masked_history_rate", "classifier_free_path_rate", "film_condition_usage", "paper_spec_assumption_sensitivity", "prior_only_control", "deterministic_seeded_inference", "evaluator_owned_seen_mask", "finite_recovery_trajectory"],
        "resource_probes": ["training_updates", "inference_field_evaluations", "spectral_rank", "lanczos_precompute_cost", "per_user_latency", "peak_device_memory", "precompute_cost", "trajectory_norms"],
        "metric_episode": "exact_general_cf_dev_validation_only",
        "hidden_fallback": "FORBIDDEN",
    },
    episode_contract={
        "required_outcomes": ["ndcg_at_10", "recall_at_10", "coverage", "tail_recall", "field_evaluations", "latency", "peak_memory", "trajectory_stability"],
        "failure_classes": ["MECHANISM_REFUTED", "DENOISER_OR_FIELD_UNDERFIT", "TRAJECTORY_INSTABILITY", "PRIOR_OR_CONDITION_DOMINANCE", "RESOURCE_INFEASIBLE", "PROTOCOL_OR_IMPLEMENTATION_INVALID"],
        "outcome_classes": ["SUPPORTED", "MECHANISM_REFUTED", "DENOISER_OR_FIELD_UNDERFIT", "TRAJECTORY_INSTABILITY", "PRIOR_OR_CONDITION_DOMINANCE", "RESOURCE_INFEASIBLE", "PROTOCOL_OR_IMPLEMENTATION_INVALID", "INCONCLUSIVE"],
        "outcome_memory_lanes": {
            "SUPPORTED": "MECHANISM_POSITIVE",
            "MECHANISM_REFUTED": "MECHANISM_NEGATIVE",
            "DENOISER_OR_FIELD_UNDERFIT": "OPTIMIZATION_DIAGNOSTIC",
            "TRAJECTORY_INSTABILITY": "OPTIMIZATION_DIAGNOSTIC",
            "PRIOR_OR_CONDITION_DOMINANCE": "MECHANISM_NEGATIVE",
            "RESOURCE_INFEASIBLE": "RESOURCE_DIAGNOSTIC",
            "PROTOCOL_OR_IMPLEMENTATION_INVALID": "PROTOCOL_DIAGNOSTIC",
            "INCONCLUSIVE": "INCONCLUSIVE",
        },
        "negative_update_keys": ["forward_path", "spectral_anisotropy", "spectral_noise_bounds", "personal_signal_retention", "time_weighting", "condition_dominance", "conditioning_dropout", "film_modulation", "graph_hops", "latent_bottleneck", "prior_strength", "solver_steps", "guidance_scale"],
        "slot_update_keys": {
            "STATE_REPRESENTATION": ["latent_bottleneck"],
            "FORWARD_PATH": ["forward_path", "spectral_anisotropy", "spectral_noise_bounds", "personal_signal_retention"],
            "TIME_SCHEDULE": ["time_weighting"],
            "CONDITIONING": ["condition_dominance", "conditioning_dropout", "graph_hops"],
            "GENERATIVE_DYNAMICS": ["forward_path", "spectral_anisotropy", "spectral_noise_bounds", "time_weighting", "film_modulation"],
            "PERSONAL_INFORMATION_PRESERVATION": ["personal_signal_retention"],
            "DENOISING_FLOW_OBJECTIVE": ["time_weighting"],
            "PRIOR": ["prior_strength"],
            "RECOVERY_SOLVER": ["solver_steps"],
            "GUIDANCE_CALIBRATION": ["guidance_scale", "condition_dominance"],
            "SCORE_HEAD": ["guidance_scale"],
        },
    },
)


class DiffusionFlowCFProviderV1(DeclarativeMechanismSpaceProvider):
    def __init__(self) -> None:
        super().__init__(SPEC)


PROVIDER = DiffusionFlowCFProviderV1()

__all__ = ["PROVIDER", "SPACE_ID", "FAMILY_ID", "DiffusionFlowCFProviderV1"]
