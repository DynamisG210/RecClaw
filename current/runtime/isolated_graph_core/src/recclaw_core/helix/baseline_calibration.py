"""Pre-outcome development-baseline calibration for Helix V31.

The baseline is not copied from the V29 online-test campaign.  It is the exact
strict BL-ICF LightGCN root program, rebound to the V31 development profile and
measured by the same validation-only worker/runtime used by both formal arms.
"""

from __future__ import annotations

import math
from statistics import fmean, stdev
from typing import Any, Mapping, Sequence

from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    campaign_development_validation_profile_ref,
    executable_mechanism,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.compilation_cache import (
    compile_campaign_program,
)
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    DEVELOPMENT_EVALUATOR,
    DEVELOPMENT_SPLIT,
    validate_execution_recipe,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (
    CAMPAIGN_DEVELOPMENT_TRAINING_RELEASE_RESOURCE,
    campaign_training_runtime_release,
)
from recclaw_core.mechanism_space.canonical import deep_thaw


BASELINE_PLAN_SCHEMA_V31 = "recclaw.helix.development-baseline-plan.v31"
BASELINE_RECEIPT_SCHEMA_V31 = "recclaw.helix.development-baseline-receipt.v31"
BASELINE_SEEDS_V31 = (55098, 55099, 55100, 55101, 55102)
_T_CRITICAL_95_DF4 = 2.7764451051977987


class BaselineCalibrationError(ValueError):
    pass


def development_lightgcn_recipe_v31() -> dict[str, Any]:
    """Return the exact validation-only recipe for the strict LightGCN root."""

    mechanism = executable_mechanism("LIGHTGCN")
    program = deep_thaw(mechanism.mechanism_program)
    profile_ref = campaign_development_validation_profile_ref()
    program["profile_ref"] = profile_ref
    report = compile_campaign_program(program)
    if not report.is_valid:
        raise BaselineCalibrationError("V31 LightGCN root program does not compile")
    comparator_ref = "recclaw.strict-bl-icf-lightgcn-root.v31"
    comparator_digest = sha256_digest(
        {
            "comparator_ref": comparator_ref,
            "entrypoint": mechanism.entrypoint,
            "entrypoint_source_sha256": mechanism.entrypoint_source_sha256,
            "mechanism_program_digest": report.mechanism_program_digest,
            "mechanism_semantics_digest": report.mechanism_semantics_digest,
            "profile_ref": profile_ref,
        }
    )

    recipe = mechanism.execution_recipe()
    recipe.update(
        {
            "capability_digest": comparator_digest,
            "capability_family": "LIGHTGCN",
            "capability_ref": comparator_ref,
            "candidate_id": str(report.candidate_id),
            "comparator_digest": comparator_digest,
            "comparator_ref": comparator_ref,
            "dataset": "ml-1m",
            "evaluator": DEVELOPMENT_EVALUATOR,
            "execution_role": "COMPARATOR",
            "mechanism_program_digest": str(report.mechanism_program_digest),
            "mechanism_semantics_digest": str(report.mechanism_semantics_digest),
            "profile_digest": profile_ref["profile_digest"],
            "profile_ref": profile_ref["profile_id"],
            "split": DEVELOPMENT_SPLIT,
        }
    )
    validate_execution_recipe(recipe)
    return canonical_value(recipe)


def development_baseline_plan_v31() -> dict[str, Any]:
    recipe = development_lightgcn_recipe_v31()
    release = campaign_training_runtime_release(
        CAMPAIGN_DEVELOPMENT_TRAINING_RELEASE_RESOURCE
    )
    payload = {
        "schema": BASELINE_PLAN_SCHEMA_V31,
        "status": "PRECOMMITTED_AWAITING_CALIBRATION",
        "comparator": {
            "comparator_id": "STRICT_BL_ICF_LIGHTGCN_ROOT_V31",
            "entrypoint": recipe["entrypoint"],
            "candidate_id": recipe["candidate_id"],
            "mechanism_program_digest": recipe["mechanism_program_digest"],
            "mechanism_semantics_digest": recipe["mechanism_semantics_digest"],
            "execution_recipe": recipe,
            "execution_recipe_digest": sha256_digest(recipe),
        },
        "runtime": {
            "release_id": release.release_id,
            "release_digest": release.digest,
            "online_metric_source": "BEST_VALID_RESULT",
            "online_partition_role": "DEVELOPMENT_VALIDATION",
            "native_gpu_id": 2,
            "cuda_visible_devices": "MUST_BE_UNSET",
        },
        "calibration": {
            "metric": "NDCG@10",
            "epochs_requested": 100,
            "ordered_seed_panel": BASELINE_SEEDS_V31,
            "required_success_count": len(BASELINE_SEEDS_V31),
            "aggregation": "ARITHMETIC_MEAN",
            "uncertainty": "T_INTERVAL_95_PERCENT_DF4",
            "provider_calls": 0,
            "heldout_access": "FORBIDDEN",
        },
    }
    return {**canonical_value(payload), "plan_digest": sha256_digest(payload)}


def finalize_development_baseline_v31(
    observations: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    frozen_plan = canonical_value(
        dict(development_baseline_plan_v31() if plan is None else plan)
    )
    if frozen_plan.get("schema") != BASELINE_PLAN_SCHEMA_V31:
        raise BaselineCalibrationError("baseline plan schema mismatch")
    plan_without_digest = {
        key: value for key, value in frozen_plan.items() if key != "plan_digest"
    }
    if frozen_plan.get("plan_digest") != sha256_digest(plan_without_digest):
        raise BaselineCalibrationError("baseline plan digest mismatch")

    calibration = frozen_plan["calibration"]
    expected_seeds = tuple(calibration["ordered_seed_panel"])
    runtime = frozen_plan["runtime"]
    comparator = frozen_plan["comparator"]
    by_seed: dict[int, Mapping[str, Any]] = {}
    for raw in observations:
        observation = canonical_value(dict(raw))
        seed = observation.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise BaselineCalibrationError("baseline observation seed is invalid")
        if seed in by_seed:
            raise BaselineCalibrationError("duplicate baseline calibration seed")
        if seed not in expected_seeds:
            raise BaselineCalibrationError("unplanned baseline calibration seed")
        if observation.get("exit_status") != "SUCCESS":
            raise BaselineCalibrationError("baseline calibration requires all successes")
        if observation.get("runtime_release_digest") != runtime["release_digest"]:
            raise BaselineCalibrationError("baseline runtime release drift")
        if observation.get("execution_recipe_digest") != comparator[
            "execution_recipe_digest"
        ]:
            raise BaselineCalibrationError("baseline execution recipe drift")
        if observation.get("metric_source") != "BEST_VALID_RESULT" or observation.get(
            "online_partition_role"
        ) != "DEVELOPMENT_VALIDATION":
            raise BaselineCalibrationError("baseline observation is not development-only")
        if any("test" in str(key).lower() or "heldout" in str(key).lower() for key in observation):
            raise BaselineCalibrationError("baseline observation exposes heldout/test data")
        metrics = observation.get("metrics")
        value = metrics.get("ndcg@10") if isinstance(metrics, Mapping) else None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise BaselineCalibrationError("baseline NDCG@10 is invalid")
        by_seed[seed] = observation

    if tuple(sorted(by_seed)) != tuple(sorted(expected_seeds)):
        raise BaselineCalibrationError("baseline seed panel is incomplete")
    values = [float(by_seed[seed]["metrics"]["ndcg@10"]) for seed in expected_seeds]
    mean = fmean(values)
    sample_std = stdev(values)
    sem = sample_std / math.sqrt(len(values))
    margin = _T_CRITICAL_95_DF4 * sem
    evidence_rows = [
        {
            "seed": seed,
            "ndcg_at_10": by_seed[seed]["metrics"]["ndcg@10"],
            "observation_digest": sha256_digest(by_seed[seed]),
        }
        for seed in expected_seeds
    ]
    receipt = {
        "schema": BASELINE_RECEIPT_SCHEMA_V31,
        "status": "FROZEN_BEFORE_ARM_LAUNCH",
        "plan_digest": frozen_plan["plan_digest"],
        "comparator_id": comparator["comparator_id"],
        "execution_recipe_digest": comparator["execution_recipe_digest"],
        "runtime_release_digest": runtime["release_digest"],
        "metric": "NDCG@10",
        "partition_role": "DEVELOPMENT_VALIDATION",
        "ordered_seed_panel": expected_seeds,
        "observations": evidence_rows,
        "ndcg_at_10": mean,
        "sample_std": sample_std,
        "standard_error": sem,
        "confidence_interval_95": {
            "lower": max(0.0, mean - margin),
            "upper": min(1.0, mean + margin),
        },
        "provider_calls": 0,
        "heldout_access": "NONE",
    }
    return {**canonical_value(receipt), "receipt_digest": sha256_digest(receipt)}


__all__ = [
    "BASELINE_PLAN_SCHEMA_V31",
    "BASELINE_RECEIPT_SCHEMA_V31",
    "BASELINE_SEEDS_V31",
    "BaselineCalibrationError",
    "development_baseline_plan_v31",
    "development_lightgcn_recipe_v31",
    "finalize_development_baseline_v31",
]
