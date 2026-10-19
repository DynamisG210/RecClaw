"""P4's frozen raw-data protocol at the existing worker boundary."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from recclaw_core.experiments.helix_abc_v1.canonical import canonical_value, sha256_digest
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    P4_SPARSE_SPECTRAL_EVALUATOR, P4_SPARSE_SPECTRAL_SPLIT,
)

RAW_SHA256 = "e943abb91013a54c385828fdf5ab4ce49e957ca3a772adb30cde2a7d5539b389"
P4_EVAL_ARGS = {"split": {"RS": [0.8, 0.1, 0.1]}, "group_by": "user", "order": "RO", "mode": "full"}
P4_FIXED_CONFIG = {
    "benchmark_filename": None,
    "eval_args": P4_EVAL_ARGS,
    "load_col": {"inter": ["user_id", "item_id"]},
    "train_batch_size": 2048, "eval_batch_size": 65536,
    "metric_decimal_place": 10, "valid_metric": "NDCG@10",
    "eval_step": 1, "reproducibility": True,
}


def p4_fit_mode(config: Mapping[str, Any]) -> str:
    """Read the declared fitting method; old coefficient packages retain BPR."""
    mode = config.get("p4_fit_mode", "BPR_COEFFICIENTS")
    if mode not in {"BPR_COEFFICIENTS", "TRAIN_ONLY_PRECOMPUTE"}:
        raise ValueError("unknown declared P4 fitting mode")
    return mode


def p4_fit_input_contract(config: Mapping[str, Any]) -> tuple[str, float | None]:
    """Read explicit research semantics, never infer masking from prose."""
    mode = config.get("p4_fit_input_mode")
    fraction = config.get("p4_mask_fraction")
    if mode == "FULL_TRAIN":
        if fraction is not None:
            raise ValueError("FULL_TRAIN does not take p4_mask_fraction")
    elif mode == "MASKED_TRAIN":
        if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction < 1:
            raise ValueError("MASKED_TRAIN requires 0 < p4_mask_fraction < 1")
    else:
        raise ValueError("new P4 fitting requires explicit FULL_TRAIN or MASKED_TRAIN")
    if config.get("p4_zero_action") not in {"COEFFICIENTS_ZERO", "NOT_DECLARED"}:
        raise ValueError("new P4 fitting must explicitly declare its zero-action contract")
    return mode, fraction


def p4_fitting_inputs(model: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Give the fitting function a coherent view, separate from final scoring."""
    import numpy as np
    import torch
    from recclaw_core.experiments.helix_abc_v1.p4_fagsp_parent import frozen_fagsp_scores

    mode, fraction = p4_fit_input_contract(config)
    original = model.p4_residual_inputs()
    train = original["train_csr"]
    rows, cols = train.nonzero()
    if mode == "MASKED_TRAIN":
        count = int(np.ceil(float(fraction) * len(rows)))
        selected = np.random.default_rng(original["seed"]).choice(len(rows), count, replace=False)
        rows, cols = rows[selected], cols[selected]
        masked = train.copy().tolil()
        masked[rows, cols] = 0
        masked = masked.tocsr()
        masked.eliminate_zeros()
        fitted: dict[str, Any] = {}
        parent = frozen_fagsp_scores(masked, residual_inputs=fitted)
        device = original["parent_scores"].device
        inputs = {key: torch.from_numpy(value).to(device) if isinstance(value, np.ndarray) else value
                  for key, value in fitted.items()}
        coo = inputs["train_csr"].tocoo()
        inputs["train_matrix"] = torch.sparse_coo_tensor(
            torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64)),
            torch.from_numpy(coo.data.astype(np.float64)), size=coo.shape,
        ).coalesce().to(device)
        inputs["parent_scores"] = torch.from_numpy(parent).to(device)
        inputs["seed"] = original["seed"]
    else:
        inputs = dict(original)
    device = inputs["parent_scores"].device
    inputs.update(
        target_rows=torch.as_tensor(rows, dtype=torch.long, device=device),
        target_cols=torch.as_tensor(cols, dtype=torch.long, device=device),
        target_values=torch.ones(len(rows), dtype=torch.float64, device=device),
    )
    return inputs


def is_p4_recipe(recipe: Mapping[str, Any]) -> bool:
    return (recipe.get("split") == P4_SPARSE_SPECTRAL_SPLIT
            and canonical_value(recipe.get("evaluator")) == P4_SPARSE_SPECTRAL_EVALUATOR)


def actual_execution_contract(record: Mapping[str, Any]) -> dict[str, Any]:
    """Read an explicit construction/execution record; never invent parent values."""
    contract = record.get("execution_contract")
    if not isinstance(contract, Mapping) or set(contract) != {
        "base_model_config", "capability_family", "model", "config"
    } or not isinstance(contract.get("config"), Mapping):
        raise ValueError("P4 requires a four-field actual execution_contract from the parent execution record")
    if contract["base_model_config"] != "P4SparseSpectral" or contract["model"] != "FreshCandidateModel":
        raise ValueError("P4 execution record must identify its residual-base adapter ABI")
    if contract["capability_family"] != "P4_SPARSE_SPECTRAL_OPERATOR":
        raise ValueError("P4 execution record has a different capability family")
    if not record.get("source") or not record.get("evidence_class"):
        raise ValueError("P4 execution record requires source and evidence_class")
    config = contract["config"]
    for key, value in P4_FIXED_CONFIG.items():
        if key in config and canonical_value(config[key]) != canonical_value(value):
            raise ValueError(f"P4 execution record changes frozen protocol field {key}")
    if config.get("seed") != 2026:
        raise ValueError("P4's historical validation comparator is bound to seed2026")
    return canonical_value(contract)


def merge_mechanism_delta(parent: Mapping[str, Any], delta_json: str) -> dict[str, Any]:
    delta = json.loads(delta_json)
    if not isinstance(delta, dict):
        raise ValueError("mechanism_config_json must encode an object")
    p4_fit_mode({**parent["config"], **delta})
    immutable = {**P4_FIXED_CONFIG, "seed": 2026}
    for key, value in immutable.items():
        if key in delta and canonical_value(delta[key]) != canonical_value(value):
            raise ValueError(f"P4 mechanism delta changes frozen protocol field {key}")
    for key in ("epochs", "stopping_step", "recclaw_trainer_entrypoint", "frozen_parent", "frozen_parent_id", "frozen_parent_execution_digest"):
        if key in delta and canonical_value(delta[key]) != canonical_value(parent["config"].get(key)):
            raise ValueError(f"P4 mechanism delta changes execution-owned field {key}")
    return canonical_value({**dict(parent), "config": {**dict(parent["config"]), **delta}})


def p4_worker_config(recipe: Mapping[str, Any], *, seed: int, epochs: int) -> dict[str, Any]:
    if not is_p4_recipe(recipe) or seed != 2026:
        raise ValueError("P4 worker requires its exact seed2026 raw-RS811 validation protocol")
    if recipe["base_model_config"] != "P4SparseSpectral":
        raise ValueError("P4 worker requires the residual-base adapter")
    return {**P4_FIXED_CONFIG, "seed": seed, "epochs": epochs, "stopping_step": min(5, epochs)}


def dataset_roots_for_recipe(recipe, *, default_root: Path, default_partition):
    if not is_p4_recipe(recipe):
        digest = hashlib.sha256((default_root / "search_partition_manifest.json").read_bytes()).hexdigest()
        return default_root, default_root / "ml-1m", digest, default_partition
    root_text = os.environ.get("RECCLAW_P4_DATASET_ROOT")
    if not root_text:
        raise ValueError("P4 requires RECCLAW_P4_DATASET_ROOT pointing to raw ML-1M")
    root = Path(root_text).expanduser().resolve()
    digest = hashlib.sha256((root / "ml-1m/ml-1m.inter").read_bytes()).hexdigest()
    if digest != RAW_SHA256:
        raise ValueError("P4 raw ML-1M SHA256 mismatch")
    manifest = {"schema": "recclaw.p4-sparse-spectral.dataset-manifest.v1", "dataset": "ml-1m", "raw_dataset_sha256": digest, "data_root": str(root)}
    return root, root / "ml-1m", sha256_digest(manifest), manifest
