from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from recclaw_core.experiments.helix_abc_v1.canonical import bytes_sha256


ROOT = Path(__file__).resolve().parents[3]
ORIGINAL_SOURCE_SHA256 = (
    "ade9963f19c1d5e90e261a71b6cf21c28af92af86f9f511920a39e18b75c8df3"
)
REALIZATION_ROOT = (
    ROOT
    / "src/recclaw_core/experiments/helix_abc_v1/resources/"
    "f1_resource_compatible_realization_v1"
)
REALIZATION_SOURCE = REALIZATION_ROOT / "recclaw_ext/candidate.py"
MINI_DATA = (
    Path(__file__).resolve().parent / "fixtures/innovation_spine/data"
)


def _sealed_source() -> Path:
    configured = os.environ.get("RECCLAW_F1_ORIGINAL_SOURCE")
    if configured:
        return Path(configured)
    receipt = json.loads(
        (
            ROOT / "docs/research_line/vnext/F1_OPEN_META_CANONICAL_RECEIPT.json"
        ).read_text(encoding="utf-8")
    )
    external_root = Path(receipt["external_receipt_ref"]).parent
    return (
        external_root
        / "execution/candidates/slot-01/"
        "innovation-candidate-e2fdeaacdf44bbcd5f85f905/recclaw_ext/candidate.py"
    )


def _load_candidate(path: Path, module_name: str) -> type[Any]:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"unable to load candidate source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FreshCandidateModel


def _models(tmp_path: Path):
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed

    original_class = _load_candidate(_sealed_source(), "f1_sealed_candidate_v1")
    realization_class = _load_candidate(
        REALIZATION_SOURCE, "f1_resource_compatible_realization_v1"
    )
    import recbole

    recbole_root = Path(recbole.__file__).resolve().parents[1]
    config = Config(
        model="BPR",
        dataset="mini",
        config_file_list=[
            str(recbole_root / "recbole/properties/model/BPR.yaml"),
            str(ROOT / "configs/task_ml1m.yaml"),
            str(ROOT / "configs/lightgcn_metrics.yaml"),
        ],
        config_dict={
            "benchmark_filename": ["train", "valid", "test"],
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "data_path": str(MINI_DATA),
            "epochs": 1,
            "eval_batch_size": 32,
            "eval_step": 1,
            "reproducibility": True,
            "seed": 53102,
            "show_progress": False,
            "state": "ERROR",
            "stopping_step": 1,
            "topk": [3],
            "train_batch_size": 8,
            "use_gpu": os.environ.get("RECCLAW_EQUIVALENCE_USE_GPU") == "1",
            "valid_metric": "NDCG@3",
        },
    )
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, _valid_data, _test_data = data_preparation(config, dataset)
    original = original_class(config, train_data._dataset)
    realization = realization_class(config, train_data._dataset)
    realization.load_state_dict(original.state_dict(), strict=True)
    original.score_chunk_size = 3
    realization.score_chunk_size = 3
    realization.full_sort_user_chunk_size = 2
    interaction = next(iter(train_data))
    return original, realization, interaction


def _assert_close(left: torch.Tensor, right: torch.Tensor) -> None:
    if left.is_cuda or right.is_cuda:
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    else:
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)


def _exercise_equivalence(tmp_path: Path, *, stability_weight: float) -> None:
    from recbole.data.interaction import Interaction

    original, realization, interaction = _models(tmp_path)
    original.stability_weight = stability_weight
    realization.stability_weight = stability_weight

    original.eval()
    realization.eval()
    with torch.no_grad():
        _assert_close(original.predict(interaction), realization.predict(interaction))
        users = interaction[original.USER_ID][:4]
        full_sort_input = Interaction({original.USER_ID: users})
        original_full = original.full_sort_predict(full_sort_input)
        realization_full = realization.full_sort_predict(full_sort_input)
    _assert_close(original_full, realization_full)
    assert original_full.shape == realization_full.shape
    assert original_full.numel() == int(users.shape[0]) * int(original.n_items)

    original.train()
    realization.train()
    original.zero_grad(set_to_none=True)
    realization.zero_grad(set_to_none=True)
    original_loss = original.calculate_loss(interaction)
    realization_loss = realization.calculate_loss(interaction)
    _assert_close(original_loss, realization_loss)
    original_loss.backward()
    realization_loss.backward()

    original_parameters = dict(original.named_parameters())
    realization_parameters = dict(realization.named_parameters())
    assert original_parameters.keys() == realization_parameters.keys()
    for name in original_parameters:
        original_grad = original_parameters[name].grad
        realization_grad = realization_parameters[name].grad
        assert original_grad is not None, name
        assert realization_grad is not None, name
        _assert_close(original_grad, realization_grad)
    assert realization.latest_diagnostics == pytest.approx(
        original.latest_diagnostics, rel=1e-6, abs=1e-7
    )


def test_lineage_parameters_and_computation_only_change(tmp_path: Path) -> None:
    original_source = _sealed_source()
    assert bytes_sha256(original_source.read_bytes()) == ORIGINAL_SOURCE_SHA256
    assert bytes_sha256(REALIZATION_SOURCE.read_bytes()) != ORIGINAL_SOURCE_SHA256
    original, realization, interaction = _models(tmp_path)

    assert dict(original.named_parameters()).keys() == dict(
        realization.named_parameters()
    ).keys()
    assert sum(parameter.numel() for parameter in original.parameters()) == sum(
        parameter.numel() for parameter in realization.parameters()
    )
    assert original.support_topk == realization.support_topk == 1
    assert original.stability_weight == realization.stability_weight == 0.2

    original_key_calls = 0
    realization_key_calls = 0

    def count_original(*_args):
        nonlocal original_key_calls
        original_key_calls += 1

    def count_realization(*_args):
        nonlocal realization_key_calls
        realization_key_calls += 1

    original_hook = original.support_key.register_forward_hook(count_original)
    realization_hook = realization.support_key.register_forward_hook(
        count_realization
    )
    try:
        original.calculate_loss(interaction)
        realization.calculate_loss(interaction)
    finally:
        original_hook.remove()
        realization_hook.remove()
    assert original_key_calls == 4
    assert realization_key_calls == 1


def test_scores_loss_diagnostics_and_all_gradients_are_equivalent(
    tmp_path: Path,
) -> None:
    _exercise_equivalence(tmp_path, stability_weight=0.2)


def test_mechanism_off_scores_loss_diagnostics_and_gradients_are_equivalent(
    tmp_path: Path,
) -> None:
    _exercise_equivalence(tmp_path, stability_weight=0.0)


def test_full_sort_reuses_key_per_user_chunk_not_per_item_chunk(
    tmp_path: Path,
) -> None:
    from recbole.data.interaction import Interaction

    original, realization, interaction = _models(tmp_path)
    users = interaction[original.USER_ID][:4]
    full_sort_input = Interaction({original.USER_ID: users})
    original_key_calls = 0
    realization_key_calls = 0

    def count_original(*_args):
        nonlocal original_key_calls
        original_key_calls += 1

    def count_realization(*_args):
        nonlocal realization_key_calls
        realization_key_calls += 1

    original_hook = original.support_key.register_forward_hook(count_original)
    realization_hook = realization.support_key.register_forward_hook(
        count_realization
    )
    try:
        with torch.no_grad():
            original_scores = original.full_sort_predict(full_sort_input)
            realization_scores = realization.full_sort_predict(full_sort_input)
    finally:
        original_hook.remove()
        realization_hook.remove()
    _assert_close(original_scores, realization_scores)
    assert original_key_calls == math.ceil(original.n_items / 3)
    assert realization_key_calls == math.ceil(int(users.shape[0]) / 2)
