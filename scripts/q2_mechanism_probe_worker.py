#!/usr/bin/env python3
"""Run the frozen Q2 resource prefix and cheap mechanism probes on one GPU."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any


class ResourceDeadlineExpired(TimeoutError):
    """The preregistered Q0R2 300 second resource deadline expired."""


def _load_candidate_class(source_path: Path) -> type:
    spec = importlib.util.spec_from_file_location(
        "recclaw_q2_selected_candidate", source_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("selected candidate module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = getattr(module, "FreshCandidateModel", None)
    if not isinstance(candidate, type):
        raise RuntimeError("selected candidate entrypoint is unavailable")
    return candidate


def _float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "numel") and int(value.numel()) == 1:
        value = value.item()
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("non-finite probe observation")
    return result


def _tensor_max_abs(value: Any) -> float:
    return _float(value.detach().abs().max())


def _interaction_from_rows(interaction: Any, rows: list[int], model: Any) -> Any:
    import torch
    from recbole.data.interaction import Interaction

    index = torch.tensor(rows, dtype=torch.long, device=interaction[model.USER_ID].device)
    return Interaction(
        {
            model.USER_ID: interaction[model.USER_ID].index_select(0, index),
            model.ITEM_ID: interaction[model.ITEM_ID].index_select(0, index),
            model.NEG_ITEM_ID: interaction[model.NEG_ITEM_ID].index_select(0, index),
        }
    )


def _concat_interactions(interactions: list[Any], model: Any) -> Any:
    import torch
    from recbole.data.interaction import Interaction

    return Interaction(
        {
            field: torch.cat([item[field] for item in interactions], dim=0)
            for field in (model.USER_ID, model.ITEM_ID, model.NEG_ITEM_ID)
        }
    )


def _origin_blind_probe_interaction(
    model: Any,
    preallocated_train_batches: tuple[Any, ...],
    contract: dict[str, Any],
) -> Any:
    population = contract["probe_population"]
    selected = []
    limit = int(population["maximum_examples_per_probe_batch"])
    for batch_index in population["probe_train_batch_indices"]:
        batch = preallocated_train_batches[int(batch_index)]
        users = batch[model.USER_ID].detach().cpu().tolist()
        positives = batch[model.ITEM_ID].detach().cpu().tolist()
        negatives = batch[model.NEG_ITEM_ID].detach().cpu().tolist()
        eligible = [
            row
            for row, (user, positive) in enumerate(zip(users, positives))
            if model._get_user_history(int(user), int(positive)).numel() >= 2
        ]
        eligible.sort(
            key=lambda row: (
                int(users[row]),
                int(positives[row]),
                int(negatives[row]),
                row,
            )
        )
        if eligible:
            selected.append(_interaction_from_rows(batch, eligible[:limit], model))
    if not selected:
        raise RuntimeError("no structurally eligible origin-blind probe examples")
    return _concat_interactions(selected, model)


def _loss_statistics(model: Any, interaction: Any) -> dict[str, Any]:
    import torch

    users = interaction[model.USER_ID]
    positives = interaction[model.ITEM_ID]
    negatives = interaction[model.NEG_ITEM_ID]
    device = model._user_embedding_layer().weight.device
    users = users.to(device)
    positives = positives.to(device)
    negatives = negatives.to(device)
    model.train()
    model.set_mechanism_enabled(True)
    model.zero_grad(set_to_none=True)
    full_loss = model.calculate_loss(interaction)
    full_loss.backward()
    gradient_norms = {
        name: _float(parameter.grad.norm()) if parameter.grad is not None else 0.0
        for name, parameter in model.named_parameters()
        if name == "support_gate.weight"
    }
    user_e = model._user_embedding_layer()(users)
    pos_e = model._item_embedding_layer()(positives)
    neg_e = model._item_embedding_layer()(negatives)
    stability = model._stability_penalty(
        users, user_e, pos_e, neg_e, exclude_items=positives
    )
    weighted_stability = float(model._STABILITY_WEIGHT) * _float(stability)
    model.zero_grad(set_to_none=True)
    model.set_mechanism_enabled(False)
    off_loss = model.calculate_loss(interaction)
    off_loss.backward()
    model.set_mechanism_enabled(True)
    return {
        "full_loss": _float(full_loss),
        "mechanism_off_loss": _float(off_loss),
        "stability_contribution": weighted_stability,
        "full_off_abs_delta": abs(_float(full_loss) - _float(off_loss)),
        "mechanism_gradient_l2": gradient_norms,
    }


def _gate_statistics(model: Any, interaction: Any, contract: dict[str, Any]) -> dict[str, Any]:
    import torch

    device = model._user_embedding_layer().weight.device
    users = interaction[model.USER_ID].to(device)
    positives = interaction[model.ITEM_ID].to(device)
    raw_logits: list[float] = []
    saturated = 0
    eligible_users: set[int] = set()
    distinct_users = {int(value) for value in users.detach().cpu().tolist()}
    signatures: set[tuple[float, ...]] = set()
    for row in range(users.size(0)):
        user_id = int(users[row].item())
        positive = int(positives[row].item())
        history = model._get_user_history(user_id, positive)
        if history.numel() < 2:
            continue
        eligible_users.add(user_id)
        history = history.to(device)
        history_e = model._item_embedding_layer()(history)
        user_e = model._user_embedding_layer()(users[row : row + 1]).expand_as(
            history_e
        )
        logits = model.support_gate(torch.cat([user_e, history_e], dim=-1)).squeeze(-1)
        weights = torch.softmax(logits, dim=0)
        raw_logits.extend(float(value) for value in logits.detach().cpu().tolist())
        saturated += int(_float(weights.max()) >= float(contract["numeric_rules"]["support_saturation_weight"]))
        centered = logits - logits.mean()
        signatures.add(
            tuple(round(float(value), 7) for value in centered.detach().cpu().tolist())
        )
    if not raw_logits:
        return {
            "distinct_user_count": len(distinct_users),
            "eligible_user_count": 0,
            "raw_logit_std": 0.0,
            "support_saturation_rate": 1.0,
            "distinct_centered_logit_signatures": 0,
        }
    logits_tensor = torch.tensor(raw_logits, dtype=torch.float64)
    return {
        "distinct_user_count": len(distinct_users),
        "eligible_user_count": len(eligible_users),
        "raw_logit_count": len(raw_logits),
        "raw_logit_min": _float(logits_tensor.min()),
        "raw_logit_max": _float(logits_tensor.max()),
        "raw_logit_mean": _float(logits_tensor.mean()),
        "raw_logit_std": _float(logits_tensor.std(unbiased=False)),
        "support_saturation_rate": saturated / max(1, len(eligible_users)),
        "distinct_centered_logit_signatures": len(signatures),
    }


def _routing_statistics(model: Any, interaction: Any) -> dict[str, Any]:
    import torch

    users = interaction[model.USER_ID].to(model._user_embedding_layer().weight.device)
    positives = interaction[model.ITEM_ID].to(users.device)
    negatives = interaction[model.NEG_ITEM_ID].to(users.device)
    call_count = 0

    def counted_gate(_module: Any, _inputs: Any, _output: Any) -> None:
        nonlocal call_count
        call_count += 1

    hook = model.support_gate.register_forward_hook(counted_gate)
    try:
        model.eval()
        model.set_mechanism_enabled(True)
        with torch.no_grad():
            user_e = model._user_embedding_layer()(users)
            pos_e = model._item_embedding_layer()(positives)
            neg_e = model._item_embedding_layer()(negatives)
            summaries, _, _ = model._history_summary(users, exclude_items=positives)
            full_positive = torch.sum((user_e + summaries) * pos_e, dim=-1)
            full_margin = torch.sum((user_e + summaries) * (pos_e - neg_e), dim=-1)
            model.set_mechanism_enabled(False)
            off_positive = torch.sum(user_e * pos_e, dim=-1)
            off_margin = torch.sum(user_e * (pos_e - neg_e), dim=-1)
    finally:
        hook.remove()
        model.set_mechanism_enabled(True)
    return {
        "support_gate_call_count": call_count,
        "positive_score_max_abs_delta": _tensor_max_abs(full_positive - off_positive),
        "pairwise_margin_max_abs_delta": _tensor_max_abs(full_margin - off_margin),
        "candidate_declared_graph_propagation_present": bool(
            hasattr(model, "norm_adj_matrix")
        ),
        "traced_path": "support_gate_to_history_summary_to_user_repr_to_score_and_margin",
    }


def _target_conditioning_statistics(model: Any, contract: dict[str, Any]) -> dict[str, Any]:
    import torch

    device = model._user_embedding_layer().weight.device
    minimum = int(contract["probe_population"]["minimum_eligible_users_with_two_supports"])
    compared = 0
    maximum_delta = 0.0
    order_changes = 0
    for user_id, history_tensor in enumerate(model._user_histories):
        history = sorted(set(int(value) for value in history_tensor.tolist()))
        if len(history) < 4:
            continue
        first_target, second_target = history[:2]
        common = [value for value in history if value not in {first_target, second_target}]
        common_tensor = torch.tensor(common, dtype=torch.long, device=device)
        user_tensor = torch.tensor([user_id], dtype=torch.long, device=device)
        history_e = model._item_embedding_layer()(common_tensor)
        user_e = model._user_embedding_layer()(user_tensor).expand_as(history_e)
        # The selected implementation has no target input to support_gate.  We
        # execute both declared target cases anyway and compare their common set.
        first_logits = model.support_gate(torch.cat([user_e, history_e], dim=-1)).squeeze(-1)
        second_logits = model.support_gate(torch.cat([user_e, history_e], dim=-1)).squeeze(-1)
        first_centered = first_logits - first_logits.mean()
        second_centered = second_logits - second_logits.mean()
        maximum_delta = max(
            maximum_delta, _tensor_max_abs(first_centered - second_centered)
        )
        order_changes += int(
            not torch.equal(
                torch.argsort(first_logits, stable=True),
                torch.argsort(second_logits, stable=True),
            )
        )
        compared += 1
        if compared >= minimum:
            break
    return {
        "compared_user_count": compared,
        "max_centered_common_logit_delta": maximum_delta,
        "pairwise_order_change_count": order_changes,
        "target_input_observed_in_gate": False,
    }


def _copy_shared_embeddings(candidate: Any, parent: Any) -> None:
    import torch

    with torch.no_grad():
        parent.user_embedding.weight.copy_(candidate._user_embedding_layer().weight)
        parent.item_embedding.weight.copy_(candidate._item_embedding_layer().weight)


def _parent_equivalence_statistics(
    candidate: Any, parent: Any, interaction: Any
) -> dict[str, Any]:
    import torch
    from recbole.data.interaction import Interaction

    device = candidate._user_embedding_layer().weight.device
    users = interaction[candidate.USER_ID].to(device)
    positives = interaction[candidate.ITEM_ID].to(device)
    unique_users = torch.unique(users, sorted=True)
    pair = Interaction(
        {
            candidate.USER_ID: users,
            candidate.ITEM_ID: positives,
            candidate.NEG_ITEM_ID: interaction[candidate.NEG_ITEM_ID].to(device),
        }
    )
    user_only = Interaction({candidate.USER_ID: unique_users})
    _copy_shared_embeddings(candidate, parent)
    candidate.eval()
    parent.eval()
    candidate.set_mechanism_enabled(False)
    with torch.no_grad():
        candidate_predict = candidate.predict(pair)
        parent_predict = parent.predict(pair)
        candidate_full = candidate.full_sort_predict(user_only)
        parent_full = parent.full_sort_predict(user_only)
    candidate.train()
    parent.train()
    candidate.zero_grad(set_to_none=True)
    candidate_loss = candidate.calculate_loss(pair)
    candidate_loss.backward()
    candidate_gradients = {
        "user": candidate._user_embedding_layer().weight.grad.detach().clone(),
        "item": candidate._item_embedding_layer().weight.grad.detach().clone(),
    }
    parent.zero_grad(set_to_none=True)
    parent_loss = parent.calculate_loss(pair)
    parent_loss.backward()
    parent_gradients = {
        "user": parent.user_embedding.weight.grad.detach(),
        "item": parent.item_embedding.weight.grad.detach(),
    }
    gradient_delta = max(
        _tensor_max_abs(candidate_gradients[name] - parent_gradients[name])
        for name in ("user", "item")
    )
    gradient_reference = max(
        _tensor_max_abs(parent_gradients[name]) for name in ("user", "item")
    )
    candidate.set_mechanism_enabled(True)
    return {
        "declared_parent_class": (
            f"{parent.__class__.__module__}:{parent.__class__.__name__}"
        ),
        "max_abs_deltas": {
            "predict": _tensor_max_abs(candidate_predict - parent_predict),
            "full_sort_predict": _tensor_max_abs(candidate_full - parent_full),
            "calculate_loss": abs(_float(candidate_loss) - _float(parent_loss)),
            "shared_parameter_gradients": gradient_delta,
        },
        "reference_max_abs_values": {
            "predict": _tensor_max_abs(parent_predict),
            "full_sort_predict": _tensor_max_abs(parent_full),
            "calculate_loss": abs(_float(parent_loss)),
            "shared_parameter_gradients": gradient_reference,
        },
    }


def _removed_summary(weights: Any, embeddings: Any, removed_index: int) -> Any:
    import torch

    keep = torch.ones(weights.size(0), dtype=torch.bool, device=weights.device)
    keep[removed_index] = False
    kept_weights = weights[keep]
    kept_weights = kept_weights / kept_weights.sum().clamp_min(1e-12)
    return torch.sum(kept_weights.unsqueeze(-1) * embeddings[keep], dim=0)


def _discriminative_statistics(
    model: Any, interaction: Any, contract: dict[str, Any]
) -> dict[str, Any]:
    import torch

    seed = int(contract["probe_population"]["seed"])
    device = model._user_embedding_layer().weight.device
    users = interaction[model.USER_ID].to(device)
    positives = interaction[model.ITEM_ID].to(device)
    negatives = interaction[model.NEG_ITEM_ID].to(device)
    learned_damage: list[float] = []
    control_damage: list[float] = []
    with torch.no_grad():
        for row in range(users.size(0)):
            user_id = int(users[row].item())
            positive_id = int(positives[row].item())
            history = model._get_user_history(user_id, positive_id)
            if history.numel() < 2:
                continue
            history = history.to(device)
            history_e = model._item_embedding_layer()(history)
            user_e = model._user_embedding_layer()(users[row : row + 1]).squeeze(0)
            pos_e = model._item_embedding_layer()(positives[row : row + 1]).squeeze(0)
            neg_e = model._item_embedding_layer()(negatives[row : row + 1]).squeeze(0)
            user_expand = user_e.unsqueeze(0).expand_as(history_e)
            logits = model.support_gate(
                torch.cat([user_expand, history_e], dim=-1)
            ).squeeze(-1)
            weights = torch.softmax(logits, dim=0)
            top = int(torch.argmax(weights).item())
            non_top = [index for index in range(weights.size(0)) if index != top]
            token = f"{seed}:{user_id}:{positive_id}".encode("ascii")
            control = non_top[int(hashlib.sha256(token).hexdigest(), 16) % len(non_top)]
            summary = torch.sum(weights.unsqueeze(-1) * history_e, dim=0)
            direction = pos_e - neg_e
            original_margin = torch.sum((user_e + summary) * direction)
            learned_margin = torch.sum(
                (user_e + _removed_summary(weights, history_e, top)) * direction
            )
            control_margin = torch.sum(
                (user_e + _removed_summary(weights, history_e, control)) * direction
            )
            learned_damage.append(_float(torch.abs(learned_margin - original_margin)))
            control_damage.append(_float(torch.abs(control_margin - original_margin)))
    return {
        "eligible_example_count": len(learned_damage),
        "learned_removal_mean_abs_margin_damage": (
            sum(learned_damage) / len(learned_damage) if learned_damage else None
        ),
        "control_removal_mean_abs_margin_damage": (
            sum(control_damage) / len(control_damage) if control_damage else None
        ),
        "control_selection": "SHA256(seed,user_id,positive_item_id)_FROM_NON_TOP_SUPPORTS",
    }


def _throughput_summary(telemetry: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for phase in ("TRAIN", "EVAL"):
        rows = [row for row in telemetry.get("phase_records", []) if row["phase"] == phase]
        batches = sum(int(row.get("batch_count", 0)) for row in rows)
        wall_ms = sum(int(row.get("wall_time_ms", 0)) for row in rows)
        result[phase.lower()] = {
            "completed_phases": len(rows),
            "batch_count": batches,
            "wall_time_ms": wall_ms,
            "batches_per_second": batches / (wall_ms / 1000.0) if wall_ms else None,
            "peak_allocated_mib": max(
                (
                    float(row["peak_allocated_mib"])
                    for row in rows
                    if isinstance(row.get("peak_allocated_mib"), (int, float))
                ),
                default=None,
            ),
            "peak_reserved_mib": max(
                (
                    float(row["peak_reserved_mib"])
                    for row in rows
                    if isinstance(row.get("peak_reserved_mib"), (int, float))
                ),
                default=None,
            ),
        }
    return result


def _sealed_resource_snapshot(root: Path) -> list[dict[str, Any]]:
    from recclaw_core.experiments.helix_abc_v1.mechanism_characterization import (
        bytes_sha256,
    )

    return [
        {
            "path": str(path.relative_to(root)),
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": bytes_sha256(path.read_bytes()),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]


def _load_prior_resource_for_continuation(
    *,
    output_root: Path,
    sealed_resource_root: Path,
    repo_root: Path,
    contract_sha256: str,
    import_gate_path: Path,
) -> tuple[dict[str, Any], str, Path, list[dict[str, Any]]]:
    """Bind the completed resource prefix without executing any training again."""

    from recclaw_core.experiments.helix_abc_v1.mechanism_characterization import (
        bytes_sha256,
        canonical_json_bytes,
        write_new_json,
    )

    resource_result_path = sealed_resource_root / "RESOURCE_PROBE_RESULT.json"
    telemetry_path = sealed_resource_root / "RESOURCE_TELEMETRY.json"
    original_binding_path = sealed_resource_root / "INPUT_BINDING_BEFORE_OUTCOME.json"
    for required_path in (resource_result_path, telemetry_path, original_binding_path):
        if not required_path.is_file():
            raise RuntimeError(f"continuation prerequisite is missing: {required_path}")

    sealed_snapshot = _sealed_resource_snapshot(sealed_resource_root)
    write_new_json(
        output_root / "SEALED_RESOURCE_ROOT_BEFORE_CONTINUATION.json",
        {
            "schema": "recclaw.research-line.q2-sealed-resource-snapshot.v1",
            "status": "SNAPSHOT_BEFORE_CONTINUATION",
            "development_only": True,
            "sealed_resource_root_writes": 0,
            "files": sealed_snapshot,
        },
    )

    resource_result = json.loads(resource_result_path.read_text(encoding="utf-8"))
    resource_status = str(resource_result.get("status"))
    if resource_status not in {"SUCCESS", "RESOURCE_CENSORED", "RESOURCE_DEFERRED"}:
        raise RuntimeError("prior resource status is not an admissible physical result")
    expected_telemetry_sha256 = str(resource_result.get("telemetry_sha256"))
    observed_telemetry_sha256 = bytes_sha256(telemetry_path.read_bytes())
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    correction_applied = False
    removed_field = None
    if observed_telemetry_sha256 != expected_telemetry_sha256:
        active_progress = telemetry.get("active_progress")
        if not isinstance(active_progress, dict) or "last_loss_observation" not in active_progress:
            raise RuntimeError("resource telemetry drift is not the observed loss-wrapper mutation")
        removed_field = "active_progress.last_loss_observation"
        active_progress.pop("last_loss_observation")
        reconstructed_sha256 = bytes_sha256(canonical_json_bytes(telemetry))
        if reconstructed_sha256 != expected_telemetry_sha256:
            raise RuntimeError("resource telemetry cannot be exactly reconstructed")
        correction_applied = True

    authoritative_path = output_root / "RESOURCE_TELEMETRY_AUTHORITATIVE.json"
    authoritative_sha256 = write_new_json(authoritative_path, telemetry)
    if authoritative_sha256 != expected_telemetry_sha256:
        raise RuntimeError("authoritative resource telemetry digest mismatch")
    write_new_json(
        output_root / "RESOURCE_TELEMETRY_INTEGRITY_CORRECTION.json",
        {
            "schema": "recclaw.research-line.q2-resource-telemetry-integrity.v1",
            "status": "AUTHORITATIVE_COPY_VERIFIED",
            "development_only": True,
            "mechanism_effect_updates": 0,
            "resource_probe_reexecuted": False,
            "sealed_resource_root_writes": 0,
            "source_telemetry_modified": False,
            "observed_mutated_telemetry_sha256": observed_telemetry_sha256,
            "resource_result_recorded_telemetry_sha256": expected_telemetry_sha256,
            "authoritative_telemetry_sha256": authoritative_sha256,
            "correction_applied": correction_applied,
            "removed_post_resource_field": removed_field,
            "removed_field_belongs_to_resource_result": False,
            "reconstruction_method": (
                "copy sealed telemetry in memory, remove the sole post-resource "
                "field, and write a new authoritative artifact in the continuation root"
            ),
            "root_cause": (
                "resource telemetry loss wrapper remained installed after restore; "
                "the first post-resource loss call mutated only active_progress"
            ),
        },
    )

    stderr_path = sealed_resource_root.parent / "launcher.stderr"
    if not stderr_path.is_file():
        raise RuntimeError("initial mechanism probe stderr is missing")
    stderr_bytes = stderr_path.read_bytes()
    if b"Expected all tensors to be on the same device" not in stderr_bytes:
        raise RuntimeError("initial mechanism probe failure fingerprint drift")
    write_new_json(
        output_root / "INITIAL_MECHANISM_PROBE_FAILURE.json",
        {
            "schema": "recclaw.research-line.q2-initial-mechanism-probe-failure.v1",
            "status": "GENERAL_CALLER_DEVICE_PLACEMENT_FAILURE",
            "development_only": True,
            "resource_probe_reexecuted": False,
            "sealed_resource_root_writes": 0,
            "sealed_resource_result_sha256": bytes_sha256(
                resource_result_path.read_bytes()
            ),
            "sealed_contaminated_telemetry_sha256": observed_telemetry_sha256,
            "candidate_source_patched": False,
            "mechanism_effect_updates": 0,
            "stderr_sha256": bytes_sha256(stderr_bytes),
            "error_fingerprint": "Expected all tensors to be on the same device",
        },
    )

    continuation_binding_path = output_root / "CONTINUATION_INPUT_BINDING_BEFORE_RESULT.json"
    write_new_json(
        continuation_binding_path,
        {
            "schema": "recclaw.research-line.q2-continuation-input-binding.v1",
            "development_only": True,
            "held_out_reads": 0,
            "q2_contract_sha256": contract_sha256,
            "original_input_binding_sha256": bytes_sha256(
                original_binding_path.read_bytes()
            ),
            "resource_probe_result_sha256": bytes_sha256(
                resource_result_path.read_bytes()
            ),
            "authoritative_resource_telemetry_sha256": authoritative_sha256,
            "normal_package_import_gate_sha256": bytes_sha256(
                import_gate_path.read_bytes()
            ),
            "consumer_source_sha256": bytes_sha256(
                (
                    repo_root
                    / "src/recclaw_core/experiments/helix_abc_v1/"
                    "mechanism_characterization.py"
                ).read_bytes()
            ),
            "physical_worker_source_sha256": bytes_sha256(
                Path(__file__).resolve().read_bytes()
            ),
            "resource_probe_reexecuted": False,
            "candidate_source_patched": False,
            "caller_fix": "move origin-blind Interaction to the model device",
            "model_state": "FRESH_UNTRAINED",
            "checkpoint_status": "NOT_ASSESSED_PROTOCOL_NO_CHECKPOINT",
            "checkpoint_missingness_class": "PROTOCOL_CONSUMER_MISSINGNESS",
            "allowed_evidence_scope": [
                "STRUCTURAL_PARTICIPATION",
                "DEVICE_PATH",
                "DECLARED_PARENT_EQUIVALENCE",
                "TARGET_CONDITIONING_EQUIVALENCE",
            ],
            "learned_activation_or_mechanism_effect_allowed": False,
            "outcomes_present_when_written": 1,
        },
    )
    return (
        resource_result,
        bytes_sha256(resource_result_path.read_bytes()),
        continuation_binding_path,
        sealed_snapshot,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--recbole-root", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--continue-mechanism-only", action="store_true")
    parser.add_argument("--sealed-resource-root")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    recbole_root = Path(args.recbole_root).resolve()
    output_root = Path(args.output_root).resolve()
    contract_path = Path(args.contract).resolve()
    if args.continue_mechanism_only:
        if output_root.exists():
            raise RuntimeError(f"Q2 continuation output root already exists: {output_root}")
        if not args.sealed_resource_root:
            raise RuntimeError("--sealed-resource-root is required for continuation")
        sealed_resource_root = Path(args.sealed_resource_root).resolve()
        if not sealed_resource_root.is_dir():
            raise RuntimeError(
                f"sealed resource root is missing: {sealed_resource_root}"
            )
        import_gate_path = output_root.parent / "IMPORT_GATE_CONTINUATION_SEALED.json"
    else:
        if output_root.exists():
            raise RuntimeError(f"Q2 output root already exists: {output_root}")
        if args.sealed_resource_root:
            raise RuntimeError("--sealed-resource-root is continuation-only")
        sealed_resource_root = output_root
        import_gate_path = output_root.parent / "IMPORT_GATE.json"
    import_gate = json.loads(import_gate_path.read_text(encoding="utf-8"))
    if (
        import_gate.get("schema") != "recclaw.q2-normal-package-import-gate.v1"
        or import_gate.get("status") != "PASS"
        or import_gate.get("normal_package_import") is not True
        or import_gate.get("physical_probe_started") is not False
    ):
        raise RuntimeError("accepted-runtime normal package import Gate did not pass")
    output_root.mkdir(parents=True)

    sys.path.insert(0, str(repo_root / "src"))
    sys.path.insert(0, str(repo_root / "scripts"))
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(recbole_root))
    from recclaw_core.experiments.helix_abc_v1.mechanism_characterization import (
        bytes_sha256,
        canonical_json_bytes,
        classify_mechanism_evidence,
        evaluate_probe_statistics,
        full_ablation_allowed,
        load_contract,
        q3_evidence_package,
        validate_selected_package,
        write_new_json,
    )
    from recclaw_core.experiments.helix_abc_v1.resource_scheduling import (
        build_fixed_batch_prefix_contract,
    )
    from campaign_train_worker import (
        _finalize_resource_telemetry,
        _install_resource_telemetry,
        _preallocate_batches,
    )
    import run_candidate

    contract = load_contract(contract_path)
    selected = validate_selected_package(repo_root, contract)
    contract_sha256 = bytes_sha256(contract_path.read_bytes())
    prefix_contract = build_fixed_batch_prefix_contract()
    prefix_bytes = canonical_json_bytes(prefix_contract)
    expected_prefix_sha256 = contract["resource_probe"]["prefix_contract_sha256"]
    if bytes_sha256(prefix_bytes) != expected_prefix_sha256:
        raise RuntimeError("frozen Q0R2 prefix contract digest drift")
    binding = {
        "schema": "recclaw.research-line.q2-physical-input-binding.v1",
        "development_only": True,
        "held_out_reads": 0,
        "q2_contract_sha256": contract_sha256,
        "q1_commit": contract["input"]["q1_commit"],
        "q1_tree": contract["input"]["q1_tree"],
        "candidate_package_digest": contract["input"]["candidate_package_digest"],
        "candidate_source_tree_digest": contract["input"]["candidate_source_tree_digest"],
        "prefix_contract_sha256": expected_prefix_sha256,
        "normal_package_import_gate_sha256": bytes_sha256(
            import_gate_path.read_bytes()
        ),
        "consumer_source_sha256": bytes_sha256(
            (
                repo_root
                / "src/recclaw_core/experiments/helix_abc_v1/"
                "mechanism_characterization.py"
            ).read_bytes()
        ),
        "physical_worker_source_sha256": bytes_sha256(
            Path(__file__).resolve().read_bytes()
        ),
        "outcomes_present_when_written": 0,
    }
    binding_path = sealed_resource_root / "INPUT_BINDING_BEFORE_OUTCOME.json"
    contract_snapshot_path = sealed_resource_root / "PROBE_CONTRACT_BEFORE_OUTCOME.json"
    prefix_path = sealed_resource_root / "Q0R2_FIXED_BATCH_PREFIX_BEFORE_OUTCOME.json"
    if not args.continue_mechanism_only:
        write_new_json(binding_path, binding)
        write_new_json(contract_snapshot_path, contract)
        write_new_json(prefix_path, prefix_contract)
        for path in (binding_path, contract_snapshot_path, prefix_path):
            path.chmod(0o444)
    else:
        if json.loads(contract_snapshot_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("pre-outcome Q2 contract snapshot drift")
        if bytes_sha256(prefix_path.read_bytes()) != expected_prefix_sha256:
            raise RuntimeError("pre-outcome Q0R2 prefix snapshot drift")

    run_candidate.install_optional_dependency_stubs()
    run_candidate.patch_recbole_runtime_compat()
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.trainer import Trainer
    from recbole.utils import init_seed
    from recclaw_ext.models.composable_v2 import LightGCNComposableV2

    candidate_class = _load_candidate_class(selected["candidate_source_path"])
    common_config = {
        "benchmark_filename": ["train", "dev", "dev"],
        "checkpoint_dir": str(output_root / "checkpoints"),
        "data_path": str(Path(args.data_path).resolve()),
        "epochs": int(contract["resource_probe"]["epochs"]),
        "eval_step": 1,
        "reproducibility": True,
        "seed": int(contract["resource_probe"]["seed"]),
        "show_progress": False,
        "state": "ERROR",
        "stopping_step": int(contract["resource_probe"]["epochs"]),
        "use_gpu": True,
    }
    task_configs = [
        repo_root / "configs/task_ml1m.yaml",
        repo_root / "configs/lightgcn_metrics.yaml",
    ]
    candidate_config = Config(
        model="BPR",
        dataset="ml-1m",
        config_file_list=[
            str(recbole_root / "recbole/properties/model/BPR.yaml"),
            *(str(path) for path in task_configs),
        ],
        config_dict=common_config,
    )
    init_seed(candidate_config["seed"], candidate_config["reproducibility"])
    dataset = create_dataset(candidate_config)
    train_data, valid_data, _unused_development_alias = data_preparation(
        candidate_config, dataset
    )
    init_seed(candidate_config["seed"], candidate_config["reproducibility"])
    train_batches = _preallocate_batches(
        train_data, tuple(int(value) for value in prefix_contract["train_batch_indices"])
    )
    valid_batches: tuple[Any, ...] = ()
    if not args.continue_mechanism_only:
        init_seed(candidate_config["seed"], candidate_config["reproducibility"])
        valid_batches = _preallocate_batches(
            valid_data,
            tuple(int(value) for value in prefix_contract["eval_batch_indices"]),
        )
    init_seed(candidate_config["seed"], candidate_config["reproducibility"])
    model = candidate_class(candidate_config, train_data._dataset).to(
        candidate_config["device"]
    )
    if args.continue_mechanism_only:
        (
            resource_result,
            resource_result_sha256,
            result_binding_path,
            sealed_resource_snapshot,
        ) = (
            _load_prior_resource_for_continuation(
                output_root=output_root,
                sealed_resource_root=sealed_resource_root,
                repo_root=repo_root,
                contract_sha256=contract_sha256,
                import_gate_path=import_gate_path,
            )
        )
        resource_status = str(resource_result["status"])
    else:
        trainer = Trainer(candidate_config, model)
        telemetry_path = output_root / "RESOURCE_TELEMETRY.json"
        worker_started_ns = time.monotonic_ns()
        telemetry, restore = _install_resource_telemetry(
            trainer,
            torch=__import__("torch"),
            train_data=train_data,
            valid_data=valid_data,
            telemetry_path=telemetry_path,
            prefix_contract=prefix_contract,
            preallocated_train_batches=train_batches,
            preallocated_valid_batches=valid_batches,
            worker_started_ns=worker_started_ns,
        )
        resource_error: dict[str, Any] | None = None
        resource_status = "SUCCESS"
        deadline = int(contract["resource_probe"]["deadline_seconds"])
        previous_handler = signal.getsignal(signal.SIGALRM)

        def deadline_expired(_signum: int, _frame: Any) -> None:
            raise ResourceDeadlineExpired(
                f"Q0R2 resource deadline expired after {deadline}s"
            )

        signal.signal(signal.SIGALRM, deadline_expired)
        signal.setitimer(signal.ITIMER_REAL, deadline)
        try:
            trainer.fit(train_data, valid_data, saved=False, show_progress=False)
        except ResourceDeadlineExpired as error:
            resource_status = "RESOURCE_CENSORED"
            resource_error = {
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        except Exception as error:  # feasibility remains separate from mechanism
            resource_status = "RESOURCE_DEFERRED"
            resource_error = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            restore()
        telemetry = _finalize_resource_telemetry(telemetry)
        resource_result = {
            "schema": "recclaw.research-line.q2-resource-probe-result.v1",
            "status": resource_status,
            "feasibility_status": (
                "RESOURCE_FEASIBLE"
                if resource_status == "SUCCESS"
                else "RESOURCE_DEFERRED"
            ),
            "execution_purpose": "RESOURCE_PROBE_ONLY",
            "epochs_requested": int(contract["resource_probe"]["epochs"]),
            "epochs_completed": int(telemetry.get("epochs_completed", 0)),
            "deadline_seconds": deadline,
            "held_out_reads": 0,
            "mechanism_effect_updates": 0,
            "resource_failure_updates_mechanism_effect": False,
            "loss_trend": telemetry.get("loss_trend", []),
            "throughput_and_memory": _throughput_summary(telemetry),
            "telemetry_sha256": bytes_sha256(telemetry_path.read_bytes()),
            "failure": resource_error,
        }
        resource_result_path = output_root / "RESOURCE_PROBE_RESULT.json"
        resource_result_sha256 = write_new_json(resource_result_path, resource_result)
        result_binding_path = binding_path

    # Cheap mechanism probes deliberately run only after the resource probe.
    probe_interaction = _origin_blind_probe_interaction(model, train_batches, contract)
    probe_interaction = probe_interaction.to(candidate_config["device"])
    parent_config = Config(
        model="LightGCN",
        dataset="ml-1m",
        config_file_list=[
            str(recbole_root / "recbole/properties/model/LightGCN.yaml"),
            *(str(path) for path in task_configs),
        ],
        config_dict={
            **common_config,
            **contract["declared_mechanism"]["closest_parent_config"],
        },
    )
    parent = LightGCNComposableV2(parent_config, train_data._dataset).to(
        parent_config["device"]
    )
    statistics = {
        "loss_participation": _loss_statistics(model, probe_interaction),
        "gate_activation": _gate_statistics(model, probe_interaction, contract),
        "routing_and_propagation": _routing_statistics(model, probe_interaction),
        "target_conditioning": _target_conditioning_statistics(model, contract),
        "mechanism_off_parent_equivalence": _parent_equivalence_statistics(
            model, parent, probe_interaction
        ),
        "discriminative_prediction": (
            {
                "eligible_example_count": 0,
                "learned_removal_mean_abs_margin_damage": None,
                "control_removal_mean_abs_margin_damage": None,
                "assessment": "NOT_ASSESSED_PROTOCOL_NO_CHECKPOINT",
                "reason": (
                    "resource-only training used saved=False and the continued "
                    "process has a fresh untrained model"
                ),
            }
            if args.continue_mechanism_only
            else _discriminative_statistics(model, probe_interaction, contract)
        ),
    }
    evidence = evaluate_probe_statistics(statistics, contract)
    if args.continue_mechanism_only:
        for name in ("loss_participation", "routing_and_propagation"):
            evidence[name]["evidence_scope"] = "UNTRAINED_STRUCTURAL_ONLY"
            evidence[name]["mechanism_effect_assessed"] = False
        gate_structural_status = evidence["gate_activation"]["status"]
        evidence["gate_activation"].update(
            {
                "status": "NOT_ASSESSED_PROTOCOL_NO_CHECKPOINT",
                "structural_variability_status": gate_structural_status,
                "evidence_scope": "UNTRAINED_STRUCTURAL_ONLY",
                "learned_activation_assessed": False,
                "mechanism_effect_assessed": False,
            }
        )
        for name in (
            "target_conditioning",
            "mechanism_off_parent_equivalence",
        ):
            evidence[name]["evidence_scope"] = "UNTRAINED_EQUIVALENCE_ONLY"
            evidence[name]["mechanism_effect_assessed"] = False
        evidence["discriminative_prediction"].update(
            {
                "status": "NOT_ASSESSED_PROTOCOL_NO_CHECKPOINT",
                "evidence_scope": "NOT_ASSESSED",
                "mechanism_effect_assessed": False,
            }
        )
    state = classify_mechanism_evidence(evidence)
    sealed_resource_attestation_sha256 = None
    if args.continue_mechanism_only:
        sealed_resource_snapshot_after = _sealed_resource_snapshot(
            sealed_resource_root
        )
        if sealed_resource_snapshot_after != sealed_resource_snapshot:
            raise RuntimeError("sealed resource root changed during continuation")
        sealed_resource_attestation_sha256 = write_new_json(
            output_root / "SEALED_RESOURCE_ROOT_READ_ONLY_ATTESTATION.json",
            {
                "schema": (
                    "recclaw.research-line.q2-sealed-resource-read-only-attestation.v1"
                ),
                "status": "UNCHANGED",
                "development_only": True,
                "sealed_resource_root_writes": 0,
                "before_equals_after": True,
                "files": sealed_resource_snapshot_after,
            },
        )
    effect_update_allowed = (
        not args.continue_mechanism_only
        and state in {"ACTIVE_SUPPORTED", "ACTIVE_CONTRADICTED"}
    )
    physical_result = {
        "schema": "recclaw.research-line.q2-mechanism-physical-result.v1",
        "status": "MECHANISM_STATE_DECIDED",
        "development_only": True,
        "scientific_effect_claim": False,
        "held_out_reads": 0,
        "q2_contract_sha256": contract_sha256,
        "input_binding_sha256": bytes_sha256(result_binding_path.read_bytes()),
        "candidate_package_digest": contract["input"]["candidate_package_digest"],
        "resource_probe_result_sha256": resource_result_sha256,
        "resource_status": resource_status,
        "resource_updates_mechanism_effect": False,
        "mechanism_statistics": statistics,
        "mechanism_evidence": evidence,
        "mechanism_state": state,
        "mechanism_effect_update_allowed": effect_update_allowed,
        "model_state": (
            "FRESH_UNTRAINED_AFTER_RESOURCE_ONLY_NO_CHECKPOINT"
            if args.continue_mechanism_only
            else "IN_PROCESS_RESOURCE_PROBE_MODEL"
        ),
        "protocol_consumer_missingness": (
            "NOT_ASSESSED_PROTOCOL_NO_CHECKPOINT"
            if args.continue_mechanism_only
            else None
        ),
        "protocol_consumer_missingness_is_resource_failure": False,
        "resource_probe_executions": 1,
        "resource_probe_reexecuted": False,
        "sealed_resource_root_writes": 0,
        "sealed_resource_read_only_attestation_sha256": (
            sealed_resource_attestation_sha256
        ),
        "full_ablation_eligible": full_ablation_allowed(
            state=state, evidence=evidence, resource_status=resource_status
        ),
        "full_ablation_executed": False,
        "initial_mechanism_probe_failures": int(args.continue_mechanism_only),
        "general_caller_fixes": int(args.continue_mechanism_only),
        "retries": 0,
    }
    physical_path = output_root / "Q2_PHYSICAL_RESULT.json"
    physical_sha256 = write_new_json(physical_path, physical_result)
    q3 = q3_evidence_package(
        contract_sha256=contract_sha256,
        physical_result_sha256=physical_sha256,
        state=state,
        evidence=evidence,
        resource_status=resource_status,
        full_ablation_executed=False,
    )
    q3.update(
        {
            "mechanism_effect_update_allowed": effect_update_allowed,
            "protocol_consumer_missingness": physical_result[
                "protocol_consumer_missingness"
            ],
            "protocol_consumer_missingness_is_resource_failure": False,
            "model_state": physical_result["model_state"],
            "negative_evidence_is_q3_consumable": state
            in {"INACTIVE", "ACTIVE_CONTRADICTED", "NON_IDENTIFIABLE"},
        }
    )
    write_new_json(output_root / "Q3_MECHANISM_EVIDENCE_PACKAGE.json", q3)
    print(
        json.dumps(
            {
                "mechanism_state": state,
                "resource_status": resource_status,
                "full_ablation_eligible": physical_result["full_ablation_eligible"],
                "output_root": str(output_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
