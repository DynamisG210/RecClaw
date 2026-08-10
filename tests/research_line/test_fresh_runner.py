"""Contract tests for the thin Research Line to fresh-run adapter."""

from __future__ import annotations

import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    COMMON_DATASET,
    COMMON_EVALUATOR,
    COMMON_SPLIT,
    ExperimentBindingV1,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    SearchCandidateBindingV1,
    SearchProfileEntryOriginV1,
)
from recclaw_core.research_line import fresh_runner as fresh_runner_module
from recclaw_core.research_line.fresh_runner import (
    FreshRunnerError,
    make_fresh_runner,
)
from recclaw_core.research_line.gpu_reservation_provider import (
    make_nvidia_smi_gpu_reservation_provider,
)


_HELIX_TEST_ROOT = Path(__file__).resolve().parents[1] / "experiments" / "helix_abc_v1"
sys.path.insert(0, str(_HELIX_TEST_ROOT))

from test_e0_search_adapter import _proposal  # noqa: E402


def _digest(label: str) -> str:
    return sha256_digest({"fresh_runner_test": label})


def _fixed_lightgcn() -> tuple[dict[str, Any], SearchCandidateBindingV1]:
    proposal = _proposal(
        candidate_id="cand-fresh-lightgcn",
        mechanism_id="LIGHTGCN",
        mechanism_axis="graph_propagation",
        mechanism_program={"program": "fixed-lightgcn"},
        protocol_digest=_digest("fixed-protocol"),
    )
    binding = SearchCandidateBindingV1(
        proposal=proposal,
        capability_ref="cap:fixed-lightgcn",
        capability_digest=_digest("fixed-capability"),
        executable_entrypoint="recbole.model.general_recommender.lightgcn:LightGCN",
        entry_origin=SearchProfileEntryOriginV1.FIXED_66,
        mechanism_semantics_digest=_digest("fixed-semantics"),
    )
    return {
        "capability_family": "LIGHTGCN",
        "capability_ref": binding.capability_ref,
        "capability_digest": binding.capability_digest,
        "profile_ref": "profile:fixed-66",
        "profile_digest": _digest("fixed-profile"),
        "mechanism_id": proposal.mechanism_id,
        "mechanism_semantics_digest": binding.mechanism_semantics_digest,
        "entrypoint": binding.executable_entrypoint,
        "entrypoint_source_sha256": _digest("fixed-entrypoint-source"),
        "model": "LightGCN",
        "base_model_config": "LightGCN",
        "config": {},
        "dataset": COMMON_DATASET,
        "split": COMMON_SPLIT,
        "evaluator": dict(COMMON_EVALUATOR),
        "execution_role": "CANDIDATE",
    }, binding


def _qualified_custom() -> tuple[dict[str, Any], SearchCandidateBindingV1]:
    proposal = _proposal(
        candidate_id="cand-fresh-qualified-custom",
        mechanism_id="QUALIFIED_CUSTOM",
        mechanism_axis="message_transform",
        mechanism_program={"program": "qualified-custom"},
        protocol_digest=_digest("protocol"),
    )
    binding = SearchCandidateBindingV1(
        proposal=proposal,
        capability_ref="cap:qualified-custom",
        capability_digest=_digest("qualified-capability"),
        executable_entrypoint="recclaw_ext.generated.custom:CustomFamilyModel",
        entry_origin=SearchProfileEntryOriginV1.QUALIFIED_REGISTRY,
        mechanism_semantics_digest=_digest("qualified-semantics"),
    )
    recipe = {
        "capability_family": "QUALIFIED_CUSTOM_FAMILY",
        "capability_ref": binding.capability_ref,
        "capability_digest": binding.capability_digest,
        "profile_ref": "profile:fresh-runner-test",
        "profile_digest": _digest("profile"),
        "mechanism_id": proposal.mechanism_id,
        "mechanism_semantics_digest": binding.mechanism_semantics_digest,
        "entrypoint": binding.executable_entrypoint,
        "entrypoint_source_sha256": _digest("entrypoint-source"),
        "model": "CustomFamilyModel",
        "base_model_config": "NGCF",
        "config": {"custom_gain": 1},
        "dataset": COMMON_DATASET,
        "split": COMMON_SPLIT,
        "evaluator": dict(COMMON_EVALUATOR),
        "execution_role": "CANDIDATE",
        "candidate_package_ref": "package:qualified-custom",
        "candidate_package_digest": _digest("package"),
        "candidate_root_ref": "root:qualified-custom",
        "candidate_root_digest": _digest("root"),
        "candidate_source_tree_digest": _digest("source-tree"),
    }
    return recipe, binding


def _fake_launch(calls: list[dict[str, Any]]):
    def launch(**kwargs: Any) -> dict[str, Any]:
        calls.append(dict(kwargs))
        recipe = kwargs["execution_recipe"]
        experiment_binding = ExperimentBindingV1.from_execution_recipe(
            recipe,
            candidate_root=kwargs["candidate_root"],
            dataset_manifest_digest=_digest("dataset-manifest"),
            seed=kwargs["seed"],
            epochs=kwargs["epochs"],
            timeout_seconds=kwargs["timeout_seconds"],
            execution_purpose=kwargs["execution_purpose"],
            resource_telemetry=kwargs["resource_telemetry"],
            watchdog_seconds=kwargs["watchdog_seconds"],
            prefix_contract_digest=None,
            run_id=kwargs["run_id"],
            round_id="round:fresh-runner-test",
            claim_id="claim:fresh-runner-test",
            permit_digest=_digest("permit"),
            runtime_binding_digest=_digest("runtime-binding"),
            runtime_release_digest=_digest("runtime-release"),
            runner_abi="test.fresh-runner.v1",
            filesystem_capability_digest=_digest("filesystem-capability"),
        )
        result = {
            "exit_status": "SUCCESS",
            "metrics": {"ndcg@10": 0.5},
            "experiment_binding": experiment_binding.canonical_dict(),
            "experiment_binding_ref": experiment_binding.ref,
            "experiment_binding_digest": experiment_binding.digest,
            "binding_digest": experiment_binding.digest,
            "wall_time_ms": 7,
            "seed": experiment_binding.seed,
            "execution_recipe_digest": experiment_binding.execution_recipe_digest,
        }
        if kwargs.get("gpu_id") is not None:
            result.update(
                {
                    "gpu_id": kwargs["gpu_id"],
                    "physical_gpu_id": str(kwargs["gpu_id"]),
                    "selection_mode": (
                        fresh_runner_module.DIRECT_GPU_SELECTION_MODE
                    ),
                    "cuda_visible_devices": None,
                }
            )
        if kwargs["expected_recbole_source_tree_digest"] is not None:
            result["recbole_source_identity"] = {
                "source_tree_digest": kwargs[
                    "expected_recbole_source_tree_digest"
                ]
            }
        return result

    return launch


def test_one_adapter_preserves_fixed_and_qualified_family_and_run_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_recipe, fixed_binding = _fixed_lightgcn()
    qualified_recipe, qualified_binding = _qualified_custom()
    qualified_root = tmp_path / "qualified" / "capability"
    (qualified_root / "recclaw_ext").mkdir(parents=True)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        fresh_runner_module,
        "run_development_training",
        _fake_launch(calls),
    )
    runner = make_fresh_runner(
        repo_root=tmp_path / "repo",
        side_root=tmp_path / "side",
        run_id="fresh-run-007",
        seed=2026,
        epochs=3,
        timeout_seconds=77,
        execution_purpose="DEVELOPMENT_PILOT_OFFLINE_TOPN",
        candidate_root_by_capability={qualified_binding.capability_ref: qualified_root},
    )

    fixed_result = runner(fixed_recipe, fixed_binding)
    qualified_result = runner(qualified_recipe, qualified_binding)

    assert len(calls) == 2
    assert calls[0]["candidate_root"] is None
    assert calls[1]["candidate_root"] == qualified_root.resolve()
    for call, recipe in zip(calls, (fixed_recipe, qualified_recipe), strict=True):
        assert call["repo_root"] == (tmp_path / "repo").resolve()
        assert call["side_root"] == (tmp_path / "side").resolve()
        assert call["run_id"] == "fresh-run-007"
        assert call["seed"] == 2026
        assert call["epochs"] == 3
        assert call["timeout_seconds"] == 77
        assert call["execution_purpose"] == "DEVELOPMENT_PILOT_OFFLINE_TOPN"
        assert call["run_identity"] == "recclaw-research-line-v1"
        assert call["authority"] == (
            "user-delegated-research-line-physical-execution"
        )
        assert call["resource_telemetry"] is False
        assert call["watchdog_seconds"] is None
        assert call["entrypoint"] == recipe["entrypoint"]
        assert call["source_sha256"] == recipe["entrypoint_source_sha256"]
        assert call["execution_recipe"] == recipe

    fixed_experiment_binding = ExperimentBindingV1.from_canonical_dict(
        fixed_result["experiment_binding"]
    )
    qualified_experiment_binding = ExperimentBindingV1.from_canonical_dict(
        qualified_result["experiment_binding"]
    )
    assert fixed_experiment_binding.model == fixed_recipe["model"] == "LightGCN"
    assert fixed_experiment_binding.candidate_root_path is None
    assert qualified_experiment_binding.model == qualified_recipe["model"]
    assert qualified_experiment_binding.capability_family == "QUALIFIED_CUSTOM_FAMILY"
    assert qualified_experiment_binding.candidate_root_path == str(
        qualified_root.resolve()
    )
    assert qualified_experiment_binding.model != "BPR"


def test_qualified_capability_without_local_root_fails_before_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recipe, binding = _qualified_custom()
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        fresh_runner_module,
        "run_development_training",
        _fake_launch(calls),
    )
    runner = make_fresh_runner(
        repo_root=tmp_path / "repo",
        side_root=tmp_path / "side",
        run_id="fresh-run-missing-root",
        seed=2026,
        epochs=3,
        timeout_seconds=77,
        execution_purpose="DEVELOPMENT_PILOT_OFFLINE_TOPN",
        candidate_root_by_capability={},
    )

    with pytest.raises(FreshRunnerError, match="candidate-local root"):
        runner(recipe, binding)

    assert calls == []


def test_physical_runtime_source_and_resource_inputs_reach_binding(
    tmp_path: Path,
) -> None:
    recipe, binding = _fixed_lightgcn()
    source_digest = _digest("recbole-source-tree")
    calls: list[dict[str, Any]] = []
    runner = make_fresh_runner(
        repo_root=tmp_path / "repo",
        side_root=tmp_path / "side",
        run_id="fresh-run-physical-inputs",
        seed=2026,
        epochs=3,
        timeout_seconds=77,
        execution_purpose="DEVELOPMENT_PILOT_OFFLINE_TOPN",
        candidate_root_by_capability={},
        recbole_commit_identity="7b02be5ec80a88310f2d04a27a82adfcbb5dc211",
        expected_recbole_source_tree_digest=source_digest,
        resource_telemetry=True,
        watchdog_seconds=71,
        cuda_visible_devices="1",
        launch=_fake_launch(calls),
    )

    result = runner(recipe, binding)
    experiment_binding = ExperimentBindingV1.from_canonical_dict(
        result["experiment_binding"]
    )

    assert calls[0]["recbole_commit_identity"] == (
        "7b02be5ec80a88310f2d04a27a82adfcbb5dc211"
    )
    assert calls[0]["expected_recbole_source_tree_digest"] == source_digest
    assert calls[0]["cuda_visible_devices"] == "1"
    assert calls[0]["final_worker_ceiling_seconds"] == 3600
    assert experiment_binding.resource_telemetry is True
    assert experiment_binding.watchdog_seconds == 71


def test_direct_gpu_id_reaches_launcher_without_cvd_and_preserves_identity(
    tmp_path: Path,
) -> None:
    recipe, binding = _fixed_lightgcn()
    calls: list[dict[str, Any]] = []
    runner = make_fresh_runner(
        repo_root=tmp_path / "repo",
        side_root=tmp_path / "side",
        run_id="fresh-run-direct-gpu",
        seed=2026,
        epochs=3,
        timeout_seconds=77,
        execution_purpose="DEVELOPMENT_PILOT_OFFLINE_TOPN",
        candidate_root_by_capability={},
        gpu_id=2,
        launch=_fake_launch(calls),
    )

    result = runner.run_with_physical_context(
        recipe,
        binding,
        physical_context=_physical_context(
            binding,
            round_index=1,
            attempt_index=0,
            seed=2026,
        ),
    )

    assert calls[0]["gpu_id"] == 2
    assert calls[0]["cuda_visible_devices"] is None
    assert result["gpu_id"] == 2
    assert result["physical_gpu_id"] == "2"
    assert result["selection_mode"] == fresh_runner_module.DIRECT_GPU_SELECTION_MODE
    assert result["physical_identity"]["gpu_id"] == 2
    assert result["physical_identity"]["physical_gpu_id"] == "2"
    assert result["physical_identity"]["selection_mode"] == (
        fresh_runner_module.DIRECT_GPU_SELECTION_MODE
    )
    assert result["physical_identity"]["cuda_visible_devices"] is None


def test_gpu_id_and_legacy_cvd_are_mutually_exclusive_before_launch(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, Any]] = []

    with pytest.raises(FreshRunnerError, match="mutually exclusive"):
        make_fresh_runner(
            repo_root=tmp_path / "repo",
            side_root=tmp_path / "side",
            run_id="fresh-run-direct-conflict",
            seed=2026,
            epochs=3,
            timeout_seconds=77,
            execution_purpose="DEVELOPMENT_PILOT_OFFLINE_TOPN",
            candidate_root_by_capability={},
            cuda_visible_devices="2",
            gpu_id=2,
            launch=_fake_launch(calls),
        )

    assert calls == []


@pytest.mark.parametrize("invalid_gpu_id", [-1, True, "2"])
def test_direct_gpu_id_rejects_invalid_selectors(
    tmp_path: Path,
    invalid_gpu_id: object,
) -> None:
    with pytest.raises(FreshRunnerError, match="gpu_id"):
        make_fresh_runner(
            repo_root=tmp_path / "repo",
            side_root=tmp_path / "side",
            run_id="fresh-run-invalid-gpu",
            seed=2026,
            epochs=3,
            timeout_seconds=77,
            execution_purpose="DEVELOPMENT_PILOT_OFFLINE_TOPN",
            candidate_root_by_capability={},
            gpu_id=invalid_gpu_id,  # type: ignore[arg-type]
        )


def _physical_context(
    binding: SearchCandidateBindingV1,
    *,
    round_index: int,
    attempt_index: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema": "recclaw.research-line.physical-execution-context.v1",
        "campaign_id": "campaign:fresh-physical-context",
        "round_index": round_index,
        "opportunity_ref": f"opportunity:{round_index}",
        "attempt_index": attempt_index,
        "candidate_id": binding.proposal.candidate_id,
        "binding_digest": binding.digest,
        "candidate_semantic_digest": binding.mechanism_semantics_digest,
        "research_context_digest": _digest(f"context:{round_index}"),
        "profile_digest": _digest("profile:physical-context"),
        "seed": str(seed),
    }


def test_context_aware_calls_allocate_unique_physical_run_ids_without_recipe_drift(
    tmp_path: Path,
) -> None:
    recipe, binding = _fixed_lightgcn()
    calls: list[dict[str, Any]] = []
    runner = make_fresh_runner(
        repo_root=tmp_path / "repo",
        side_root=tmp_path / "side",
        run_id="fresh-run-context",
        seed=54303,
        epochs=1,
        timeout_seconds=77,
        execution_purpose="DEVELOPMENT_PILOT_OFFLINE_TOPN",
        candidate_root_by_capability={},
        launch=_fake_launch(calls),
    )
    contexts = (
        _physical_context(binding, round_index=1, attempt_index=0, seed=61001),
        _physical_context(binding, round_index=1, attempt_index=1, seed=61001),
        _physical_context(binding, round_index=2, attempt_index=0, seed=61001),
    )

    results = tuple(
        runner.run_with_physical_context(
            recipe,
            binding,
            physical_context=context,
        )
        for context in contexts
    )

    run_ids = tuple(result["physical_identity"]["run_id"] for result in results)
    assert len(set(run_ids)) == 3
    assert all(run_id.startswith("fresh-run-context:physical:") for run_id in run_ids)
    assert tuple(call["run_id"] for call in calls) == run_ids
    assert tuple(call["seed"] for call in calls) == (61001, 61001, 61001)
    assert all(
        result["experiment_binding"]["execution_recipe_digest"]
        == sha256_digest(recipe)
        for result in results
    )


def test_context_aware_path_rejects_sentinel_worker_seed_and_reaches_gpu_identity(
    tmp_path: Path,
) -> None:
    recipe, binding = _fixed_lightgcn()
    calls: list[dict[str, Any]] = []
    runner = make_fresh_runner(
        repo_root=tmp_path / "repo",
        side_root=tmp_path / "side",
        run_id="fresh-run-gpu-identity",
        seed=54303,
        epochs=1,
        timeout_seconds=77,
        execution_purpose="DEVELOPMENT_PILOT_OFFLINE_TOPN",
        candidate_root_by_capability={},
        cuda_visible_devices="2",
        final_worker_ceiling_seconds=1234,
        launch=_fake_launch(calls),
    )
    context = _physical_context(
        binding,
        round_index=1,
        attempt_index=0,
        seed=62001,
    )
    result = runner.run_with_physical_context(
        recipe,
        binding,
        physical_context=context,
    )

    assert calls[0]["cuda_visible_devices"] == "2"
    assert calls[0]["final_worker_ceiling_seconds"] == 1234
    assert result["physical_identity"]["cuda_visible_devices"] == "2"
    assert result["physical_identity"]["final_worker_ceiling_seconds"] == 1234
    assert result["physical_identity"]["reservation_status"] == (
        fresh_runner_module.GPU_RESERVATION_STATUS_UNMEASURED
    )
    sentinel_context = {**context, "seed": "NEXT_DEVELOPMENT_SEED"}
    with pytest.raises(FreshRunnerError, match="not a worker seed"):
        runner.run_with_physical_context(
            recipe,
            binding,
            physical_context=sentinel_context,
        )
    assert len(calls) == 1


def test_required_context_provider_receives_flag_and_evidence_reaches_launch(
    tmp_path: Path,
) -> None:
    recipe, binding = _fixed_lightgcn()
    calls: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []

    def command_runner(argv: tuple[str, ...], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if "--query-compute-apps" in argv:
            stdout = ""
        else:
            stdout = (
                "2, GPU-test-uuid, NVIDIA GeForce RTX 4090, "
                "00000000:02:00.0, 24564, Default\n"
            )
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    collector = make_nvidia_smi_gpu_reservation_provider(
        cuda_visible_devices="2",
        reservation_authority_ref="sealed:gpu5:gpu2:reservation-001",
        command_runner=command_runner,
        hostname=socket.gethostname(),
        clock=lambda: datetime(2026, 8, 10, 1, 2, 3, tzinfo=timezone.utc),
    )

    def provider(request: Any) -> Any:
        requests.append(dict(request))
        return collector(request)

    provider.provider_identity = collector.provider_identity  # type: ignore[attr-defined]
    runner = make_fresh_runner(
        repo_root=tmp_path / "repo",
        side_root=tmp_path / "side",
        run_id="fresh-run-required",
        seed=54303,
        epochs=1,
        timeout_seconds=77,
        execution_purpose="DEVELOPMENT_PILOT_OFFLINE_TOPN",
        candidate_root_by_capability={},
        cuda_visible_devices="2",
        require_gpu_reservation_evidence=True,
        gpu_reservation_evidence_provider=provider,
        launch=_fake_launch(calls),
    )
    context = _physical_context(
        binding,
        round_index=1,
        attempt_index=0,
        seed=62001,
    )

    result = runner.run_with_physical_context(recipe, binding, context)
    physical_run_id = result["physical_identity"]["run_id"]

    assert requests[0]["physical_run_id"] == physical_run_id
    assert requests[0]["cuda_visible_devices"] == "2"
    assert requests[0]["require_gpu_reservation_evidence"] is True
    assert calls[0]["run_id"] == physical_run_id
    assert calls[0]["seed"] == 62001
    evidence = calls[0]["gpu_reservation_evidence"]
    assert evidence["identity"]["reservation_owner_ref"] == physical_run_id
    assert result["physical_identity"]["reservation_digest"] == (
        evidence["reservation_digest"]
    )
    assert runner.gpu_reservation_provider_identity() == (
        collector.provider_identity()
    )


def test_required_context_rejects_missing_provider_or_device_before_launch(
    tmp_path: Path,
) -> None:
    recipe, binding = _fixed_lightgcn()
    context = _physical_context(
        binding,
        round_index=1,
        attempt_index=0,
        seed=62001,
    )
    calls: list[dict[str, Any]] = []
    runner_without_provider = make_fresh_runner(
        repo_root=tmp_path / "repo",
        side_root=tmp_path / "side",
        run_id="fresh-run-required-no-provider",
        seed=54303,
        epochs=1,
        timeout_seconds=77,
        execution_purpose="DEVELOPMENT_PILOT_OFFLINE_TOPN",
        candidate_root_by_capability={},
        cuda_visible_devices="2",
        require_gpu_reservation_evidence=True,
        launch=_fake_launch(calls),
    )
    with pytest.raises(FreshRunnerError, match="needs a provider"):
        runner_without_provider.run_with_physical_context(
            recipe,
            binding,
            context,
        )

    runner_without_device = make_fresh_runner(
        repo_root=tmp_path / "repo",
        side_root=tmp_path / "side",
        run_id="fresh-run-required-no-device",
        seed=54303,
        epochs=1,
        timeout_seconds=77,
        execution_purpose="DEVELOPMENT_PILOT_OFFLINE_TOPN",
        candidate_root_by_capability={},
        require_gpu_reservation_evidence=True,
        gpu_reservation_evidence_provider=lambda request: None,
        launch=_fake_launch(calls),
    )
    with pytest.raises(FreshRunnerError, match="explicit"):
        runner_without_device.run_with_physical_context(
            recipe,
            binding,
            context,
        )
    assert calls == []


def test_required_context_rejects_provider_returning_none_without_launch(
    tmp_path: Path,
) -> None:
    recipe, binding = _fixed_lightgcn()
    calls: list[dict[str, Any]] = []
    runner = make_fresh_runner(
        repo_root=tmp_path / "repo",
        side_root=tmp_path / "side",
        run_id="fresh-run-required-empty",
        seed=54303,
        epochs=1,
        timeout_seconds=77,
        execution_purpose="DEVELOPMENT_PILOT_OFFLINE_TOPN",
        candidate_root_by_capability={},
        cuda_visible_devices="2",
        require_gpu_reservation_evidence=True,
        gpu_reservation_evidence_provider=lambda request: None,
        launch=_fake_launch(calls),
    )
    context = _physical_context(
        binding,
        round_index=1,
        attempt_index=0,
        seed=62001,
    )

    with pytest.raises(FreshRunnerError, match="returned no evidence"):
        runner.run_with_physical_context(recipe, binding, context)
    assert calls == []
