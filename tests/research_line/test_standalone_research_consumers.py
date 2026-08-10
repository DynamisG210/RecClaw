from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Mapping

import pytest

from recclaw_core.experiments.helix_abc_v1 import fresh_r1
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.search_adapter import adapt_current_search_profile
from recclaw_core.research_line import standalone as standalone_module
from recclaw_core.research_line.single_round import ResearchBaselineSourceV1
from recclaw_core.research_line.standalone import (
    StandaloneCampaignError,
    StandaloneResearchConfig,
    compose_standalone_campaign,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from test_fresh_runner import _fake_launch  # noqa: E402
from test_provider import _call_result, _proposal_response  # noqa: E402


def _config(run_root: Path, *, round_count: int = 2) -> StandaloneResearchConfig:
    profile = adapt_current_search_profile(campaign_id="research:standalone-consumers")
    return StandaloneResearchConfig(
        repo_root=REPO_ROOT,
        run_root=run_root,
        api_config_source={
            "release_digest": "standalone-consumer-provider-fixture",
            "source_kind": "no-provider-test",
        },
        campaign_id="research:standalone-consumers",
        baseline_source=ResearchBaselineSourceV1.from_identity(
            source_ref="research-source:standalone-consumers",
            source_sha256="2" * 64,
            comparator_ref="research-baseline:standalone-consumers",
            comparator_digest="1" * 64,
            frozen_ndcg_at_10=0.4,
            protocol_digest=profile.protocol_digest,
            seed=54303,
        ),
        seed=54303,
        epochs=1,
        round_count=round_count,
        observation_seed_schedule=(54303, 54304) if round_count == 2 else None,
    )


def _implementation_response() -> dict[str, Any]:
    source = """import torch
from torch import nn

from recbole.model.general_recommender.bpr import BPR


class FreshCandidateModel(BPR):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.user_gate = nn.Linear(self.embedding_size, self.embedding_size)
        self.item_gate = nn.Linear(self.embedding_size, self.embedding_size)

    def _score(self, user_e, item_e):
        gate = torch.sigmoid(self.user_gate(user_e) + self.item_gate(item_e))
        return torch.mul(user_e * gate, item_e).sum(dim=-1)

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        positive = interaction[self.ITEM_ID]
        negative = interaction[self.NEG_ITEM_ID]
        user_e = self.user_embedding(user)
        positive_e = self.item_embedding(positive)
        negative_e = self.item_embedding(negative)
        return self.loss(
            self._score(user_e, positive_e),
            self._score(user_e, negative_e),
        )

    def predict(self, interaction):
        user_e = self.user_embedding(interaction[self.USER_ID])
        item_e = self.item_embedding(interaction[self.ITEM_ID])
        return self._score(user_e, item_e)

    def full_sort_predict(self, interaction):
        user_e = self.user_embedding(interaction[self.USER_ID])[:, None, :]
        item_e = self.item_embedding.weight[None, :, :]
        return self._score(user_e, item_e).reshape(-1)
"""
    return {
        "schema": "recclaw.research-line.fresh-r1-implementation-response.v1",
        "proposals": [
            {
                "entrypoint": "recclaw_ext.candidate:FreshCandidateModel",
                "files": [
                    {"path": "recclaw_ext/__init__.py", "content": "# fixture\n"},
                    {"path": "recclaw_ext/candidate.py", "content": source},
                ],
                "implementation_summary": "Standalone Research consumer fixture.",
            }
        ],
    }


def _provider_call(calls: list[dict[str, Any]]):
    def call(**kwargs: Any) -> fresh_r1.ProviderAttemptResult:
        logical_call_id = str(kwargs["logical_call_id"])
        calls.append(dict(kwargs))
        if ":implementation:" in logical_call_id:
            response = _implementation_response()
        else:
            role = next(
                role
                for role in (
                    "frontier_architect",
                    "mechanism_composer",
                    "failure_analyst",
                    "experiment_designer",
                )
                if f":{role}:proposal" in logical_call_id
            )
            response = _proposal_response(role)
            if "offline-replay:challenger:" in logical_call_id:
                proposal = dict(response["proposals"][0])
                proposal["mechanism_change"] = (
                    f"{proposal['mechanism_change']} ({role}-challenger)"
                )
                response = {**response, "proposals": [proposal]}
        return _call_result(logical_call_id=logical_call_id, response=response)

    return call


def _resource_profile(calls: list[dict[str, Any]], kwargs: Mapping[str, Any]) -> dict[str, Any]:
    candidate_ref = str(kwargs["candidate_ref"])
    candidate_package_digest = kwargs["candidate_package_digest"]
    source_sha256 = str(kwargs["source_sha256"])
    compute_pattern = str(kwargs["compute_pattern"])
    calls.append(dict(kwargs))
    identity = {
        "candidate_ref": candidate_ref,
        "candidate_package_digest": candidate_package_digest,
        "candidate_source_sha256": source_sha256,
        "compute_pattern": compute_pattern,
    }
    prediction = {
        "completion_probability": 0.9,
        "estimated_total_wall_time_seconds": 30.0,
        "model": "FIXED_BATCH_THROUGHPUT_LINEAR_EXTRAPOLATION_V3",
        "peak_memory_prediction_mib": 128.0,
        "prediction_interval_seconds": [20.0, 40.0],
    }
    return {
        **identity,
        "completion_probability": 0.9,
        "effect_fields_consumed": [],
        "full_run_budget_after_probes_seconds": float(
            kwargs["total_budget_seconds"]
        ),
        "held_out_reads": 0,
        "outcome_fields_consumed": [],
        "prediction": prediction,
        "prediction_interval_seconds": [20.0, 40.0],
        "probe": {
            "completed_eval_batches": 1,
            "completed_train_batches": 4,
            "peak_gpu_memory_mib": 128.0,
            "telemetry_present": True,
            "wall_time_ms": 17,
        },
        "probe_process": {
            "exit_code": 0,
            "process_isolated": True,
            "start_method": "spawn",
            "status": "RESULT",
        },
        "profile_digest": sha256_digest(identity),
        "status": "RESOURCE_ADMITTED",
    }


def _fake_resource_probe(calls: list[dict[str, Any]]):
    def probe(repo_root: Path, **kwargs: Any) -> dict[str, Any]:
        assert repo_root == REPO_ROOT
        assert kwargs["cuda_visible_devices"] == kwargs.get(
            "cuda_visible_devices"
        )
        return _resource_profile(calls, kwargs)

    return probe


def test_standalone_consumers_activate_innovation_and_meta_across_rounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        standalone_module,
        "run_disposable_fixed_batch_resource_probe",
        _fake_resource_probe(resource_calls),
    )
    provider_calls: list[dict[str, Any]] = []
    launch_calls: list[dict[str, Any]] = []
    config = _config(tmp_path / "campaign")
    composition = compose_standalone_campaign(
        config,
        provider_call=_provider_call(provider_calls),
        launch=_fake_launch(launch_calls),
    )

    records = composition.run(2)

    assert len(records) == 2
    first = records[0].result
    assert first.innovation is not None
    assert first.innovation.activation_ready
    capability = first.innovation.capability
    assert capability is not None
    capability_id = capability.capability_id
    state = composition.campaign.state
    assert capability_id in state.qualified_execution_by_capability
    assert capability_id in state.candidate_root_by_capability
    assert capability_id in (state.resource_profile_by_capability or {})
    assert state.active_profile.profile_ref != composition.profile.profile_ref
    profiles = state.resource_profile_by_capability or {}
    profile = profiles[capability_id]
    assert profile["compute_pattern"].startswith(
        standalone_module.RESOURCE_COMPUTE_PATTERN_SCHEMA + ":"
    )
    assert "predicted_gpu_worker_seconds" not in profile

    second = records[1].result
    next_inputs = standalone_module._round_input_factory(
        config,
        provider=composition.provider,
        implementer=composition.implementer,
        portfolio_candidates=config.portfolio_candidates,
    )(records[1].state_before)
    assert capability_id in next_inputs.qualified_execution_by_capability
    assert capability_id in next_inputs.candidate_root_by_capability
    assert capability_id in next_inputs.resource_profile_by_capability
    assert second.search_acquisition is not None

    assert first.meta_research is not None
    assert first.meta_research.shadow_evaluation.verdict == "PASS"
    assert first.meta_research.promotion_decision.verdict == "PROMOTE"
    assert first.meta_research.activation_receipt is not None
    assert first.meta_research.activated_policy is not None
    assert second.meta_research is None
    assert records[1].state_before.policy.digest == (
        first.meta_research.activated_policy.digest
    )
    assert resource_calls
    assert all(call["cuda_visible_devices"] is None for call in resource_calls)
    assert provider_calls


def test_standalone_resume_rejects_research_consumer_composition_drift(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "resume", round_count=1)
    compose_standalone_campaign(
        config,
        provider_call=lambda **_kwargs: pytest.fail("Provider boundary reached"),
        launch=lambda **_kwargs: pytest.fail("runner boundary reached"),
    )
    original_interval = standalone_module.META_REPLAY_INTERVAL
    standalone_module.META_REPLAY_INTERVAL = original_interval + 1
    try:
        with pytest.raises(StandaloneCampaignError, match="execution inputs"):
            compose_standalone_campaign(
                config,
                resume=True,
                provider_call=lambda **_kwargs: pytest.fail(
                    "Provider boundary reached"
                ),
                launch=lambda **_kwargs: pytest.fail("runner boundary reached"),
            )
    finally:
        standalone_module.META_REPLAY_INTERVAL = original_interval


@pytest.mark.parametrize(
    ("device_kwargs", "expected_cvd", "expected_gpu_id"),
    [
        ({"cuda_visible_devices": "0"}, "0", None),
        ({"gpu_id": 2}, None, 2),
        ({}, None, None),
    ],
    ids=("legacy-cvd", "direct-gpu-id", "no-gpu"),
)
def test_missing_reservation_evidence_is_unmeasured_and_pattern_is_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_kwargs: Mapping[str, Any],
    expected_cvd: str | None,
    expected_gpu_id: int | None,
) -> None:
    resource_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        standalone_module,
        "run_disposable_fixed_batch_resource_probe",
        _fake_resource_probe(resource_calls),
    )
    config = replace(
        _config(tmp_path / "unmeasured", round_count=1),
        final_worker_ceiling_seconds=1234,
        **device_kwargs,
    )
    inputs = standalone_module._production_innovation_inputs(
        config,
        lambda _request: {},
        campaign_id=config.campaign_id,
        seed=config.seed,
        round_index=1,
    )
    candidate_root = tmp_path / "candidate"
    source_path = candidate_root / "recclaw_ext" / "candidate.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("# local fixture\n", encoding="utf-8")
    recipe = {
        "model": "FreshCandidateModel",
        "base_model_config": "BPR",
        "config": {"embedding_size": 8},
        "entrypoint": "recclaw_ext.candidate:FreshCandidateModel",
        "dataset": "mini",
        "split": {"strategy": "holdout", "ordering": "timestamp"},
        "evaluator": {"metrics": ["NDCG"], "topk": [10]},
        "execution_role": "CANDIDATE",
    }

    profile = inputs.resource_probe(
        candidate_root=candidate_root,
        source_path=source_path,
        entrypoint=recipe["entrypoint"],
        source_sha256="a" * 64,
        execution_recipe=recipe,
        probe_root=tmp_path / "probe",
        candidate_ref="capability:one",
        candidate_package_digest="b" * 64,
    )

    assert resource_calls[0]["cuda_visible_devices"] == expected_cvd
    assert resource_calls[0]["gpu_id"] == expected_gpu_id
    assert resource_calls[0]["gpu_reservation_evidence"] is None
    assert resource_calls[0]["total_budget_seconds"] == 1234
    assert "predicted_gpu_worker_seconds" not in profile
    pattern = profile["compute_pattern"]
    assert pattern.startswith(standalone_module.RESOURCE_COMPUTE_PATTERN_SCHEMA + ":")
    assert standalone_module._resource_compute_pattern(
        {**recipe, "capability_ref": "different", "candidate_source_tree_digest": "c" * 64}
    ) == pattern


def test_reservation_mismatch_fails_before_resource_probe_launch(
    tmp_path: Path,
) -> None:
    probe_calls: list[dict[str, Any]] = []
    evidence_requests: list[Mapping[str, Any]] = []

    def evidence_provider(request: Mapping[str, Any]) -> None:
        evidence_requests.append(request)
        return {
            "schema": fresh_r1.GPU_RESERVATION_EVIDENCE_SCHEMA,
            "reservation_ref": "reservation:wrong-device",
            "identity": {
                "host": "test-host",
                "physical_gpu_id": "2",
                "cuda_visible_devices": "0",
            },
        }

    config = replace(
        _config(tmp_path / "mismatch", round_count=1),
        cuda_visible_devices="0",
        gpu_reservation_evidence_provider=evidence_provider,
    )
    inputs = standalone_module._production_innovation_inputs(
        config,
        lambda _request: {},
        campaign_id=config.campaign_id,
        seed=config.seed,
        round_index=1,
    )
    recipe = {
        "model": "FreshCandidateModel",
        "base_model_config": "BPR",
        "config": {"embedding_size": 8},
        "entrypoint": "recclaw_ext.candidate:FreshCandidateModel",
        "dataset": "mini",
        "split": {"strategy": "holdout", "ordering": "timestamp"},
        "evaluator": {"metrics": ["NDCG"], "topk": [10]},
        "execution_role": "CANDIDATE",
    }
    with pytest.raises(StandaloneCampaignError, match="physical_gpu_id"):
        inputs.resource_probe(
            candidate_root=tmp_path / "candidate",
            source_path=tmp_path / "candidate" / "candidate.py",
            entrypoint=recipe["entrypoint"],
            source_sha256="a" * 64,
            execution_recipe=recipe,
            probe_root=tmp_path / "probe",
            candidate_ref="capability:wrong-device",
            candidate_package_digest="b" * 64,
        )
    assert evidence_requests
    assert evidence_requests[0]["physical_run_id"] == (
        standalone_module._resource_probe_run_id("capability:wrong-device")
    )
    assert probe_calls == []
