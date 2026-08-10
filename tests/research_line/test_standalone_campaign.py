from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    adapt_current_search_profile,
)
from recclaw_core.research_line.campaign import ResearchCampaign
from recclaw_core.research_line.interfaces import ResearchTaskQueueV2
from recclaw_core.research_line.single_round import ResearchBaselineSourceV1
from recclaw_core.research_line.standalone import (
    StandaloneCampaignError,
    StandaloneResearchConfig,
    compose_standalone_campaign,
    load_portfolio_candidates,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from test_fresh_runner import _fake_launch  # noqa: E402
from test_provider import _call_result, _proposal_response  # noqa: E402


def _config(run_root: Path) -> StandaloneResearchConfig:
    profile = adapt_current_search_profile(campaign_id="research:standalone-fixture")
    return StandaloneResearchConfig(
        repo_root=REPO_ROOT,
        run_root=run_root,
        api_config_source={
            "release_digest": "provider-config-fixture",
            "source_kind": "no-provider-test",
        },
        campaign_id="research:standalone-fixture",
        baseline_source=ResearchBaselineSourceV1.from_identity(
            source_ref="research-source:fixture",
            source_sha256="2" * 64,
            comparator_ref="research-baseline:fixture",
            comparator_digest="1" * 64,
            frozen_ndcg_at_10=0.4,
            protocol_digest=profile.protocol_digest,
            seed=54303,
        ),
        seed=54303,
        epochs=1,
        round_count=2,
    )


def _forbidden_boundary(**_kwargs: object) -> object:
    raise AssertionError("no Provider or GPU boundary may be called in this test")


def _direct_discovery_calls(
    calls: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        call
        for call in calls
        if str(call["logical_call_id"]).endswith(":proposal")
        and ":offline-replay:" not in str(call["logical_call_id"])
    ]


def test_standalone_entry_composes_campaign_and_resumes_without_provider_or_gpu(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "standalone")
    composed = compose_standalone_campaign(
        config,
        provider_call=_forbidden_boundary,
        launch=_forbidden_boundary,
    )

    assert isinstance(composed.campaign, ResearchCampaign)
    assert composed.campaign.state.next_round_index == 1
    assert composed.campaign.state.carryover_proposals
    queue = ResearchTaskQueueV2.from_dict(
        composed.campaign.state.context.scientific_memory["global_memory"][
            "task_queue"
        ]
    )
    assert queue.tasks == ()
    assert composed.provider.call_traces == ()
    assert composed.implementer.call_traces == ()
    assert composed.manifest["research_only"] is True
    assert composed.manifest["portfolio"]["enabled"] is False

    resumed = compose_standalone_campaign(
        config,
        resume=True,
        provider_call=_forbidden_boundary,
        launch=_forbidden_boundary,
    )
    assert resumed.campaign.state.digest == composed.campaign.state.digest
    assert resumed.campaign.state.context.context_ref == (
        "research-context:research:standalone-fixture:round-1"
    )
    assert resumed.provider.call_traces == ()
    assert resumed.implementer.call_traces == ()


def test_standalone_run_persists_metric_round_and_resumes_without_provider_or_gpu(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "metric-round")
    provider_calls: list[dict[str, object]] = []

    def provider_call(**kwargs: object):
        provider_calls.append(dict(kwargs))
        logical_call_id = str(kwargs["logical_call_id"])
        role = logical_call_id.split(":")[-2]
        return _call_result(
            logical_call_id=logical_call_id,
            response=_proposal_response(role),
        )

    launch_calls: list[dict[str, object]] = []
    composition = compose_standalone_campaign(
        config,
        provider_call=provider_call,
        launch=_fake_launch(launch_calls),
    )
    assert composition.runner.config.timeout_seconds == 3600
    assert composition.runner.config.watchdog_seconds == 3600

    results = composition.run(1)

    assert len(results) == 1
    assert results[0].result.has_metric_bearing_attempt
    assert results[0].result.candidate_run is not None
    assert composition.campaign.state.next_round_index == 2
    assert composition.campaign.round_checkpoint_path(1).is_file()
    assert composition.campaign.round_trace_path(1).is_file()
    assert len(_direct_discovery_calls(provider_calls)) == 4
    assert len(provider_calls) > 4
    assert len(launch_calls) == 1
    assert launch_calls[0]["recbole_commit_identity"] == (
        "7b02be5ec80a88310f2d04a27a82adfcbb5dc211"
    )

    resumed = compose_standalone_campaign(
        config,
        resume=True,
        provider_call=provider_call,
        launch=_fake_launch(launch_calls),
    )
    assert resumed.campaign.state.digest == composition.campaign.state.digest
    assert resumed.campaign.state.next_round_index == 2

    cached = resumed.run(1)
    assert len(cached) == 1
    assert cached[0].result.has_metric_bearing_attempt
    assert len(_direct_discovery_calls(provider_calls)) == 8
    assert len(provider_calls) > 8
    assert len(launch_calls) == 2


def test_standalone_frozen_seed_schedule_and_physical_manifest_identity(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path / "scheduled")
    config = replace(
        base,
        observation_seed_schedule=(54303, 54304),
        cuda_visible_devices="2",
        final_worker_ceiling_seconds=1234,
    )
    provider_calls: list[dict[str, object]] = []

    def provider_call(**kwargs: object):
        provider_calls.append(dict(kwargs))
        logical_call_id = str(kwargs["logical_call_id"])
        role = logical_call_id.split(":")[-2]
        return _call_result(
            logical_call_id=logical_call_id,
            response=_proposal_response(role),
        )

    launch_calls: list[dict[str, object]] = []
    composition = compose_standalone_campaign(
        config,
        provider_call=provider_call,
        launch=_fake_launch(launch_calls),
    )
    results = composition.run(2)

    assert len(results) == 2
    assert [call["seed"] for call in launch_calls] == [54303, 54304]
    assert len({call["run_id"] for call in launch_calls}) == 2
    assert all(call["cuda_visible_devices"] == "2" for call in launch_calls)
    assert all(call["final_worker_ceiling_seconds"] == 1234 for call in launch_calls)
    assert composition.manifest["observation_seed_schedule"]["digest"]
    assert composition.manifest["runner"]["cuda_visible_devices"] == "2"
    assert composition.manifest["runner"]["final_worker_ceiling_seconds"] == 1234
    for round_index, seed in ((1, 54303), (2, 54304)):
        payload = json.loads(
            (tmp_path / "scheduled" / f"ROUND_{round_index:02d}_ATTEMPT_00_PHYSICAL_OBSERVATION.json").read_text(
                encoding="utf-8"
            )
        )
        assert payload["physical_run_id"] == launch_calls[round_index - 1]["run_id"]
        assert payload["physical_seed"] == seed
        assert payload["physical_context_digest"]
        assert payload["cuda_visible_devices"] == "2"
        assert payload["final_worker_ceiling_seconds"] == 1234
        assert payload["reservation_status"] == (
            "UNMEASURED_NO_EXCLUSIVE_RESERVATION"
        )


def test_standalone_direct_gpu_id_reaches_fresh_runner_without_cvd(
    tmp_path: Path,
) -> None:
    base = _config(tmp_path / "direct-gpu")
    config = replace(base, round_count=1, gpu_id=2)
    provider_calls: list[dict[str, object]] = []

    def provider_call(**kwargs: object):
        provider_calls.append(dict(kwargs))
        logical_call_id = str(kwargs["logical_call_id"])
        role = logical_call_id.split(":")[-2]
        return _call_result(
            logical_call_id=logical_call_id,
            response=_proposal_response(role),
        )

    launch_calls: list[dict[str, object]] = []
    composition = compose_standalone_campaign(
        config,
        provider_call=provider_call,
        launch=_fake_launch(launch_calls),
    )

    results = composition.run(1)

    assert len(results) == 1
    assert launch_calls[0]["gpu_id"] == 2
    assert launch_calls[0]["cuda_visible_devices"] is None
    assert composition.runner.config.gpu_id == 2
    assert composition.runner.config.cuda_visible_devices is None
    assert results[0].result.candidate_run is not None


def test_standalone_resume_rejects_source_identity_drift(tmp_path: Path) -> None:
    config = _config(tmp_path / "standalone")
    compose_standalone_campaign(
        config,
        provider_call=_forbidden_boundary,
        launch=_forbidden_boundary,
    )
    drifted = replace(
        config,
        baseline_source=ResearchBaselineSourceV1.from_identity(
            source_ref="research-source:fixture",
            source_sha256="3" * 64,
            comparator_ref="research-baseline:fixture",
            comparator_digest="1" * 64,
            frozen_ndcg_at_10=0.4,
            protocol_digest=config.baseline_source.protocol_digest,
            seed=config.seed,
        ),
    )
    with pytest.raises(StandaloneCampaignError, match="identity"):
        compose_standalone_campaign(
            drifted,
            resume=True,
            provider_call=_forbidden_boundary,
            launch=_forbidden_boundary,
        )


def test_portfolio_profile_loader_rejects_missing_identity_bound_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "portfolio.json"
    path.write_text('{"candidates":[{"candidate_id":"candidate-1"}]}\n')
    with pytest.raises(StandaloneCampaignError, match="incomplete"):
        load_portfolio_candidates(path)


def test_standalone_rejects_receipt_backed_baseline_contract(tmp_path: Path) -> None:
    receipt_source = ResearchBaselineSourceV1.from_receipt_path(
        tmp_path / "baseline-receipt.json"
    )
    with pytest.raises(StandaloneCampaignError, match="identity-backed"):
        replace(_config(tmp_path / "standalone"), baseline_source=receipt_source)
