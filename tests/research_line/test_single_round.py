from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    SearchMemoryWriterV1,
)
from recclaw_core.experiments.helix_abc_v1.research_contracts import (
    DISCOVERY_PRODUCERS,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    adapt_current_search_profile,
)
from recclaw_core.research_line import single_round as single_round_module
from recclaw_core.research_line.single_round import (
    _RecordedExperimentRunner,
    ResearchBaselineSourceV1,
    compose_single_round,
    execute_single_round,
)
from recclaw_core.research_line.runtime import InnovationRuntimeInputs


def _api_config(path: Path) -> Path:
    path.write_text(
        "base_url = 'https://primary.example/v1'\n"
        "api_key = 'primary-secret'\n"
        "base_url = 'https://fallback.example/v1'\n"
        "api_key = 'fallback-secret'\n",
        encoding="utf-8",
    )
    return path


def _incumbent_receipt(path: Path, *, campaign_id: str) -> Path:
    profile = adapt_current_search_profile(campaign_id=campaign_id)
    path.write_text(
        json.dumps(
            {
                "typed_episode": {
                    "protocol_digest": profile.protocol_digest,
                    "comparator_ref": "fixture:bpr",
                    "comparator_digest": "1" * 64,
                },
                "outcome_summary": {
                    "seed": 54303,
                    "baseline_metrics": {"ndcg@10": 0.2068},
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _research_baseline_identity(campaign_id: str) -> ResearchBaselineSourceV1:
    profile = adapt_current_search_profile(campaign_id=campaign_id)
    return ResearchBaselineSourceV1.from_identity(
        source_ref=f"research-baseline:test:{campaign_id}",
        source_sha256="2" * 64,
        comparator_ref="fixture:bpr",
        comparator_digest="1" * 64,
        frozen_ndcg_at_10=0.2068,
        protocol_digest=profile.protocol_digest,
        seed=54303,
    )


def _search_data_root(path: Path, *, stale: bool = False, mutate: bool = False) -> Path:
    dataset_root = path / "ml-1m"
    dataset_root.mkdir(parents=True)
    payloads = {
        "ml-1m.train.inter": b"train\n",
        "ml-1m.dev.inter": b"dev\n",
        "ml-1m.heldout.inter": b"heldout\n",
        "ml-1m.user": b"user\n",
        "ml-1m.item": b"item\n",
    }
    for name, payload in payloads.items():
        (dataset_root / name).write_bytes(payload)
    if mutate:
        (dataset_root / "ml-1m.heldout.inter").write_bytes(b"changed\n")
    if stale:
        manifest = {
            "counts": {"development_validation": 1, "train": 1},
            "search_files": {
                name: hashlib.sha256(payloads[name]).hexdigest()
                for name in ("ml-1m.train.inter", "ml-1m.dev.inter")
            },
        }
    else:
        manifest = {
            "schema": "recclaw.round-test-feedback-partition-manifest.v1",
            "dataset": "ml-1m",
            "split": "SHA256_SEEDED_WITHIN_USER_80_10_10",
            "checkpoint_selection": "BEST_DEVELOPMENT_VALIDATION_NDCG_AT_10",
            "round_metric": "BEST_CHECKPOINT_TEST_RESULT",
            "adaptive_reuse": "ROUND_TEST_FEEDBACK",
            "files": {
                name: hashlib.sha256(payloads[name]).hexdigest()
                for name in payloads
            },
        }
    (path / "search_partition_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_preflight_requires_an_explicit_research_baseline_source(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="explicit Research baseline source"):
        compose_single_round(
            repo_root=tmp_path / "repo",
            run_root=tmp_path / "physical-round",
            api_config_path=tmp_path / "llm_api.md",
            campaign_id="test-single-round-no-baseline",
        )

    assert not (tmp_path / "physical-round").exists()


def test_preflight_accepts_explicit_research_baseline_identity_without_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    campaign_id = "test-single-round-explicit-identity"
    search_root = _search_data_root(tmp_path / "formal-search")
    monkeypatch.setattr(single_round_module.fresh_r1, "SEARCH_DATA_ROOT", search_root)
    baseline = _research_baseline_identity(campaign_id)

    composition = compose_single_round(
        repo_root=repo_root,
        run_root=tmp_path / "physical-round",
        api_config_path=_api_config(tmp_path / "llm_api.md"),
        campaign_id=campaign_id,
        research_baseline_source=baseline,
    )

    assert composition.baseline_source == baseline
    assert composition.incumbent_receipt_path is None
    assert composition.incumbent["source_ref"] == baseline.source_ref
    assert composition.incumbent["source_receipt"] is None
    assert composition.incumbent["source_receipt_sha256"] == baseline.source_sha256
    assert composition.manifest["experiment"]["baseline_source"]["kind"] == "IDENTITY"


def test_preflight_rejects_ambiguous_receipt_and_research_source(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="either research_baseline_source"):
        compose_single_round(
            repo_root=tmp_path / "repo",
            run_root=tmp_path / "physical-round",
            api_config_path=tmp_path / "llm_api.md",
            campaign_id="test-single-round-ambiguous-baseline",
            incumbent_receipt_path=tmp_path / "incumbent.json",
            research_baseline_source=_research_baseline_identity(
                "test-single-round-ambiguous-baseline"
            ),
        )


def test_preflight_composes_real_round_without_calls_or_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    api_config = _api_config(tmp_path / "llm_api.md")
    run_root = tmp_path / "physical-round"
    search_root = _search_data_root(tmp_path / "formal-search")
    monkeypatch.setattr(single_round_module.fresh_r1, "SEARCH_DATA_ROOT", search_root)

    composition = compose_single_round(
        repo_root=repo_root,
        run_root=run_root,
        api_config_path=api_config,
        campaign_id="test-single-round",
        incumbent_receipt_path=_incumbent_receipt(
            tmp_path / "incumbent.json",
            campaign_id="test-single-round",
        ),
    )

    assert not run_root.exists()
    assert len(composition.profile.entries) == 66
    assert composition.context.active_profile_digest == composition.profile.profile_digest
    assert composition.context.policy == composition.policy.to_dict()
    assert set(composition.context.scientific_memory["by_role"]) == set(
        DISCOVERY_PRODUCERS
    )
    assert composition.manifest["context"]["digest"] == composition.context.digest
    assert composition.manifest["provider"]["model"] == "gpt-5.4"
    assert composition.manifest["provider"]["credential_count"] == 2
    assert composition.manifest["provider"]["ordered_endpoint_digests"] == [
        sha256_digest({"base_url": "https://primary.example/v1"}),
        sha256_digest({"base_url": "https://fallback.example/v1"}),
    ]
    assert composition.manifest["experiment"]["experiment_opportunities"] == 1
    assert composition.manifest["experiment"]["seed"] == 54303
    assert len(composition.manifest["experiment"]["dataset_manifest_sha256"]) == 64
    assert set(composition.manifest["experiment"]["dataset_file_hashes"]) == {
        "ml-1m.train.inter",
        "ml-1m.dev.inter",
        "ml-1m.heldout.inter",
        "ml-1m.user",
        "ml-1m.item",
    }
    assert composition.incumbent["frozen_ndcg@10"] == 0.2068
    assert "primary-secret" not in str(composition.manifest)
    assert "fallback-secret" not in str(composition.manifest)


def test_preflight_rejects_stale_development_search_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    api_config = _api_config(tmp_path / "llm_api.md")
    run_root = tmp_path / "physical-round"
    search_root = _search_data_root(tmp_path / "stale-search", stale=True)
    monkeypatch.setattr(single_round_module.fresh_r1, "SEARCH_DATA_ROOT", search_root)

    with pytest.raises(ValueError, match="formal round-test-feedback"):
        compose_single_round(
            repo_root=repo_root,
            run_root=run_root,
            api_config_path=api_config,
            campaign_id="test-single-round-stale-data",
            incumbent_receipt_path=_incumbent_receipt(
                tmp_path / "incumbent.json",
                campaign_id="test-single-round-stale-data",
            ),
        )
    assert not run_root.exists()


def test_preflight_rejects_declared_byte_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    api_config = _api_config(tmp_path / "llm_api.md")
    run_root = tmp_path / "physical-round"
    search_root = _search_data_root(tmp_path / "mismatched-search", mutate=True)
    monkeypatch.setattr(single_round_module.fresh_r1, "SEARCH_DATA_ROOT", search_root)

    with pytest.raises(ValueError, match="hash mismatch"):
        compose_single_round(
            repo_root=repo_root,
            run_root=run_root,
            api_config_path=api_config,
            campaign_id="test-single-round-mismatched-data",
            incumbent_receipt_path=_incumbent_receipt(
                tmp_path / "incumbent.json",
                campaign_id="test-single-round-mismatched-data",
            ),
        )
    assert not run_root.exists()


def test_execute_reaches_unified_runtime_without_external_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    search_root = _search_data_root(tmp_path / "formal-search")
    monkeypatch.setattr(single_round_module.fresh_r1, "SEARCH_DATA_ROOT", search_root)
    composition = compose_single_round(
        repo_root=repo_root,
        run_root=tmp_path / "physical-round",
        api_config_path=_api_config(tmp_path / "llm_api.md"),
        campaign_id="test-single-round-runtime-edge",
        incumbent_receipt_path=_incumbent_receipt(
            tmp_path / "incumbent.json",
            campaign_id="test-single-round-runtime-edge",
        ),
    )
    captured: dict[str, object] = {}

    class ExpectedStop(RuntimeError):
        pass

    def stop_at_runtime(**kwargs: object) -> object:
        captured.update(kwargs)
        raise ExpectedStop("stop before Provider or experiment")

    monkeypatch.setattr(single_round_module, "run_research_round", stop_at_runtime)

    with pytest.raises(ExpectedStop):
        execute_single_round(composition)

    assert isinstance(captured["memory_writer"], SearchMemoryWriterV1)
    assert captured["context"] == composition.context
    assert captured["active_profile"] == composition.profile
    innovation = captured["innovation_inputs"]
    assert isinstance(innovation, InnovationRuntimeInputs)
    candidate_root = (
        composition.run_root
        / "innovation_candidates"
        / "attempt-00"
        / "candidate-under-test"
    )
    candidate_policy = replace(
        innovation.policy,
        execution_contract={
            "capability_family": "TEST_FAMILY",
            "model": "TestCandidate",
            "base_model_config": "BPR",
            "config": {},
        },
    )
    fixture = innovation.fixture_factory(
        candidate_policy,
        0,
        candidate_root,
    )
    assert not fixture.checkpoint_dir.is_relative_to(candidate_root)
    assert fixture.checkpoint_dir == (
        composition.run_root
        / "innovation_qualification"
        / "attempt-00"
        / "qualification"
        / "checkpoints"
    )
    assert fixture.runtime_identity_ref == candidate_policy.runtime_identity_ref
    assert fixture.runtime_identity_digest == candidate_policy.runtime_identity_digest
    assert (composition.run_root / "PRE_OUTCOME_MANIFEST.json").is_file()
    assert (composition.run_root / "ROUND_FAILURE.json").is_file()


def test_physical_observation_is_durable_before_interpreter(tmp_path: Path) -> None:
    output_path = tmp_path / "PHYSICAL_EXPERIMENT_OBSERVATION.json"
    calls: list[tuple[object, object]] = []

    def physical_runner(recipe: object, binding: object) -> dict[str, object]:
        calls.append((recipe, binding))
        return {"exit_status": "SUCCESS", "metrics": {"ndcg@10": 0.2056}}

    runner = _RecordedExperimentRunner(physical_runner, output_path)
    result = runner({"recipe": "fixture"}, {"binding": "fixture"})

    assert calls == [({"recipe": "fixture"}, {"binding": "fixture"})]
    assert result["metrics"]["ndcg@10"] == 0.2056
    assert output_path.is_file()
    assert '"ndcg@10":0.2056' in output_path.read_text(encoding="utf-8")
