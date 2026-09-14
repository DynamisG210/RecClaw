"""Portable publication regressions; no network calls, data, or training."""

from dataclasses import dataclass, replace
from pathlib import Path
import py_compile
from types import SimpleNamespace

import pytest

from recclaw_core.experiments.helix_abc_v1.research_capability import initial_research_policy
from recclaw_core.research_line import runtime, standalone
from recclaw_core.research_line.campaign import ResearchCampaign
from recclaw_core.research_line.fixed_search_policy import fixed_search_policy
from recclaw_core.research_line.fresh_runner import FreshExperimentRunner, FreshRunnerError
from recclaw_core.research_line.single_round import ResearchBaselineSourceV1


@dataclass
class Context:
    policy: dict
    scientific_memory: dict


@dataclass
class State:
    policy: object
    context: Context
    next_round_index: int = 2


def composition_fixture(monkeypatch):
    initial = initial_research_policy()
    learned = replace(initial, version=2)
    calls = []
    native_call = lambda role, view, **kwargs: (role, view, kwargs)
    producer = SimpleNamespace(_call=native_call)
    campaign = SimpleNamespace(
        search_space_adapter=SimpleNamespace(provider_context=lambda view: dict(view)),
        post_round_state_transition=lambda state, *_: replace(state, next_round_index=3),
    )
    composition = SimpleNamespace(
        policy=learned, provider=SimpleNamespace(_producer=producer), campaign=campaign,
        manifest={"policy": {"value": initial.to_dict()}},
    )
    monkeypatch.setattr(runtime, "_acquire_innovation_spec",
                        lambda candidates, **kwargs: calls.append((candidates, kwargs)))
    monkeypatch.setattr(runtime, "_search_ranking_inputs",
                        lambda context: ({"executed"}, "outcome-task", {"loss": 0.7}))
    return composition, initial, learned, calls


@pytest.mark.parametrize("interrupt", [False, True])
def test_fixed_policy_four_consumers_and_scoped_restore(monkeypatch, interrupt):
    composition, initial, learned, calls = composition_fixture(monkeypatch)
    producer = composition.provider._producer
    originals = (producer._call, runtime._acquire_innovation_spec,
                 runtime._search_ranking_inputs, composition.campaign.post_round_state_transition)
    role = initial.producer_token_allocation[0][0]
    memory = {"failure": "slow training", "cost": 25, "score": 0.4, "source": "actual code"}
    view = {"producer_role": role, "policy": learned.to_dict(),
            "producer_token_fraction": 0.9, "scientific_memory": memory,
            "research_portfolio": [{"idea": "refine previous mechanism"}]}
    state = State(learned, Context(learned.to_dict(), memory))
    try:
        with fixed_search_policy(composition):
            _, projected, kwargs = producer._call(role, view, trace="kept")
            assert projected["policy"] == initial.to_dict()
            assert projected["producer_token_fraction"] == dict(initial.producer_token_allocation)[role]
            assert projected["scientific_memory"] == memory
            assert projected["research_portfolio"] == view["research_portfolio"]
            assert view["policy"] == learned.to_dict()
            assert kwargs == {"trace": "kept"}
            runtime._acquire_innovation_spec(["candidate"], research_policy=learned, context=state.context)
            assert calls[0][0] == ["candidate"]
            assert calls[0][1]["research_policy"].to_dict() == initial.to_dict()
            assert calls[0][1]["context"] is state.context
            assert runtime._search_ranking_inputs(state.context) == ({"executed"}, None, {})
            updated = composition.campaign.post_round_state_transition(state, None, 2, "opportunity")
            assert updated.policy.to_dict() == initial.to_dict()
            assert updated.context.policy == initial.to_dict()
            assert updated.context.scientific_memory == memory
            assert updated.next_round_index == 3
            if interrupt:
                raise RuntimeError("interrupted")
    except RuntimeError:
        assert interrupt
    assert originals == (producer._call, runtime._acquire_innovation_spec,
                         runtime._search_ranking_inputs, composition.campaign.post_round_state_transition)


def test_fixed_mode_is_part_of_resume_identity(tmp_path):
    baseline = ResearchBaselineSourceV1.from_identity(
        source_ref="fixture", source_sha256="a" * 64,
        comparator_ref="parent", comparator_digest="b" * 64,
        frozen_ndcg_at_10=0.2, protocol_digest="c" * 64, seed=54201,
    )
    adaptive = standalone.StandaloneResearchConfig(
        repo_root=tmp_path, run_root=tmp_path / "run", api_config_source=tmp_path / "api.toml",
        campaign_id="fixture", baseline_source=baseline,
        frozen_profile_ref={"profile_ref": "fixture"},
    )
    fixed = replace(adaptive, search_policy_mode="fixed")
    old_manifest = {"execution": standalone._execution_identity(adaptive)}
    new_manifest = {"execution": standalone._execution_identity(fixed)}
    assert "search_policy_mode" not in old_manifest["execution"]
    assert new_manifest["execution"]["search_policy_mode"] == "fixed"
    assert standalone._resume_execution_identity_compatible(old_manifest, adaptive)
    assert standalone._resume_execution_identity_compatible(new_manifest, fixed)
    assert not standalone._resume_execution_identity_compatible(old_manifest, fixed)
    assert not standalone._resume_execution_identity_compatible(new_manifest, adaptive)
    with pytest.raises(standalone.StandaloneCampaignError, match="search_policy_mode"):
        replace(adaptive, search_policy_mode="invalid")


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_composition_run_activates_selected_mode(monkeypatch, mode):
    fixture, initial, learned, calls = composition_fixture(monkeypatch)
    composition = standalone.StandaloneResearchComposition(
        config=SimpleNamespace(search_policy_mode=mode), campaign=fixture.campaign,
        provider=fixture.provider, implementer=None, runner=None, profile=None,
        policy=fixture.policy, manifest=fixture.manifest,
    )
    def run(self, count):
        runtime._acquire_innovation_spec([], research_policy=learned)
        return (count,)
    monkeypatch.setattr(standalone.StandaloneResearchComposition, "_run", run)
    assert composition.run(2) == (2,)
    expected = initial if mode == "fixed" else learned
    assert calls[0][1]["research_policy"].to_dict() == expected.to_dict()


@pytest.mark.parametrize("relative", ["model.py", "recclaw_ext/models/model.py"])
def test_source_readout_ignores_bytecode_but_not_source_drift(tmp_path, relative):
    source = tmp_path / relative
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("VALUE = 3\n")
    digest = runtime.sha256_digest({"files": runtime.snapshot_candidate_tree(tmp_path)})
    expected = runtime._verified_candidate_source_files(tmp_path, digest, [relative])
    py_compile.compile(str(source), doraise=True)
    assert runtime._verified_candidate_source_files(tmp_path, digest, [relative]) == expected
    extra = source.parent / "__pycache__" / "extra.py"
    extra.write_text("VALUE = 4\n")
    with pytest.raises(ValueError, match="file set"):
        runtime._verified_candidate_source_files(tmp_path, digest, [relative])
    extra.unlink()
    source.write_text("VALUE = 9\n")
    with pytest.raises(ValueError, match="digest drift"):
        runtime._verified_candidate_source_files(tmp_path, digest, [relative])
    source.unlink()
    with pytest.raises(ValueError, match="file set"):
        runtime._verified_candidate_source_files(tmp_path, digest, [relative])


def test_prebinding_revisions_retain_mechanism_without_mixing_candidates():
    rows = [
        {"candidate_id": "a", "spec_digest": "one", "core_mechanism_contrast": "routing",
         "next_discriminative_task": "reduce repeated sampling", "failure": "first"},
        {"candidate_id": "b", "spec_digest": "two", "core_mechanism_contrast": "cosine",
         "failure": "second"},
        {"candidate_id": "a", "spec_digest": "one", "failure": "timeout", "repair_attempt": 1},
    ]
    result = SimpleNamespace(innovation=SimpleNamespace(attempts=rows))
    summaries = ResearchCampaign._prebinding_candidate_failure_summaries(result)
    assert summaries[0]["core_mechanism_contrast"] == "routing"
    assert summaries[0]["next_discriminative_task"] == "reduce repeated sampling"
    assert summaries[0]["failure"] == "timeout"
    assert summaries[0]["revision_count"] == 2
    assert summaries[1]["core_mechanism_contrast"] == "cosine"


@pytest.mark.parametrize("relative", ["model.py", "recclaw_ext/models/model.py", None])
def test_candidate_root_accepts_flat_and_nested_python(tmp_path, relative):
    from recclaw_core.research_line.fresh_runner import SearchProfileEntryOriginV1

    if relative:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("VALUE = 1\n")
    binding = SimpleNamespace(entry_origin=SearchProfileEntryOriginV1.QUALIFIED_REGISTRY,
                              capability_ref="fixture")
    runner = SimpleNamespace(config=SimpleNamespace(candidate_root_by_capability={"fixture": tmp_path}))
    if relative:
        assert FreshExperimentRunner._candidate_root(runner, binding, {}) == tmp_path
    else:
        with pytest.raises(FreshRunnerError, match="unavailable"):
            FreshExperimentRunner._candidate_root(runner, binding, {})
