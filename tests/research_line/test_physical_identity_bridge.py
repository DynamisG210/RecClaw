from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import pytest

from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.experiment_binding import (
    COMMON_EVALUATOR,
    ExperimentBindingV1,
)
from recclaw_core.experiments.helix_abc_v1.research_capability import (
    SearchMemoryWriterV1,
    initial_research_policy,
)
from recclaw_core.experiments.helix_abc_v1.search_adapter import (
    adapt_current_search_profile,
)
from recclaw_core.research_line.campaign import (
    CampaignRoundInputs,
    CampaignState,
    ResearchCampaign,
)

TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))

from test_runtime import (  # noqa: E402
    _bindings,
    _context,
    _environment,
    _fixed_proposals,
    _incumbent,
    _router,
    _runner,
)


class _CrashOnceContextRunner:
    def __init__(self, calls: list[dict[str, Any]]) -> None:
        self.calls = calls
        self.crash_on_attempt_one = True
        self.successful_run_ids: list[str] = []

    def run_with_physical_context(
        self,
        recipe: dict[str, Any],
        binding: Any,
        physical_context: Any,
    ) -> dict[str, Any]:
        context = dict(physical_context)
        self.calls.append(context)
        if context["attempt_index"] == 1 and self.crash_on_attempt_one:
            self.crash_on_attempt_one = False
            raise RuntimeError("crash before second physical launch")
        context_digest = sha256_digest(context)
        run_id = f"bridge-run:physical:{context_digest}"
        seed = int(context["seed"])
        status = "RESOURCE_CENSORED" if context["attempt_index"] == 0 else "SUCCESS"
        result = dict(_runner([], status=status, seed=seed)(recipe, binding))
        experiment_binding = ExperimentBindingV1.from_canonical_dict(
            result["experiment_binding"]
        )
        experiment_binding = replace(
            experiment_binding,
            run_id=run_id,
            seed=seed,
        )
        result.update(
            {
                "experiment_binding": experiment_binding.canonical_dict(),
                "experiment_binding_ref": experiment_binding.ref,
                "experiment_binding_digest": experiment_binding.digest,
                "binding_digest": experiment_binding.digest,
                "seed": seed,
                "physical_identity": {
                    "schema": "recclaw.research-line.physical-execution-identity.v1",
                    "run_id": run_id,
                    "seed": seed,
                    "context_digest": context_digest,
                    "cuda_visible_devices": None,
                    "reservation_digest": None,
                    "reservation_status": "UNMEASURED_NO_EXCLUSIVE_RESERVATION",
                    "final_worker_ceiling_seconds": 3600,
                },
            }
        )
        self.successful_run_ids.append(run_id)
        return result


def test_campaign_resume_reuses_sealed_attempt_and_allocates_only_remaining_identity(
    tmp_path: Path,
) -> None:
    profile = adapt_current_search_profile(campaign_id="campaign:physical-identity")
    policy = initial_research_policy()
    context = _context(profile, policy=policy)
    proposals = tuple(_fixed_proposals(profile).values())
    initial = CampaignState.initial(
        context=context,
        active_profile=profile,
        policy=policy,
        incumbent_observation=_incumbent(),
        carryover_proposals=proposals,
    )

    def producer(_role: str, _view: dict[str, Any]) -> Any:
        raise RuntimeError("frozen proposal pool")

    def inputs(state: CampaignState) -> CampaignRoundInputs:
        return CampaignRoundInputs(
            producer_bindings=_bindings(state.context, state.active_profile),
            resolver_environment=_environment(state.active_profile),
            budget_snapshot={
                "experiment_opportunities": 1,
                "round_attempt_budget": 2,
            },
            router=_router(),
            metric_contract_digest=sha256_digest(COMMON_EVALUATOR),
            observation_seed="63001",
            next_discriminative_test="use the next independent opportunity",
            attempt_scheduler=True,
            max_attempts_per_round=2,
        )

    calls: list[dict[str, Any]] = []
    runner = _CrashOnceContextRunner(calls)
    root = tmp_path / "campaign"
    campaign = ResearchCampaign(
        root=root,
        state=initial,
        producer=producer,
        runner=runner,
        round_inputs=inputs,
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
    )

    with pytest.raises(RuntimeError, match="crash before second physical launch"):
        campaign.run_round()
    assert [item["attempt_index"] for item in calls] == [0, 1]
    assert len(runner.successful_run_ids) == 1
    manifest = (root / "ROUND_01_ATTEMPT_MANIFEST.json").read_text(
        encoding="utf-8"
    )
    assert '"attempt_index":0' in manifest
    assert not (root / "ROUND_01_CHECKPOINT.pkl").is_file()

    resumed = ResearchCampaign.resume(
        root=root,
        producer=producer,
        runner=runner,
        round_inputs=inputs,
        memory_writer=SearchMemoryWriterV1("DEVELOPMENT_ONLY/SEARCH_MEMORY"),
    )
    record = resumed.run_round()

    assert record.result.has_metric_bearing_attempt
    assert [item["attempt_index"] for item in calls] == [0, 1, 1]
    assert len(runner.successful_run_ids) == 2
    assert len(set(runner.successful_run_ids)) == 2
    assert record.result.attempts[0].attempt_index == 0
    assert record.result.attempts[1].attempt_index == 1
