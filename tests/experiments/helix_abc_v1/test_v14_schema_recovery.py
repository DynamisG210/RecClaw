from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v14 import (
    V14_EXECUTABLE_PROFILE_DIGEST,
    V14_PILOT_ROUNDS_PER_ARM,
    V14_PILOT_SEARCH_SEED,
    V14PilotStoreContractV1,
)
from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    campaign_proposal_schema,
    campaign_runtime_profile,
)
from recclaw_core.experiments.helix_abc_v1.contracts import ArmCode
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (
    validate_provider_strict_schema,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (
    POLICY_BUNDLE_DIGEST_V19,
    MetaV18CampaignRuntimeV1,
    MetaV19CampaignRuntimeV1,
)
from recclaw_core.experiments.helix_abc_v1.original_main import (
    ORIGINAL_MAIN_COMMIT,
    PinnedOriginalMainAdapterV1,
)
from recclaw_core.experiments.helix_abc_v1.real_canary import (
    RealCanaryProposalBrokerV1,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (
    campaign_training_runtime_release,
)


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src/recclaw_core/experiments/helix_abc_v1"
RESOURCE_ROOT = PACKAGE / "resources"


class _NoCallUpstream:
    model = "no-call-fixture"
    max_total_tokens_per_call = 20_000

    def __init__(self) -> None:
        self.calls = 0

    def call_with_session(self, **_kwargs: Any) -> Any:
        self.calls += 1
        raise AssertionError("construction must not call the Provider")


def _runtime() -> MetaV19CampaignRuntimeV1:
    return MetaV19CampaignRuntimeV1(
        checkpoint_path=(
            RESOURCE_ROOT / "meta_vnext_policy_checkpoint_v19.json"
        ),
        experiment_id=V14PilotStoreContractV1.create().experiment_id,
        search_seed=V14_PILOT_SEARCH_SEED,
        scheduled_rounds=V14_PILOT_ROUNDS_PER_ARM,
        task_scale=1.0,
        task_density=0.2843119865332499,
    )


def test_v14_strict_schema_profile_and_releases_are_exact() -> None:
    validate_provider_strict_schema(campaign_proposal_schema())
    profile = campaign_runtime_profile()
    store = V14PilotStoreContractV1.create()
    assert profile["executable_profile_digest"] == (
        V14_EXECUTABLE_PROFILE_DIGEST
    )
    assert {
        item.bl_icf_search_space_digest for item in store.arm_policies
    } == {V14_EXECUTABLE_PROFILE_DIGEST}
    by_arm = {item.arm: item for item in store.arm_policies}
    assert by_arm[ArmCode.B].non_guard_projection() == (
        by_arm[ArmCode.C].non_guard_projection()
    )
    release = campaign_training_runtime_release()
    assert release.release_id == "TRAINING_RUNTIME_RELEASE_V9"


def test_v19_rebind_preserves_v18_learned_policy() -> None:
    runtime = _runtime()
    v18_checkpoint = json.loads(
        (
            RESOURCE_ROOT / "meta_vnext_policy_checkpoint_v18.json"
        ).read_text(encoding="utf-8")
    )
    assert runtime.policy_bundle_digest == POLICY_BUNDLE_DIGEST_V19
    assert runtime.checkpoint["coefficient_action"] == (
        "INHERIT_EXACT_V18_NO_COEFFICIENT_CHANGE"
    )
    assert runtime.checkpoint["parent_slow_policy_digest"] == (
        v18_checkpoint["parent_slow_policy_digest"]
    )
    assert runtime.checkpoint["router_configuration_digest"] == (
        v18_checkpoint["router_configuration_digest"]
    )
    assert MetaV19CampaignRuntimeV1.runtime_repair_digest != (
        MetaV18CampaignRuntimeV1.runtime_repair_digest
    )


def test_v14_broker_construction_uses_pinned_main_and_shared_v19() -> None:
    upstream = _NoCallUpstream()
    runtime = _runtime()
    broker = RealCanaryProposalBrokerV1.create_v13(
        upstream=upstream,
        template_path=RESOURCE_ROOT / "campaign_anchor_programs_v1.json",
        repository_root=ROOT,
        search_seed=V14_PILOT_SEARCH_SEED,
        adaptive_memory=True,
        campaign_meta_runtime=runtime,
    )
    assert isinstance(broker.original_controller, PinnedOriginalMainAdapterV1)
    assert ORIGINAL_MAIN_COMMIT == (
        "2d8c881354e1b536a6c66d7dfbb977e0c5090e50"
    )
    assert broker.campaign_meta_runtime is runtime
    assert broker.producer_control_enabled is True
    assert upstream.calls == 0


def test_v14_freeze_build_and_entrypoint_import_are_side_effect_free() -> None:
    from scripts.freeze_v14_pilot_contract import (
        DEFAULT_OUTPUT_ROOT,
    )

    contract = json.loads(
        (
            ROOT
            / "docs/research_line/continuous_program/"
            "V14_FROZEN_CHAIN_PILOT_CONTRACT.json"
        ).read_text(encoding="utf-8")
    )
    assert contract["record_schema"] == "recclaw.v14-pilot-contract.v1"
    assert contract["pilot_started"] is False
    assert contract["original"]["adapter"] == "PinnedOriginalMainAdapterV1"
    assert contract["meta"]["policy_bundle_digest"] == (
        POLICY_BUNDLE_DIGEST_V19
    )
    assert contract["training"]["release_id"] == (
        "TRAINING_RUNTIME_RELEASE_V6"
    )
    root_existed_before_import = DEFAULT_OUTPUT_ROOT.exists()
    runpy.run_path(str(ROOT / "scripts/run_v14_pilot.py"))
    assert DEFAULT_OUTPUT_ROOT.exists() is root_existed_before_import
