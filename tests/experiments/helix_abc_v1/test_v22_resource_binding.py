from __future__ import annotations

import inspect
from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v16 import (
    v16_arm_policies,
)
from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v22 import (
    v22_arm_policies,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_pilot import (
    MetaV17PilotOrchestratorV1,
)
from recclaw_core.experiments.helix_abc_v1.real_pilot import (
    RealPilotOrchestratorV1,
)


ROOT = Path(__file__).resolve().parents[3]


def test_v22_preserves_the_frozen_v16_treatment_policies() -> None:
    assert tuple(item.to_dict() for item in v22_arm_policies()) == tuple(
        item.to_dict() for item in v16_arm_policies()
    )


def test_resource_envelope_is_constructor_bound_through_canonical_core() -> None:
    real = inspect.signature(RealPilotOrchestratorV1).parameters
    meta = inspect.signature(MetaV17PilotOrchestratorV1).parameters
    assert "_resource_ceilings" in real
    assert "_resource_ceilings" in meta

    source = (
        ROOT
        / "src/recclaw_core/experiments/helix_abc_v1/campaign_pilot_v22.py"
    ).read_text(encoding="utf-8")
    assert "real_pilot_module" not in source
    assert "_resource_ceilings=expected_ceilings" in source
    assert "pilot_budget =" not in source
