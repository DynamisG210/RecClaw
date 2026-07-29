from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v16 import (
    V16_EXECUTABLE_PROFILE_DIGEST,
    v16_arm_policies,
)
from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v24 import (
    V24_EXECUTABLE_PROFILE_DIGEST,
    v24_arm_policies,
)
from recclaw_core.experiments.helix_abc_v1.canonical import sha256_digest
from recclaw_core.experiments.helix_abc_v1.meta_vnext_pilot import (
    MetaV17PilotOrchestratorV1,
)
from recclaw_core.experiments.helix_abc_v1.real_pilot import (
    RealPilotOrchestratorV1,
)


ROOT = Path(__file__).resolve().parents[3]
RESOURCE_ROOT = (
    ROOT
    / "src/recclaw_core/experiments/helix_abc_v1/resources"
)


def test_v24_preserves_the_frozen_v16_treatment_and_profile() -> None:
    assert V24_EXECUTABLE_PROFILE_DIGEST == V16_EXECUTABLE_PROFILE_DIGEST
    assert tuple(item.to_dict() for item in v24_arm_policies()) == tuple(
        item.to_dict() for item in v16_arm_policies()
    )


def test_v24_resource_envelope_is_constructor_bound() -> None:
    real = inspect.signature(RealPilotOrchestratorV1).parameters
    meta = inspect.signature(MetaV17PilotOrchestratorV1).parameters
    assert "_resource_ceilings" in real
    assert "_resource_ceilings" in meta

    source = (
        ROOT
        / "src/recclaw_core/experiments/helix_abc_v1/campaign_pilot_v24.py"
    ).read_text(encoding="utf-8")
    assert "_resource_ceilings=expected_ceilings" in source
    assert "pilot_budget =" not in source


def test_v24_provider_transport_forbids_provider_authored_parent() -> None:
    schema_path = RESOURCE_ROOT / "campaign_proposal_response_v2.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    proposal = schema["properties"]["proposals"]["items"]
    assert proposal["properties"]["parent_candidate_id"] == {"type": "null"}
    assert "parent_candidate_id" in proposal["required"]

    release = json.loads(
        (
            RESOURCE_ROOT / "lab_api_broker_release_v1_v24_schema_v6.json"
        ).read_text(encoding="utf-8")
    )
    expected_release_digest = release.pop("release_digest")
    assert release["response_schema_digest"] == hashlib.sha256(
        schema_path.read_bytes()
    ).hexdigest()
    assert sha256_digest(release) == expected_release_digest
    assert release["model"] == "gpt-5.4"
    assert release["retry_count"] == 0
