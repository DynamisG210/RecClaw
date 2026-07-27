#!/usr/bin/env python3
"""Run the treatment-free V14 Original and Research schema probe once."""

from __future__ import annotations

import json
from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.campaign_runtime import (
    campaign_projection,
    program_from_proposal,
)
from recclaw_core.experiments.helix_abc_v1.canary_broker import (
    original_canary_prompt,
    research_canary_prompt,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (
    LabApiCanaryBrokerV1,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src/recclaw_core/experiments/helix_abc_v1"
RESOURCE_ROOT = PACKAGE / "resources"
PROBE_ROOT = Path("/root/projects/RecClaw_v14_schema_conformance_v1")
CONFIG = Path("/root/projects/RecClaw_v2_0_Final_Reference/llm_api.md")


def main() -> None:
    if PROBE_ROOT.exists():
        raise RuntimeError("V14 schema conformance root is not fresh")
    broker = LabApiCanaryBrokerV1(
        PROBE_ROOT / "broker_private",
        schema_path=(
            RESOURCE_ROOT / "campaign_proposal_response_v1.schema.json"
        ),
        config_path=CONFIG,
        model="gpt-5.4",
        max_total_tokens_per_call=20_000,
        release_manifest_path=(
            RESOURCE_ROOT
            / "lab_api_broker_release_v1_v14_schema_v5.json"
        ),
    )
    projection = campaign_projection()
    try:
        original = broker.call_with_session(
            logical_call_id="v14-schema-original-shape-v1",
            proposal_generation_session_id=(
                "v14-schema-original-shape-session-v1"
            ),
            prompt=original_canary_prompt(
                round_index=1,
                search_seed=9216,
                phase_name="treatment-free schema conformance",
                catalog_projection=projection,
                original_state={},
            ),
            expected_proposal_count=4,
            max_total_tokens=20_000,
        )
        research = broker.call_with_session(
            logical_call_id="v14-schema-research-shape-v1",
            proposal_generation_session_id=(
                "v14-schema-research-shape-session-v1"
            ),
            prompt=research_canary_prompt(
                role="mechanism_composer",
                round_index=1,
                search_seed=9216,
                phase_name="treatment-free schema conformance",
                catalog_projection=projection,
                policy_directive={
                    "proposal_intent": "DISCOVERY",
                    "parent_policy": "EXPLICIT_ROOT_REQUEST",
                },
            ),
            expected_proposal_count=1,
            max_total_tokens=20_000,
        )
        for proposal in original.response["proposals"]:
            program_from_proposal(proposal)
            if proposal["original_priority"] not in {
                "high",
                "medium",
                "low",
            }:
                raise RuntimeError(
                    "Original conformance proposal lacks priority"
                )
        for proposal in research.response["proposals"]:
            program_from_proposal(proposal)
            if proposal["original_priority"] is not None:
                raise RuntimeError(
                    "Research conformance proposal populated Original priority"
                )
        print(
            json.dumps(
                {
                    "broker_release_digest": broker.release.release_digest,
                    "original_response_digest": original.response_digest,
                    "physical_calls": 2,
                    "research_response_digest": research.response_digest,
                    "retry_count": 0,
                    "verdict": "PASS",
                },
                sort_keys=True,
            )
        )
    finally:
        broker.close()


if __name__ == "__main__":
    main()
