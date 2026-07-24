#!/usr/bin/env python3
"""Execute the single authorized post-M6E Pilot V5 once."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from run_m6_pilot import execute, file_sha256  # noqa: E402

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.m6e_conformance import (  # noqa: E402
    require_m6e_conformance_packet,
)
from recclaw_core.experiments.helix_abc_v1.precanary_orchestration import (  # noqa: E402
    PrivateTreatmentAssignmentV1,
)
from recclaw_core.experiments.helix_abc_v1.real_pilot import (  # noqa: E402
    FRESH_PILOT_V5_SEARCH_SEED,
    FreshPilotOrchestratorV3,
    PilotStoreContractV3,
    pilot_budget,
)
from recclaw_core.experiments.helix_abc_v1.runtime_release import (  # noqa: E402
    common_release_projection_digest,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_contracts import (  # noqa: E402
    TrainingExecutionPurposeV1,
)
from recclaw_core.experiments.helix_abc_v1.training_runtime_release import (  # noqa: E402
    TRAINING_RUNNER_ABI,
    training_runtime_release_digest,
)


CONTRACT_PATH = (
    ROOT
    / "docs"
    / "research_line"
    / "m6"
    / "DEVELOPMENT_PILOT_CONTRACT_V5.json"
)
SEALED_PILOT_SEEDS = frozenset({9201, 9202, 9203, 9204})


def verify_contract_v5(contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    content = dict(contract)
    expected = content.pop("content_digest")
    if sha256_digest(content) != expected:
        raise RuntimeError("Pilot V5 contract content digest mismatch")
    for relative, expected_hash in contract["source"]["files"].items():
        if file_sha256(ROOT / relative) != expected_hash:
            raise RuntimeError(f"Pilot V5 source identity mismatch: {relative}")
    exact_files = {
        Path(contract["broker"]["response_schema_path"]): contract["broker"][
            "response_schema_sha256"
        ],
        Path(contract["bl_icf"]["template_fixture_path"]): contract["bl_icf"][
            "template_fixture_sha256"
        ],
        Path(contract["training"]["profile_path"]): contract["training"][
            "profile_sha256"
        ],
        Path(contract["broker"]["codex_executable"]): contract["broker"][
            "codex_executable_sha256"
        ],
    }
    for path, expected_hash in exact_files.items():
        if file_sha256(path) != expected_hash:
            raise RuntimeError(f"Pilot V5 external identity mismatch: {path}")
    if contract["status"] != "FROZEN_PRE_OUTCOME":
        raise RuntimeError("Pilot V5 contract is not frozen")

    expected_store = PilotStoreContractV3.create()
    expected_pilot = {
        "experiment_id": expected_store.experiment_id,
        "ordinary_execution_seed": expected_store.ordinary_execution_seed,
        "rounds_per_arm": expected_store.scheduled_slots_per_arm_seed,
        "search_seeds": list(expected_store.search_seeds),
        "store_contract_identity_digest": expected_store.identity_digest,
    }
    if contract["record_schema"] != "recclaw.development-pilot-contract.v3":
        raise RuntimeError("fresh Pilot contract schema is not V3")
    if contract["pilot"] != expected_pilot:
        raise RuntimeError("fresh Pilot contract does not bind V5 store identity")
    assignment = PrivateTreatmentAssignmentV1.create(
        expected_store.experiment_id,
        nonce="M6-PILOT-9205-OPAQUE-V5",
    )
    if contract["assignment"] != {
        "commitment": assignment.commitment,
        "opaque": True,
    }:
        raise RuntimeError("fresh Pilot V5 treatment assignment mismatch")
    if contract["pilot"]["search_seeds"] != [FRESH_PILOT_V5_SEARCH_SEED]:
        raise RuntimeError("Pilot V5 seed is not the frozen smallest unused seed")
    if SEALED_PILOT_SEEDS.intersection(contract["pilot"]["search_seeds"]):
        raise RuntimeError("sealed Pilot seed reuse is forbidden")

    if (
        contract["training"]["runner_abi"] != TRAINING_RUNNER_ABI
        or contract["training"]["runtime_release_digest"]
        != training_runtime_release_digest()
        or contract["training"]["execution_purpose"]
        != TrainingExecutionPurposeV1.PILOT.value
    ):
        raise RuntimeError("Pilot V5 training release identity mismatch")
    if contract["budget_per_arm_round"] != pilot_budget().to_dict():
        raise RuntimeError("Pilot V5 budget differs from the frozen Pilot")
    if (
        contract["bl_icf"]["common_release_projection_digest"]
        != common_release_projection_digest()
    ):
        raise RuntimeError("Pilot V5 common BL/runtime projection mismatch")
    expected_arms = {
        "A": {
            "controller": "OriginalControllerV1",
            "evidence_port": "NullEvidencePortV1",
            "physical_llm_call_ceiling_per_round": 1,
        },
        "B": {
            "controller": "ResearchLineControllerV1",
            "evidence_port": "NullEvidencePortV1",
            "physical_llm_call_ceiling_per_round": 4,
        },
        "C": {
            "controller": "ResearchLineControllerV1",
            "evidence_port": "EvidenceGuardPortV1",
            "physical_llm_call_ceiling_per_round": 4,
        },
    }
    if contract["arm_composition"] != expected_arms:
        raise RuntimeError("Pilot V5 A/B/C treatment definition mismatch")

    packet = require_m6e_conformance_packet(ROOT)
    if (
        contract["m6e"]["conformance_packet_digest"]
        != packet["content_digest"]
        or contract["m6e"]["runtime_release_digest"]
        != packet["training_runtime_release_digest"]
        or contract["m6e"]["P0"] != 0
        or contract["m6e"]["P1"] != 0
    ):
        raise RuntimeError("Pilot V5 does not bind the passing M6E gate")
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    return execute(
        args.contract.resolve(),
        args.output_root.resolve(),
        contract_verifier=verify_contract_v5,
        orchestrator_type=FreshPilotOrchestratorV3,
    )


if __name__ == "__main__":
    raise SystemExit(main())
