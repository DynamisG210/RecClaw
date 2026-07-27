#!/usr/bin/env python3
"""Freeze or verify the fresh V14 chain Pilot contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.freeze_v13_pilot_contract import (
        DEFAULT_LLM_CONFIG,
        G7_GATE,
        PROPOSAL_SCHEMA,
        RESOURCE_ROOT,
        ROOT,
        _source_files,
        build_contract,
        file_sha256,
    )
except ModuleNotFoundError:
    from freeze_v13_pilot_contract import (
        DEFAULT_LLM_CONFIG,
        G7_GATE,
        PROPOSAL_SCHEMA,
        RESOURCE_ROOT,
        ROOT,
        _source_files,
        build_contract,
        file_sha256,
    )
from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v14 import (
    V14_PILOT_ASSIGNMENT_NONCE,
    V14_PILOT_ROUNDS_PER_ARM,
    V14_PILOT_SEARCH_SEED,
    V14PilotStoreContractV1,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (
    canonical_json_bytes,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (
    CHECKPOINT_SHA256_V19,
    POLICY_BUNDLE_DIGEST_V19,
    PROMOTION_DECISION_DIGEST_V19,
    MetaV19CampaignRuntimeV1,
)


DEFAULT_OUTPUT = (
    ROOT
    / "docs"
    / "research_line"
    / "continuous_program"
    / "V14_FROZEN_CHAIN_PILOT_CONTRACT.json"
)
DEFAULT_OUTPUT_ROOT = Path("/root/projects/RecClaw_campaign_pilot_9216_v14")
META_CHECKPOINT = RESOURCE_ROOT / "meta_vnext_policy_checkpoint_v19.json"
META_PROMOTION = (
    ROOT
    / "docs"
    / "research_line"
    / "continuous_program"
    / "META_V19_TRANSPORT_REQUALIFICATION.json"
)
BROKER_RELEASE = (
    RESOURCE_ROOT / "lab_api_broker_release_v1_v14_schema_v5.json"
)
TRAINING_RELEASE = RESOURCE_ROOT / "training_runtime_release_v6.json"
RECORD_SCHEMA = "recclaw.v14-pilot-contract.v1"
SCIENTIFIC_GATE = (
    ROOT
    / "docs"
    / "research_line"
    / "continuous_program"
    / "V14_SCIENTIFIC_ATTRIBUTION_GATE.json"
)


def _v14_source_files() -> tuple[str, ...]:
    replacements = {
        "scripts/freeze_campaign_training_runtime_release_v5.py": (
            "scripts/freeze_campaign_training_runtime_release_v6.py"
        ),
        "scripts/freeze_v13_pilot_contract.py": (
            "scripts/freeze_v14_pilot_contract.py"
        ),
        "scripts/run_v13_pilot.py": "scripts/run_v14_pilot.py",
        "src/recclaw_core/experiments/helix_abc_v1/campaign_pilot_v13.py": (
            "src/recclaw_core/experiments/helix_abc_v1/"
            "campaign_pilot_v14.py"
        ),
    }
    result = [replacements.get(item, item) for item in _source_files()]
    result.extend(
        [
            "scripts/freeze_lab_api_broker_release_v14.py",
            "scripts/run_v14_schema_conformance_probe.py",
        ]
    )
    return tuple(sorted(set(result)))


def build_v14_contract(*, llm_api_config: Path) -> dict[str, object]:
    return build_contract(
        llm_api_config=llm_api_config,
        output_root=DEFAULT_OUTPUT_ROOT,
        store_contract_factory=V14PilotStoreContractV1.create,
        assignment_nonce=V14_PILOT_ASSIGNMENT_NONCE,
        search_seed=V14_PILOT_SEARCH_SEED,
        rounds_per_arm=V14_PILOT_ROUNDS_PER_ARM,
        meta_runtime_class=MetaV19CampaignRuntimeV1,
        meta_checkpoint=META_CHECKPOINT,
        meta_promotion=META_PROMOTION,
        meta_checkpoint_sha256=CHECKPOINT_SHA256_V19,
        meta_policy_bundle_digest=POLICY_BUNDLE_DIGEST_V19,
        meta_promotion_digest=PROMOTION_DECISION_DIGEST_V19,
        broker_release_path=BROKER_RELEASE,
        training_release_path=TRAINING_RELEASE,
        scientific_gate_path=SCIENTIFIC_GATE,
        record_schema=RECORD_SCHEMA,
        source_files=_v14_source_files(),
    )


def verify_v14_pilot_contract(path: Path) -> dict[str, object]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    preimage = dict(contract)
    expected = preimage.pop("content_digest")
    if sha256_digest(preimage) != expected:
        raise RuntimeError("V14 contract content digest mismatch")
    gate = json.loads(SCIENTIFIC_GATE.read_text(encoding="utf-8"))
    if (
        contract["record_schema"] != RECORD_SCHEMA
        or contract["status"] != "FROZEN_PRE_OUTCOME"
        or contract["pilot_started"] is not False
        or contract["broker"]["model"] != "gpt-5.4"
        or contract["common_substrate"]["executable_mechanism_count"] != 66
        or contract["meta"]["policy_bundle_digest"]
        != POLICY_BUNDLE_DIGEST_V19
        or contract["guard_and_fusion"]["gate_result_digest"]
        != gate["gate_result_digest"]
    ):
        raise RuntimeError("V14 contract invariant mismatch")
    expected_store = V14PilotStoreContractV1.create()
    if contract["pilot"] != {
        "experiment_id": expected_store.experiment_id,
        "ordinary_execution_seed": expected_store.ordinary_execution_seed,
        "rounds_per_arm": expected_store.scheduled_slots_per_arm_seed,
        "search_seeds": list(expected_store.search_seeds),
        "store_contract_identity_digest": expected_store.identity_digest,
    }:
        raise RuntimeError("V14 store contract identity mismatch")
    for relative, digest in contract["source"]["files"].items():
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"V14 source identity mismatch: {relative}")
    exact = {
        Path(contract["broker"]["release_manifest_path"]): contract[
            "broker"
        ]["release_manifest_sha256"],
        Path(contract["broker"]["response_schema_path"]): contract[
            "broker"
        ]["response_schema_sha256"],
        Path(contract["meta"]["checkpoint_path"]): contract["meta"][
            "checkpoint_sha256"
        ],
        Path(contract["training"]["release_manifest_path"]): contract[
            "training"
        ]["release_manifest_sha256"],
    }
    for artifact, digest in exact.items():
        if file_sha256(artifact) != digest:
            raise RuntimeError(f"V14 release identity mismatch: {artifact}")
    if Path(contract["output_root"]).exists():
        raise RuntimeError("V14 Pilot output root is no longer fresh")
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--llm-api-config",
        type=Path,
        default=DEFAULT_LLM_CONFIG,
    )
    args = parser.parse_args()
    if args.verify is not None:
        contract = verify_v14_pilot_contract(args.verify.resolve())
        print(
            json.dumps(
                {
                    "content_digest": contract["content_digest"],
                    "pilot_started": contract["pilot_started"],
                    "verdict": "PASS",
                },
                sort_keys=True,
            )
        )
        return 0
    contract = build_v14_contract(
        llm_api_config=args.llm_api_config.resolve()
    )
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError("V14 frozen contract already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json_bytes(contract) + b"\n")
    print(
        json.dumps(
            {
                "content_digest": contract["content_digest"],
                "path": output.as_posix(),
                "sha256": file_sha256(output),
                "verdict": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
