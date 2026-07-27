#!/usr/bin/env python3
"""Run the frozen V14 chain Pilot once."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC, ROOT / "scripts"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from freeze_v13_pilot_contract import DEFAULT_LLM_CONFIG  # noqa: E402
from freeze_v14_pilot_contract import (  # noqa: E402
    DEFAULT_OUTPUT,
    verify_v14_pilot_contract,
)
from recclaw_core.experiments.helix_abc_v1.campaign_pilot_v14 import (  # noqa: E402
    V14_PILOT_ROUNDS_PER_ARM,
    V14_PILOT_SEARCH_SEED,
    V14PilotOrchestratorV1,
)
from recclaw_core.experiments.helix_abc_v1.meta_vnext_campaign import (  # noqa: E402
    MetaV19CampaignRuntimeV1,
)
from run_v13_pilot import execute_campaign_pilot  # noqa: E402


def execute(contract_path: Path, llm_api_config: Path) -> int:
    return execute_campaign_pilot(
        contract_path,
        llm_api_config,
        verify_contract=verify_v14_pilot_contract,
        meta_runtime_class=MetaV19CampaignRuntimeV1,
        orchestrator_class=V14PilotOrchestratorV1,
        search_seed=V14_PILOT_SEARCH_SEED,
        rounds_per_arm=V14_PILOT_ROUNDS_PER_ARM,
        version_label="V14",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--llm-api-config",
        type=Path,
        default=DEFAULT_LLM_CONFIG,
    )
    args = parser.parse_args()
    return execute(args.contract.resolve(), args.llm_api_config.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
