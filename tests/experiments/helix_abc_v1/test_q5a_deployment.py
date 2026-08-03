from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.q5a_deployment import (  # noqa: E402
    EXPECTED_PROVIDER_ENDPOINT_DIGEST,
    EXPECTED_RECBOLE_COMMIT,
    build_q5a_deployment_manifest,
)


def test_deployment_manifest_declares_complete_static_and_runtime_boundary() -> None:
    manifest = build_q5a_deployment_manifest(
        repo_root=ROOT,
        projects_root=Path("/root/projects"),
        search_data_root=Path("/root/projects/RecClaw_campaign_dataset_v1/search"),
        recbole_root=Path("/root/projects/RecBole"),
        python_executable=Path("/root/miniconda3/envs/recbole/bin/python"),
        api_config=Path("/root/projects/RecClaw_v2_0_Final_Reference/llm_api.md"),
    )

    assert manifest["schema"] == "recclaw.research-line.q5a-deployment-manifest.v1"
    assert manifest["static_import_graph"]["unresolved_internal_imports"] == []
    assert manifest["provider"]["endpoint_digest"] == EXPECTED_PROVIDER_ENDPOINT_DIGEST
    assert manifest["runtime"]["recbole_head"] == EXPECTED_RECBOLE_COMMIT
    assert manifest["stage_consumers"]["dynamic_candidate_entrypoint"] == "candidate_package_relative:recclaw_ext/candidate.py"
