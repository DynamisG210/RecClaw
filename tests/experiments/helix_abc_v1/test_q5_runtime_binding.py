from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in os.sys.path:
    os.sys.path.insert(0, str(SRC))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from recclaw_core.experiments.helix_abc_v1.q5a_deployment import (  # noqa: E402
    build_q5a_deployment_manifest,
)
from recclaw_runtime_binding import (  # noqa: E402
    RuntimeBindingError,
    RuntimeBindingV1,
    claim_execution_owner,
    read_verified_preflight,
    read_verified_execution_gate,
    release_execution_owner,
    verify_execution_gate,
    validate_realize_prerequisites,
    write_execution_gate,
)


PYTHON = Path("/root/miniconda3/envs/recbole/bin/python").resolve()
PROJECTS = Path("/root/projects")
PREFREEZE = PROJECTS / "RecClaw_q5a_idea_feasibility/results/research_line/q5a_idea_feasibility_20260803_01/PREFREEZE_MANIFEST.json"
Q4_FIXTURE = ROOT / "results/research_line/q4_prospective_policy_comparison_20260803_01/shared_pool/FROZEN_SHARED_POOL_BEFORE_SELECTION.json"
R1_EXTERNAL = PROJECTS / "RecClaw_r1_r2_runs/fresh_r1_training_filesystem_fix_v3"
SEARCH = PROJECTS / "RecClaw_campaign_dataset_v1/search"
RECBole = PROJECTS / "RecBole"
API_CONFIG = PROJECTS / "RecClaw_v2_0_Final_Reference/llm_api.md"


def _copy_tree(source: Path, target: Path) -> None:
    shutil.copytree(
        source,
        target,
        ignore=shutil.ignore_patterns("__pycache__", ".git", ".pytest_cache"),
    )


def _relocated_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "relocated-repo"
    _copy_tree(ROOT / "src", repo / "src")
    _copy_tree(ROOT / "scripts", repo / "scripts")
    _copy_tree(ROOT / "configs", repo / "configs")
    _copy_tree(ROOT / "recclaw_ext", repo / "recclaw_ext")
    _copy_tree(ROOT / "tests/experiments/helix_abc_v1", repo / "tests/experiments/helix_abc_v1")
    _copy_tree(ROOT / "docs/research_line/vnext", repo / "docs/research_line/vnext")
    fixture_target = repo / "results/research_line/q4_prospective_policy_comparison_20260803_01/shared_pool/FROZEN_SHARED_POOL_BEFORE_SELECTION.json"
    fixture_target.parent.mkdir(parents=True)
    shutil.copy2(Q4_FIXTURE, fixture_target)
    prefreeze_target = repo / "results/research_line/q5a_idea_feasibility_20260803_01/PREFREEZE_MANIFEST.json"
    prefreeze_target.parent.mkdir(parents=True)
    shutil.copy2(PREFREEZE, prefreeze_target)

    projects_root = tmp_path / "mapped-projects"
    _copy_tree(R1_EXTERNAL, projects_root / "RecClaw_r1_r2_runs/fresh_r1_training_filesystem_fix_v3")
    search_root = tmp_path / "mapped-search"
    _copy_tree(SEARCH, search_root)
    recbole_root = tmp_path / "mapped-recbole"
    _copy_tree(RECBole / "recbole", recbole_root / "recbole")
    (recbole_root / ".git").mkdir()
    (recbole_root / ".git/HEAD").write_text("7b02be5ec80a88310f2d04a27a82adfcbb5dc211\n", encoding="utf-8")
    api_config = tmp_path / "mapped-api" / "llm_api.md"
    api_config.parent.mkdir()
    shutil.copy2(API_CONFIG, api_config)
    return repo, projects_root, search_root, recbole_root


def _manifest_and_binding(tmp_path: Path) -> tuple[Path, RuntimeBindingV1, Path, Path]:
    repo, projects_root, search_root, recbole_root = _relocated_fixture(tmp_path)
    preflight_root = tmp_path / "preflight"
    campaign_root = tmp_path / "campaign"
    api_config = tmp_path / "mapped-api/llm_api.md"
    manifest = build_q5a_deployment_manifest(
        repo_root=repo,
        projects_root=projects_root,
        search_data_root=search_root,
        recbole_root=recbole_root,
        python_executable=PYTHON,
        api_config=api_config,
        prefreeze_manifest=repo / "results/research_line/q5a_idea_feasibility_20260803_01/PREFREEZE_MANIFEST.json",
        preflight_root=preflight_root,
        campaign_root=campaign_root,
    )
    manifest_path = tmp_path / "Q5A_DEPLOYMENT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    binding = RuntimeBindingV1.from_manifest(manifest_path, repo_root=repo)
    return manifest_path, binding, repo, preflight_root


def _digest(value: dict[str, object]) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _write_pass_gate(binding: RuntimeBindingV1, repo: Path, preflight_root: Path) -> tuple[Path, Path, dict[str, object]]:
    preflight_root.mkdir(parents=True, exist_ok=True)
    preflight = {
        "schema": "recclaw.research-line.q5a-comprehensive-preflight.v1",
        "deployment_digest": binding.deployment_digest,
        "binding_digest": binding.binding_digest,
        "prefreeze_digest": binding.prefreeze_digest,
        "source_tree_digest": binding.source_tree_digest,
        "preflight_root": str(binding.preflight_root),
        "campaign_root": str(binding.campaign_root),
        "status": "PASS",
        "provider_calls": 0,
        "implementer_calls": 0,
        "qualification_calls": 0,
        "training_runs": 0,
        "held_out_reads": 0,
        "retries": 0,
    }
    preflight["preflight_digest"] = _digest(preflight)
    preflight_path = preflight_root / "PREFLIGHT.json"
    preflight_path.write_text(json.dumps(preflight, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    prefreeze_path = repo / "results/research_line/q5a_idea_feasibility_20260803_01/PREFREEZE_MANIFEST.json"
    gate = verify_execution_gate(
        binding=binding,
        preflight_receipt_path=preflight_path,
        prefreeze_manifest_path=prefreeze_path,
    )
    write_execution_gate(binding.campaign_root / "EXECUTION_GATE.json", gate)
    return preflight_path, prefreeze_path, gate


def test_relocated_preflight_binds_import_time_paths_and_renders_without_calls(tmp_path: Path) -> None:
    manifest_path, binding, repo, preflight_root = _manifest_and_binding(tmp_path)
    receipt_path = preflight_root / "PREFLIGHT.json"
    env = os.environ.copy()
    for name in (
        "RECCLAW_PROJECTS_ROOT",
        "RECCLAW_SEARCH_DATA_ROOT",
        "RECCLAW_RECBOLE_ROOT",
        "RECCLAW_API_CONFIG",
        "RECCLAW_RUNTIME_BINDING_DIGEST",
    ):
        env.pop(name, None)
    env["PYTHONPATH"] = f"{repo / 'src'}:{repo / 'scripts'}"
    command = [
        str(PYTHON),
        str(repo / "scripts/q5a_deployment_preflight.py"),
        "preflight",
        "--repo-root",
        str(repo),
        "--manifest",
        str(manifest_path),
        "--output",
        str(receipt_path),
    ]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["status"] == "PASS"
    assert receipt["provider_calls"] == receipt["implementer_calls"] == receipt["training_runs"] == 0
    assert receipt["deployment_digest"] == binding.deployment_digest
    assert receipt["preflight_root"] == str(preflight_root.resolve())
    checks = {row["name"]: row["status"] for row in receipt["checks"]}
    assert checks["runtime_binding_activation"] == "PASS"
    assert checks["path_sensitive_import_order"] == "PASS"
    assert checks["import_time_runtime_paths_exact"] == "PASS"
    assert checks["accepted_profile_registry_context"] == "PASS"
    assert checks["provider_request_rendering_no_call"] == "PASS"
    assert checks["implementer_prompt_rendering_no_call"] == "PASS"
    assert "/root/projects" not in json.dumps(receipt, sort_keys=True)


def test_gate_rejects_missing_stale_or_drifted_receipts_and_existing_attempt_root(tmp_path: Path) -> None:
    manifest_path, binding, repo, preflight_root = _manifest_and_binding(tmp_path)
    preflight_root.mkdir(parents=True)
    stale = preflight_root / "STALE.json"
    stale.write_text(json.dumps({"schema": "old", "status": "PASS"}), encoding="utf-8")
    with pytest.raises(RuntimeBindingError):
        read_verified_preflight(stale, binding)

    drifted = json.loads(manifest_path.read_text(encoding="utf-8"))
    drifted["runtime"]["search_data_root"] = str(tmp_path / "wrong-search")
    drifted_path = tmp_path / "DRIFTED.json"
    drifted_path.write_text(json.dumps(drifted, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(RuntimeBindingError):
        RuntimeBindingV1.from_manifest(drifted_path, repo_root=binding.repo_root)

    preflight = {
        "schema": "recclaw.research-line.q5a-comprehensive-preflight.v1",
        "status": "HARD_BLOCK",
    }
    blocked = preflight_root / "BLOCKED.json"
    blocked.write_text(json.dumps(preflight), encoding="utf-8")
    with pytest.raises(RuntimeBindingError):
        verify_execution_gate(
            binding=binding,
            preflight_receipt_path=blocked,
            prefreeze_manifest_path=binding.repo_root / "results/research_line/q5a_idea_feasibility_20260803_01/PREFREEZE_MANIFEST.json",
        )

    _write_pass_gate(binding, repo, preflight_root)
    attempt = binding.campaign_root / "pool-01/slot-01/physical_attempt_01"
    attempt.mkdir(parents=True)
    with pytest.raises(RuntimeBindingError):
        claim_execution_owner(binding, gate_receipt_path=binding.campaign_root / "EXECUTION_GATE.json")


def test_stale_persistent_gate_fails_closed_before_owner_claim(tmp_path: Path) -> None:
    _manifest_path, binding, repo, preflight_root = _manifest_and_binding(tmp_path)
    _write_pass_gate(binding, repo, preflight_root)
    gate_path = binding.campaign_root / "EXECUTION_GATE.json"
    stale_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    stale_gate["gate_contract"] = "OLD_GATE"
    stale_gate["gate_digest"] = _digest({key: value for key, value in stale_gate.items() if key != "gate_digest"})
    gate_path.write_text(json.dumps(stale_gate, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(RuntimeBindingError):
        claim_execution_owner(binding, gate_receipt_path=gate_path, stage="POOL")
    assert not (binding.campaign_root / "EXECUTION_OWNER.json").exists()
    assert not list(binding.campaign_root.rglob("physical_attempt_01"))


def test_runner_missing_binding_fails_before_import_or_provider(tmp_path: Path) -> None:
    output_root = tmp_path / "campaign"
    command = [
        str(PYTHON),
        str(ROOT / "scripts/run_q5a_idea_feasibility.py"),
        "pool",
        "--output-root",
        str(output_root),
        "--projects-root",
        str(PROJECTS),
        "--search-data-root",
        str(SEARCH),
        "--recbole-root",
        str(RECBole),
        "--python-executable",
        str(PYTHON),
        "--api-config",
        str(API_CONFIG),
        "--deployment-manifest",
        str(tmp_path / "missing.json"),
        "--preflight-receipt",
        str(tmp_path / "missing-preflight.json"),
        "--prefreeze-manifest",
        str(tmp_path / "missing-prefreeze.json"),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert completed.returncode != 0
    assert "recclaw_core" not in completed.stderr
    assert not output_root.exists()


def test_pass_gate_is_consumed_by_runner_before_any_attempt_and_releases_owner(tmp_path: Path) -> None:
    manifest_path, binding, repo, preflight_root = _manifest_and_binding(tmp_path)
    preflight_root.mkdir(parents=True)
    prefreeze_path = repo / "results/research_line/q5a_idea_feasibility_20260803_01/PREFREEZE_MANIFEST.json"
    receipt = {
        "schema": "recclaw.research-line.q5a-comprehensive-preflight.v1",
        "deployment_digest": binding.deployment_digest,
        "binding_digest": binding.binding_digest,
        "prefreeze_digest": binding.prefreeze_digest,
        "source_tree_digest": binding.source_tree_digest,
        "preflight_root": str(binding.preflight_root),
        "campaign_root": str(binding.campaign_root),
        "status": "PASS",
        "provider_calls": 0,
        "implementer_calls": 0,
        "qualification_calls": 0,
        "training_runs": 0,
        "held_out_reads": 0,
        "retries": 0,
    }
    receipt["preflight_digest"] = _digest(receipt)
    receipt_path = preflight_root / "PREFLIGHT.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    command = [
        str(PYTHON),
        str(repo / "scripts/run_q5a_idea_feasibility.py"),
        "pool",
        "--repo-root",
        str(repo),
        "--output-root",
        str(binding.campaign_root),
        "--projects-root",
        str(binding.projects_root),
        "--search-data-root",
        str(binding.search_data_root),
        "--recbole-root",
        str(binding.recbole_root),
        "--python-executable",
        str(PYTHON),
        "--api-config",
        str(binding.api_config),
        "--deployment-manifest",
        str(manifest_path),
        "--preflight-receipt",
        str(receipt_path),
        "--prefreeze-manifest",
        str(prefreeze_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo / 'src'}:{repo / 'scripts'}"
    completed = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert completed.returncode != 0
    assert (binding.campaign_root / "EXECUTION_GATE.json").is_file()
    released = binding.campaign_root / "EXECUTION_STAGE_POOL_RELEASED.json"
    assert released.is_file()
    assert json.loads(released.read_text(encoding="utf-8"))["status"] == "FAILED"
    assert not (binding.campaign_root / "EXECUTION_OWNER.json").exists()
    assert not list(binding.campaign_root.rglob("physical_attempt_01"))
    assert "prefreeze manifest is missing" in completed.stderr


def test_owner_claim_accepts_only_frozen_prefreeze_artifacts(tmp_path: Path) -> None:
    _manifest_path, binding, repo, preflight_root = _manifest_and_binding(tmp_path)
    _write_pass_gate(binding, repo, preflight_root)
    prefreeze = repo / "results/research_line/q5a_idea_feasibility_20260803_01/PREFREEZE_MANIFEST.json"
    shutil.copy2(prefreeze, binding.campaign_root / "PREFREEZE_MANIFEST.json")
    (binding.campaign_root / "POOL_GENERATION_PLAN.json").write_text("{}", encoding="utf-8")
    owner = claim_execution_owner(binding, gate_receipt_path=binding.campaign_root / "EXECUTION_GATE.json")
    release_execution_owner(binding, owner, status="TEST_RELEASED")
    assert (binding.campaign_root / "EXECUTION_STAGE_POOL_RELEASED.json").is_file()


def test_stage_aware_gate_pool_select_realize_lifecycle_is_no_call_and_single_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _manifest_path, binding, repo, preflight_root = _manifest_and_binding(tmp_path)
    _preflight_path, prefreeze_path, gate = _write_pass_gate(binding, repo, preflight_root)
    shutil.copy2(prefreeze_path, binding.campaign_root / "PREFREEZE_MANIFEST.json")

    pool_owner = claim_execution_owner(
        binding, gate_receipt_path=binding.campaign_root / "EXECUTION_GATE.json", stage="POOL"
    )
    with pytest.raises(RuntimeBindingError):
        claim_execution_owner(
            binding, gate_receipt_path=binding.campaign_root / "EXECUTION_GATE.json", stage="REALIZE"
        )

    pool_receipt = binding.campaign_root / "POOL_GENERATION_RECEIPT.json"
    pool_receipt.write_text(
        json.dumps(
            {
                "schema": "recclaw.research-line.q5a-pool-generation-receipt.v1",
                "all_pools_complete": True,
                "provider_calls": 0,
                "retries": 0,
                "held_out_reads": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    release_execution_owner(binding, pool_owner, status="COMPLETED")
    with pytest.raises(RuntimeBindingError):
        claim_execution_owner(
            binding, gate_receipt_path=binding.campaign_root / "EXECUTION_GATE.json", stage="POOL"
        )
    with pytest.raises(RuntimeBindingError):
        validate_realize_prerequisites(binding)

    import run_q5a_idea_feasibility as runner

    physical_calls = {"provider": 0}

    def no_provider_call(*_args: object, **_kwargs: object) -> None:
        physical_calls["provider"] += 1
        raise AssertionError("no-call lifecycle must not invoke Provider")

    monkeypatch.setattr(runner, "bounded_provider_call", no_provider_call)

    for pool_index in range(1, 4):
        pool_root = binding.campaign_root / "pools" / f"{pool_index:02d}"
        pool_root.mkdir(parents=True)
        rows = [
            {
                "preoutcome_score": {"spec_digest": f"{pool_index:02d}{slot:02d}".ljust(64, "0")},
                "stage": "OPENSPEC_FROZEN",
            }
            for slot in range(1, 9)
        ]
        raw = {"candidate_pools": {"shared": rows}}
        pool_digest = _digest({"pool_index": pool_index})
        (pool_root / "RAW_POOL_BEFORE_SELECTION.json").write_text(
            json.dumps(raw, sort_keys=True, separators=(",", ":")), encoding="utf-8"
        )
        (pool_root / "POOL_MANIFEST.json").write_text(
            json.dumps({"pool_digest": pool_digest, "candidate_pools": {"shared": rows}}, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    def fake_ranked(pool: dict[str, object], policy: str, **_kwargs: object) -> tuple[object, ...]:
        ids = [str(row["preoutcome_score"]["spec_digest"]) for row in pool["candidate_pools"]["shared"]]
        selected = ids[:2]
        return selected, [], {candidate_id: "TEST" for candidate_id in selected}, {candidate_id: float(candidate_id in selected) for candidate_id in ids}, None, None, "TEST", None, "NONE"

    monkeypatch.setattr(runner, "_ranked_selection", fake_ranked)
    f1_policy = tmp_path / "f1.json"
    outcome_policy = tmp_path / "outcome.json"
    outcome_activation = tmp_path / "activation.json"
    for path in (f1_policy, outcome_policy, outcome_activation):
        path.write_text("{}", encoding="utf-8")
    runner.select(
        type(
            "Args",
            (),
            {
                "output_root": binding.campaign_root,
                "f1_policy": f1_policy,
                "outcome_policy": outcome_policy,
                "outcome_activation": outcome_activation,
            },
        )()
    )
    assert validate_realize_prerequisites(binding)["gate_digest"] == gate["gate_digest"]

    realize_owner = claim_execution_owner(
        binding, gate_receipt_path=binding.campaign_root / "EXECUTION_GATE.json", stage="REALIZE"
    )
    (binding.campaign_root / "REALIZATION_EXECUTION_RECEIPT.json").write_text("{}", encoding="utf-8")
    release_execution_owner(binding, realize_owner, status="COMPLETED")
    with pytest.raises(RuntimeBindingError):
        claim_execution_owner(
            binding, gate_receipt_path=binding.campaign_root / "EXECUTION_GATE.json", stage="REALIZE"
        )

    stale_gate = dict(gate)
    stale_gate["gate_contract"] = "OLD_GATE"
    stale_gate["gate_digest"] = _digest({key: value for key, value in stale_gate.items() if key != "gate_digest"})
    stale_path = binding.campaign_root / "STALE_GATE.json"
    stale_path.write_text(json.dumps(stale_gate, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    with pytest.raises(RuntimeBindingError):
        read_verified_execution_gate(stale_path, binding)
    assert physical_calls["provider"] == 0
    assert not list(binding.campaign_root.rglob("physical_attempt_01"))
