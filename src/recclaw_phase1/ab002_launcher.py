"""Materialize and launch isolated AB-002 control or treatment repetitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = PROJECT_ROOT / "configs/phase1/ab002/evidence_guard_contract.json"
ARM_MANIFEST = PROJECT_ROOT / "configs/phase1/ab002/arm_manifest.json"
S0_MANIFEST = PROJECT_ROOT / "configs/phase1/ab002/s0_manifest.json"
TREATMENT_OVERLAY = PROJECT_ROOT / "phase1/overlays/treatment_agent.patch"
FROZEN_GUARD_FILES = {
    "src/recclaw_core/__init__.py": "ef861a60d2af26d298c91a1b54dc3860e34d5f7b17e3fbd096bc6e28805fb07b",
    "src/recclaw_core/exploration/__init__.py": "e7f8d2c9cdba82c7af5239208e94df70c9016a6e75c33d255546e9dd18f60438",
    "src/recclaw_core/exploration/evidence_guard.py": "d47df73feec97a01f2528cbf110b62c473d16414fcfc94ffefaaad3ff0a7c1af",
    "src/recclaw_core/exploration/original_recclaw_adapter.py": "acd25fb34be71bde445a91614f115cd4ada0aad58c66754ae3a2a6dca1d5aa35",
    "src/recclaw_core/exploration/original_recclaw_guard_hook.py": "a63a4fc1067313a8cf7799d9a1b75593221c8e4acd34d553abf8deaa32165340",
}
FROZEN_TREATMENT_OVERLAY_SHA256 = (
    "2bfeeb6f4fd7893784899278ae2c4beee5e539da57d0532670edb5102f3e24c4"
)
FROZEN_TREATMENT_AGENT_SHA256 = (
    "00705ed1dcbd28c57dec145dfebcf6a04bcadc053013e488f2555b446835e79a"
)
RUNTIME_SCRIPT_FILES = (
    "scripts/action_space.py",
    "scripts/agent.py",
    "scripts/analysis/build_candidate_search_tree.py",
    "scripts/analysis/lint_recclaw_space.py",
    "scripts/build_experience_summary.py",
    "scripts/collect_result.py",
    "scripts/compare_runs.py",
    "scripts/implement_candidate_proposal.py",
    "scripts/promote_candidate_proposal.py",
    "scripts/propose_candidate.py",
    "scripts/run_candidate.py",
    "scripts/run_reflection_pilot.py",
    "scripts/validate_candidate_proposal.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_runtime_root(runtime_root: Path) -> Path:
    resolved = runtime_root.expanduser().resolve()
    if not resolved.is_absolute():
        raise ValueError("runtime root must be absolute")
    if _inside(resolved, PROJECT_ROOT) or _inside(PROJECT_ROOT, resolved):
        raise ValueError("runtime root must be outside and must not contain the source checkout")
    return resolved


def runtime_source_files() -> list[Path]:
    files = [PROJECT_ROOT / rel for rel in RUNTIME_SCRIPT_FILES]
    files.extend(sorted((PROJECT_ROOT / "configs").glob("*.yaml")))
    files.extend(sorted((PROJECT_ROOT / "configs/candidates").glob("*.yaml")))
    files.extend(sorted((PROJECT_ROOT / "recclaw_ext").rglob("*.py")))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("runtime source allowlist is incomplete: " + ", ".join(missing))
    return sorted(set(files), key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())


def copy_allowlisted_source(destination: Path) -> list[dict[str, Any]]:
    rows = []
    for source in runtime_source_files():
        rel = source.relative_to(PROJECT_ROOT)
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        rows.append(
            {
                "path": rel.as_posix(),
                "size_bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
    return rows


def _assert_frozen_guard_source() -> None:
    for rel, expected in FROZEN_GUARD_FILES.items():
        actual = sha256_file(PROJECT_ROOT / rel)
        if actual != expected:
            raise ValueError(f"frozen Guard file drift: {rel}: {actual} != {expected}")
    overlay = sha256_file(TREATMENT_OVERLAY)
    if overlay != FROZEN_TREATMENT_OVERLAY_SHA256:
        raise ValueError(f"treatment overlay drift: {overlay}")


def materialize_arm(
    *, runtime_root: Path, arm: str, search_seed: int
) -> dict[str, Any]:
    if arm not in {"control", "treatment"}:
        raise ValueError("arm must be control or treatment")
    root = validate_runtime_root(runtime_root)
    repetition = root / "runs" / arm / f"search_seed_{search_seed}"
    if repetition.exists():
        raise FileExistsError(f"repetition root must be absent: {repetition}")
    source_root = repetition / "source"
    source_root.mkdir(parents=True)
    source_rows = copy_allowlisted_source(source_root)
    integration_rows: list[dict[str, Any]] = []
    integration_root: Path | None = None
    if arm == "treatment":
        _assert_frozen_guard_source()
        completed = subprocess.run(
            ["git", "apply", "--check", str(TREATMENT_OVERLAY)],
            cwd=source_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError("treatment overlay check failed: " + completed.stderr.strip())
        subprocess.run(
            ["git", "apply", str(TREATMENT_OVERLAY)],
            cwd=source_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        patched_agent = sha256_file(source_root / "scripts/agent.py")
        if patched_agent != FROZEN_TREATMENT_AGENT_SHA256:
            raise ValueError(f"patched treatment agent drift: {patched_agent}")
        integration_root = repetition / "treatment_integration"
        for rel, expected in FROZEN_GUARD_FILES.items():
            source = PROJECT_ROOT / rel
            target = integration_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            actual = sha256_file(target)
            if actual != expected:
                raise ValueError(f"copied Guard file drift: {rel}")
            integration_rows.append(
                {"path": rel, "size_bytes": target.stat().st_size, "sha256": actual}
            )
        contract_target = integration_root / "evidence_guard_contract.json"
        shutil.copy2(CONTRACT, contract_target)
        integration_rows.append(
            {
                "path": "evidence_guard_contract.json",
                "size_bytes": contract_target.stat().st_size,
                "sha256": sha256_file(contract_target),
            }
        )
    else:
        forbidden = [
            path
            for path in source_root.rglob("*")
            if path.is_file()
            and (
                "evidence_guard" in path.name.lower()
                or b"recclaw_core.exploration" in path.read_bytes()
            )
        ]
        if forbidden:
            raise ValueError("control source contains Guard material: " + ", ".join(map(str, forbidden)))
    record = {
        "record_type": "RECCLAW_PHASE1_AB002_RUNTIME_MATERIALIZATION",
        "schema_version": "recclaw.phase1.ab002.runtime_materialization.v1",
        "status": "LOCAL_COMPLETE",
        "run_status": "NOT_STARTED",
        "arm": arm,
        "search_seed": search_seed,
        "source_root": str(source_root),
        "source_files": source_rows,
        "integration_root": None if integration_root is None else str(integration_root),
        "integration_files": integration_rows,
        "control_guard_material_count": 0 if arm == "control" else None,
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }
    canonical_write(repetition / "materialization_manifest.json", record)
    return record


def build_launch_plan(
    *,
    runtime_root: Path,
    arm: str,
    search_seed: int,
    gpu_id: int,
    pair_id: str,
    broker_url: str,
    baseline_dir: Path,
    python_executable: str,
) -> dict[str, Any]:
    root = validate_runtime_root(runtime_root)
    repetition = root / "runs" / arm / f"search_seed_{search_seed}"
    source_root = repetition / "source"
    output_root = repetition / "outputs"
    broker_base = f"{broker_url.rstrip('/')}/v1/pairs/{pair_id}/{arm}"
    command = [
        python_executable,
        str(source_root / "scripts/run_reflection_pilot.py"),
        "--gpu-id",
        str(gpu_id),
        "--rounds",
        "20",
        "--start-round",
        "1",
        "--search-seed",
        str(search_seed),
        "--refresh-experience-every",
        "10",
        "--proposal-source",
        "llm",
        "--loop-mode",
        "auto",
        "--baseline-dir",
        str(baseline_dir.expanduser().resolve()),
        "--llm-provider",
        "compatible",
        "--llm-model",
        "gpt-5.4",
        "--llm-base-url",
        broker_base,
        "--llm-timeout",
        "300",
        "--llm-retries",
        "3",
        "--llm-api-key-env",
        "OPENAI_API_KEY",
        "--pilot-root",
        str(output_root),
        "--stamp",
        "pilot",
        "--search-intensity",
        "algorithm_first",
        "--algorithm-budget-per-window",
        "3",
    ]
    environment = {
        "RECCLAW_AB_ARM": arm,
        "RECCLAW_LLM_PAIR_ID": pair_id,
        "RECCLAW_SEARCH_SEED": str(search_seed),
        "RECCLAW_RUNTIME_ROOT": str(root),
        "OPENAI_API_KEY": "FROM_RECCLAW_BROKER_CLIENT_TOKEN",
    }
    if arm == "treatment":
        integration_root = repetition / "treatment_integration"
        environment.update(
            {
                "PYTHONPATH": str(integration_root / "src"),
                "RECCLAW_EVIDENCE_GUARD_MODE": "active",
                "RECCLAW_EVIDENCE_GUARD_CONTRACT": str(
                    integration_root / "evidence_guard_contract.json"
                ),
                "RECCLAW_EVIDENCE_GUARD_FEEDBACK": str(
                    output_root / "pilot/evidence_guard_feedback.jsonl"
                ),
            }
        )
    return {
        "record_type": "RECCLAW_PHASE1_AB002_LAUNCH_PLAN",
        "schema_version": "recclaw.phase1.ab002.launch_plan.v1",
        "status": "LOCAL_COMPLETE",
        "run_status": "NOT_STARTED",
        "arm": arm,
        "pair_id": pair_id,
        "search_seed": search_seed,
        "gpu_id": gpu_id,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "command_argv": command,
        "environment_without_secrets": environment,
        "seed_from_argument_present": False,
        "all_outputs_external": True,
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }


def _execute(plan: dict[str, Any], *, client_token_env: str) -> int:
    if os.environ.get("RECCLAW_AB002_START_AUTHORIZED") != "YES":
        raise SystemExit("execution requires RECCLAW_AB002_START_AUTHORIZED=YES")
    token = os.environ.get(client_token_env, "")
    if not token:
        raise SystemExit(f"missing broker client token env: {client_token_env}")
    source_root = Path(plan["source_root"])
    if not source_root.is_dir():
        raise SystemExit("materialize the selected arm before execution")
    output_root = Path(plan["output_root"])
    if output_root.exists():
        raise SystemExit(f"output root must be absent: {output_root}")
    baseline = Path(plan["command_argv"][plan["command_argv"].index("--baseline-dir") + 1])
    if not baseline.is_dir():
        raise SystemExit(f"baseline directory missing: {baseline}")
    env = dict(os.environ)
    for key, value in plan["environment_without_secrets"].items():
        if key != "OPENAI_API_KEY":
            env[key] = str(value)
    env["OPENAI_API_KEY"] = token
    repetition = source_root.parent
    stdout_path = repetition / "launcher_stdout.log"
    stderr_path = repetition / "launcher_stderr.log"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(
            list(plan["command_argv"]),
            cwd=source_root,
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return int(completed.returncode)


def build_parser(arm: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Prepare or launch AB-002 {arm}")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--search-seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--broker-url", required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--python-executable", default=sys.executable)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--materialize-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--client-token-env", default="RECCLAW_BROKER_CLIENT_TOKEN")
    return parser


def arm_main(arm: str, argv: list[str] | None = None) -> int:
    args = build_parser(arm).parse_args(argv)
    plan = build_launch_plan(
        runtime_root=args.runtime_root,
        arm=arm,
        search_seed=args.search_seed,
        gpu_id=args.gpu_id,
        pair_id=args.pair_id,
        broker_url=args.broker_url,
        baseline_dir=args.baseline_dir,
        python_executable=args.python_executable,
    )
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    if not Path(plan["source_root"]).exists():
        materialize_arm(
            runtime_root=args.runtime_root,
            arm=arm,
            search_seed=args.search_seed,
        )
    if args.materialize_only:
        print(json.dumps(plan, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    return _execute(plan, client_token_env=args.client_token_env)


def control_main(argv: list[str] | None = None) -> int:
    return arm_main("control", argv)


def treatment_main(argv: list[str] | None = None) -> int:
    return arm_main("treatment", argv)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare or launch one AB-002 arm")
    parser.add_argument("--arm", choices=("control", "treatment"), required=True)
    args, remaining = parser.parse_known_args(argv)
    return arm_main(args.arm, remaining)


if __name__ == "__main__":
    raise SystemExit(main())
