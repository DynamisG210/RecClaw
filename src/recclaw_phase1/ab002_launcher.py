"""Materialize and launch isolated AB-002 control or treatment repetitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from recclaw_phase1.ab002_s0 import (
    BASELINE_INPUTS,
    INITIAL_STATE_MANIFEST,
    S0_ID,
    copy_s0_source,
    validate_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = PROJECT_ROOT / "configs/phase1/ab002/experiment_contract.json"
ARM_MANIFEST = PROJECT_ROOT / "configs/phase1/ab002/arm_manifest.json"
S0_MANIFEST = INITIAL_STATE_MANIFEST
TREATMENT_OVERLAY = PROJECT_ROOT / "phase1/overlays/treatment_agent.patch"
FROZEN_GUARD_FILES = {
    "src/recclaw_core/__init__.py": "ef861a60d2af26d298c91a1b54dc3860e34d5f7b17e3fbd096bc6e28805fb07b",
    "src/recclaw_core/exploration/__init__.py": "e7f8d2c9cdba82c7af5239208e94df70c9016a6e75c33d255546e9dd18f60438",
    "src/recclaw_core/exploration/evidence_guard.py": "d47df73feec97a01f2528cbf110b62c473d16414fcfc94ffefaaad3ff0a7c1af",
    "src/recclaw_core/exploration/original_recclaw_adapter.py": "acd25fb34be71bde445a91614f115cd4ada0aad58c66754ae3a2a6dca1d5aa35",
    "src/recclaw_core/exploration/original_recclaw_guard_hook.py": "a63a4fc1067313a8cf7799d9a1b75593221c8e4acd34d553abf8deaa32165340",
}
FROZEN_TREATMENT_OVERLAY_SHA256 = (
    "48a0a65e8e940679a3880a058b2978508bc050eb1b022a4053f8f54fdb49dfc0"
)
FROZEN_TREATMENT_AGENT_SHA256 = (
    "f5a26e7def57cdb325a1e48ae69295187b615d41e319be3205da70f6d6929c1f"
)
TREATMENT_PATCH_HEADER = (
    "diff --git a/scripts/agent.py b/scripts/agent.py\n",
    "--- a/scripts/agent.py\n",
    "+++ b/scripts/agent.py\n",
)
HUNK_HEADER = re.compile(
    r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?: .*)?\n?$"
)
TREATMENT_ONLY_ENV_KEYS = (
    "PYTHONPATH",
    "RECCLAW_EVIDENCE_GUARD_MODE",
    "RECCLAW_EVIDENCE_GUARD_CONTRACT",
    "RECCLAW_EVIDENCE_GUARD_FEEDBACK",
)
UPSTREAM_ONLY_ENV_KEYS = (
    "RECCLAW_LAB_LLM_API_KEY",
    "RECCLAW_LAB_LLM_BASE_URL",
)
CANARY_SEED = 9001
FULL_SEARCH_SEEDS = (42, 43, 44, 45, 46, 47)
SEARCH_SEEDS = (CANARY_SEED, *FULL_SEARCH_SEEDS)
FULL_EXPERIMENT_DIRECTIVE = (
    "RecClaw algorithm-first pilot: keep the ML-1M full-sort general-rec protocol unchanged; "
    "prioritize LLM-driven algorithm discovery, mechanism composition, local extension implementation, "
    "smoke verification, and formal runs. Parameter-only tuning is allowed only as a small sanity/refinement "
    "budget after credible algorithm signal. Do not propose sequential recommendation in this experiment line."
)
CANARY_EXPERIMENT_DIRECTIVE = (
    "AB-002 integration Canary under the unchanged ML-1M full-sort protocol. In the first scheduled proposal "
    "batch include at least one genuinely code_required BPR or LightGCN candidate so the common implementation, "
    "training, result interpretation, and memory paths are exercised. Do not change split, sampling, candidate "
    "universe, metric, baseline, model family, or resource budget."
)
RUNTIME_GUARD_CONTRACT_KEYS = (
    "claim",
    "protocol",
    "protocol_implementation",
    "live_postcheck_binding",
    "current_evidence",
    "metric_projection",
    "seed_policy",
    "execution_environment",
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


def copy_allowlisted_source(destination: Path) -> list[dict[str, Any]]:
    return copy_s0_source(destination)


def expected_rounds(search_seed: int) -> int:
    if search_seed == CANARY_SEED:
        return 3
    if search_seed in FULL_SEARCH_SEEDS:
        return 20
    raise ValueError(f"unsupported AB-002 search seed: {search_seed}")


def expected_candidate_executions(search_seed: int) -> int:
    if search_seed == CANARY_SEED:
        return 6
    if search_seed in FULL_SEARCH_SEEDS:
        return 20
    raise ValueError(f"unsupported AB-002 search seed: {search_seed}")


def expected_pair_id(search_seed: int) -> str:
    return f"AB002-SEED-{search_seed}"


def expected_gpu_id(search_seed: int, arm: str) -> int:
    if arm not in {"control", "treatment"}:
        raise ValueError("arm must be control or treatment")
    control_gpu = 0 if search_seed == CANARY_SEED or search_seed % 2 == 0 else 1
    return control_gpu if arm == "control" else 1 - control_gpu


def validate_baseline_inputs(baseline_dir: Path) -> list[dict[str, Any]]:
    root = baseline_dir.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"baseline directory missing: {root}")
    expected_names = {str(item["filename"]) for item in BASELINE_INPUTS}
    actual_names = {path.name for path in root.glob("*.log") if path.is_file()}
    if actual_names != expected_names:
        raise ValueError(
            f"baseline log inventory drift: {sorted(actual_names)} != {sorted(expected_names)}"
        )
    rows = []
    for item in BASELINE_INPUTS:
        path = root / str(item["filename"])
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise ValueError(f"baseline log digest drift: {path.name}")
        rows.append({"path": str(path), "sha256": actual, "model": item["model"]})
    return rows


def runtime_guard_contract() -> dict[str, Any]:
    source = json.loads(CONTRACT.read_text(encoding="utf-8"))
    projected = {
        "record_type": "RECCLAW_PHASE1_AB002_RUNTIME_GUARD_CONTRACT",
        "schema_version": "recclaw.phase1.ab002.runtime_guard_contract.v1",
        "contract_id": source["contract_id"],
        **{key: source[key] for key in RUNTIME_GUARD_CONTRACT_KEYS},
        "historical_reporting_references_present": False,
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }
    serialized = json.dumps(projected, ensure_ascii=True, sort_keys=True)
    for forbidden in ("0.2908", "0.289067", "0.274367", "historical_v4m"):
        if forbidden in serialized:
            raise ValueError(f"runtime Guard contract contains reporting-only reference: {forbidden}")
    return projected


def _assert_frozen_guard_source() -> None:
    for rel, expected in FROZEN_GUARD_FILES.items():
        actual = sha256_file(PROJECT_ROOT / rel)
        if actual != expected:
            raise ValueError(f"frozen Guard file drift: {rel}: {actual} != {expected}")
    overlay = sha256_file(TREATMENT_OVERLAY)
    if overlay != FROZEN_TREATMENT_OVERLAY_SHA256:
        raise ValueError(f"treatment overlay drift: {overlay}")


def apply_frozen_treatment_overlay(source: Path, overlay: Path) -> bytes:
    """Apply the one frozen UTF-8 unified diff without an ambient Git binary."""
    source_lines = source.read_bytes().decode("utf-8").splitlines(keepends=True)
    patch_lines = overlay.read_bytes().decode("utf-8").splitlines(keepends=True)
    if tuple(patch_lines[:3]) != TREATMENT_PATCH_HEADER:
        raise ValueError("treatment overlay has an unexpected target header")

    output: list[str] = []
    source_index = 0
    patch_index = 3
    hunk_count = 0
    while patch_index < len(patch_lines):
        match = HUNK_HEADER.fullmatch(patch_lines[patch_index])
        if match is None:
            raise ValueError(f"unexpected treatment overlay line {patch_index + 1}")
        old_start = int(match.group(1))
        old_count = int(match.group(2) or "1")
        new_start = int(match.group(3))
        new_count = int(match.group(4) or "1")
        hunk_source_index = old_start - 1
        if hunk_source_index < source_index or hunk_source_index > len(source_lines):
            raise ValueError("treatment overlay hunks are overlapping or out of range")
        output.extend(source_lines[source_index:hunk_source_index])
        if new_start - 1 != len(output):
            raise ValueError("treatment overlay new-line position is inconsistent")
        source_index = hunk_source_index
        patch_index += 1
        observed_old = 0
        observed_new = 0

        while patch_index < len(patch_lines) and not patch_lines[patch_index].startswith("@@ "):
            patch_line = patch_lines[patch_index]
            if not patch_line or patch_line[0] not in {" ", "+", "-"}:
                raise ValueError(f"unsupported treatment overlay line {patch_index + 1}")
            marker = patch_line[0]
            payload = patch_line[1:]
            if marker in {" ", "-"}:
                if source_index >= len(source_lines) or source_lines[source_index] != payload:
                    raise ValueError(
                        f"treatment overlay context mismatch at source line {source_index + 1}"
                    )
                source_index += 1
                observed_old += 1
            if marker in {" ", "+"}:
                output.append(payload)
                observed_new += 1
            patch_index += 1

        if observed_old != old_count or observed_new != new_count:
            raise ValueError(
                "treatment overlay hunk count mismatch: "
                f"old {observed_old}/{old_count}, new {observed_new}/{new_count}"
            )
        hunk_count += 1

    if hunk_count == 0:
        raise ValueError("treatment overlay contains no hunks")
    output.extend(source_lines[source_index:])
    patched = "".join(output).encode("utf-8")
    if hashlib.sha256(patched).hexdigest() != FROZEN_TREATMENT_AGENT_SHA256:
        raise ValueError("patched treatment agent does not match the frozen digest")
    return patched


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
    s0_manifest = validate_records()
    source_rows = copy_allowlisted_source(source_root)
    integration_rows: list[dict[str, Any]] = []
    integration_root: Path | None = None
    if arm == "treatment":
        _assert_frozen_guard_source()
        patched_bytes = apply_frozen_treatment_overlay(
            source_root / "scripts/agent.py", TREATMENT_OVERLAY
        )
        (source_root / "scripts/agent.py").write_bytes(patched_bytes)
        patched_agent = sha256_file(source_root / "scripts/agent.py")
        if patched_agent != FROZEN_TREATMENT_AGENT_SHA256:
            raise ValueError(f"patched treatment agent drift: {patched_agent}")
        for row in source_rows:
            if row["path"] == "scripts/agent.py":
                row["common_s0_sha256"] = row["sha256"]
                row["sha256"] = patched_agent
                row["size_bytes"] = (source_root / "scripts/agent.py").stat().st_size
                row["treatment_overlay_sha256"] = FROZEN_TREATMENT_OVERLAY_SHA256
                break
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
        canonical_write(contract_target, runtime_guard_contract())
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
        "s0_id": S0_ID,
        "s0_source_tree_sha256": s0_manifest["source_tree_sha256"],
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
    if search_seed not in SEARCH_SEEDS:
        raise ValueError(f"unsupported AB-002 search seed: {search_seed}")
    if pair_id != expected_pair_id(search_seed):
        raise ValueError(
            f"pair id must bind the search seed: {pair_id} != {expected_pair_id(search_seed)}"
        )
    expected_gpu = expected_gpu_id(search_seed, arm)
    if gpu_id != expected_gpu:
        raise ValueError(
            f"GPU assignment drift for seed {search_seed} {arm}: {gpu_id} != {expected_gpu}"
        )
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    recbole_root = contract["execution_environment"]["recbole_root"]
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
        str(expected_rounds(search_seed)),
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
        "--llm-temperature",
        "0.2",
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
        "--max-implement-per-round",
        "1" if search_seed == CANARY_SEED else "2",
        "--experiment-directive",
        CANARY_EXPERIMENT_DIRECTIVE if search_seed == CANARY_SEED else FULL_EXPERIMENT_DIRECTIVE,
    ]
    environment = {
        "RECCLAW_AB_ARM": arm,
        "RECCLAW_LLM_PAIR_ID": pair_id,
        "RECCLAW_SEARCH_SEED": str(search_seed),
        "RECCLAW_RUNTIME_ROOT": str(root),
        "RECCLAW_CANDIDATE_EXECUTION_BUDGET": str(
            expected_candidate_executions(search_seed)
        ),
        "RECCLAW_CANDIDATE_EXECUTION_LEDGER": str(
            repetition / "candidate_execution_budget.jsonl"
        ),
        "RECBOLE_ROOT": recbole_root,
        "PYTHONNOUSERSITE": "1",
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
        "round_budget": expected_rounds(search_seed),
        "maximum_candidate_executions": expected_candidate_executions(search_seed),
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


def _build_execution_environment(
    plan: dict[str, Any], *, client_token: str, client_token_env: str = "RECCLAW_BROKER_CLIENT_TOKEN"
) -> dict[str, str]:
    env = dict(os.environ)
    for key in (*TREATMENT_ONLY_ENV_KEYS, *UPSTREAM_ONLY_ENV_KEYS, client_token_env):
        env.pop(key, None)
    for key, value in plan["environment_without_secrets"].items():
        if key != "OPENAI_API_KEY":
            env[key] = str(value)
    env["OPENAI_API_KEY"] = client_token
    return env


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
    validate_baseline_inputs(baseline)
    env = _build_execution_environment(
        plan,
        client_token=token,
        client_token_env=client_token_env,
    )
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
    parser.add_argument("--search-seed", type=int, choices=SEARCH_SEEDS, required=True)
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
