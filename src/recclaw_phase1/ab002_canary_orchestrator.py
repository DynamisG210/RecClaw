"""Execute the one authorized AB-002 Canary without persisting credentials."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path
from urllib import request as urlrequest

from recclaw_phase1.ab002_canary_audit import audit_canary
from recclaw_phase1.ab002_canary_probes import run_probes
from recclaw_phase1.ab002_launcher import canonical_write
from recclaw_phase1.ab002_pair_runner import run_pair
from recclaw_phase1.ab002_preflight import build_preflight


def _wait_for_broker(url: str, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"paired broker exited before health: {process.returncode}")
        try:
            with urlrequest.urlopen(f"{url.rstrip('/')}/health", timeout=1.0) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.2)
    raise TimeoutError("paired broker did not become healthy within 30 seconds")


def run_canary(
    *, runtime_root: Path, baseline_dir: Path, expected_tag: str, port: int
) -> dict[str, object]:
    root = runtime_root.expanduser().resolve()
    api_key = os.environ.get("RECCLAW_LAB_LLM_API_KEY", "")
    base_url = os.environ.get("RECCLAW_LAB_LLM_BASE_URL", "")
    if not api_key or not base_url:
        raise ValueError("non-empty lab LLM key and base URL environment variables are required")
    if root.exists():
        raise FileExistsError(f"Canary runtime root must be absent: {root}")
    root.mkdir(parents=True)
    broker_dir = root / "broker"
    broker_dir.mkdir()
    broker_db = broker_dir / "paired_llm.sqlite3"
    broker_log = broker_dir / "broker.log"
    broker_url = f"http://127.0.0.1:{port}"
    client_token = secrets.token_urlsafe(48)
    environment = dict(os.environ)
    environment["RECCLAW_BROKER_CLIENT_TOKEN"] = client_token
    environment["RECCLAW_AB002_START_AUTHORIZED"] = "YES"
    process = None
    record_path = root / "execution_records" / "canary_orchestration.json"
    record: dict[str, object] = {
        "record_type": "RECCLAW_PHASE1_AB002_CANARY_ORCHESTRATION",
        "schema_version": "recclaw.phase1.ab002.canary_orchestration.v1",
        "status": "IN_PROGRESS",
        "gate_status": "NOT_STARTED",
        "search_seed": 9001,
        "full_ab_started": False,
        "credentials_persisted": False,
        "authority": "NONE",
        "evidence_class": "DEVELOPMENT_ONLY",
        "formal_acceptance": False,
    }
    canonical_write(record_path, record)
    try:
        with broker_log.open("wb") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "recclaw_phase1.paired_llm_broker",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--db",
                    str(broker_db),
                    "--upstream-base-url",
                    base_url,
                    "--upstream-api-key-env",
                    "RECCLAW_LAB_LLM_API_KEY",
                    "--client-token-env",
                    "RECCLAW_BROKER_CLIENT_TOKEN",
                    "--upstream-timeout",
                    "300",
                    "--pair-wait",
                    "330",
                    "--max-requests-per-arm-per-pair",
                    "60",
                    "--max-requested-output-tokens-per-arm-per-pair",
                    "245760",
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            _wait_for_broker(broker_url, process)
            preflight_path = root / "preflight" / "pre_canary.json"
            preflight = build_preflight(
                runtime_root=root,
                baseline_dir=baseline_dir,
                broker_url=broker_url,
                expected_tag=expected_tag,
                upstream_key_env="RECCLAW_LAB_LLM_API_KEY",
                client_token_env="RECCLAW_BROKER_CLIENT_TOKEN",
            )
            canonical_write(preflight_path, preflight)
            previous = dict(os.environ)
            os.environ.update(environment)
            try:
                pair = run_pair(
                    runtime_root=root,
                    search_seed=9001,
                    broker_url=broker_url,
                    baseline_dir=baseline_dir,
                    python_executable=sys.executable,
                    client_token_env="RECCLAW_BROKER_CLIENT_TOKEN",
                )
            finally:
                os.environ.clear()
                os.environ.update(previous)
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
        probes_path = root / "canary" / "controlled_probes.json"
        probes = run_probes()
        canonical_write(probes_path, probes)
        report_path = root / "canary_report.json"
        report = audit_canary(
            runtime_root=root, preflight_path=preflight_path, probes_path=probes_path
        )
        canonical_write(report_path, report)
        record.update(
            {
                "status": "LOCAL_COMPLETE",
                "pair_status": pair["status"],
                "preflight_record": str(preflight_path),
                "controlled_probes_record": str(probes_path),
                "canary_report": str(report_path),
                "independent_canary_review_status": "NOT_STARTED",
                "full_ab_authorized": False,
            }
        )
        canonical_write(record_path, record)
        return record
    except BaseException as exc:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        record.update(
            {
                "status": "STOPPED",
                "failure_class": type(exc).__name__,
                "failure_detail": str(exc)[:2000],
                "full_ab_authorized": False,
            }
        )
        canonical_write(record_path, record)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute the authorized AB-002 Canary pair")
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--expected-tag", required=True)
    parser.add_argument("--port", type=int, default=18082)
    args = parser.parse_args(argv)
    record = run_canary(
        runtime_root=args.runtime_root,
        baseline_dir=args.baseline_dir,
        expected_tag=args.expected_tag,
        port=args.port,
    )
    print(json.dumps(record, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
