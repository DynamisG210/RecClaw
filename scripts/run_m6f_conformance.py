#!/usr/bin/env python3
"""Run the M6F fake/adversarial and treatment-free Broker conformance gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from recclaw_core.experiments.helix_abc_v1.broker_process import (  # noqa: E402
    BrokerConformanceReportV1,
)
from recclaw_core.experiments.helix_abc_v1.canary_broker import (  # noqa: E402
    CanaryBrokerError,
    CodexCliCanaryBrokerV1,
)
from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_digest,
)


RESOURCE_ROOT = (
    SRC
    / "recclaw_core"
    / "experiments"
    / "helix_abc_v1"
    / "resources"
)
RELEASE_PATH = RESOURCE_ROOT / "broker_process_release_v2.json"
SCHEMA_PATH = RESOURCE_ROOT / "pilot_proposal_response_v1.schema.json"


def write_json(path: Path, value: Any, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(mode)


def run_fake_suite(output_root: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "unittest",
        "tests.experiments.helix_abc_v1.test_m6f_broker_observability",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=False,
        timeout=180,
    )
    stdout_path = output_root / "private" / "fake_suite.stdout"
    stderr_path = output_root / "private" / "fake_suite.stderr"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_bytes(completed.stdout)
    stderr_path.write_bytes(completed.stderr)
    result = {
        "command_digest": sha256_digest(command),
        "return_code": completed.returncode,
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "test_module": (
            "tests.experiments.helix_abc_v1."
            "test_m6f_broker_observability"
        ),
        "verdict": "PASS" if completed.returncode == 0 else "FAIL",
    }
    write_json(output_root / "M6F_FAKE_PROCESS_SUITE.json", result)
    if completed.returncode != 0:
        raise RuntimeError("M6F fake/adversarial suite failed")
    return result


def fixed_prompt(shape: str) -> tuple[str, int]:
    if shape == "ORIGINAL_SHAPE":
        return (
            """This is a treatment-free Broker transport conformance diagnostic.
Do not use tools or inspect files. Return JSON only through the supplied schema.
Return exactly four synthetic, runnable recommender-mechanism proposal records.
Use only values admitted by the schema. The content will be discarded and must
not mention an experiment, arm, result, memory, metric outcome, or proposal history.""",
            4,
        )
    if shape == "PRODUCER_SHAPE":
        return (
            """This is a treatment-free Broker transport conformance diagnostic.
Do not use tools or inspect files. Return JSON only through the supplied schema.
Return exactly one synthetic, runnable recommender-mechanism proposal record.
Set proposal_intent to FALSIFICATION and use only values admitted by the schema.
The content will be discarded and must not mention an experiment, arm, result,
memory, metric outcome, or proposal history.""",
            1,
        )
    raise ValueError("unknown conformance shape")


def conformance_report(
    *,
    probe_id: str,
    shape: str,
    broker: CodexCliCanaryBrokerV1,
    logical_call_id: str,
    total_tokens: int,
) -> BrokerConformanceReportV1:
    receipt, outcome = broker.conformance_evidence(logical_call_id)
    payload = {
        "actual_total_tokens": total_tokens,
        "broker_release_digest": broker.release.release_digest,
        "outcome_digest": outcome.outcome_digest,
        "probe_id": probe_id,
        "probe_shape": shape,
        "receipt_digest": receipt.receipt_digest,
        "request_envelope_digest": outcome.request_envelope_digest,
        "search_round_opened": False,
        "treatment_free": True,
        "verdict": "PASS" if outcome.status == "SUCCESS" else "FAIL",
    }
    return BrokerConformanceReportV1(
        **payload, report_digest=sha256_digest(payload)
    )


def failure_root_cause_record(
    *,
    broker: CodexCliCanaryBrokerV1,
    shape: str,
    error: CanaryBrokerError,
) -> dict[str, Any]:
    if error.outcome is None or error.receipt is None:
        raise RuntimeError("real probe failed without typed process evidence")
    outcome = error.outcome
    receipt = error.receipt
    return {
        "affected_contract": "BrokerProcessReleaseV2 exact conformance",
        "cause_evidence": {
            "classifier_rule_id": outcome.classifier_rule_id,
            "redacted_excerpt": outcome.redacted_excerpt,
            "supporting_artifact_ref": outcome.supporting_artifact_ref,
        },
        "evidence_backed_cause_class": outcome.failure_class,
        "exact_failing_command_projection": {
            "argv_template_digest": broker.release.argv_template_digest,
            "broker_executable_path": broker.release.broker_executable_path,
            "broker_executable_sha256": broker.release.broker_executable_sha256,
            "cwd_policy_digest": (
                broker.release.working_directory_policy_digest
            ),
            "environment_allowlist_policy_digest": (
                broker.release.environment_allowlist_policy_digest
            ),
            "model": broker.release.model,
            "reasoning_effort": broker.release.reasoning_effort,
            "request_envelope_digest": outcome.request_envelope_digest,
            "sandbox_mode": broker.release.sandbox_mode,
        },
        "exit_receipt": receipt.to_dict(),
        "formal_acceptance": False,
        "minimal_repair": "NO_CHANGE_AUTHORIZED_UNTIL_EVIDENCE_REVIEW",
        "probe_shape": shape,
        "raw_artifact_digests": {
            "stderr_sha256": receipt.stderr_sha256,
            "stdout_sha256": receipt.stdout_sha256,
        },
        "record_schema": "recclaw.m6f-broker-root-cause-record.v1",
        "secret_redaction_status": {
            "raw_artifacts_private": True,
            "redaction_policy_digest": broker.release.redaction_policy_digest,
            "unrestricted_environment_persisted": False,
        },
        "treatment_and_budget_impact": (
            "NONE; diagnostic request opened no SearchRound and used a separate "
            "conformance budget"
        ),
        "v5_provider_level_root_cause": "UNKNOWN",
    }


def run_real_probes(output_root: Path) -> tuple[dict[str, Any], ...]:
    release = json.loads(RELEASE_PATH.read_text(encoding="utf-8"))
    broker = CodexCliCanaryBrokerV1(
        output_root / "private" / "real_broker",
        schema_path=SCHEMA_PATH,
        codex_executable=Path(release["broker_executable_path"]),
        model=release["model"],
        reasoning_effort=release["reasoning_effort"],
        service_tier="default",
        max_total_tokens_per_call=20_000,
        timeout_ms=900_000,
        cli_version=release["broker_cli_version"],
        login_mode=release["login_mode"],
        release_manifest_path=RELEASE_PATH,
    )
    reports: list[dict[str, Any]] = []
    try:
        for ordinal, shape in enumerate(
            ("ORIGINAL_SHAPE", "PRODUCER_SHAPE"), start=1
        ):
            prompt, expected_count = fixed_prompt(shape)
            logical_call_id = f"m6f-conformance-{ordinal}"
            try:
                call = broker.call_with_session(
                    logical_call_id=logical_call_id,
                    proposal_generation_session_id=(
                        f"m6f-diagnostic-session-{ordinal}"
                    ),
                    prompt=prompt,
                    expected_proposal_count=expected_count,
                )
            except CanaryBrokerError as error:
                record = failure_root_cause_record(
                    broker=broker, shape=shape, error=error
                )
                write_json(
                    output_root / "M6F_BROKER_ROOT_CAUSE_RECORD.json",
                    record,
                )
                raise
            report = conformance_report(
                probe_id=f"M6F_{shape}_V1",
                shape=shape,
                broker=broker,
                logical_call_id=logical_call_id,
                total_tokens=call.total_tokens,
            )
            write_json(
                output_root / f"M6F_BROKER_{shape}_CONFORMANCE.json",
                report.to_dict(),
            )
            reports.append(report.to_dict())
    finally:
        broker.close()
    return tuple(reports)


def execute(output_root: Path, *, real_probes: bool) -> int:
    if output_root.exists():
        raise FileExistsError("M6F conformance root already exists")
    output_root.mkdir(parents=True)
    fake = run_fake_suite(output_root)
    if not real_probes:
        write_json(
            output_root / "M6F_CONFORMANCE_PACKET.json",
            {
                "fake_suite": fake,
                "real_probes": "NOT_RUN",
                "verdict": "FAKE_PASS_REAL_PENDING",
            },
        )
        return 0
    reports = run_real_probes(output_root)
    packet = {
        "authority": "NONE",
        "broker_release_digest": json.loads(
            RELEASE_PATH.read_text(encoding="utf-8")
        )["release_digest"],
        "evidence_class": "DEVELOPMENT_ONLY",
        "fake_suite": fake,
        "formal_acceptance": False,
        "real_probe_report_digests": [
            report["report_digest"] for report in reports
        ],
        "search_rounds_opened": 0,
        "treatment_state_used": False,
        "v5_provider_level_root_cause": "UNKNOWN",
        "verdict": "PASS",
    }
    packet["packet_digest"] = sha256_digest(packet)
    write_json(output_root / "M6F_CONFORMANCE_PACKET.json", packet)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--real-probes", action="store_true")
    args = parser.parse_args()
    return execute(args.output_root.resolve(), real_probes=args.real_probes)


if __name__ == "__main__":
    raise SystemExit(main())
