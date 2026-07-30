#!/usr/bin/env python3
"""Run the single authorized non-research R1 prefreeze endpoint probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for import_root in (ROOT, SRC):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from recclaw_core.experiments.helix_abc_v1.canonical import (  # noqa: E402
    bytes_sha256,
    canonical_json_bytes,
    canonical_value,
    sha256_digest,
)
from recclaw_core.experiments.helix_abc_v1.canary_broker import (  # noqa: E402
    CanaryBrokerError,
)
from recclaw_core.experiments.helix_abc_v1.lab_api_broker import (  # noqa: E402
    LabApiCanaryBrokerV1,
    load_lab_api_credentials,
)


RESOURCE_ROOT = (
    ROOT / "src/recclaw_core/experiments/helix_abc_v1/resources"
)
SCHEMA_PATH = RESOURCE_ROOT / "fresh_open_spec_proposal_response_v1.schema.json"
SENTINEL_PATH = RESOURCE_ROOT / "fresh_open_spec_schema_probe_payload_v1.json"
RELEASE_PATH = (
    ROOT / "docs/research_line/vnext/FRESH_OPEN_SPEC_PROVIDER_RELEASE_V1.json"
)
RECEIPT_PATH = (
    ROOT
    / "docs/research_line/vnext/"
    "FRESH_OPEN_SPEC_ENDPOINT_PROBE_RECEIPT_V1.json"
)
PROBE_SCHEMA = "recclaw.research-line.fresh-open-spec-endpoint-probe.v1"
MODEL = "gpt-5.4"
PROBE_TOKEN_CEILING = 2000


def _credential_identity(config_path: Path, *, base_url: str, api_key: str) -> dict[str, str]:
    config_digest = bytes_sha256(config_path.resolve().read_bytes())
    endpoint_digest = sha256_digest({"base_url": base_url.rstrip("/")})
    identity_digest = sha256_digest(
        {
            "scheme": "RECClaw credential config identity v1",
            "config_bytes_sha256": config_digest,
            "api_key_sha256": hashlib.sha256(api_key.encode("utf-8")).hexdigest(),
            "endpoint_digest": endpoint_digest,
        }
    )
    return {
        "config_bytes_sha256": config_digest,
        "credential_identity_digest": identity_digest,
        "endpoint_digest": endpoint_digest,
    }


def _prompt(sentinel: dict[str, Any]) -> str:
    return (
        "This is a one-time transport, authentication, exact-model, and strict-"
        "JSON-schema qualification probe. It is not research ideation. Echo the "
        "following JSON object exactly, with no additions, omissions, prose, "
        "markdown, or tool use. The response is a pre-authored sentinel and must "
        "never be projected into OpenResearchSpecV1, resolved, implemented, "
        "qualified, admitted, counted, trained, or interpreted as a candidate:\n"
        + canonical_json_bytes(sentinel).decode("utf-8")
    )


def _blocked_fields(error: CanaryBrokerError | Exception) -> list[str]:
    if isinstance(error, CanaryBrokerError) and error.outcome is not None:
        failure_class = error.outcome.failure_class.value
        if failure_class == "AUTHENTICATION_ERROR":
            return ["endpoint_authentication"]
        if failure_class == "MODEL_UNAVAILABLE":
            return ["exact_gpt_5_4_support", "returned_model"]
        if failure_class in {
            "SCHEMA_VALIDATION_FAILURE",
            "CLI_CONTRACT_ERROR",
        }:
            return [
                "endpoint_authentication",
                "exact_fresh_open_spec_schema_support",
                "returned_model",
            ]
        if failure_class in {
            "CONNECTIVITY_ERROR",
            "TIMEOUT",
            "PROVIDER_ERROR",
        }:
            return [
                "endpoint_reachability",
                "endpoint_authentication",
                "exact_gpt_5_4_support",
                "exact_fresh_open_spec_schema_support",
                "returned_model",
            ]
    return [
        "endpoint_authentication",
        "exact_gpt_5_4_support",
        "exact_fresh_open_spec_schema_support",
        "returned_model",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-api-config", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    args = parser.parse_args()

    private_root = args.private_root.resolve()
    if private_root.exists():
        raise SystemExit("probe private root already exists; refusing a second probe")
    if RECEIPT_PATH.exists():
        raise SystemExit("probe receipt already exists; refusing a second probe")

    config_path = args.llm_api_config.resolve()
    base_url, api_key = load_lab_api_credentials(config_path)
    credential = _credential_identity(
        config_path,
        base_url=base_url,
        api_key=api_key,
    )
    del api_key
    sentinel = json.loads(SENTINEL_PATH.read_bytes())
    sentinel_digest = sha256_digest(sentinel)
    release = json.loads(RELEASE_PATH.read_bytes())
    broker: LabApiCanaryBrokerV1 | None = None
    physical_call_count = 0
    try:
        broker = LabApiCanaryBrokerV1(
            private_root,
            schema_path=SCHEMA_PATH,
            config_path=config_path,
            model=MODEL,
            max_total_tokens_per_call=6000,
            timeout_ms=900_000,
            release_manifest_path=RELEASE_PATH,
        )
        physical_call_count = 1
        result = broker.call_with_session(
            logical_call_id="fresh-open-spec-prefreeze-schema-probe-v1",
            proposal_generation_session_id=(
                "fresh-open-spec-prefreeze-schema-probe-session-v1"
            ),
            prompt=_prompt(sentinel),
            expected_proposal_count=1,
            max_total_tokens=PROBE_TOKEN_CEILING,
        )
        if result.returned_model != MODEL:
            raise CanaryBrokerError(
                "probe returned a model other than exact gpt-5.4"
            )
        if canonical_value(result.response) != canonical_value(sentinel):
            raise CanaryBrokerError(
                "probe response differed from the pre-authored sentinel"
            )
        receipt = {
            "schema": PROBE_SCHEMA,
            "status": "PASS",
            "classification": (
                "PASS_EXACT_GPT_5_4_ENDPOINT_AUTH_AND_FRESH_SCHEMA"
            ),
            "model_requested": MODEL,
            "returned_model": result.returned_model,
            "authentication_status": "VERIFIED",
            "endpoint_digest": credential["endpoint_digest"],
            "credential_config_digest": credential["config_bytes_sha256"],
            "credential_identity_digest": credential[
                "credential_identity_digest"
            ],
            "provider_release_ref": (
                "repo:docs/research_line/vnext/"
                "FRESH_OPEN_SPEC_PROVIDER_RELEASE_V1.json"
            ),
            "provider_release_digest": release["release_digest"],
            "response_schema_ref": (
                "repo:src/recclaw_core/experiments/helix_abc_v1/resources/"
                "fresh_open_spec_proposal_response_v1.schema.json"
            ),
            "response_schema_digest": bytes_sha256(SCHEMA_PATH.read_bytes()),
            "request_digest": result.request_digest,
            "response_digest": result.response_digest,
            "sentinel_digest": sentinel_digest,
            "physical_provider_calls": physical_call_count,
            "retry_count": 0,
            "sensitive_values_persisted": False,
            "sensitive_headers_persisted": False,
            "research_candidates_generated": 0,
            "open_specs_projected": 0,
            "resolver_calls": 0,
            "candidate_roots_created": 0,
            "candidate_admissions": 0,
            "training_runs": 0,
            "outcomes_consumed": 0,
            "held_out_reads": 0,
            "probe_private_root_digest": sha256_digest(
                {"path": private_root.as_posix()}
            ),
            "blocked_fields": [],
        }
        exit_code = 0
    except Exception as error:
        failure_class = (
            error.outcome.failure_class.value
            if isinstance(error, CanaryBrokerError)
            and error.outcome is not None
            else type(error).__name__
        )
        request_digest = (
            error.outcome.request_envelope_digest
            if isinstance(error, CanaryBrokerError)
            and error.outcome is not None
            else None
        )
        outcome_digest = (
            error.outcome.outcome_digest
            if isinstance(error, CanaryBrokerError)
            and error.outcome is not None
            else None
        )
        receipt = {
            "schema": PROBE_SCHEMA,
            "status": "BLOCKED",
            "classification": failure_class,
            "model_requested": MODEL,
            "returned_model": None,
            "authentication_status": "UNVERIFIED",
            "endpoint_digest": credential["endpoint_digest"],
            "credential_config_digest": credential["config_bytes_sha256"],
            "credential_identity_digest": credential[
                "credential_identity_digest"
            ],
            "provider_release_ref": (
                "repo:docs/research_line/vnext/"
                "FRESH_OPEN_SPEC_PROVIDER_RELEASE_V1.json"
            ),
            "provider_release_digest": release["release_digest"],
            "response_schema_ref": (
                "repo:src/recclaw_core/experiments/helix_abc_v1/resources/"
                "fresh_open_spec_proposal_response_v1.schema.json"
            ),
            "response_schema_digest": bytes_sha256(SCHEMA_PATH.read_bytes()),
            "request_digest": request_digest,
            "response_digest": None,
            "broker_outcome_digest": outcome_digest,
            "sentinel_digest": sentinel_digest,
            "physical_provider_calls": physical_call_count,
            "retry_count": 0,
            "sensitive_values_persisted": False,
            "sensitive_headers_persisted": False,
            "research_candidates_generated": 0,
            "open_specs_projected": 0,
            "resolver_calls": 0,
            "candidate_roots_created": 0,
            "candidate_admissions": 0,
            "training_runs": 0,
            "outcomes_consumed": 0,
            "held_out_reads": 0,
            "probe_private_root_digest": sha256_digest(
                {"path": private_root.as_posix()}
            ),
            "blocked_fields": _blocked_fields(error),
        }
        exit_code = 2
    finally:
        if broker is not None:
            broker.close()

    RECEIPT_PATH.write_bytes(canonical_json_bytes(receipt))
    print(
        json.dumps(
            {
                "classification": receipt["classification"],
                "physical_provider_calls": receipt[
                    "physical_provider_calls"
                ],
                "receipt_sha256": bytes_sha256(RECEIPT_PATH.read_bytes()),
                "status": receipt["status"],
            },
            sort_keys=True,
        )
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
