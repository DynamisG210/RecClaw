#!/usr/bin/env python3
"""Verify the accepted runtime can import the Q2 consumer normally."""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
from pathlib import Path

from recclaw_core.experiments.helix_abc_v1.mechanism_characterization import (
    bytes_sha256,
    load_contract,
    write_new_json,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    contract_path = Path(args.contract).resolve()
    contract = load_contract(contract_path)
    payload = {
        "schema": "recclaw.q2-normal-package-import-gate.v1",
        "status": "PASS",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "rfc8785_version": importlib.metadata.version("rfc8785"),
        "normal_package_import": True,
        "contract_schema": contract["schema"],
        "contract_sha256": bytes_sha256(contract_path.read_bytes()),
        "development_only": contract["development_only"],
        "physical_probe_started": False,
    }
    write_new_json(Path(args.output), payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
