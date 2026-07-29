#!/usr/bin/env python3
"""Launch one measured V24 gpu35 qualification canary."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def _gpu_row() -> dict[str, object]:
    query = (
        "index,name,memory.total,memory.used,memory.free,"
        "utilization.gpu,power.draw,temperature.gpu"
    )
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()
    fields = [item.strip() for item in raw.split(",")]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "index": int(fields[0]),
        "name": fields[1],
        "memory_total_mib": int(fields[2]),
        "memory_used_mib": int(fields[3]),
        "memory_free_mib": int(fields[4]),
        "gpu_util_percent": int(fields[5]),
        "power_w": float(fields[6]),
        "temperature_c": int(fields[7]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--mechanism-id", required=True)
    parser.add_argument("--search-seed", type=int, required=True)
    args = parser.parse_args()

    backend = args.backend_root.resolve()
    source = backend / "source"
    python = Path(
        "/NAS2020/Workspaces/DMGroup/tingrangan/"
        "recclaw_v15_backend_v1/runtime_exact_v2/bin/python"
    )
    recbole = Path(
        "/NAS2020/Workspaces/DMGroup/tingrangan/"
        "recclaw_v15_backend_v1/recbole"
    )
    data = Path(
        "/NAS2020/Workspaces/DMGroup/tingrangan/"
        "recclaw_v15_backend_v1/search_dataset"
    )
    canary_parent = backend / "backend_canaries_v24"
    output_root = canary_parent / (
        f"canary_{args.name}_{args.search_seed}_full_recipe_v1"
    )
    if output_root.exists():
        raise RuntimeError(f"canary output already exists: {output_root}")
    canary_parent.mkdir(parents=True, exist_ok=True)

    initial = _gpu_row()
    if (
        int(initial["memory_used_mib"]) > 512
        or int(initial["gpu_util_percent"]) > 10
    ):
        raise RuntimeError(f"gpu35 is not idle: {initial}")

    result_filename = (
        f"GPU35_V16_{args.name.upper()}_{args.search_seed}_RESULT.json"
    )
    command = [
        str(python),
        str(source / "scripts/run_v24_full_recipe_canary.py"),
        "--output-root",
        str(output_root),
        "--python",
        str(python),
        "--recbole-root",
        str(recbole),
        "--data-path",
        str(data),
        "--campaign-mechanism-id",
        args.mechanism_id,
        "--experiment-id",
        f"V24-GPU35-FULL-RECIPE-CANARY-{args.search_seed}",
        "--lineage-partition",
        "V24_GPU35_QUALIFICATION_EXCLUDED_FROM_PILOT_AND_MAIN",
        "--result-filename",
        result_filename,
        "--search-seed",
        str(args.search_seed),
    ]
    monitor_rows: list[dict[str, object]] = [initial]
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(2):
            try:
                monitor_rows.append(_gpu_row())
            except Exception as error:  # measurement failure is recorded
                monitor_rows.append(
                    {
                        "timestamp_utc": datetime.now(
                            timezone.utc
                        ).isoformat(),
                        "monitor_error": repr(error),
                    }
                )

    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source), str(source / "src"), str(recbole)]
    )
    started_utc = datetime.now(timezone.utc).isoformat()
    started_ns = time.monotonic_ns()
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    process = subprocess.run(
        command,
        cwd=source,
        env=environment,
        check=False,
    )
    stop.set()
    thread.join(timeout=5)
    monitor_rows.append(_gpu_row())
    elapsed_ms = int((time.monotonic_ns() - started_ns) / 1_000_000)

    monitor_path = canary_parent / (
        f"canary_{args.name}_{args.search_seed}_gpu_monitor.csv"
    )
    fieldnames = sorted(
        {key for row in monitor_rows for key in row}
    )
    with monitor_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(monitor_rows)

    status = {
        "command": command,
        "elapsed_ms": elapsed_ms,
        "exit_code": process.returncode,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "initial_gpu": initial,
        "mechanism_id": args.mechanism_id,
        "monitor_path": monitor_path.as_posix(),
        "name": args.name,
        "output_root": output_root.as_posix(),
        "result_path": (output_root / result_filename).as_posix(),
        "search_seed": args.search_seed,
        "started_utc": started_utc,
    }
    status_path = canary_parent / (
        f"canary_{args.name}_{args.search_seed}_status.json"
    )
    status_path.write_text(
        json.dumps(status, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(status, sort_keys=True))
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
