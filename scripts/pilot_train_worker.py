#!/usr/bin/env python3
"""Run one package-owned RecBole Pilot training execution."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--recbole-root", required=True)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    recbole_root = Path(args.recbole_root).resolve()
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "scripts"))
    sys.path.insert(0, str(recbole_root))

    import numpy as np

    if not hasattr(np, "float_"):
        np.float_ = np.float64
    if not hasattr(np, "int_"):
        np.int_ = np.int64
    if not hasattr(np, "complex_"):
        np.complex_ = np.complex128
    if not hasattr(np, "unicode_"):
        np.unicode_ = np.str_
    if not hasattr(np, "string_"):
        np.string_ = np.bytes_

    import run_candidate

    run_candidate.install_optional_dependency_stubs()
    from recbole.quick_start import run

    run_candidate.patch_recbole_runtime_compat()
    output_path = Path(args.output_path)
    log_path = Path(args.log_path)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object]
    exit_code = 0
    cwd_before = Path.cwd()
    argv_before = sys.argv[:]
    try:
        os.chdir(project_root)
        sys.argv = [
            "pilot_train_worker.py",
            f"--model={args.model}",
            f"--dataset={args.dataset}",
        ]
        config_files = [
            project_root / "configs" / "task_ml1m.yaml",
            project_root / "configs" / "lightgcn_metrics.yaml",
        ]
        with log_path.open("w", encoding="utf-8", errors="replace") as handle:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                result = run(
                    args.model,
                    args.dataset,
                    config_file_list=[str(path) for path in config_files],
                    config_dict={
                        "checkpoint_dir": str(checkpoint_dir),
                        "data_path": str(Path(args.data_path).resolve()),
                        "epochs": args.epochs,
                        "eval_step": 1,
                        "reproducibility": True,
                        "seed": args.seed,
                        "show_progress": False,
                        "state": "ERROR",
                        "stopping_step": args.epochs,
                        "use_gpu": True,
                    },
                    saved=False,
                )
        payload = {
            "best_valid_result": result.get("best_valid_result", {}),
            "best_valid_score": result.get("best_valid_score"),
            "exit_status": "SUCCESS",
            "model": args.model,
            "test_result": result.get("test_result", {}),
        }
    except Exception as error:  # noqa: BLE001 - failure is a Pilot outcome.
        exit_code = 1
        payload = {
            "error_message": str(error),
            "error_type": type(error).__name__,
            "exit_status": "RUNTIME_FAILURE",
            "model": args.model,
            "traceback": traceback.format_exc(),
        }
        with log_path.open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(payload["traceback"])
    finally:
        os.chdir(cwd_before)
        sys.argv = argv_before
    output_path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
