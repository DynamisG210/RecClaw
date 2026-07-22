#!/usr/bin/env python3
"""Run one RecClaw candidate selected by candidate_id.

This runner is the single candidate execution entrypoint for RecClaw v0.5.
It reads configs/candidate_registry.yaml and supports the current candidate
contract:

- runner_type: config_only | model | posthoc
- wired: true | false
- entrypoint: import path for executable model/posthoc class
- consumes: config keys actually read by the executable path
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.machinery
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
import types
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
CANDIDATE_DIR = CONFIG_DIR / "candidates"
RESULT_DIR = PROJECT_ROOT / "results" / "candidates"
RESULTS_CSV = PROJECT_ROOT / "results" / "results.csv"
OVERRIDE_DIR = PROJECT_ROOT / "results" / "overrides"
REGISTRY_PATH = CONFIG_DIR / "candidate_registry.yaml"
COLLECT_SCRIPT = SCRIPT_DIR / "collect_result.py"

BASE_CONFIGS = {
    "BPR": "bpr.yaml",
    "LightGCN": "lightgcn.yaml",
}
RUNNABLE_TYPES = {"config_only", "model"}


class Tee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            try:
                stream.write(text)
                stream.flush()
            except ValueError:
                continue
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except ValueError:
                continue


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_registry(path: Path = REGISTRY_PATH) -> list[dict[str, Any]]:
    payload = load_yaml(path)
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError(f"registry candidates must be a list: {path}")
    return candidates


def find_candidate(candidate_id: str, registry_path: Path = REGISTRY_PATH) -> dict[str, Any]:
    for candidate in load_registry(registry_path):
        if candidate.get("candidate_id") == candidate_id:
            return dict(candidate)
    raise KeyError(f"candidate_id not found in registry: {candidate_id}")


def normalize_base_model(raw: str | None, override: str | None = None) -> str:
    text = (override or raw or "").lower()
    if "lightgcn" in text:
        return "LightGCN"
    if "bpr" in text:
        return "BPR"
    raise ValueError(f"could not infer base model from: {raw!r}")


def model_name_from_entrypoint(entrypoint: str) -> str:
    if ":" not in entrypoint:
        raise ValueError(f"entrypoint must use module:attribute format: {entrypoint}")
    return entrypoint.split(":", 1)[1]


def resolve_recbole_root(explicit: str | None = None) -> Path:
    candidates = [
        explicit,
        os.environ.get("RECBOLE_ROOT"),
        os.environ.get("RECBole_ROOT"),
        str(PROJECT_ROOT.parent / "RecBole"),
    ]
    skipped: list[str] = []
    for value in candidates:
        if not value:
            continue
        try:
            path = Path(value).expanduser().resolve()
            if (path / "run_recbole.py").exists():
                return path
        except OSError as exc:
            skipped.append(f"{value} ({type(exc).__name__}: {exc})")
            continue
    detail = f"; skipped inaccessible candidates: {', '.join(skipped)}" if skipped else ""
    raise FileNotFoundError(
        "could not find RecBole. Set RECBOLE_ROOT to a checkout containing run_recbole.py"
        + detail
    )


def parse_override(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise ValueError(f"override must use key=value format: {value}")
    key, raw = value.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"override key is empty: {value}")
    return key, yaml.safe_load(raw)


def write_override_file(run_id: str, overrides: list[str], override_dir: Path = OVERRIDE_DIR) -> Path | None:
    if not overrides:
        return None
    payload = dict(parse_override(item) for item in overrides)
    override_dir.mkdir(parents=True, exist_ok=True)
    path = override_dir / f"{run_id}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=True)
    return path


def import_object(spec: str) -> Any:
    module_name, attr = spec.split(":", 1)
    module = __import__(module_name, fromlist=[attr])
    return getattr(module, attr)


def patch_recbole_model_lookup(local_models: dict[str, Any]) -> None:
    import recbole.config.configurator as configurator
    import recbole.quick_start.quick_start as quick_start
    import recbole.utils as recbole_utils

    original_get_model = recbole_utils.get_model

    def get_model(model_name: str) -> Any:
        if model_name in local_models:
            return local_models[model_name]
        return original_get_model(model_name)

    recbole_utils.get_model = get_model
    configurator.get_model = get_model
    quick_start.get_model = get_model


def patch_recbole_runtime_compat() -> None:
    import copy
    import recbole.evaluator.collector as collector_module
    from scipy.sparse import dok_matrix

    def dok_update(self: Any, values: Any) -> None:
        iterator = values.items() if hasattr(values, "items") else values
        for key, value in iterator:
            self[key] = value

    dok_matrix.update = dok_update  # type: ignore[method-assign]

    def get_data_struct(self: Any) -> Any:
        for key, value in list(self.data_struct._data_dict.items()):
            if hasattr(value, "cpu"):
                self.data_struct._data_dict[key] = value.cpu()
        returned_struct = copy.deepcopy(self.data_struct)
        for key in ["rec.topk", "rec.meanrank", "rec.score", "rec.items", "data.label"]:
            if key in self.data_struct:
                del self.data_struct[key]
        return returned_struct

    collector_module.Collector.get_data_struct = get_data_struct


def install_optional_dependency_stubs() -> None:
    """Provide tiny stubs for optional RecBole imports unused by this runner."""
    try:
        __import__("ray")
    except ModuleNotFoundError:
        ray_module = types.ModuleType("ray")
        tune_module = types.ModuleType("ray.tune")
        ray_module.tune = tune_module
        sys.modules["ray"] = ray_module
        sys.modules["ray.tune"] = tune_module

    try:
        __import__("colorlog")
    except ModuleNotFoundError:
        colorlog_module = types.ModuleType("colorlog")

        class ColoredFormatter(logging.Formatter):
            def __init__(
                self,
                fmt: str | None = None,
                datefmt: str | None = None,
                *args: Any,
                **kwargs: Any,
            ) -> None:
                kwargs.pop("log_colors", None)
                super().__init__(fmt=fmt, datefmt=datefmt, *args, **kwargs)

            def format(self, record: logging.LogRecord) -> str:
                if not hasattr(record, "log_color"):
                    record.log_color = ""
                return super().format(record)

        colorlog_module.ColoredFormatter = ColoredFormatter
        sys.modules["colorlog"] = colorlog_module

    try:
        __import__("colorama")
    except ModuleNotFoundError:
        colorama_module = types.ModuleType("colorama")
        colorama_module.init = lambda *args, **kwargs: None
        sys.modules["colorama"] = colorama_module

    if "torch.utils.tensorboard" not in sys.modules:
        tensorboard_module = types.ModuleType("torch.utils.tensorboard")

        class SummaryWriter:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass

            def __getattr__(self, name: str) -> Any:
                def noop(*args: Any, **kwargs: Any) -> None:
                    return None

                return noop

            def close(self) -> None:
                return None

        tensorboard_module.SummaryWriter = SummaryWriter
        sys.modules["torch.utils.tensorboard"] = tensorboard_module

    try:
        __import__("texttable")
    except ModuleNotFoundError:
        texttable_module = types.ModuleType("texttable")

        class Texttable:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.rows: list[Any] = []

            def add_rows(self, rows: list[Any], *args: Any, **kwargs: Any) -> None:
                self.rows.extend(rows)

            def set_cols_align(self, *args: Any, **kwargs: Any) -> None:
                return None

            def set_cols_valign(self, *args: Any, **kwargs: Any) -> None:
                return None

            def draw(self) -> str:
                return "\n".join(" | ".join(map(str, row)) for row in self.rows)

        texttable_module.Texttable = Texttable
        sys.modules["texttable"] = texttable_module

    try:
        __import__("tqdm")
    except ModuleNotFoundError:
        tqdm_module = types.ModuleType("tqdm")
        tqdm_module.__spec__ = importlib.machinery.ModuleSpec("tqdm", loader=None)

        def tqdm(iterable: Any = None, *args: Any, **kwargs: Any) -> Any:
            return iterable if iterable is not None else []

        tqdm_module.tqdm = tqdm
        sys.modules["tqdm"] = tqdm_module

    try:
        __import__("thop")
    except Exception:
        thop_module = types.ModuleType("thop")
        profile_module = types.ModuleType("thop.profile")
        vision_module = types.ModuleType("thop.vision")
        basic_hooks_module = types.ModuleType("thop.vision.basic_hooks")
        profile_module.register_hooks = {}

        def count_parameters(*args: Any, **kwargs: Any) -> None:
            return None

        basic_hooks_module.count_parameters = count_parameters
        thop_module.profile = profile_module
        thop_module.vision = vision_module
        vision_module.basic_hooks = basic_hooks_module
        sys.modules["thop"] = thop_module
        sys.modules["thop.profile"] = profile_module
        sys.modules["thop.vision"] = vision_module
        sys.modules["thop.vision.basic_hooks"] = basic_hooks_module


def run_recbole_in_process(
    *,
    recbole_root: Path,
    model_name: str,
    dataset: str,
    config_files: list[Path],
    log_path: Path,
    local_model_entrypoint: str | None,
    checkpoint_dir: Path,
    ) -> int:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(recbole_root))
    import torch  # noqa: F401
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

    install_optional_dependency_stubs()
    if local_model_entrypoint:
        local_model = import_object(local_model_entrypoint)
        patch_recbole_model_lookup({model_name: local_model})

    from recbole.quick_start import run

    patch_recbole_runtime_compat()

    argv_backup = sys.argv[:]
    cwd_backup = Path.cwd()
    sys.argv = [
        "run_candidate.py",
        f"--model={model_name}",
        f"--dataset={dataset}",
    ]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(PROJECT_ROOT)
        with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
            tee = Tee(sys.stdout, log_handle)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                result = run(
                    model_name,
                    dataset,
                    config_file_list=[str(path) for path in config_files],
                    config_dict={
                        "data_path": str(recbole_root / "dataset"),
                        "checkpoint_dir": str(checkpoint_dir),
                    },
                )
                print(json.dumps(result, ensure_ascii=True, default=str, indent=2))
        return 0
    finally:
        os.chdir(cwd_backup)
        sys.argv = argv_backup


def collect_result(
    *,
    log_path: Path,
    run_id: str,
    config_change: str,
    candidate_id: str,
    exit_code: int,
    result_dir: Path = RESULT_DIR,
    results_csv: Path = RESULTS_CSV,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(COLLECT_SCRIPT),
        str(log_path),
        "--append-csv",
        str(results_csv),
        "--run-id",
        run_id,
        "--config-change",
        config_change,
        "--log-path",
        str(log_path),
        "--notes",
        f"candidate_id={candidate_id}",
    ]
    if exit_code != 0:
        cmd.extend(["--status-override", "crash"])
    completed = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    if completed.returncode != 0:
        raise RuntimeError(f"collect_result.py failed with code {completed.returncode}")
    parsed = json.loads(completed.stdout)
    result_json_path = result_dir / f"{run_id}.json"
    with result_json_path.open("w", encoding="utf-8") as handle:
        json.dump(parsed, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    parsed["result_json_path"] = str(result_json_path)
    return parsed


def cleanup_checkpoint_dir(run_checkpoint_dir: Path, checkpoint_root: Path) -> None:
    try:
        resolved_run_dir = run_checkpoint_dir.resolve()
        resolved_root = checkpoint_root.resolve()
        resolved_run_dir.relative_to(resolved_root)
    except (OSError, ValueError):
        raise RuntimeError(f"refuse to cleanup checkpoint dir outside root: {run_checkpoint_dir}")
    if resolved_run_dir == resolved_root:
        raise RuntimeError(f"refuse to cleanup checkpoint root directly: {checkpoint_root}")
    if resolved_run_dir.exists():
        shutil.rmtree(resolved_run_dir)


def validate_candidate(candidate_id: str, candidate: dict[str, Any]) -> tuple[str, str]:
    runner_type = str(candidate.get("runner_type") or "").strip()
    entrypoint = str(candidate.get("entrypoint") or "").strip()
    if runner_type == "posthoc":
        raise NotImplementedError(
            "posthoc candidates need a trained-model score-adjustment flow; "
            "run_candidate.py currently supports config_only/model candidates."
        )
    if not candidate.get("wired", False):
        raise NotImplementedError(f"candidate is not wired yet: {candidate_id}")
    if runner_type not in RUNNABLE_TYPES:
        raise NotImplementedError(
            f"unsupported runner_type for {candidate_id}: {runner_type or '<missing>'}"
        )
    if not entrypoint:
        raise ValueError(f"candidate entrypoint is missing: {candidate_id}")
    return runner_type, entrypoint


def build_plan(
    candidate_id: str,
    base_model_override: str | None,
    overrides: list[str],
    create_override: bool = True,
    result_dir: Path = RESULT_DIR,
    override_dir: Path = OVERRIDE_DIR,
    registry_path: Path = REGISTRY_PATH,
) -> dict[str, Any]:
    candidate = find_candidate(candidate_id, registry_path)
    runner_type, entrypoint = validate_candidate(candidate_id, candidate)
    candidate_config_path = CANDIDATE_DIR / f"{candidate_id}.yaml"
    candidate_config = load_yaml(candidate_config_path)
    base_model = normalize_base_model(candidate.get("base_model"), base_model_override)

    if runner_type == "model":
        model_name = str(candidate_config.get("model") or model_name_from_entrypoint(entrypoint))
        local_model_entrypoint = entrypoint
    else:
        model_name = str(candidate_config.get("model") or model_name_from_entrypoint(entrypoint))
        local_model_entrypoint = None

    dataset = str(load_yaml(CONFIG_DIR / "task_ml1m.yaml").get("dataset") or "ml-1m")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"candidate_{candidate_id}_{timestamp}"
    override_path = write_override_file(run_id, overrides, override_dir) if create_override else None

    config_files = [
        CONFIG_DIR / "task_ml1m.yaml",
        CONFIG_DIR / BASE_CONFIGS[base_model],
        CONFIG_DIR / "lightgcn_metrics.yaml",
    ]
    if candidate_config_path.exists():
        config_files.append(candidate_config_path)
    if override_path is not None:
        config_files.append(override_path)

    config_change = "+".join(path.name for path in config_files)
    return {
        "candidate": candidate,
        "runner_type": runner_type,
        "entrypoint": entrypoint,
        "local_model_entrypoint": local_model_entrypoint,
        "candidate_config": candidate_config,
        "candidate_config_path": str(candidate_config_path)
        if candidate_config_path.exists()
        else "",
        "base_model": base_model,
        "model_name": model_name,
        "dataset": dataset,
        "run_id": run_id,
        "log_path": result_dir / f"{run_id}.log",
        "config_files": config_files,
        "config_change": config_change,
        "overrides": overrides,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a RecClaw candidate by candidate_id.")
    parser.add_argument("candidate_id", help="candidate_id from configs/candidate_registry.yaml")
    parser.add_argument(
        "--registry-path",
        default=os.environ.get("RECCLAW_REGISTRY_PATH"),
        help=(
            "Optional candidate registry YAML. Defaults to configs/candidate_registry.yaml; "
            "random baselines can pass a temporary registry containing replayable agent candidates."
        ),
    )
    parser.add_argument("--base-model", choices=sorted(BASE_CONFIGS), help="Override ambiguous base_model")
    parser.add_argument("--recbole-root", help="Path to the adjacent RecBole checkout")
    parser.add_argument(
        "--result-dir",
        default=os.environ.get("RECCLAW_RESULT_DIR"),
        help=(
            "Directory for candidate logs and per-run JSON files. Defaults to "
            "results/candidates; can also be set with RECCLAW_RESULT_DIR."
        ),
    )
    parser.add_argument(
        "--results-csv",
        default=os.environ.get("RECCLAW_RESULTS_CSV"),
        help=(
            "CSV file to append collected results to. Defaults to results/results.csv, "
            "or result-dir/../results.csv when --result-dir is customized."
        ),
    )
    parser.add_argument(
        "--override-dir",
        default=os.environ.get("RECCLAW_OVERRIDE_DIR"),
        help=(
            "Directory for temporary override YAML files. Defaults to results/overrides; "
            "can also be set with RECCLAW_OVERRIDE_DIR."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=os.environ.get("RECCLAW_CHECKPOINT_DIR"),
        help=(
            "Directory for RecBole checkpoints. Defaults to results/checkpoints; "
            "can also be set with RECCLAW_CHECKPOINT_DIR."
        ),
    )
    parser.add_argument(
        "--cleanup-checkpoints",
        action="store_true",
        default=os.environ.get("RECCLAW_CLEANUP_CHECKPOINTS", "").lower() in {"1", "true", "yes"},
        help=(
            "Write this run's checkpoints into a run-specific subdirectory and delete it "
            "after result collection."
        ),
    )
    parser.add_argument(
        "--checkpoint-per-run",
        action="store_true",
        default=os.environ.get("RECCLAW_CHECKPOINT_PER_RUN", "").lower() in {"1", "true", "yes"},
        help="Write this run's checkpoints into a run-specific subdirectory without deleting it.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Append a temporary YAML override as key=value, useful for smoke tests",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved run plan only")
    args = parser.parse_args()

    custom_result_dir = bool(args.result_dir)
    result_dir = Path(args.result_dir).expanduser().resolve() if args.result_dir else RESULT_DIR
    results_csv = (
        Path(args.results_csv).expanduser().resolve()
        if args.results_csv
        else (result_dir.parent / RESULTS_CSV.name if custom_result_dir else RESULTS_CSV)
    )
    override_dir = Path(args.override_dir).expanduser().resolve() if args.override_dir else OVERRIDE_DIR
    registry_path = Path(args.registry_path).expanduser().resolve() if args.registry_path else REGISTRY_PATH

    result_dir.mkdir(parents=True, exist_ok=True)
    try:
        plan = build_plan(
            args.candidate_id,
            args.base_model,
            args.overrides,
            create_override=not args.dry_run,
            result_dir=result_dir,
            override_dir=override_dir,
            registry_path=registry_path,
        )
    except NotImplementedError as exc:
        print(f"Candidate is not runnable yet: {exc}", file=sys.stderr)
        return 2

    recbole_root = resolve_recbole_root(args.recbole_root)
    checkpoint_root = (
        Path(args.checkpoint_dir).expanduser().resolve()
        if args.checkpoint_dir
        else PROJECT_ROOT / "results" / "checkpoints"
    )
    use_run_checkpoint_dir = bool(args.cleanup_checkpoints or args.checkpoint_per_run)
    run_checkpoint_dir = checkpoint_root / plan["run_id"] if use_run_checkpoint_dir else checkpoint_root
    printable_plan = {
        "candidate_id": args.candidate_id,
        "base_model": plan["base_model"],
        "model_name": plan["model_name"],
        "runner_type": plan["runner_type"],
        "entrypoint": plan["entrypoint"],
        "wired": bool(plan["candidate"].get("wired")),
        "consumes": plan["candidate"].get("consumes", []),
        "dataset": plan["dataset"],
        "run_id": plan["run_id"],
        "log_path": str(plan["log_path"]),
        "result_dir": str(result_dir),
        "results_csv": str(results_csv),
        "override_dir": str(override_dir),
        "registry_path": str(registry_path),
        "config_files": [str(path) for path in plan["config_files"]],
        "overrides": plan["overrides"],
        "recbole_root": str(recbole_root),
        "checkpoint_dir": str(run_checkpoint_dir),
        "cleanup_checkpoints": bool(args.cleanup_checkpoints),
        "checkpoint_per_run": bool(use_run_checkpoint_dir),
    }
    print(json.dumps(printable_plan, ensure_ascii=True, indent=2))
    if args.dry_run:
        return 0

    exit_code = 0
    try:
        exit_code = run_recbole_in_process(
            recbole_root=recbole_root,
            model_name=plan["model_name"],
            dataset=plan["dataset"],
            config_files=plan["config_files"],
            log_path=plan["log_path"],
            local_model_entrypoint=plan["local_model_entrypoint"],
            checkpoint_dir=run_checkpoint_dir,
        )
    except Exception as exc:  # noqa: BLE001 - capture crash details into the run log.
        exit_code = 1
        with plan["log_path"].open("a", encoding="utf-8", errors="replace") as handle:
            handle.write(f"\nCRASH: {type(exc).__name__}: {exc}\n")
            handle.write(traceback.format_exc())
        print(f"Candidate run failed: {type(exc).__name__}: {exc}", file=sys.stderr)

    parsed = collect_result(
        log_path=plan["log_path"],
        run_id=plan["run_id"],
        config_change=plan["config_change"],
        candidate_id=args.candidate_id,
        exit_code=exit_code,
        result_dir=result_dir,
        results_csv=results_csv,
    )
    if args.cleanup_checkpoints:
        cleanup_checkpoint_dir(run_checkpoint_dir, checkpoint_root)
    summary = {
        "candidate_id": args.candidate_id,
        "run_id": plan["run_id"],
        "status": parsed.get("status"),
        "log_path": str(plan["log_path"]),
        "result_json_path": parsed.get("result_json_path"),
        "exit_code": exit_code,
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
