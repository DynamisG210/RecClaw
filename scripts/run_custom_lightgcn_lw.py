#!/usr/bin/env python3
"""Run the local LightGCNLW scaffold with a local RecBole checkout.

The parser is intentionally stdlib-only so ``--help`` works even when the active
interpreter does not have torch or RecBole installed. Use a RecBole-capable
Python environment for actual training runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from logging import getLogger
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_FILES = (
    CONFIG_DIR / "task_ml1m.yaml",
    CONFIG_DIR / "lightgcn.yaml",
    CONFIG_DIR / "lightgcn_metrics.yaml",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the local LightGCNLW scaffold with a local RecBole checkout."
    )
    parser.add_argument(
        "--dataset",
        default="ml-1m",
        help="Dataset name passed to RecBole (default: ml-1m).",
    )
    parser.add_argument(
        "--config-files",
        nargs="*",
        default=[],
        metavar="CONFIG",
        help="Additional config files layered after the default task/model/metrics configs.",
    )
    parser.add_argument(
        "--recbole-root",
        help="Path to a local RecBole checkout. Defaults to RECBOLE_ROOT/RECBole_ROOT or ../RecBole.",
    )
    parser.add_argument(
        "--saved",
        dest="saved",
        action="store_true",
        default=True,
        help="Save the best checkpoint during training (default: enabled).",
    )
    parser.add_argument(
        "--no-saved",
        dest="saved",
        action="store_false",
        help="Disable checkpoint saving.",
    )
    return parser


def resolve_config_path(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(PROJECT_ROOT / raw_path)

    for path in candidates:
        if path.is_file():
            return path.resolve()

    raise FileNotFoundError(f"config file not found: {raw_path}")


def resolve_config_files(extra_config_files: list[str]) -> list[str]:
    config_files: list[str] = []
    for path in DEFAULT_CONFIG_FILES:
        if not path.is_file():
            raise FileNotFoundError(f"missing default config file: {path}")
        config_files.append(str(path.resolve()))

    for raw_path in extra_config_files:
        config_files.append(str(resolve_config_path(raw_path)))

    return config_files


def resolve_recbole_root(explicit_root: str | None) -> Path:
    candidates = [
        explicit_root,
        os.environ.get("RECBOLE_ROOT"),
        os.environ.get("RECBole_ROOT"),
        str(PROJECT_ROOT.parent / "RecBole"),
    ]

    for raw_path in candidates:
        if not raw_path:
            continue
        path = Path(raw_path).expanduser().resolve()
        if (path / "run_recbole.py").is_file() and (path / "recbole").is_dir():
            return path

    raise FileNotFoundError(
        "could not find a usable RecBole checkout. "
        "Set --recbole-root or RECBOLE_ROOT/RECBole_ROOT."
    )


def extend_python_path(recbole_root: Path) -> None:
    for path in (PROJECT_ROOT, recbole_root):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def import_runtime_dependencies(recbole_root: Path):
    extend_python_path(recbole_root)

    try:
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.utils import get_trainer, init_logger, init_seed
        from recclaw_ext.models import LightGCNLW
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Failed to import RecBole runtime dependencies. "
            "Use a Python environment with torch and RecBole available."
        ) from exc

    return (
        Config,
        create_dataset,
        data_preparation,
        get_trainer,
        init_logger,
        init_seed,
        LightGCNLW,
    )


def build_config(config_class, model_class, dataset: str, config_files: list[str]):
    original_argv = sys.argv[:]
    try:
        # RecBole's Config also inspects sys.argv, so hide this runner's CLI flags.
        sys.argv = [sys.argv[0]]
        return config_class(
            model=model_class,
            dataset=dataset,
            config_file_list=config_files,
        )
    finally:
        sys.argv = original_argv


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass

    return value


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    recbole_root = resolve_recbole_root(args.recbole_root)
    config_files = resolve_config_files(args.config_files)
    runtime = import_runtime_dependencies(recbole_root)
    (
        Config,
        create_dataset,
        data_preparation,
        get_trainer,
        init_logger,
        init_seed,
        LightGCNLW,
    ) = runtime

    # LightGCNLW is a local class, so this runner layers the LightGCN configs explicitly
    # instead of relying on a LightGCNLW.yaml file inside RecBole.
    config = build_config(Config, LightGCNLW, args.dataset, config_files)

    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)

    train_data, valid_data, test_data = data_preparation(config, dataset)

    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    model = LightGCNLW(config, train_data._dataset).to(config["device"])
    logger.info(model)

    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=args.saved,
        show_progress=config["show_progress"],
    )
    test_result = trainer.evaluate(
        test_data,
        load_best_model=args.saved,
        show_progress=config["show_progress"],
    )

    return {
        "model": config["model"],
        "dataset": config["dataset"],
        "valid_metric": config["valid_metric"],
        "config_files": config_files,
        "recbole_root": str(recbole_root),
        "saved": args.saved,
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        result = run_experiment(args)
    except (FileNotFoundError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")

    print(json.dumps(to_jsonable(result), indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
