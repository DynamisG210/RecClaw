#!/usr/bin/env python3
"""Build a no-run DiffRec single-parent launch from explicit real inputs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from recclaw_core.search_spaces.diffusion_flow_cf_v1.ready import (  # noqa: E402
    DiffusionFlowReadyError,
    build_diffusion_flow_ready_launch,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and statically validate a self-contained DiffRec READY "
            "descriptor. This command never trains or launches RecClaw."
        )
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--parent-model", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--catalog-items", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        path = build_diffusion_flow_ready_launch(
            template_path=args.template,
            parent_model_path=args.parent_model,
            parent_result_path=args.parent_result,
            train_path=args.train,
            dev_path=args.dev,
            catalog_items_path=args.catalog_items,
            output_root=args.output_root,
        )
    except (OSError, DiffusionFlowReadyError, ValueError) as error:
        parser.error(str(error))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
