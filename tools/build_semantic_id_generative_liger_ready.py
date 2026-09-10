#!/usr/bin/env python3
"""Build the exact LIGER single-parent READY launch without running it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from recclaw_core.search_spaces.semantic_id_generative_v1.ready import (  # noqa: E402
    build_semantic_id_generative_ready_launch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--template",
        type=Path,
        default=(
            ROOT
            / "configs/single_parent_search_spaces/"
            "semantic_id_generative_liger_single_parent_v1.template.json"
        ),
    )
    parser.add_argument("--parent-runner", type=Path, required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--content-asset", type=Path, required=True)
    parser.add_argument("--structured-field-manifest", type=Path, required=True)
    parser.add_argument("--catalog-mapping", type=Path, required=True)
    parser.add_argument("--sid-mapping", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ready_path = build_semantic_id_generative_ready_launch(
        template_path=args.template,
        parent_runner_path=args.parent_runner,
        parent_result_path=args.parent_result,
        train_path=args.train,
        dev_path=args.dev,
        content_asset_path=args.content_asset,
        structured_manifest_path=args.structured_field_manifest,
        catalog_mapping_path=args.catalog_mapping,
        sid_mapping_path=args.sid_mapping,
        output_root=args.output_root,
    )
    print(ready_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
