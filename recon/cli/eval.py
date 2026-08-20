"""CLI: evaluation entry point.

Usage:
    python -m recon.cli.eval \\
        --decoded-dir results/decoded/ \\
        --reference-dir data/references/ \\
        --output eval_report.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ..engine.evaluator import Evaluator
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate decoded outputs.")
    parser.add_argument(
        "--decoded-dir",
        type=str,
        required=True,
        help="Directory with decoded *.txt files.",
    )
    parser.add_argument(
        "--reference-dir",
        type=str,
        required=True,
        help="Directory with reference *.txt files (same basenames).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_report.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--per-subject",
        action="store_true",
        help="Also compute per-subject breakdown.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    decoded_dir = Path(args.decoded_dir)
    ref_dir = Path(args.reference_dir)

    if not decoded_dir.is_dir():
        raise FileNotFoundError(f"Decoded dir not found: {decoded_dir}")
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"Reference dir not found: {ref_dir}")

    ev = Evaluator()
    n_matched = 0
    n_missing = 0
    for decoded_path in sorted(decoded_dir.glob("*.txt")):
        ref_path = ref_dir / decoded_path.name
        if not ref_path.exists():
            logger.warning("No reference for %s, skipping", decoded_path)
            n_missing += 1
            continue
        pred = decoded_path.read_text(encoding="utf-8").strip()
        ref = ref_path.read_text(encoding="utf-8").strip()
        ev.add(pred=pred, ref=ref)
        n_matched += 1

    logger.info("Matched %d files, %d missing", n_matched, n_missing)
    report = ev.compute()
    out = report.to_dict()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    logger.info("Results:")
    for k, v in out.items():
        logger.info("  %s: %s", k, v)
    logger.info("Saved report to %s", out_path)


if __name__ == "__main__":
    main()