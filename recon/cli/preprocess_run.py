"""Runner utilities for the preprocessing pipeline (Phase 2).

The dispatcher is ``scripts/run_preprocess.sh`` (ssh + worker slices, the
same pattern as ``launch_multi_node.sh``). This module provides the
mgmt-side ``status`` command: it scans the output directories and reports
per-stage progress from the sidecar files — the filesystem IS the task
queue, so progress = sidecar coverage.

Usage:
    python -m recon.cli.preprocess_run status
    python -m recon.cli.preprocess_run status --subjects 1 2 --stages meg_zresp
"""
from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose, initialize
from omegaconf import OmegaConf


def _artifact_patterns(cfg, stage: str) -> dict[str, list[str]]:
    """Expected artifact name templates per (subject, story) per stage."""
    layer = cfg.layer
    return {
        "gpt_char_features": [f"wordvectors/story{{story}}_{L}.npy" for L in cfg.layers],
        "semantic_downsample": [f"downsample/ds_{{story}}_{layer}.npy"],
        "semantic_delay": [f"MEG/zstim/sub{{sub}}_zstim_{layer}_story{{story}}.npy"],
        "meg_zresp": [f"MEG/zresp/zresp{{sub}}_{{story}}.npy"],
        "meg_context": [f"MEG/zresp/zresp{{sub}}_{{story}}_context_{cfg.n_context}.npy"],
        "fmri_cube": [f"zresp/cube/zresp{{sub}}_{{story}}.npy"],
    }[stage]


# Stages whose artifacts are per-story only (all subjects share the same
# story stimuli/time-alignment).
_SUBJECT_INDEPENDENT_STAGES = {"gpt_char_features", "semantic_downsample"}

_ALL_STAGES = [
    "gpt_char_features",
    "semantic_downsample",
    "semantic_delay",
    "meg_zresp",
    "meg_context",
    "fmri_cube",
]


def _status(args: argparse.Namespace) -> None:
    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="preprocess", overrides=[f"paths={args.paths}"])
    root = Path(cfg.paths.processed_root)

    subjects = args.subjects or list(range(1, 13))
    stories = args.stories or list(range(1, 61))
    stages = args.stages or _ALL_STAGES

    print(f"{'stage':22s} {'done':>6s} {'total':>6s}  pct")
    print("-" * 46)
    for stage in stages:
        patterns = _artifact_patterns(cfg, stage)
        subs = [1] if stage in _SUBJECT_INDEPENDENT_STAGES else subjects
        done = total = 0
        for sub in subs:
            for story in stories:
                for pat in patterns:
                    total += 1
                    artifact = root / pat.format(sub=sub, story=story)
                    sidecar = artifact.with_suffix(artifact.suffix + ".meta.json")
                    if sidecar.exists():
                        done += 1
        pct = 100.0 * done / total if total else 0.0
        print(f"{stage:22s} {done:6d} {total:6d}  {pct:5.1f}%")


def _stats(args: argparse.Namespace) -> None:
    """Dataset size statistics: coverage, sample counts, split-aware totals."""
    import numpy as np

    from recon.data.drdr import discover_drdr

    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="preprocess", overrides=[f"paths={args.paths}"])
    root = Path(cfg.paths.processed_root)

    subjects = args.subjects or list(range(1, 13))
    index = discover_drdr(root, modality="meg", max_subjects=max(subjects))
    if args.subjects:
        from recon.engine.trainer import _filter_index
        from omegaconf import OmegaConf
        index = _filter_index(index, OmegaConf.create({"subjects": subjects}))

    print(f"{'sub':>4s} {'stories':>8s} {'samples':>9s}  zresp shape (first)")
    print("-" * 52)
    total = 0
    for sub in index.subjects:
        sub_pairs = [p for p in index.pairs if p[0] == sub]
        n_samples = 0
        shape = None
        for _, story in sub_pairs:
            zresp = root / "MEG" / "zresp" / f"zresp{sub}_{story}.npy"
            if not zresp.exists():
                continue
            a = np.load(zresp, mmap_mode="r")
            n_samples += a.shape[0]
            if shape is None:
                shape = tuple(a.shape)
        total += n_samples
        print(f"{sub:4d} {len(sub_pairs):8d} {n_samples:9d}  {shape}")

    print("-" * 52)
    print(f"total: {index.n_subjects()} subjects, {index.n_pairs()} (sub, story) pairs, "
          f"{total} samples")
    avg = total / index.n_pairs() if index.n_pairs() else 0
    print(f"avg {avg:.0f} samples/story")

    if args.split:
        from omegaconf import OmegaConf
        from recon.data.split import split_index
        split = split_index(
            index,
            OmegaConf.create({
                "method": args.split,
                "val_ratio": args.val_ratio,
                "test_ratio": args.test_ratio,
                "seed": args.seed,
                "test_subjects": args.test_subjects,
            }),
        )
        counts = {"train": 0, "val": 0, "test": 0}
        n_stories = {"train": 0, "val": 0, "test": 0}
        for name, pairs in (("train", split.train), ("val", split.val), ("test", split.test)):
            for sub, story in pairs:
                zresp = root / "MEG" / "zresp" / f"zresp{sub}_{story}.npy"
                if zresp.exists():
                    counts[name] += np.load(zresp, mmap_mode="r").shape[0]
                    n_stories[name] += 1
        print(f"\nsplit[{args.split}]: "
              f"train {n_stories['train']} stories / {counts['train']} samples | "
              f"val {n_stories['val']} / {counts['val']} | "
              f"test {n_stories['test']} / {counts['test']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocessing runner utilities.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="Scan sidecars and report per-stage progress.")
    p_status.add_argument("--subjects", type=int, nargs="+", default=None)
    p_status.add_argument("--stories", type=int, nargs="+", default=None)
    p_status.add_argument("--stages", nargs="+", default=None)
    p_status.add_argument("--paths", default="cluster", help="paths config: cluster | local")

    p_stats = sub.add_parser("stats", help="Dataset size: coverage, samples, split-aware totals.")
    p_stats.add_argument("--subjects", type=int, nargs="+", default=None)
    p_stats.add_argument("--paths", default="cluster", help="paths config: cluster | local")
    p_stats.add_argument("--split", default=None, help="none | ratio | holdout (adds split-aware totals)")
    p_stats.add_argument("--val-ratio", type=float, default=0.15)
    p_stats.add_argument("--test-ratio", type=float, default=0.15)
    p_stats.add_argument("--seed", type=int, default=42)
    p_stats.add_argument("--test-subjects", type=int, nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "status":
        _status(args)
    elif args.command == "stats":
        _stats(args)
    else:  # pragma: no cover
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()