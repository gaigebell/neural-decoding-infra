"""CLI: batch decoding for a training run's test set (D2).

Discovers the test split from the run's ``run_metadata.json`` (recomputing
the split from the saved config — works for holdout / ratio / explicit),
decodes every test (subject, story) pair with the aligned flow, and
builds per-story reference texts.

Usage:
    python -m recon.cli.decode_batch \
        --run-dir outputs/ckpt/megA_loso12 --checkpoint best_val.pt \
        --processed-root /home/test/reconstruction/results_v2 \
        --output-dir decoded/megA_loso12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from ..data.drdr import discover_drdr, load_meg_story
from ..data.reference import build_reference
from ..data.split import split_index
from ..decoders.beam import DecodingConfig
from ..engine.trainer import _filter_index
from ..models.registry import build_model
from ..utils.logging import get_logger
from .decode import _LOCAL_GPT_PATH, _CLUSTER_GPT_PATH, decode_story

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch-decode a run's test set.")
    p.add_argument("--run-dir", type=str, required=True,
                   help="Run directory containing run_metadata.json and the checkpoint.")
    p.add_argument("--checkpoint", type=str, default="best_val.pt",
                   help="Checkpoint filename inside --run-dir.")
    p.add_argument("--split", type=str, default="test", help="test | val")
    p.add_argument("--processed-root", type=str, default=None)
    p.add_argument("--char-time-dir", type=str, default=None,
                   help="Directory with story_{N}_char_time.mat (default: auto).")
    p.add_argument("--gpt-path", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--limit", type=int, default=0, help="Max pairs to decode (0 = all).")
    p.add_argument("--beam-width", type=int, default=200)
    p.add_argument("--lm-mass", type=float, default=0.9)
    p.add_argument("--sim-ratio", type=float, default=0.15)
    p.add_argument("--select-layer", type=int, default=10)
    p.add_argument("--n-context", type=int, default=5)
    return p.parse_args()


def _resolve_char_time_dir(processed_root: Path, args: argparse.Namespace) -> Path:
    if args.char_time_dir:
        return Path(args.char_time_dir)
    for rel in (
        Path("mydata/derivatives/annotations/time_align/char-level"),
        Path("annotations/time_align/char-level"),
    ):
        cand = processed_root.parent / rel
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "char-time dir not found — pass --char-time-dir explicitly")


def _test_pairs(run_dir: Path, processed_root: Path, split_name: str):
    """Recompute the split from the run's saved config."""
    meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    cfg = OmegaConf.create(meta["config"])
    data_cfg = cfg.data
    index = discover_drdr(processed_root, modality="meg")
    index = _filter_index(index, data_cfg)
    split = split_index(index, data_cfg.get("split", OmegaConf.create({"method": "none"})))
    pairs = getattr(split, split_name)
    logger.info("%s set: %d (subject, story) pairs", split_name, len(pairs))
    return pairs, data_cfg


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    processed_root = Path(args.processed_root) if args.processed_root else (
        Path("E:/results") if sys.platform == "win32"
        else Path("/home/test/reconstruction/results_v2"))
    gpt_path = Path(args.gpt_path) if args.gpt_path else (
        Path(_LOCAL_GPT_PATH) if sys.platform == "win32" else Path(_CLUSTER_GPT_PATH))
    char_time_dir = _resolve_char_time_dir(processed_root, args)
    out_dir = Path(args.output_dir or f"decoded/{run_dir.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_dir = out_dir / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)

    pairs, data_cfg = _test_pairs(run_dir, processed_root, args.split)
    if args.limit:
        pairs = pairs[: args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ───────────── model ─────────────
    ckpt_path = run_dir / args.checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_cfg = OmegaConf.create(
        {"name": (ckpt.get("config") or {}).get("model", {}).get("name", "meg_model_a"),
         **dict((ckpt.get("config") or {}).get("model", {}))})
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ───────────── GPT-2 ─────────────
    from transformers import BertTokenizer, GPT2LMHeadModel
    tokenizer = BertTokenizer.from_pretrained(str(gpt_path))
    gpt_model = GPT2LMHeadModel.from_pretrained(str(gpt_path)).to(device)
    gpt_model.eval()

    decoding_cfg = DecodingConfig(
        beam_width=args.beam_width, lm_mass=args.lm_mass, sim_ratio=args.sim_ratio,
        select_layer=args.select_layer, max_chars=0, device=str(device),
    )
    n_context = int(args.n_context or data_cfg.get("n_context", 5))

    for i, (sub, story) in enumerate(pairs):
        logger.info("[%d/%d] decoding sub=%d story=%d", i + 1, len(pairs), sub, story)
        mat = char_time_dir / f"story_{story}_char_time.mat"
        story_arr = np.asarray(load_meg_story(processed_root, sub, story, n_context=n_context))
        try:
            text, reference = decode_story(
                model, gpt_model, tokenizer, story_arr, mat,
                n_context=n_context, decoding_cfg=decoding_cfg)
        except Exception as e:  # keep going on per-story failures
            logger.error("decode failed sub=%d story=%d: %s", sub, story, e)
            continue
        (out_dir / f"sub{sub}_story{story}.txt").write_text(text, encoding="utf-8")
        # reference is subject-independent — write once per story
        ref_path = ref_dir / f"story{story}.txt"
        if not ref_path.exists():
            ref_path.write_text(reference, encoding="utf-8")

    logger.info("Batch decode done. Outputs: %s / references: %s", out_dir, ref_dir)


if __name__ == "__main__":
    main()
