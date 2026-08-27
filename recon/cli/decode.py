"""CLI: decoding entry point.

Usage:
    python -m recon.cli.decode \\
        --checkpoint results/ckpt/epoch_1/checkpoint_epoch_1.pt \\
        --subject 1 --story 1 --processed-root E:/results \\
        --max-chars 100 --output decoded.txt

    # Random-init model (pipeline smoke; the model predicts garbage):
    python -m recon.cli.decode --subject 1 --story 1 --max-chars 50 ...

See ``docs/guides/06-run-decoding.md``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from ..data.drdr import load_meg_story
from ..decoders.beam import BeamSearchDecoder, DecodingConfig
from ..models.registry import build_model
from ..utils.logging import get_logger

logger = get_logger(__name__)

_LOCAL_GPT_PATH = "D:/allforwork/Liu_Lab/_Reconstruction/gpt2-chinese-cluecorpussmall"
_CLUSTER_GPT_PATH = "/home/test/reconstruction/pretrained/gpt2-chinese-cluecorpussmall"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Decode one story from a trained checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a .pt checkpoint. If omitted, a random-init model is "
        "used (pipeline smoke only — predictions are meaningless).",
    )
    parser.add_argument(
        "--subject", type=int, required=True, help="Subject ID (1-12)."
    )
    parser.add_argument(
        "--story", type=int, required=True, help="Story ID (1-60)."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output text file path. Default: <results_dir>/decoded/"
        "sub{subject}_story{story}.txt",
    )
    parser.add_argument(
        "--processed-root",
        type=str,
        default=None,
        help="Root of the preprocessed results. Default: E:/results on "
        "Windows, /home/test/reconstruction/results otherwise.",
    )
    parser.add_argument(
        "--gpt-path",
        type=str,
        default=None,
        help="Path to the GPT-2 Chinese model directory. Default: local "
        "gpt2-chinese-cluecorpussmall checkout on Windows, cluster "
        "pretrained dir otherwise.",
    )
    parser.add_argument(
        "--model-name", type=str, default="meg_model_a",
        help="Model name (must match a registered model). Used when no "
        "checkpoint is given, or no config is saved inside the checkpoint.",
    )
    parser.add_argument(
        "--model-config", type=str, default=None,
        help="Optional path to a model config YAML (overrides checkpoint config).",
    )
    parser.add_argument(
        "--n-context", type=int, default=5,
        help="MEG context window size (matches the training config).",
    )
    parser.add_argument(
        "--beam-width", type=int, default=200, help="Beam width."
    )
    parser.add_argument(
        "--lm-mass", type=float, default=0.9, help="Nucleus sampling mass."
    )
    parser.add_argument(
        "--sim-ratio", type=float, default=0.15,
        help="Weight of cosine similarity vs LM logprob.",
    )
    parser.add_argument(
        "--select-layer", type=int, default=10,
        help="GPT-2 layer for candidate semantic features.",
    )
    parser.add_argument(
        "--max-chars", type=int, default=2000, help="Max characters to decode."
    )
    return parser.parse_args()


def _default_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    """Resolve processed_root and gpt_path defaults per platform."""
    if args.processed_root:
        processed_root = Path(args.processed_root)
    elif sys.platform == "win32":
        processed_root = Path("E:/results")
    else:
        processed_root = Path("/home/test/reconstruction/results")

    if args.gpt_path:
        gpt_path = Path(args.gpt_path)
    elif sys.platform == "win32":
        gpt_path = Path(_LOCAL_GPT_PATH)
    else:
        gpt_path = Path(_CLUSTER_GPT_PATH)

    for label, p in (("processed_root", processed_root), ("gpt_path", gpt_path)):
        if not p.exists():
            raise FileNotFoundError(f"{label} does not exist: {p} (override with --{label.replace('_', '-')})")
    return processed_root, gpt_path


def main() -> None:
    """Decode one story end-to-end."""
    args = _parse_args()
    processed_root, gpt_path = _default_paths(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Decoding: sub=%d story=%d device=%s", args.subject, args.story, device
    )

    # ───────────────────── Brain model ─────────────────────

    if args.checkpoint:
        logger.info("Loading checkpoint: %s", args.checkpoint)
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        saved_model_cfg = (ckpt.get("config") or {}).get("model") or {}
        model_cfg = OmegaConf.create(
            {"name": args.model_name, **dict(saved_model_cfg)}
        )
        if args.model_config:
            model_cfg = OmegaConf.load(args.model_config)
        model = build_model(model_cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        logger.warning(
            "No checkpoint given — using a random-init model. "
            "Decoded text is meaningless; this is a pipeline smoke only."
        )
        model_cfg = (
            OmegaConf.load(args.model_config)
            if args.model_config
            else OmegaConf.create(
                {"name": args.model_name, "n_channels": 306, "n_context": args.n_context}
            )
        )
        model = build_model(model_cfg).to(device)
    model.eval()

    # ───────────────────── GPT-2 ─────────────────────

    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    logger.info("Loading GPT-2 from %s", gpt_path)
    gpt_tokenizer = GPT2Tokenizer.from_pretrained(str(gpt_path))
    gpt_model = GPT2LMHeadModel.from_pretrained(str(gpt_path)).to(device)
    gpt_model.eval()

    # ───────────────────── Brain data ─────────────────────

    logger.info("Loading MEG story: sub=%d story=%d", args.subject, args.story)
    story = np.asarray(
        load_meg_story(processed_root, args.subject, args.story, n_context=args.n_context)
    )
    # Copy: memmap-backed arrays are read-only; torch needs a writable base
    brain_tensor = torch.from_numpy(story.copy()).float().to(device)  # (T, ctx, C)
    logger.info("Story tensor: %s", tuple(brain_tensor.shape))

    # ───────────────────── Decode ─────────────────────

    decoding_cfg = DecodingConfig(
        beam_width=args.beam_width,
        lm_mass=args.lm_mass,
        sim_ratio=args.sim_ratio,
        select_layer=args.select_layer,
        max_chars=args.max_chars,
        device=str(device),
    )
    decoder = BeamSearchDecoder(
        brain_encoder=model,
        gpt_model=gpt_model,
        gpt_tokenizer=gpt_tokenizer,
        config=decoding_cfg,
    )

    logger.info("Starting beam search decode...")
    text = decoder.decode(brain_tensor)
    logger.info("Decoded %d characters", len(text))

    out_path = Path(
        args.output or f"decoded/sub{args.subject}_story{args.story}.txt"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
