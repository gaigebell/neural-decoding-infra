"""CLI: decoding entry point.

Usage:
    python -m recon.cli.decode \\
        --checkpoint /path/to/best.pth \\
        --subject 1 \\
        --story 60 \\
        --output decoded.txt

See ``docs/guides/06-run-decoding.md``.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..data.drdr import load_meg_sample, load_stim_sample
from ..data.datasets.meg import MEGDataset
from ..decoders.beam import BeamSearchDecoder, DecodingConfig
from ..models.registry import build_model
from ..utils.logging import get_logger

logger = get_logger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Decode one story from a trained checkpoint."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .pt checkpoint file.",
    )
    parser.add_argument(
        "--subject", type=int, required=True, help="Subject ID (1-12)."
    )
    parser.add_argument(
        "--story", type=int, required=True, help="Story ID (1-60)."
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output text file path."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/test/reconstruction",
        help="Root of the DRDR data.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meg_model_a",
        help="Model name (must match a registered model).",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default=None,
        help="Optional path to model config YAML.",
    )
    parser.add_argument(
        "--beam-width", type=int, default=200, help="Beam width."
    )
    parser.add_argument(
        "--lm-mass", type=float, default=0.9, help="Nucleus sampling mass."
    )
    parser.add_argument(
        "--max-chars", type=int, default=2000, help="Max characters to decode."
    )
    parser.add_argument(
        "--gpt-path",
        type=str,
        default="/home/test/pretrained/gpt2-chinese",
        help="Path to GPT-2 Chinese model.",
    )
    return parser.parse_args()


def main() -> None:
    """Decode one story end-to-end."""
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Decoding: sub=%d story=%d output=%s", args.subject, args.story, args.output)

    # Load checkpoint
    logger.info("Loading checkpoint: %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = OmegaConf.create({"name": args.model_name, **(ckpt.get("config", {}).get("model", {}))})
    if args.model_config:
        model_cfg = OmegaConf.load(args.model_config)

    # Build model and load weights
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load GPT-2 (lazy import — only when actually decoding)
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    logger.info("Loading GPT-2 from %s", args.gpt_path)
    gpt_tokenizer = GPT2Tokenizer.from_pretrained(args.gpt_path)
    gpt_model = GPT2LMHeadModel.from_pretrained(args.gpt_path).to(device)
    gpt_model.eval()

    # Load brain data for the story
    logger.info("Loading brain data: sub=%d story=%d", args.subject, args.story)
    brain = load_meg_sample(
        data_root=Path(args.data_root),
        subject_id=args.subject,
        story_id=args.story,
        n_context=int(getattr(model_cfg, "n_context", 5)),
    )
    # Convert to tensor
    if hasattr(brain.x, "cpu"):
        brain_tensor = brain.x
    else:
        import numpy as np
        brain_tensor = torch.from_numpy(brain.x).float()
    if brain_tensor.ndim == 2:
        brain_tensor = brain_tensor.unsqueeze(0)  # add T dim
    brain_tensor = brain_tensor.to(device)

    # Build decoder
    decoding_cfg = DecodingConfig(
        beam_width=args.beam_width,
        lm_mass=args.lm_mass,
        max_chars=args.max_chars,
    )
    decoder = BeamSearchDecoder(
        brain_encoder=model,
        gpt_model=gpt_model,
        gpt_tokenizer=gpt_tokenizer,
        config=decoding_cfg,
    )

    # Run decoding
    logger.info("Starting beam search decode...")
    text = decoder.decode(brain_tensor)
    logger.info("Decoded %d characters", len(text))

    # Save output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()