"""CLI: decoding entry point (aligned flow, D1-A).

The decode flow mirrors training exactly: each character is anchored to
its onset zresp row (``char_grid``), the model input is the context
window at ``row - (n_context - 1)`` (zero-padded warmup, same as
training), and the decode length equals the number of valid characters
("tell the model how many characters to fill"). A ``--length`` override
provides the insertion point for a future word-count estimator
(``LengthPredictor``, plan D1-B).

Usage:
    python -m recon.cli.decode --checkpoint <best_val.pt> \
        --subject 1 --story 60 --max-chars 0 \
        --processed-root E:/results

See ``docs/planning/decode-eval-phase.md`` and
``docs/guides/10-user-manual.md``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from ..data.drdr import load_meg_story
from ..decoders.alignment import char_grid, decode_input_rows
from ..decoders.beam import BeamSearchDecoder, DecodingConfig
from ..models.registry import build_model
from ..utils.logging import get_logger

logger = get_logger(__name__)

_LOCAL_GPT_PATH = "D:/allforwork/Liu_Lab/_Reconstruction/gpt2-chinese-cluecorpussmall"
_CLUSTER_GPT_PATH = "/home/test/reconstruction/llm/gpt2-chinese-cluecorpussmall"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode one story from a trained checkpoint (aligned flow)."
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a .pt checkpoint. Omitted = random-init smoke.")
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--story", type=int, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Output text path (default: decoded/sub{s}_story{t}.txt).")
    parser.add_argument("--processed-root", type=str, default=None,
                        help="Preprocessed results root (default: platform auto).")
    parser.add_argument("--char-time", type=str, default=None,
                        help="Path to story_{N}_char_time.mat (default: derived from --processed-root).")
    parser.add_argument("--gpt-path", type=str, default=None,
                        help="GPT-2 model dir (default: platform auto).")
    parser.add_argument("--model-name", type=str, default="meg_model_a")
    parser.add_argument("--model-config", type=str, default=None)
    parser.add_argument("--n-context", type=int, default=5,
                        help="MEG context window (must match training).")
    parser.add_argument("--length", type=int, default=None,
                        help="Decode length override (LengthPredictor insertion "
                             "point; default = number of valid characters).")
    parser.add_argument("--save-reference", action="store_true",
                        help="Also write the reference text next to the output.")
    parser.add_argument("--beam-width", type=int, default=200)
    parser.add_argument("--lm-mass", type=float, default=0.9)
    parser.add_argument("--sim-ratio", type=float, default=0.15)
    parser.add_argument("--select-layer", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=0,
                        help="Cap on decode steps (0 = no cap).")
    return parser.parse_args()


def _defaults(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    if args.processed_root:
        processed_root = Path(args.processed_root)
    elif sys.platform == "win32":
        processed_root = Path("E:/results")
    else:
        processed_root = Path("/home/test/reconstruction/results_v2")

    if args.gpt_path:
        gpt_path = Path(args.gpt_path)
    elif sys.platform == "win32":
        gpt_path = Path(_LOCAL_GPT_PATH)
    else:
        gpt_path = Path(_CLUSTER_GPT_PATH)

    char_time = Path(args.char_time) if args.char_time else None
    return processed_root, gpt_path, char_time


def _build_aligned_features(
    model: torch.nn.Module,
    story: np.ndarray,  # (T, n_context, C) — the context-window story
    char_time: Path,
    n_context: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    """Per-character brain features with training-identical inputs.

    Returns (features (N, 768), chars) where N = number of valid chars.
    """
    chars, rows = char_grid(char_time, n_rows=story.shape[0])
    anchors = decode_input_rows(rows, n_context=n_context)  # row - (ctx-1)

    inputs = np.zeros((len(chars), n_context, story.shape[2]), dtype=np.float32)
    for i, a in enumerate(anchors):
        if a >= 0:
            inputs[i] = story[a]
    batch = torch.from_numpy(inputs).to(device)  # (N, ctx, C)
    with torch.no_grad():
        out = model(batch)
        if isinstance(out, tuple):
            out = out[0]
    return out, chars


def decode_story(
    model: torch.nn.Module,
    gpt_model,
    tokenizer,
    story: np.ndarray,
    char_time: Path,
    n_context: int,
    decoding_cfg: DecodingConfig,
    length: int | None = None,
) -> tuple[str, str]:
    """Decode one story with the aligned flow.

    Returns (decoded_text, reference_text). ``length`` is the optional
    decode-length override (LengthPredictor insertion point).
    """
    features, chars = _build_aligned_features(model, story, char_time, n_context, decoding_cfg.device)
    n_chars = len(chars)
    if length is not None:
        if length > n_chars:
            raise ValueError(f"length {length} > valid characters {n_chars}")
        features = features[:length]
        n_chars = length

    decoder = BeamSearchDecoder(
        brain_encoder=model, gpt_model=gpt_model, gpt_tokenizer=tokenizer,
        config=decoding_cfg,
    )
    text = decoder.decode(features)
    return text, "".join(chars[:n_chars])


def main() -> None:
    args = _parse_args()
    processed_root, gpt_path, char_time = _defaults(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Decoding sub=%d story=%d device=%s", args.subject, args.story, device)

    # ───────────── brain model ─────────────
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        saved_model_cfg = (ckpt.get("config") or {}).get("model") or {}
        from omegaconf import OmegaConf
        model_cfg = OmegaConf.create({"name": args.model_name, **dict(saved_model_cfg)})
        if args.model_config:
            model_cfg = OmegaConf.load(args.model_config)
        model = build_model(model_cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        logger.warning("No checkpoint — random-init model (pipeline smoke only)")
        from omegaconf import OmegaConf
        model_cfg = (OmegaConf.load(args.model_config) if args.model_config
                     else OmegaConf.create({"name": args.model_name, "n_channels": 306, "n_context": args.n_context}))
        model = build_model(model_cfg).to(device)
    model.eval()

    # ───────────── GPT-2 ─────────────
    from transformers import BertTokenizer, GPT2LMHeadModel
    logger.info("Loading GPT-2 from %s", gpt_path)
    tokenizer = BertTokenizer.from_pretrained(str(gpt_path))
    gpt_model = GPT2LMHeadModel.from_pretrained(str(gpt_path)).to(device)
    gpt_model.eval()

    # ───────────── story + char-time ─────────────
    story = np.asarray(
        load_meg_story(processed_root, args.subject, args.story, n_context=args.n_context)
    )  # (T, ctx, C)
    logger.info("Story tensor: %s", tuple(story.shape))
    if char_time is None:
        candidates = [
            # local: E:/reconstruction/mydata/...  (sibling of E:/results)
            processed_root.parent / "reconstruction" / "mydata" / "derivatives" / "annotations",
            # local direct layout
            processed_root.parent / "mydata" / "derivatives" / "annotations",
            # cluster: annotations/ directly under the reconstruction root
            processed_root.parent / "annotations",
        ]
        for cand in candidates:
            mat = cand / "time_align" / "char-level" / f"story_{args.story}_char_time.mat"
            if mat.exists():
                char_time = mat
                break
    if not char_time.exists():
        raise FileNotFoundError(
            f"char-time mat not found: {char_time} (pass --char-time explicitly)")

    # ───────────── decode (aligned flow) ─────────────
    decoding_cfg = DecodingConfig(
        beam_width=args.beam_width,
        lm_mass=args.lm_mass,
        sim_ratio=args.sim_ratio,
        select_layer=args.select_layer,
        max_chars=args.max_chars or 0,
        device=str(device),
    )
    text, reference = decode_story(
        model, gpt_model, tokenizer, story, char_time,
        n_context=args.n_context, decoding_cfg=decoding_cfg, length=args.length,
    )
    logger.info("Decoded %d characters", len(text))

    out_path = Path(args.output or f"decoded/sub{args.subject}_story{args.story}.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    logger.info("Saved to %s", out_path)

    if args.save_reference:
        ref_path = out_path.with_name(out_path.stem + ".ref.txt")
        ref_path.write_text(reference, encoding="utf-8")
        logger.info("Reference saved to %s", ref_path)


if __name__ == "__main__":
    main()
