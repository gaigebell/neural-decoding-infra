"""CLI: evaluation entry point (dual 口径 + multi-metric, D2/D6).

Metrics (selectable via ``--metrics``, default all):
- ``crr`` / ``cer`` / ``topk`` — character-level, in TWO 口径:
  full characters (legacy-aligned) and Chinese-only (punct/digits filtered;
  punct placement is free in Chinese, so this reflects semantic recovery)
- ``perplexity`` — decoded text fluency under the LM (already in evaluator)
- ``sem_sim`` — BERT (bert-base-chinese) mean-pooled embedding cosine
  between decoded and reference (psycholinguistic semantic fidelity;
  legacy used SentenceBERT — same family)
- ``length_ratio`` / ``unique_ratio`` — completeness diagnostics

Usage:
    python -m recon.cli.eval --decoded-dir decoded/<run> \
        --reference-dir decoded/<run>/references --output report.json

Decoded filenames are ``sub{S}_story{T}.txt``; references ``story{T}.txt``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from ..engine.evaluator import character_error_rate, character_recognition_rate, topk_accuracy
from ..utils.logging import get_logger

logger = get_logger(__name__)

_CJK_RE = re.compile(r"[一-鿿]")

ALL_METRICS = ["crr", "cer", "perplexity", "sem_sim", "length_ratio", "unique_ratio"]

_LOCAL_GPT_PATH = Path("D:/allforwork/Liu_Lab/_Reconstruction/gpt2-chinese-cluecorpussmall")
_CLUSTER_GPT_PATH = Path("/home/test/reconstruction/llm/gpt2-chinese-cluecorpussmall")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate decoded outputs.")
    p.add_argument("--decoded-dir", type=str, required=True)
    p.add_argument("--reference-dir", type=str, required=True)
    p.add_argument("--output", type=str, default="eval_report.json")
    p.add_argument("--metrics", nargs="+", default=None,
                   help=f"Subset of {ALL_METRICS} (default: all).")
    p.add_argument("--per-subject", action="store_true")
    p.add_argument("--no-chinese-only", action="store_true",
                   help="Skip the Chinese-only 口径 (report full-char only).")
    return p.parse_args()


def _chinese_only(text: str) -> str:
    return "".join(_CJK_RE.findall(text))


def _sem_sim(decoded: list[str], refs: list[str]) -> float:
    """BERT mean-pooled embedding cosine similarity (avg over pairs)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    logger.info("Loading bert-base-chinese for semantic similarity...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModel.from_pretrained("bert-base-chinese")
    model.eval()

    def embed(text: str) -> torch.Tensor:
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = model(**ids).last_hidden_state  # (1, L, 768)
        mask = ids["attention_mask"].unsqueeze(-1)
        return (out * mask).sum(dim=1) / mask.sum(dim=1)  # mean pool

    sims = []
    for d, r in zip(decoded, refs):
        if not d or not r:
            sims.append(0.0)
            continue
        a, b = embed(d), embed(r)
        sims.append(float(torch.nn.functional.cosine_similarity(a, b, dim=-1)))
    return sum(sims) / len(sims) if sims else 0.0


def _fluency_perplexity(decoded: list[str], gpt_path: Path) -> float:
    """Decoded-text fluency: exp(mean token NLL) under GPT-2."""
    import math

    import torch
    import torch.nn.functional as F
    from transformers import BertTokenizer, GPT2LMHeadModel

    tokenizer = BertTokenizer.from_pretrained(str(gpt_path))
    model = GPT2LMHeadModel.from_pretrained(str(gpt_path))
    model.eval()
    total_ll, n_tokens = 0.0, 0
    with torch.no_grad():
        for text in decoded:
            if len(text) < 2:
                continue
            ids = tokenizer.encode(text, return_tensors="pt")
            logits = model(ids).logits[0]
            ll = F.cross_entropy(logits[:-1], ids[0, 1:], reduction="sum")
            total_ll += float(ll)
            n_tokens += len(ids[0]) - 1
    return math.exp(total_ll / n_tokens) if n_tokens else float("inf")


def _eval_one(decoded: list[str], refs: list[str], metrics: list[str], gpt_path: Path) -> dict:
    out: dict = {}
    if any(m in metrics for m in ("crr", "cer", "topk")):
        out["crr_full"] = character_recognition_rate(decoded, refs)
        out["cer_full"] = character_error_rate(decoded, refs)
        out["top5_full"] = topk_accuracy([list(d) for d in decoded], refs, k=5)
        out["crr_chinese"] = character_recognition_rate(
            [_chinese_only(d) for d in decoded], [_chinese_only(r) for r in refs])
        out["cer_chinese"] = character_error_rate(
            [_chinese_only(d) for d in decoded], [_chinese_only(r) for r in refs])
    if "perplexity" in metrics:
        out["perplexity"] = _fluency_perplexity(decoded, gpt_path)
    if "sem_sim" in metrics:
        out["sem_sim_bert"] = _sem_sim(decoded, refs)
    if "length_ratio" in metrics:
        out["length_ratio"] = (
            sum(len(d) for d in decoded) / sum(len(r) for r in refs) if refs else 0.0)
    if "unique_ratio" in metrics:
        joined = "".join(decoded)
        out["unique_ratio_decoded"] = len(set(joined)) / len(joined) if joined else 0.0
    return out


def main() -> None:
    args = _parse_args()
    metrics = args.metrics or ALL_METRICS
    decoded_dir = Path(args.decoded_dir)
    ref_dir = Path(args.reference_dir)
    if not decoded_dir.is_dir():
        raise FileNotFoundError(f"decoded dir not found: {decoded_dir}")
    if not ref_dir.is_dir():
        raise FileNotFoundError(f"reference dir not found: {ref_dir}")

    pairs: list[tuple[int, str, str]] = []  # (subject, decoded, ref)
    n_missing = 0
    for f in sorted(decoded_dir.glob("sub*_story*.txt")):
        m = re.match(r"sub(\d+)_story(\d+)\.txt", f.name)
        if not m:
            continue
        sub, story = int(m.group(1)), int(m.group(2))
        ref_path = ref_dir / f"story{story}.txt"
        if not ref_path.exists():
            logger.warning("no reference for %s — skipping", f.name)
            n_missing += 1
            continue
        pairs.append((sub, f.read_text(encoding="utf-8").strip(),
                      ref_path.read_text(encoding="utf-8").strip()))
    logger.info("matched %d files, %d missing", len(pairs), n_missing)

    report: dict = {"n_samples": len(pairs)}
    if pairs:
        decoded, refs = [d for _, d, _ in pairs], [r for _, _, r in pairs]
        gpt_path = _LOCAL_GPT_PATH if sys.platform == "win32" else _CLUSTER_GPT_PATH
        report["metrics"] = _eval_one(decoded, refs, metrics, gpt_path)

    if args.per_subject:
        per: dict = {}
        for sub in sorted({s for s, _, _ in pairs}):
            sub_pairs = [(d, r) for s, d, r in pairs if s == sub]
            per[f"sub{sub}"] = _eval_one([d for d, _ in sub_pairs],
                                         [r for _, r in sub_pairs], metrics, gpt_path)
        report["per_subject"] = per

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Report:\n%s", json.dumps(report, indent=2, ensure_ascii=False))
    logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
