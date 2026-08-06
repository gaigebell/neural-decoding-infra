# Guide 07: Run evaluation

> **Audience**: Anyone measuring model performance.

---

## What evaluation computes

Given decoded character sequences and ground-truth, compute:

| Metric | What it measures | Typical value |
|---|---|---|
| **CRR** | Character Recognition Rate (top-1 accuracy) | 0–100% |
| **CER** | Character Error Rate (edit distance / length) | 0–100% |
| **Top-k** | Top-k accuracy (k=5, 10) | 0–100% |
| **BLEU-n** | N-gram overlap (n=1, 2, 4) | 0–1 |
| **Perplexity** | LM perplexity of decoded sequence | lower = better |

## Prerequisites

- Decoded outputs (from [Guide 06](06-run-decoding.md))
- Ground-truth text (from preprocessing pipeline)

## Run evaluation

```bash
# On a compute node
python -m recon.cli.eval \
    --decoded-dir /home/test/reconstruction/results/decoded/ \
    --reference-dir /home/test/reconstruction/data/eval_references/ \
    --output /home/test/reconstruction/results/eval_report.json
```

The expected output format:

```json
{
  "overall": {
    "crr": 12.3,
    "cer": 78.5,
    "top5": 28.7,
    "bleu1": 0.45,
    "bleu2": 0.21,
    "bleu4": 0.05,
    "perplexity": 142.3
  },
  "per_subject": {
    "1": {"crr": 13.5, "cer": 75.2, ...},
    "2": {"crr": 11.8, "cer": 80.1, ...}
  },
  "per_story": {...}
}
```

## CLI options

```
python -m recon.cli.eval [OPTIONS]

  --decoded-dir PATH        Directory with decoded *.txt files (required)
  --reference-dir PATH      Directory with reference *.txt files (required)
  --output PATH             Where to save metrics JSON (default: eval_report.json)
  --metrics [crr cer topk bleu perp]  Which metrics to compute (default: all)
  --per-subject             Also compute per-subject breakdown
  --per-story               Also compute per-story breakdown
```

## What W&B logs

- `eval/crr` — overall
- `eval/cer`
- `eval/top5`, `eval/top10`
- `eval/per_subject/{sub_id}/crr` — per subject

## Reproducibility

For paper-quality numbers, always:

1. Use the same checkpoint
2. Decode the same set of stories (held-out test set)
3. Reference the run ID + config in the report

```bash
# Save experiment snapshot
cp configs/experiment/003-eval-baseline.yaml results/eval_reports/run_id/config.yaml
```

## Common pitfalls

| Pitfall | How to avoid |
|---|---|
| Comparing across different splits | Lock the test split in `data/test_stories=[...]` |
| Reporting on training data | Always use `data.subjects=[h.O.O]` (held-out) |
| Different preprocessing | Pin preprocessing version in report |
| Different LM | Same GPT-2 layer 10 always |

## See also

- [Architecture 06: Evaluation](../architecture/06-evaluation.md)
- [Guide 06: Run decoding](06-run-decoding.md)
- [Standard 04: Testing](../standards/04-testing.md)

---

Maintained by owner.