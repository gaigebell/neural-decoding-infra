# Guide 06: Run decoding

> **Audience**: Anyone running inference / decoding on a trained model.

---

## What decoding does

Given a trained model, decode a story's worth of brain signals into Chinese characters. Uses:

1. Brain encoder → predicted semantic vector per word position
2. GPT-2 (frozen) with KV cache → nucleus-sampled candidates
3. Beam search → best character sequence

## Prerequisites

- A trained checkpoint (from [Guide 04](04-run-training.md))
- GPT-2 Chinese pretrained weights (downloaded once)

## Quickstart

```bash
# On a compute node
ssh cn3
cd /home/test/reconstruction/neural-decoding-infra

# Decode one story (subject 1, story 60)
python -m recon.cli.decode \
    --checkpoint /home/test/reconstruction/results/ckpt/run_id/best.pth \
    --subject 1 \
    --story 60 \
    --output /home/test/reconstruction/results/decoded/sub01_story60.txt
```

Expected: ~30 seconds (vs 30 minutes in the old pipeline).

## CLI options

```
python -m recon.cli.decode [OPTIONS]

  --checkpoint PATH       Path to .pth checkpoint (required)
  --subject INT           Subject ID (1-12) (required)
  --story INT             Story ID (1-60) (required)
  --output PATH           Where to save decoded text (required)
  --beam-width INT        Beam width (default 200)
  --lm-mass FLOAT         Nucleus mass (default 0.9)
  --lm-ratio FLOAT        Nucleus ratio (default 0.1)
  --device STR            Device (default cuda:0)
```

## Batch decoding (multiple stories)

```bash
# Decode stories 50-60 for subject 1
for story in 50 51 52 53 54 55 56 57 58 59 60; do
    python -m recon.cli.decode \
        --checkpoint /path/to/best.pth \
        --subject 1 \
        --story $story \
        --output /home/test/reconstruction/results/decoded/sub01_story${story}.txt
done
```

For cross-subject sweeps:

```bash
for sub in 1 2 3 4 5 6 7 8 9 10 11 12; do
    for story in 55 56 57 58 59 60; do
        python -m recon.cli.decode \
            --checkpoint /path/to/best.pth \
            --subject $sub \
            --story $story \
            --output /home/test/reconstruction/results/decoded/sub${sub}_story${story}.txt
    done
done
```

## How it works (in 30 seconds)

```python
# Pseudocode
brain_features = encoder(zresp)  # (T, 768)
beam = [Hypothesis()]  # empty hypothesis
for t in range(T):
    candidates = lm.beam_propose(beam, context=t)  # nucleus sample, with KV cache
    candidate_embs = lm.embed(candidates)  # (K, 768)
    scores = cosine_sim(brain_features[t], candidate_embs)  # (K,)
    beam = update_beam(beam, candidates, scores, beam_width=200)
return beam[0].words  # best sequence
```

See [Architecture 05: Decoding engine](../architecture/05-decoding-engine.md) for full details.

## Where output goes

| File | Content |
|---|---|
| `*.txt` | Decoded character sequence (one story) |
| `*.json` | Beam candidates + scores (for debugging) |
| W&B | Per-word confidence, beam entropy (if logged) |

## Common issues

| Issue | Fix |
|---|---|
| "Checkpoint not found" | Check path; ensure full path |
| OOM during decoding | Lower `beam-width` |
| Wrong shape mismatch | Ensure checkpoint matches current model config |
| Slow decoding (>5 min/story) | KV cache not enabled? Check `recon/decoders/beam.py` |

## See also

- [Architecture 05: Decoding engine](../architecture/05-decoding-engine.md)
- [Guide 04: Run training](04-run-training.md)
- [Guide 07: Run evaluation](07-run-evaluation.md)

---

Maintained by owner.