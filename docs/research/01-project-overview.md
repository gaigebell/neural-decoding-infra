# 01 — Project Overview

> **Status**: Living document
> **Audience**: Researchers, collaborators, future-self
> **Last updated**: 2026-07-28

---

## 1. What we are studying

This project investigates whether **Chinese characters being read** can be reconstructed from **non-invasive brain signals** — specifically functional MRI (fMRI) and magnetoencephalography (MEG).

The grand goal is a real-time, non-invasive "mind-reading" interface that decodes continuous Chinese text from a person's brain activity while they read naturally.

## 2. Why it matters

- **Accessibility**: Restoring written communication for patients with locked-in syndrome, ALS, or severe paralysis.
- **Neuroscience**: Probing whether semantic representations are spatially and temporally consistent across individuals and modalities.
- **Foundation models for the brain**: Testing whether large pretrained models (e.g., BrainOmni) can serve as universal feature extractors across subjects and recording modalities.

## 3. Scientific approach

We follow the **semantic alignment paradigm** established by [Tang et al., Nature Neuroscience 2023](https://www.biorxiv.org/content/10.1101/2022.09.29.509744v3):

1. Represent the brain signal at each time step as a vector in a high-dimensional **semantic space** (we use a frozen GPT-2 Chinese model with hidden layer 10, dim=768).
2. Train a deep model to map from brain signals → semantic vectors.
3. Decode the predicted semantic vector back to characters using a beam search with nucleus-sampling language model prior.

Compared to direct character classification, this approach:
- Decouples signal modeling from language modeling.
- Allows swapping in different language models without retraining the brain encoder.
- Yields more interpretable intermediate representations.

## 4. Why Chinese (vs English)?

- The literature on continuous character decoding is dominated by English (e.g., [Tang et al. 2023] uses English stories).
- Chinese character-level decoding is harder (large alphabet ~6k characters, smaller character boundaries) but more useful (Chinese has no spaces between words).
- Existing baseline in our lab (师姐李昕玲's ridge regression work) is Chinese; building on it lets us compare directly.

## 5. Key references

| Paper | What we use from it |
|---|---|
| **Tang et al., Nat. Neurosci. 2023** (`Semantic reconstruction of continuous language from non-invasive brain recordings.pdf`) | The semantic-alignment framework; GPT-2 as semantic space; beam-search decoding. |
| **Decoding Continuous Character-based Language from Non-invasive Brain Recordings** | Direct inspiration for our pipeline. |
| **BrainOmni** (Wang et al., 2024) | Pretrained universal brain feature extractor we are integrating. |
| **师姐李昕岭论文** | Our ridge-regression baseline; data source. |

## 6. Key challenges

| Challenge | Current status | Our approach |
|---|---|---|
| Subject variability | Single-subject only | Cross-subject joint training |
| Limited training data per subject | 60 stories × 12 subjects | Scaling to all subjects |
| Brain signal noise | High | Information-bottleneck regularization |
| Decoding latency | ~30 min per story | Batch decoding + GPT KV cache (target: <30 sec) |
| Modality fusion (fMRI + MEG) | Separate models | Shared semantic space → joint model |

## 7. North-star metrics

For the current phase (next 6 months):

| Metric | Current baseline | Target |
|---|---|---|
| Per-character accuracy (top-1) | ~5% (subject 1 only) | ≥15% (joint 12 subjects) |
| Decoding latency (story 60) | ~30 min | <30 sec |
| Cross-subject generalization | N/A | Demonstrated |
| BrainOmni integration | Prototype | Production-ready |

## 8. Out of scope

To prevent scope creep:

- ❌ Real-time BCI (we focus on offline decoding of recorded data).
- ❌ Invasive modalities (ECoG / single-unit) — strictly non-invasive.
- ❌ Image / audio decoding — only text.
- ❌ Causal language modeling — we only decode what the subject already read.

---

## See also

- [Data card](02-data-card.md) — what data we use
- [Cluster card](03-cluster-card.md) — what compute we have
- [Architecture overview](../architecture/01-overview.md) — how the system is built
- [ADR-0002: Pydantic schema](../decisions/0002-pydantic-schema.md) — how we represent data

---

Maintained by owner. Update when project scope or goals change.