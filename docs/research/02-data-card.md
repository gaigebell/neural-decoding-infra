# 02 — Data Card

> **Status**: Living document
> **Audience**: Anyone who touches the data
> **Last updated**: 2026-07-28

This document follows the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) convention.

---

## 1. Dataset summary

| Field | Value |
|---|---|
| **Name** | DRDR (Decoding Reading) Chinese narrative-reading dataset |
| **Source** | 师姐李昕岭's thesis data; originally from Tang et al. 2023 (English) and extended to Chinese |
| **Subjects** | 12 native Chinese speakers (sub-01 ~ sub-12) |
| **Modalities** | MEG (306 channels), fMRI (MNI 3D + CIFTI surface) |
| **Task** | Continuous reading of 60 Chinese stories |
| **Total runs** | ~60 runs × 12 subjects = ~720 sessions |
| **Storage size** | ~250 GB (raw) + ~50 GB (preprocessed) |
| **Distribution** | Internal to lab; not publicly shareable |
| **Path (cluster)** | `/home/test/reconstruction/` (NFS from mgmt) |
| **Path (local mirror)** | `E:/reconstruction/` on owner's dev machine |

## 2. Collection

- **Subjects**: Right-handed native Chinese speakers, recruited from [university student pool].
- **Stimuli**: 60 Chinese stories adapted from public-domain Chinese fictions and news articles, ranging ~1,500–3,000 characters each.
- **Task design**: Subjects read each story silently while MEG and fMRI are recorded simultaneously.
- **Recording**:
  - MEG: Elekta/Neuromag VectorView system, 306 channels (102 magnetometers + 204 gradiometers), sampled at 1 kHz.
  - fMRI: 3T Siemens scanner, TR=2 s, voxel size 2×2×2 mm, whole-brain coverage.
- **Ethics**: IRB-approved; written informed consent; subjects anonymized as sub-01 … sub-12.

## 3. Structure

Following BIDS-like conventions:

```
reconstruction/
├── mydata/
│   ├── dataset_description.json
│   ├── participants.tsv
│   ├── participants.json
│   ├── CHANGES
│   ├── README
│   └── derivatives/
│       ├── annotations/
│       │   ├── embeddings/{bert,gpt,word2vec}/
│       │   ├── frequency/{char-level,word-level}/
│       │   ├── quiz/
│       │   ├── scripts/
│       │   ├── syntactic_annotations/{constituency_parsing,dependency_parsing,part_of_speech}/
│       │   └── time_align/{char-level,word-level}/
│       └── preprocessed_data/
│           └── sub-XX/{MEG,MNI,CIFTI}/
└── results/
    ├── MEG/{downsample,noise,word_rate,zresp,zstim}/
    ├── clean-wordvectors/
    ├── wordvectors/
    └── ...
```

### Per-subject layout

```
sub-XX/
├── MEG/                                       # 60 runs/subject
│   ├── sub-XX_task-RDR_run-YY_meg.fif
│   ├── sub-XX_task-RDR_run-YY_meg.json
│   ├── sub-XX_task-RDR_run-YY_channels.tsv
│   └── sub-XX_task-RDR_run-YY_events.tsv
├── MNI/                                       # ~64 runs/subject
│   ├── mask.nii
│   └── sub-XX_task-RDR_run-YY_bold.nii.gz
└── CIFTI/                                     # ~64 runs/subject
    └── sub-XX_task-RDR_run-YY_bold.dtseries.nii
```

### Per-subject counts (as of 2026-07)

| Subject | MEG runs | MNI runs | CIFTI runs |
|---|---|---|---|
| sub-01 | 60 | 56 | 64 |
| sub-02 | 60 | 64 | 64 |
| sub-03 | 60 | 64 | 64 |
| sub-04 | 56 | 64 | 64 |
| sub-05 | 56 | 64 | 64 |
| sub-06 | 55 | 59 | 61 |
| sub-07 | 60 | 59 | 64 |
| sub-08 | 60 | 51 | 64 |
| sub-09 | 60 | 64 | 64 |
| sub-10 | 60 | 59 | 64 |
| sub-11 | 60 | 64 | 64 |
| sub-12 | 60 | 61 | 64 |

> Asymmetric run counts are real (some subjects dropped runs due to motion / equipment issues). Do not assume equal coverage.

## 4. Preprocessing outputs

### MEG (already preprocessed on dev machine)

| File | Shape | Notes |
|---|---|---|
| `results/MEG/zresp1_YY.npy` | (T, C) or (T, n_context, C) | Z-scored, downsampled, chunked |
| `results/MEG/zresp1_YY_context_5.npy` | (T, 5, C) | + context window |
| `results/MEG/zresp1_YY_brainomni_*.pt` | dict {x, pos, sensor_type} | BrainOmni input format |

### fMRI

| File | Shape | Notes |
|---|---|---|
| `E:/reconstruction/mydata/derivatives/preprocessed_data/sub-XX/MNI/sub-XX_task-RDR_run-YY_bold.nii.gz` | (T, X, Y, Z) | 4D BOLD volume |
| `mask.nii` | (X, Y, Z) bool | Brain mask |

### Stimulus (zstim)

| File | Shape | Notes |
|---|---|---|
| `results/zstim/subX_zstim_10_storyYY.npy` | (T, 4×768) | GPT-2 layer 10, 4 delays concatenated |

## 5. Semantic features

- **Model**: GPT-2 Chinese (~110M params, layer 10, embedding dim 768)
- **Context window**: 5 characters
- **Trim**: first/last 5 characters dropped
- **Delays**: STIM_DELAYS=[1,2,3,4], RESP_DELAYS=[-4,-3,-2,-1]
- **Layer weighting**: `[0.1, 0.7, 0.5, 0.3]` (configurable)

## 6. Sensitive attributes

- Brain data is **biometric** and must be treated as sensitive.
- Subjects are de-identified (sub-01 … sub-12).
- Data is **not public**; do not commit to GitHub, do not share externally.
- Any publication/figure must ensure no subject can be re-identified from results.

## 7. Known issues

- **English stories** are present in some subjects' runs (50+ stories flagged); need to exclude during training. See `english` dict in `prepipeline.py`.
- **Run dropouts** vary by subject; subject-level min/max is non-trivial.
- **Time alignment** files (`time_align/word-level/time_i.npy`) assume first_char_onset_time=12s and interval=0.4s; verify before each preprocessing run.

## 8. Versions

| Date | Version | Notes |
|---|---|---|
| 2024-09 | v1 | Initial ridge-regression baseline (师姐's work). |
| 2025-11 | v2 | MEG_model_A + fMRI3dCIB added; single-subject evaluation. |
| 2026-07 | v3 | BrainOmni preprocessing pipeline added. |
| TBD | v4 | HDF5 reformatted (planned). |

## 9. How to access

```bash
# On cluster (any compute node)
ls /home/test/reconstruction/
ls /home/test/reconstruction/mydata/derivatives/preprocessed_data/sub-01/

# On dev machine (owner only)
ls E:/reconstruction/
ls E:/results/MEG/zresp/

# NOT accessible: via GitHub (private data, not in repo)
```

## 10. See also

- [Project overview](01-project-overview.md) — why this data
- [Cluster card](03-cluster-card.md) — where it lives
- [Data pipeline architecture](../architecture/02-data-pipeline.md) — how it's loaded
- ADR-0002 — pydantic schema for data

---

Maintained by owner. Update when data structure, version, or known issues change.