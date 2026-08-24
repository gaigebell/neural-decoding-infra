# 02 — Data Pipeline

> **Status**: Living document — updated as the system evolves
> **Audience**: Anyone who needs to understand how data flows from disk to GPU
> **Last updated**: 2026-08-20

---

## 1. Purpose

This document describes the **data pipeline**: how raw recordings become
training batches. It answers "where does data live, in what form, at each
stage" and "what does one sample's journey look like".

For the *design method* behind these choices (how to think about scale,
IO efficiency, and tradeoffs), see the companion tutorial
[`docs/tutorials/01-data-pipeline-design.md`](../tutorials/01-data-pipeline-design.md).

The pipeline splits into two halves:

- **Offline preprocessing** (one-time, legacy code): raw BIDS → aligned
  per-story `.npy` arrays.
- **Runtime pipeline** (this repo): `.npy` arrays → validated samples →
  batched tensors → GPU.

---

## 2. Data forms at every stage

| Stage | Form | Location | Size (subject 1) |
|---|---|---|---|
| ① Raw (BIDS) | `.fif` (MEG), `.nii.gz` (fMRI), `.hdf5` (GPT embeddings), `.tsv` | `E:/reconstruction/mydata/` | ~100 GB-class |
| ② Preprocessed | per-(story, timestep)-aligned `.npy` | `E:/results/` | MEG ~1.8 GB; fMRI cube **~97 GB** |
| ③ Runtime sample | memmap view → `MEGSample` / `fMRISample` | RAM (page cache + small resident) | 6 KB ~ 3.4 MB each |
| ④ Batch tensors | `BrainStimBatch` (torch.Tensor) | RAM → GPU | MEG batch 64 ≈ 400 KB; fMRI batch 4 ≈ 14 MB |
| ⑤ Products | checkpoints / decoded txt / eval json | `results_dir/` | small |

The ②→③ boundary is what the legacy pipeline (`prepipeline.py`,
`process_brain_feature.py`, `datasetbuilderMEG.py`) produced: it turned
"unaligned, format-heterogeneous recordings" into **matrices where every
row is one timestep, brain signal and stimulus aligned row-by-row**.
Nothing downstream ever touches BIDS again.

---

## 3. Preprocessed shapes (the ground truth for everything downstream)

Measured on real data:

```
MEG/zresp/zresp1_1.npy              (1011, 306)       float32   # one timestep per row, 306 channels reduced
MEG/zresp/zresp1_1_context_5.npy    (1011, 5, 306)    float32   # each row holds a 5-step context window
MEG/zstim/sub1_zstim_10_story1.npy  (1011, 3072)      float64   # GPT-2 layer 10, 4 delays concatenated
zresp/cube/zresp1_1.npy             (570, 91,109,91)  float32   # fMRI volumes inside the mask cube
zstim/sub1_zstim_10_story1.npy      (570, 3072)       float64   # same stimulus features at fMRI sampling rate
mask.nii                            (91, 109, 91)     int16     # brain mask
```

Two schema decisions follow from these shapes:

- MEG has **no time window** (one scalar per channel per step), so the
  canonical schema uses `(C, 1)` / `(ctx, C, 1)` — trailing dim is the
  per-step time window, always 1 here, shared with future windowed data.
- `3072 = 4 × 768` is **4 concatenated delay vectors**. The legacy
  pipeline weighted them with `[0.1, 0.7, 0.5, 0.3]` into a 768-dim
  target — that is [`weight_delays`](../../recon/data/drdr.py) in this repo.

---

## 4. The runtime pipeline, in four stages

### 4.1 Discover — [`drdr.py:discover_drdr`](../../recon/data/drdr.py)

Scans the filesystem and answers "which (subject, story) pairs exist",
producing a pure in-memory `DRDRIndex` (tens of KB).

Two decisions worth noting:

1. **A pair counts only if BOTH response and stimulus files exist.**
   Measured: fMRI stories 56–60 have cubes but no zstim (47/52 pairs).
   The legacy code discovered late ("file missing → Warning → None");
   this repo moves the defect to discovery so everything downstream can
   assume a clean index.
2. Only plain filenames (`zresp{sub}_{story}.npy`) are scanned, so the
   `_context_5` / `_chunked` variants don't inflate the story count.

### 4.2 Dataset — [`datasets/meg.py`](../../recon/data/datasets/meg.py), [`datasets/fmri.py`](../../recon/data/datasets/fmri.py)

`torch.utils.data.Dataset` subclasses with a shared structure:

```python
# construction: build the flat sample table + empty caches (reads nothing)
self._items: list[(sub, story, t)]            # every (story, timestep) expanded
self._story_resp: dict[(sub, story), memmap]  # filled on first touch
self._story_stim: dict[(sub, story), ndarray]

def __getitem__(self, idx):
    sub, story, t = self._items[idx]
    resp = self._resp(sub, story)   # np.load(mmap_mode="r") on first touch
    ...
```

**Life of one sample** (MEG, n_context=5, index lands on story 1 step 7):

1. Table lookup `(1, 1, 7)`; first touch of story 1 opens a memmap
   (**no data copied** — only a page mapping is created).
2. Alignment shift: `j = 7 - (5 - 1) = 3` → row 3 of the memmap
   `(5, 306)` → `row[..., None]` → `(5, 306, 1)`. This is the exact
   equivalent of the legacy prepend-4-zeros zip trick; test
   `test_alignment_shift_zero_padding` locks the semantics (first
   n_context-1 samples all-zero, then real data).
3. Target: the stimulus was delay-weighted **once per story** at load
   time and cached as `(T, 768)`; here it's a plain row slice.
4. Wrap in a pydantic sample (`MEGChunkedSample`) — shape/dtype
   validation happens **here**, at the boundary.

The fMRI variant differs only in the slice size (a 3.4 MB 3D volume)
and a mask loaded once and shared by all samples.

### 4.3 Collate — [`collate.py`](../../recon/data/collate.py)

`collate_brain_stim_pairs`: `np.stack` → `torch.from_numpy`, producing a
`BrainStimBatch` (`brain.x` / `brain.pos` / `stim.zstim` …). This is the
**single place** where samples become batches; every consumer (trainer,
future evaluator/decoder paths) gets one canonical batch shape. Empty
batches and mixed modalities are rejected (both unit-tested).

### 4.4 DataLoader assembly — [`trainer.py`](../../recon/engine/trainer.py)

`build_dataloaders` wires it together: `shuffle=True` (or
`DistributedSampler` under DDP), `num_workers`, `pin_memory`,
`collate_fn`. Three data sources (fake / drdr / drdr_fmri) share one
`Trainer` code path — the smoke test exercises exactly this path with
fake data.

---

## 5. Read-efficiency analysis

**Per-`__getitem__` disk cost:**

| Sample | Bytes read | Nature |
|---|---|---|
| MEG context | 5×306×4 B ≈ **6 KB** | contiguous, within one row |
| fMRI volume | 91×109×91×4 B ≈ **3.4 MB** | contiguous (~830 × 4 KB pages) |

**What memmap buys**: `np.load(mmap_mode="r")` defers loading to the OS
page cache, 4 KB at a time. Dataset construction reads **nothing**; only
touched rows fault in. RAM = resident weighted-stim caches + recently
touched pages, not the dataset size.

**Numbers** (subject 1, 3-story MEG run):

- On disk: resp 3×5.9 MB + stim 3×24.8 MB ≈ 92 MB
- Resident: weighted stim cache 3×3.1 MB ≈ 9 MB; the rest is page cache,
  evicted by the OS as needed
- Legacy comparison: `datasetbuilderMEG` loaded the whole dataset into a
  `List[(tensor, tensor)]` — impossible at 12-subject scale

**Two honest costs**:

1. **Global shuffle vs memmap**: `shuffle=True` randomizes the whole
   (sub, story, t) table, so consecutive batches jump across stories and
   page-cache hit rate drops; on NFS every miss is a network RPC. Worst
   on the first epoch. Mitigations (P2): shuffle within story blocks, or
   pre-warm. Invisible at MEG scale; must be solved before large fMRI runs.
2. **mmap on NFS**: local SATA page fault ~0.1 ms; NFS fault is a round
   trip ~0.2–1 ms. `num_workers=2` hides this by prefetching batches
   while the GPU computes. (Windows spawn deadlocks with workers — the
   trainer degrades to 0 with a warning; Linux forks fine.)

**The 16× shrink**: 3072 float64 → 768 float32 (4× fewer elements × 8→4
bytes). `weight_delays` runs once per story; everything after flows at
768-dim float32.

---

## 6. Space analysis (three ledgers)

| Level | Content | Scale |
|---|---|---|
| Disk (`E:/results`) | fMRI cubes dominate: 47 × 2.06 GB ≈ **97 GB** (linear variant doubles it); MEG 12 subjects ≈ 22 GB | ~200 GB-class |
| Training RAM | data side ≈ resident stim caches + page cache, **10–50 MB**; model params a few M | far below any eager scheme |
| GPU memory | batch (MEG 64 ≈ 400 KB; fMRI 4 ≈ 14 MB) + activations (fMRI conv1 ≈ 74 MB at 32 ch) + model | comfortable on 8 GB |

Why keep the expensive cube: 3D CNNs need spatial structure. The linear
(902,629-dim) and ROI (37,853-dim) variants coexist on disk for
fully-connected/ROI models; the adapter currently reads only the cube.

---

## 7. Why this design (each claim with evidence)

1. **Schema as single source of truth (ADR-0002)** — pydantic validates
   shape/dtype at the boundary. Evidence: during real-data integration it
   caught a real bug (context data wrapped in the wrong sample class) at
   construction, three lines from the cause; without validation this is a
   silent shape mismatch that only explodes mid-training.
2. **Adapter isolation** — only `drdr.py` knows file names/locations;
   models, trainer, decoder see schemas. Evidence: fake / drdr /
   drdr_fmri share one Trainer and one test suite; a new dataset = a new
   adapter, zero training-code changes.
3. **Memmap lazy loading = scalability** — dataset size has no cap
   (97 GB fMRI runs the same as 100 MB MEG); RAM grows with stories
   *touched*, not dataset size. Evidence: real fMRI training ran with
   < 50 MB data-side RAM; the legacy eager `List` scheme cannot run at
   12-subject scale.
4. **Expensive work done once** — `weight_delays` per story, cached;
   per-sample cost is a row slice (microseconds of Python).
5. **Alignment logic in exactly one place** — the `t - (n_context - 1)`
   semantics live in `MEGDataset.__getitem__` + one locking test. The
   legacy code hid it inside a prepend-then-zip trick readers had to
   reverse-engineer.
6. **Single collate entry point** — batch shapes are defined once,
   reused by all four dataloader builders; error handling (empty /
   mixed-modality) has dedicated tests.
7. **Defects moved earlier** — missing zstim is filtered at discovery,
   not surfaced as a load-time `None`.
8. **Platform differences made explicit** — the Windows spawn guard
   warns loudly instead of behaving silently; it fired during local
   development, which is how it was found and fixed.

**Known limitations** (also in
[`guides/09-manual-review-checklist.md`](../guides/09-manual-review-checklist.md)):
global shuffle vs page cache (to solve before large fMRI runs); NFS
first-read coldness (pre-warm in P2).

---

## 8. Related documents

- Tutorial (how to *think* about data pipeline design):
  [`docs/tutorials/01-data-pipeline-design.md`](../tutorials/01-data-pipeline-design.md)
- Adapter how-to: [`guides/02-write-data-adapter.md`](../guides/02-write-data-adapter.md)
- Data schema rationale: [`decisions/0002-pydantic-schema.md`](../decisions/0002-pydantic-schema.md)
- Data card: [`research/02-data-card.md`](../research/02-data-card.md)
