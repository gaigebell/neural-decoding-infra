# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- Repository bootstrap (Week 00)
- Industry-standard documentation structure under `docs/`:
  - `docs/README.md` (Diátaxis-based documentation index)
  - `docs/architecture/01-overview.md` (system architecture)
  - `docs/research/` (project overview, data card, cluster card)
  - `docs/decisions/0001`–`0006` (six ADRs)
  - `docs/standards/01`–`07` (Python, docstring, git, testing, config, docs, **dependencies**)
  - `docs/guides/01`–`08` (setup, data adapter, model, training, multi-node, decoding, eval, debug)
  - `docs/dev-logs/2026-07-28-week-00-bootstrap.md` (first dev log)
- Dependency management infrastructure (方案 C)
- **P0 + P1 baseline implementation** (initial):
  - `recon/data/`
    - `schema.py` — Pydantic schemas: `StimSample`, `MEGSample`, `MEGChunkedSample`, `fMRISample`, `BrainOmniSample`, `BrainStimPair`
    - `fake_data.py` — synthetic data generators + CLI
    - `drdr.py` — DRDR dataset adapter
    - `datasets/meg.py` — `MEGDataset` (PyTorch wrapper)
    - `datasets/fmri.py` — `fMRIDataset`
  - `recon/models/`
    - `registry.py` — `MODEL_REGISTRY` + `register_model` + `build_model`
    - `fmri/fmri3dcib.py` — 3D CNN + Information Bottleneck
    - `meg/meg_model_a.py` — spatial-temporal attention + BiGRU
  - `recon/engine/`
    - `trainer.py` — unified training engine (DDP-aware, AMP, checkpointing, W&B)
    - `evaluator.py` — CRR, CER, Top-k, Perplexity + `Evaluator` aggregator
  - `recon/decoders/beam.py` — batched beam search with GPT-2 KV cache (target: 60× speedup)
  - `recon/cli/`
    - `train.py` — Hydra entry point for training
    - `decode.py` — single-story decoding CLI
    - `eval.py` — metrics CLI
  - `recon/utils/`
    - `optional.py` — `require_optional`, `is_installed`, `extra_installed`
    - `logging.py` — `get_logger` + `WandBLogger` wrapper
  - `configs/`
    - `train.yaml` — default Hydra entry
    - `model/{fmri3dcib,meg_model_a}.yaml`
    - `data/{drdr,drdr_fmri,fake}.yaml`
    - `paths/{cluster,local,ci}.yaml`
  - `tests/`
    - `unit/test_optional.py` — optional dep tests
    - `unit/test_schema.py` — pydantic schema tests
    - `unit/test_fake_data.py` — fake data + CLI tests
    - `unit/test_models.py` — model forward / loss / backward tests
    - `unit/test_evaluator.py` — metrics tests
    - `integration/test_smoke.py` — Tier 0 end-to-end smoke

### Fixed

- `BrainStimPair` discriminated union: replaced invalid
  `discriminator="__class__.__name__"` with per-class `modality: Literal[...]`
  fields + `discriminator="modality"` (pydantic v2 requirement).
- `MEGChunkedSample` inherited the parent's 2D `x` validator; the child now
  overrides `_check_x` for 3D (context, channels, time) input.
- `fake_fmri` mask meshgrid produced a transposed shape; now matches volume.
- `Trainer._train_epoch`: removed broken `next(islice(...))` leftover and the
  local `import itertools` (UnboundLocalError); `_InfiniteIterator` now
  re-creates its iterator from the original iterable on exhaustion.
- `Trainer` targets: `stim.zstim` may hold several concatenated delays
  (e.g. 4 × 768 = 3072); the trainer now slices the target to the model's
  output dim (`pred.shape[-1]`).
- `MEGModelA.forward` now mean-pools the per-step time window for 4D input
  (B, n_context, n_channels, n_time).
- `train.smoke=true` now actually runs 1 epoch regardless of `train.epochs`.
- Windows: DataLoader `num_workers > 0` deadlocks under `python -m` (spawn);
  the trainer now forces `num_workers=0` with a warning. Fake-data pairs use
  module-level `_PairDataset` so workers can pickle it.
- `paths.results_root` renamed to `paths.results_dir` everywhere (matches
  `${paths.results_dir}` interpolations in `train.yaml` / `TrainConfig`).
- Tests: `test_hydra_config_loads` now uses real Hydra composition; CER
  expectation corrected to reference-length normalization (1/3); fake-data
  CLI assertion fixed (channels-first axis).

### Added

- `recon/data/collate.py`: `collate_brain_stim_pairs` — the single place
  where validated samples become batched tensors (meg / meg_chunked / fmri /
  brainomni); wired into all four dataloader builders.
- `tests/unit/test_collate.py`: shape/dtype/mixed-modality tests for collate.
- `tests/integration/test_real_data.py`: Tier 1 integration tests against
  the real DRDR data (discovery, delay weighting, MEG alignment shift,
  collate + model forward, fMRI volume/mask). Auto-skipped when data is
  absent; resolve root via `RECON_PROCESSED_ROOT` / `E:/results` /
  `/home/test/reconstruction/results`.

### Changed

- `recon/data/drdr.py` rewritten against the real preprocessed layout
  (verified on the owner's dev machine, 2026-08-16):
  - `discover_drdr(processed_root, modality)` — meg scans
    `MEG/zresp/zresp{sub}_{story}.npy`, fmri scans `zresp/cube/...`;
    pairs are valid only when both response and stimulus files exist
    (fMRI: 47 of 52 cube stories have zstim).
  - `weight_delays` — ports the original `process_stim_data`
    (`[0.1, 0.7, 0.5, 0.3]` delay weighting) → (T, 768) float32 targets.
  - Story loaders use memmaps (`mmap_mode="r"`); fMRI cube stories are
    ~2 GB on disk, so only touched volumes are paged in.
- `recon/data/datasets/meg.py` / `fmri.py` rewritten: per-time-step
  iteration over all (subject, story, t); per-story memmap cache; MEG
  context variant emits `MEGChunkedSample` (n_context, C, 1) with the
  original alignment shift (response window from `t - (n_context - 1)`,
  zero-padded at the start); plain variant emits `MEGSample` (C, 1).
- `StimSample` canonical `zstim` is now the delay-weighted 768-dim target;
  raw 4×768 vector moved to optional `zstim_raw`. `fake_data` pairs and
  collate updated to match.
- Paths configs gained `processed_root` (local `E:/results`, cluster
  `/home/test/reconstruction/results`); `data=drdr*` requires it.
- `configs/model/fmri3dcib.yaml`: `input_shape: [91, 109, 91]` (real cube
  shape; backbone ends in AdaptiveMaxPool3d so any shape works).
- `configs/data/drdr.yaml` / `drdr_fmri.yaml`: `weights`, `stories`,
  `mask_path` options.

### Fixed

- Real-data smoke verified end-to-end on CPU: MEG (3 stories → 12 samples,
  1 epoch, loss 1.20) and fMRI (1 story → 2 steps, 1 epoch, loss 0.71).
- `discover_drdr` no longer lists stories whose zstim is missing.

### Planned

- Cluster Tier 1 smoke (real data, 1 node, 1 epoch)
- First 12-subject baseline run
- Multi-node DDP launch validation
- BrainOmni alignment head (in `recon/models/brainomni/`)
- First reproducible baseline (v0.1.0)

---

## [0.0.0] — 2026-07-28

### Added

- Initial repo creation
- README placeholder

---

## Versioning policy

- **Major (x.0.0)**: Breaking API change. Bump when adding/removing/changing public APIs.
- **Minor (0.x.0)**: New feature, backward-compatible. Bump when adding a new model variant, new dataset, or significant capability.
- **Patch (0.0.x)**: Bug fix or docs-only change.

Tags follow `vMAJOR.MINOR.PATCH`. We tag releases manually after a milestone
(reproducible baseline, paper submission, etc.).

---

## Release history

| Date | Version | Milestone |
|---|---|---|
| 2026-07-28 | 0.0.0 | Repo bootstrap |
| TBD | 0.1.0 | First reproducible baseline (target: end of August 2026) |