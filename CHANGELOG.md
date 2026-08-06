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
  - `docs/decisions/0001`–`0006` (six ADRs covering Hydra, schemas, SLURM, ssh launch, no-local-GPU, unified Trainer)
  - `docs/standards/01`–`06` (Python, docstring, git, testing, config, docs conventions)
  - `docs/guides/01`–`08` (setup, data adapter, model, training, multi-node, decoding, eval, debug)
  - `docs/dev-logs/2026-07-28-week-00-bootstrap.md` (first dev log)
- Dependency management infrastructure (方案 C):
  - `pyproject.toml` extras (`brainomni`, `large-lm`, `data-public`, `all`, `dev-all`)
  - `recon/utils/optional.py` — `is_installed`, `extra_installed`, `require_optional`
  - `recon/encoders/brainomni.py` — lazy-import example with friendly error messages
  - `recon/models/registry.py` — `MODEL_REGISTRY` + `register_model` + `build_model`
  - `tests/unit/test_optional.py` — unit tests (no extras required)
  - `docs/standards/07-dependencies.md` — full conventions
  - `Makefile` extended: `install-all-extras`, `install-brainomni`, `install-cluster`
  - `.github/workflows/lint.yml` — extras matrix + self-hosted runner for cluster
  - `environment.yml` — migrated to single-env + pip `[all]` install

### Planned

- Week 01: code skeleton (`recon/` package) + first fMRI model + first cluster smoke
- Week 02: MEG model + multi-node DDP launch script
- Week 03: decoder speedup (batched + KV cache)
- Week 04: evaluator + first 12-subject baseline run

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