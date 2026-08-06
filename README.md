# neural-decoding-infra

> AI infrastructure for training and inference of non-invasive brain-to-text decoding.

[![status](https://img.shields.io/badge/status-WIP-yellow)]()
[![python](https://img.shields.io/badge/python-3.10+-blue)]()
[![license](https://img.shields.io/badge/license-Proprietary-red)]()

---

## What is this?

This repository hosts the **industrial-grade infrastructure** for training and evaluating deep-learning models that decode Chinese characters from non-invasive brain signals (fMRI and MEG/EEG).

It is the successor to the original `recon/` research codebase. The goal of this rebuild is to turn one-shot research scripts into a reproducible, extensible, team-friendly framework that can scale from a single GPU to multi-node DDP training.

## Quickstart

```bash
# 1. Clone (private repo — requires GitHub access)
git clone git@github.com:<owner>/neural-decoding-infra.git
cd neural-decoding-infra

# 2. Install (editable, for development)
pip install -e ".[dev]"

# 3. Run smoke test (requires cluster GPU access; see docs/guides/01-setup-env.md)
python -m recon.cli.train --config-path=configs paths=cluster train.smoke=true

# 4. Run lint + unit tests
make test
make lint
```

## Project structure

```
neural-decoding-infra/
├── recon/                  # Main code package
├── tests/                  # Tests (unit + integration)
├── configs/                # Hydra configuration files
├── scripts/                # Operational scripts (launch, smoke, sync)
├── docs/                   # All documentation
│   ├── architecture/       # Design docs ("why this way?")
│   ├── decisions/          # ADRs ("what did we decide and why?")
│   ├── standards/          # Coding standards ("how should I write?")
│   ├── guides/             # How-to guides ("how do I do X?")
│   ├── dev-logs/           # Weekly development logs ("what did we do?")
│   └── research/           # Research context ("what are we studying?")
├── pyproject.toml
├── environment.yml
├── Makefile
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE
```

For a deep dive, start with **[docs/architecture/01-overview.md](docs/architecture/01-overview.md)**.

## Documentation map

| Document | Purpose | Audience |
|---|---|---|
| `docs/architecture/` | **Why** the system is designed this way | Everyone |
| `docs/decisions/` | **What** key decisions were made and trade-offs | Everyone |
| `docs/standards/` | **How** to write code/docs (conventions) | Contributors |
| `docs/guides/` | **How-to** recipes for common tasks | Operators |
| `docs/dev-logs/` | **What** happened week by week | Future-self |
| `docs/research/` | **What** we are researching (scientific context) | Researchers |

## Status

🚧 **Work in progress** — currently in bootstrap phase (Week 0, July 2026).

See [`docs/dev-logs/`](docs/dev-logs/) for the latest progress and [`CHANGELOG.md`](CHANGELOG.md) for version history.

## Cluster

This project trains on a 4-node GPU cluster (PH402, 8 GPUs total). For cluster topology and access instructions, see [`docs/research/03-cluster-card.md`](docs/research/03-cluster-card.md).

## Contributing

This is currently a single-owner project. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the conventions any future contributor must follow.

## License

Proprietary. See [`LICENSE`](LICENSE). Do not redistribute without permission.

## Contact

Project owner: see `git log` for current maintainers.