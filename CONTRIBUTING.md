# Contributing

> **Status**: Single-owner project, but the conventions here apply to anyone who joins later.

---

## TL;DR

1. Read [`docs/standards/`](docs/standards/) before writing code
2. Read [`docs/decisions/`](docs/decisions/) before questioning a design choice
3. Use [Conventional Commits](docs/standards/03-git-workflow.md)
4. Run `make lint` and `make test` before pushing
5. Reference ADRs in commit messages and PRs

---

## Code

All code lives under `recon/`. Read:

- [`docs/standards/01-python-style.md`](docs/standards/01-python-style.md) — PEP 8 + project rules
- [`docs/standards/02-docstring.md`](docs/standards/02-docstring.md) — Google-style docstrings
- [`docs/standards/04-testing.md`](docs/standards/04-testing.md) — pytest conventions

Key rules:

- Type hints on all public functions
- No hardcoded paths (use configs)
- No `print()` — use `recon.utils.logging`
- All paths go through Hydra config

## Adding new functionality

| You want to... | Read |
|---|---|
| Add a new dataset | [`docs/guides/02-write-data-adapter.md`](docs/guides/02-write-data-adapter.md) |
| Add a new model | [`docs/guides/03-write-model.md`](docs/guides/03-write-model.md) |
| Add a new training config | [`docs/standards/05-configuration.md`](docs/standards/05-configuration.md) |
| Add a new decoder | [`docs/architecture/05-decoding-engine.md`](docs/architecture/05-decoding-engine.md) |

## Documentation

- Use the templates in each `docs/<section>/README.md`
- Update the docs index (`docs/README.md`) when adding a file
- Link liberally
- One concept per file

See [`docs/standards/06-documentation.md`](docs/standards/06-documentation.md).

## Commits

Follow [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<scope>): <short summary>

<body>
```

Examples:

- `feat(trainer): add DDP gradient synchronization`
- `fix(decoder): correct KV cache offset`
- `docs(adr): add 0007-quantization-decision`

See [`docs/standards/03-git-workflow.md`](docs/standards/03-git-workflow.md).

## Pull requests

- Branch from `main`
- Self-review the diff before requesting review
- Fill the PR template (or use the default GitHub one)
- Reference relevant ADRs
- Reference relevant docs

## Testing

- Unit tests for any new logic
- Tests must run on CPU (no GPU required for CI)
- See [`docs/standards/04-testing.md`](docs/standards/04-testing.md)

```bash
make test           # all unit tests
make test-coverage  # with coverage
make lint           # ruff + mypy
```

## Architecture decisions

If you're proposing a significant change:

1. Search [`docs/decisions/`](docs/decisions/) for relevant existing ADRs
2. If none exists, write a new ADR (use [the template](docs/decisions/README.md))
3. Discuss with the owner before implementing

Don't bundle "implementation + decision" into a single PR. Separate them.

## Don't commit

- ❌ Data files (`*.npy`, `*.pt`, `*.npz`, `*.fif`, `*.nii*`)
- ❌ Checkpoints (`*.pth`, `ckpt/`)
- ❌ Logs (`logs/`, `*.log`)
- ❌ Secrets (`.env`, API keys)
- ❌ `__pycache__/`, `*.pyc`

See [`.gitignore`](.gitignore).

## Communication

- Issues: GitHub issue tracker
- Questions: GitHub Discussions (when enabled)
- Reviews: PR comments

## License

By contributing, you agree that your contributions will be licensed under the
project's [LICENSE](LICENSE).