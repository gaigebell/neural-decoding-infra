# Standard 03: Git Workflow

> **Audience**: Anyone making commits or PRs.
> **Status**: Active.

---

## 1. Branch strategy

We use a simplified **trunk-based development**:

- `main` — always deployable; protected branch
- `feat/<name>` — feature branches, short-lived (≤ 1 week)
- `fix/<name>` — bug fix branches
- `docs/<name>` — documentation-only changes
- `exp/<name>` — experimental / throwaway branches

No long-lived develop/release branches. All features merge to `main` via PR.

## 2. Commit messages

We follow **Conventional Commits**:

```
<type>(<scope>): <short summary>

<body>

<footer>
```

### Types

| Type | When | Example |
|---|---|---|
| `feat` | New user-facing feature | `feat(trainer): add cross-subject joint training` |
| `fix` | Bug fix | `fix(decoder): correct KV cache offset for batch>1` |
| `docs` | Documentation only | `docs(adr): add 0007-quantization-decision` |
| `refactor` | Code change that fixes neither bug nor adds feature | `refactor(data): extract MEG loader to a separate function` |
| `perf` | Performance improvement | `perf(decoder): batch nucleus sampling 10x faster` |
| `test` | Add or fix tests | `test(data): add pydantic validation unit tests` |
| `chore` | Build / tooling / non-code | `chore(deps): pin torch==2.4.0` |
| `revert` | Revert a previous commit | `revert: feat(trainer): drop KV cache change` |

### Scope

Use the module name (or sub-module) most affected:

- `data`, `model`, `trainer`, `decoder`, `eval`, `cli`, `config`, `docs`, `cluster`

### Summary

- ≤ 72 characters
- Imperative mood ("add", not "added")
- No trailing period
- Lowercase first letter (after type)

### Body

- Wrap at 72 chars
- Explain **what** and **why**, not how (the diff shows how)
- Reference issue / ADR if relevant: `Refs ADR-0006.`

### Footer

- `Refs: ADR-NNNN` — links to an architecture decision
- `Refs: #N` — links to a GitHub issue (if used)
- `BREAKING CHANGE: <description>` — for breaking API changes

### Example

```
feat(trainer): add DDP gradient synchronization

Previously, training with >1 GPU did not synchronize gradients,
causing NaN losses at scale. Now uses torch.distributed for
gradient all-reduce via DDP wrapper.

Refs: ADR-0006
```

## 3. Pull request process

1. **Branch** from `main`: `git checkout -b feat/my-feature main`
2. **Commit** in logical chunks (not one giant commit)
3. **Push** and **open PR** against `main`
4. **Fill PR template** (see `.github/PULL_REQUEST_TEMPLATE.md`):
   - Summary of changes
   - How to test
   - ADR / doc references
5. **Self-review** the diff before requesting review
6. **Pass CI** (lint + unit tests + cluster smoke)
7. **Merge** with **squash commit** (clean history on `main`)

### PR size guideline

- **<200 lines**: ideal
- **200–500**: acceptable if well-scoped
- **>500**: probably should be split

## 4. Don't commit

These must NEVER be committed (enforced by `.gitignore`):

- Data files (`*.npy`, `*.pt`, `*.npz`, `*.fif`, `*.nii*`)
- Checkpoints (`ckpt/`, `*.pth`)
- Logs (`logs/`, `*.log`, `wandb/`)
- Environment files (`.env`, except `.env.example`)
- Compiled artifacts (`__pycache__/`, `*.pyc`, `build/`, `dist/`)
- Editor configs (`.vscode/`, `.idea/`)

For large files that must be tracked, use **git-lfs** (`.gitattributes`).

## 5. Tags and releases

- Use **semantic versioning**: `vMAJOR.MINOR.PATCH`
- Tag a release when:
  - A milestone is reached (baseline reproducible, paper submitted, etc.)
  - API breaks require downstream updates
- Tag format: `git tag -a v0.1.0 -m "First reproducible baseline"`

## 6. Code review checklist

When reviewing a PR, check:

- [ ] Commit messages follow Conventional Commits
- [ ] No forbidden files in diff (use `git diff --stat main`)
- [ ] New code has type hints
- [ ] New code has docstrings (public APIs)
- [ ] Tests added for new logic
- [ ] No hardcoded paths
- [ ] No new top-level dependencies (check `pyproject.toml`)
- [ ] Linked ADR if design changed
- [ ] Update `CHANGELOG.md` if user-visible

## 7. Useful commands

```bash
# See what's actually changed (vs what's staged)
git diff --stat

# Check no secrets in diff
git diff | grep -iE "password|api[_-]?key|secret"

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Interactive rebase to clean history before PR
git rebase -i main

# Find when a line was introduced
git blame -L 10,20 path/to/file.py
```

## 8. See also

- [Conventional Commits spec](https://www.conventionalcommits.org/)
- [`.gitignore`](../../.gitignore)
- [`.pre-commit-config.yaml`](../../.pre-commit-config.yaml)
- [Standard 06: Documentation](06-documentation.md)

---

Maintained by owner. Update when workflow rules change.