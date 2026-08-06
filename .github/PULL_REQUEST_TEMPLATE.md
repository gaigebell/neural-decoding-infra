# Pull Request

## What

<!-- One-line summary of the change -->

## Why

<!-- What problem does this solve? Link to issue / ADR. -->

Refs: <!-- ADR-NNNN or #issue-number, if any -->

## Type of change

<!-- Check all that apply -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation only
- [ ] Refactor (no functional change)

## Testing

- [ ] Local unit tests pass (`make test-unit`)
- [ ] Lint passes (`make lint`)
- [ ] Cluster Tier 0 smoke (fake data, 1 node) — owner runs after merge
- [ ] Cluster Tier 1 smoke (real data, 1 node) — owner runs if needed
- [ ] W&B run link: <!-- https://wandb.ai/... -->

## Docs

<!-- Check all docs that need updating -->
- [ ] No doc changes needed
- [ ] Architecture doc updated (`docs/architecture/`)
- [ ] ADR added (`docs/decisions/NNNN-...`)
- [ ] Standard added/updated (`docs/standards/`)
- [ ] Guide added/updated (`docs/guides/`)
- [ ] CHANGELOG.md updated

## Self-review checklist

- [ ] Commit messages follow Conventional Commits
- [ ] No hardcoded paths
- [ ] No new top-level dependencies (or justified in PR description)
- [ ] Type hints on new public functions
- [ ] Google-style docstrings on new public functions/classes
- [ ] Tests added for new logic
- [ ] `git diff --stat main` looks reasonable (<500 lines)

## Screenshots / logs

<!-- If applicable, paste W&B screenshots or relevant log snippets -->