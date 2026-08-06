# ADR-0001: Why Hydra over plain YAML

- **Status**: Accepted
- **Date**: 2026-07-28
- **Deciders**: owner

## Context and problem statement

We need a way to configure a complex training pipeline that has many interdependent settings:

- Model hyperparameters (architecture, dims)
- Data loader settings (paths, batch size, splits)
- Training settings (optimizer, scheduler, epochs, DDP config)
- Decoder settings (beam width, nucleus mass)
- Path overrides for different environments (cluster / local / CI)

Plain YAML is simple but loses composition. We considered:

1. **Plain YAML + argparse** — manual loading, no composition, no sweeps
2. **Hydra + OmegaConf** — compositional, swappable, supports sweeps
3. **OmegaConf only (no Hydra)** — composition but no CLI integration
4. **JSON schemas (e.g., dacite)** — typed but heavy

## Decision drivers

- Configs will be edited often (every experiment)
- Sweeps over hyperparameters are a primary use case
- Multiple configs must compose (model + data + decoder)
- Path overrides between cluster / local must be easy
- Single-owner project — minimize custom tooling

## Considered options

### Option A: Plain YAML + argparse

- **Pro**: Zero dependencies, dead simple
- **Con**: No composition; can't `defaults:`; sweep = shell loop

### Option B: Hydra + OmegaConf

- **Pro**: Compositional configs, sweep support, rich CLI, well-documented
- **Pro**: Industry standard (used at Meta, FAIR, many others)
- **Con**: Adds a dependency; some learning curve; "magic" feel

### Option C: OmegaConf only

- **Pro**: Typed, composable, no CLI layer
- **Con**: Have to build CLI manually; no sweep integration

### Option D: dacite + dataclass

- **Pro**: Type-safe config
- **Con**: Heavyweight, doesn't solve composition

## Decision outcome

**Chosen option**: **Option B (Hydra + OmegaConf)**.

Hydra's `defaults:` list lets us define base configs and override per experiment. Its sweep tooling (`hydra-multirun`) is essential for hyperparameter search. The dependency is mature and battle-tested.

### Consequences

- ✅ Good: Easy composition; clean CLI; sweeps trivial; swappable defaults
- ✅ Good: Industry standard — any new contributor is likely familiar
- ❌ Bad: Adds ~3 deps (`hydra-core`, `omegaconf`, `hydra-optuna-sweeper` optional)
- ❌ Bad: Some "magic" — `@hydra.main` decorator hides entrypoint
- ❓ Risk: Hydra's syntax evolves; v1.x → v2.x had breaking changes. Pin major version.

## Pros and cons of the options

### Hydra + OmegaConf (chosen)
- ✅ Pro: Compositional, sweep support, path overrides trivial
- ✅ Pro: `@hydra.main` is concise
- ❌ Con: One more dependency
- ❌ Con: `defaults:` list ordering is subtle

### Plain YAML + argparse
- ✅ Pro: No new dependencies
- ❌ Con: Sweeps are painful
- ❌ Con: Composition is manual

## Links

- [Standards: Configuration](../standards/05-configuration.md)
- [Architecture: Configuration flow](01-overview.md#5-configuration-flow)
- Hydra docs: https://hydra.cc/