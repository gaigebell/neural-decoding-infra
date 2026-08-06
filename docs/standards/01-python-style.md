# Standard 01: Python Style

> **Audience**: Anyone writing Python in this repo.
> **Status**: Active — enforced via `ruff` (see `.pre-commit-config.yaml`).

---

## 1. Baseline

We follow **PEP 8** with these specifics:

| Tool | Purpose | Config file |
|---|---|---|
| `ruff format` | Formatting (Black-compatible) | `pyproject.toml` |
| `ruff check` | Linting (pyflakes + isort + more) | `pyproject.toml` |
| `mypy` | Static type checking | `pyproject.toml` |

All three are run by `make lint` and pre-commit hooks.

## 2. Line length

- **Max line length**: 100 characters (Black default is 88; we extend slightly because of long config strings)

## 3. Imports

Use `ruff` (isort rules) — never manually sort imports.

```python
# ✅ Good
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from recon.configs import load_config
from recon.data import MEGSample
```

Rules:

- Standard library → third-party → local
- Alphabetical within each group
- One blank line between groups

## 4. Naming

| What | Convention | Example |
|---|---|---|
| Modules / packages | `lowercase` | `recon`, `recon.data`, `recon.models.fmri` |
| Classes | `PascalCase` | `Trainer`, `MEGSample`, `InformationBottleneckAligner` |
| Functions | `snake_case` | `train_epoch`, `load_zresp`, `compute_loss` |
| Variables | `snake_case` | `train_loader`, `subject_id`, `zresp_shape` |
| Constants | `UPPER_SNAKE` | `MAX_EPOCHS`, `DEFAULT_BATCH_SIZE` |
| Private | `_leading_underscore` | `_validate_sample`, `_init_weights` |
| Type variables | `PascalCase` | `SampleT = TypeVar("SampleT")` |
| Acronyms in names | Capitalize first only | `MegSample`, `FmriModel` (not `MEGSample`) |

## 5. Type hints

**Required** for all public functions and class methods. Private helpers may omit if obvious.

```python
def load_sample(story_id: int, subject_id: int) -> MEGSample:
    ...

def compute_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, float]]:
    ...
```

For numpy arrays, use `numpy.typing.NDArray`:

```python
from numpy.typing import NDArray

def zscore(data: NDArray[np.float32]) -> NDArray[np.float32]:
    ...
```

## 6. Strings

- **Quote style**: double quotes for all strings (Black default)
- **f-strings** for interpolation, never `%` or `.format()`
- **No string concatenation** with `+` for > 2 strings

```python
# ✅ Good
f"Subject {subject_id}, story {story_id}: shape {arr.shape}"

# ❌ Bad
"Subject %d, story %d: shape %s" % (subject_id, story_id, arr.shape)
"Subject " + str(subject_id) + ", story " + str(story_id)
```

## 7. Comments and docstrings

- See [Standard 02: Docstring style](02-docstring.md)
- Inline comments: use sparingly, prefer self-documenting code
- Block comments for non-obvious logic only

## 8. Error handling

- **Be specific**: catch specific exceptions, not bare `except`
- **Fail loud**: prefer raising over silently swallowing
- **Context**: re-raise with `from` to preserve traceback

```python
# ✅ Good
try:
    raw = np.load(path)
except FileNotFoundError as e:
    raise DataLoadError(f"ZRESP not found: {path}") from e

# ❌ Bad
try:
    raw = np.load(path)
except:
    raw = None
```

## 9. Project-specific rules

These override or extend PEP 8:

1. **No global mutable state**. Use config / DI.
2. **No `print()`** for important output — use `recon.utils.logging.get_logger()`.
3. **All paths go through Hydra config**. Never hardcode `/home/test/...` in code.
4. **All randomness must be seedable**. `torch.manual_seed`, `np.random.seed`, `random.seed`.
5. **Tensor device**: never assume. Always accept `device` arg or get from trainer.
6. **DDP safety**: never do `dist.all_reduce` without checking `dist.is_initialized()`.
7. **Heavy imports must be lazy**. See [Standard 07: Dependencies](07-dependencies.md).

## 10. What we DON'T do

- ❌ `# noqa` without justification
- ❌ `from X import *`
- ❌ Mutable default arguments
- ❌ `assert` for runtime checks (use explicit `if/raise`)
- ❌ `os.path.join` for new code — use `pathlib.Path`
- ❌ `type: ignore` without comment explaining why
- ❌ Top-level imports of optional heavy packages (BrainOmni, vllm, etc.)

## 11. Pre-commit

Install once per machine:

```bash
pip install pre-commit
pre-commit install
```

After this, every `git commit` will run `ruff format` and `ruff check --fix` automatically.

## 12. See also

- [Standard 02: Docstring style](02-docstring.md)
- [Standard 04: Testing](04-testing.md)
- [Standard 07: Dependencies](07-dependencies.md)
- [`.pre-commit-config.yaml`](../../.pre-commit-config.yaml)
- [`pyproject.toml`](../../pyproject.toml)

---

Maintained by owner. Update when style conventions change.