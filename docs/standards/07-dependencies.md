# Standard 07: Dependency Management

> **Audience**: Anyone adding a dependency or working on a heavy integration.
> **Status**: Active.

---

## 1. Core principle

**Single conda env + `pyproject.toml` extras + lazy imports.**

This is the standard for all major AI/ML Python projects (HuggingFace Transformers, PyTorch Lightning, Detectron2, AllenNLP). It solves the "too many packages, can't predict what's needed" problem by:

1. **Single env**: No `conda activate X` switching, no cross-env import errors.
2. **Optional extras**: Heavy integrations are opt-in via `pip install -e ".[extra]"`.
3. **Lazy imports**: The package is always importable; heavy modules load only when used.
4. **Cluster `[all]`**: The cluster always has every extra — no env wrangling.

See ADR-0005 for the related "no local GPU testing" decision (why the cluster is the only verification environment).

## 2. Layout

```
pyproject.toml              # Declares core deps + extras
environment.yml             # Conda-side: lockfile for cluster env (uses [all])
.env.example                # Documented env vars (WANDB_API_KEY, etc.)

recon/
├── __init__.py             # Always importable, no heavy deps
├── utils/
│   └── optional.py         # require_optional() / is_installed()
├── encoders/
│   └── brainomni.py        # Lazy import example
└── models/
    └── registry.py         # MODEL_REGISTRY
```

## 3. When to add a new `extra`

Open a new extra in `pyproject.toml` when **any** of the following hold:

| Condition | Threshold |
|---|---|
| Package size | > 100 MB |
| Install time | > 5 minutes |
| Strict version pin | Yes (specific torch / CUDA) |
| Used by | One or few modules |
| Conflict risk | High with other packages |

**Examples that warrant a new extra**:
- `brainomni` — pretraining brain encoder, large checkpoints, restricted access
- `large-lm` — vLLM + FlashAttention, 5+ GB, GPU-specific
- `data-public` — HuggingFace `datasets`, large transitive deps

**Examples that go in core** (no extra):
- `torch` (everyone needs it)
- `numpy`, `pydantic`, `transformers` (foundational)
- `wandb` (always used for tracking)

## 4. Naming conventions

- **Extra name**: snake_case (e.g., `brainomni`, `large-lm`, `data-public`)
- **Package name**: matches the primary Python package for availability check
- **Both should be added** to `_EXTRA_TO_PACKAGE` in `recon/utils/optional.py`

## 5. The lazy-import pattern

For every heavy integration:

```python
# recon/encoders/brainomni.py
from ..utils.optional import require_optional

def build_brainomni_encoder(cfg):
    # 1. Friendly check FIRST
    require_optional("brainomni", hint="pip install -e '.[brainomni]'")

    # 2. Lazy imports — only happen when this function is called
    from brainomni.model import BrainOmni  # noqa: E402
    from braintokenizer.model import BrainTokenizer  # noqa: E402

    # ... actual implementation
```

**Why this order matters**:
- The `require_optional` check gives a clear error message
- The lazy import only triggers after the check passes
- Module-level imports are NOT used (so `import recon` never fails)

## 6. Anti-patterns

❌ **Top-level import of heavy package**:
```python
# recon/encoders/brainomni.py
from brainomni.model import BrainOmni  # Crashes on import!
```

❌ **Bare ModuleNotFoundError**:
```python
def build_brainomni():
    from brainomni.model import BrainOmni
    # ↑ raises ModuleNotFoundError, unhelpful message
```

❌ **Try/except that swallows the error**:
```python
try:
    from brainomni.model import BrainOmni
except ImportError:
    return None  # User has no idea why
```

❌ **Hardcoded path to model weights**:
```python
ENCODER = BrainOmni.from_pretrained("/home/user/checkpoints/brainomni")
# ↑ Breaks on other machines; pass via config instead
```

❌ **Multiple envs for "isolation"**:
```bash
conda activate recon-core        # for main code
conda activate recon-brainomni   # for BrainOmni work
# ↑ Cross-env imports break; this is what we're moving away from
```

## 7. Adding a new extra — checklist

When adding a new optional integration:

- [ ] Add to `pyproject.toml` under `[project.optional-dependencies]`
- [ ] Add the extra name → primary package mapping in
      `recon/utils/optional.py::_EXTRA_TO_PACKAGE`
- [ ] Implement the module with lazy imports (see §5)
- [ ] Add a unit test in `tests/unit/test_optional.py` (or similar)
- [ ] Add a Hydra config in `configs/<section>/<name>.yaml`
- [ ] Update `environment.yml` to include it in `[all]`
- [ ] (If heavy) Add a CI matrix entry in `.github/workflows/lint.yml`
- [ ] Document in `docs/standards/07-dependencies.md` if it has quirks

## 8. The `[all]` extra

`[all]` is the cluster's installation target. It should include every other extra:

```toml
all = [
    "neural-decoding-infra[brainomni,large-lm,data-public]",
]
```

When adding a new extra, **update `[all]`** in the same commit.

## 9. The `[dev]` extra

`[dev]` includes test/lint tooling (pytest, ruff, mypy, etc.). It does NOT include heavy ML extras by default. For local dev with everything, use `[dev-all]`.

## 10. Version constraints

When pinning versions, **be conservative**:

```toml
# ✅ Good: range, allows patches
"pydantic>=2.5,<3"

# ✅ Good: lower bound only
"numpy>=1.24"

# ❌ Bad: overly tight pin
"pydantic==2.5.3"

# ❌ Bad: no lower bound on fast-moving package
"transformers"  # could break tomorrow
```

Document non-obvious pins with a comment.

## 11. Verifying extras

Quick smoke test for each extra after install:

```bash
# Activate env
conda activate recon

# Core should always work
python -c "import recon; print('core OK')"

# Each extra
pip install -e ".[brainomni]"
python -c "from recon.encoders.brainomni import is_brainomni_available; print('brainomni OK' if is_brainomni_available() else 'brainomni NOT available')"

# All at once (cluster install)
pip install -e ".[all]"
python -c "from recon.utils.optional import extra_installed; print({e: extra_installed(e) for e in ['brainomni', 'large-lm', 'data-public']})"
```

## 12. See also

- [`pyproject.toml`](../../pyproject.toml) — extra declarations
- [`environment.yml`](../../environment.yml) — cluster env spec
- [`recon/utils/optional.py`](../../recon/utils/optional.py) — the helpers
- [`recon/encoders/brainomni.py`](../../recon/encoders/brainomni.py) — example integration
- [`recon/models/registry.py`](../../recon/models/registry.py) — model registry
- ADR-0005 — why no local GPU testing
- [Standard 04: Testing](04-testing.md) — testing extras

---

Maintained by owner. Update when extra conventions change.