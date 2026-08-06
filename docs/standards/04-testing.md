# Standard 04: Testing

> **Audience**: Anyone writing or running tests.
> **Status**: Active.

---

## 1. Philosophy

Three rules govern testing in this project:

1. **Any new logic deserves at least one test.** No exceptions.
2. **Tests must be runnable without a GPU.** ([ADR-0005](../decisions/0005-no-local-gpu-test.md))
3. **Tests must be fast.** Single test ≤ 1 second; full unit suite ≤ 1 minute.

## 2. Test layers

| Layer | Where | What | Who runs | When |
|---|---|---|---|---|
| **L0** | `tests/unit/` | Pure functions, no I/O, no GPU | Anyone, anywhere | Every commit |
| **L1** | `tests/integration/` | Component boundaries, fake data | Anyone, anywhere | PR creation |
| **L2** | cluster | Fake-data GPU smoke | Cluster cron | After merge |
| **L3** | cluster | Real-data 1-epoch | Owner manually | Before experiment |

L0 and L1 run on **any** dev machine via `make test`.
L2 and L3 run **only** on the cluster.

## 3. Framework

- **pytest** as the test runner
- **pytest-cov** for coverage
- **pytest-mock** for monkeypatching
- **hypothesis** (optional) for property-based testing

```bash
# Run all unit tests
make test-unit

# Run with coverage
make test-coverage

# Run a specific test
pytest tests/unit/test_schema.py::test_megsample_shape -v
```

## 4. Directory structure

Tests mirror source structure:

```
recon/data/schema.py
  ↳ tests/unit/test_schema.py

recon/models/registry.py
  ↳ tests/unit/test_registry.py

recon/engine/trainer.py
  ↳ tests/integration/test_trainer_smoke.py
```

Test files start with `test_` (pytest convention).

## 5. Test naming

```python
def test_<unit>_<behavior>_<expected>():
    ...
```

Examples:

- `test_zscore_zero_mean()` — `zscore()` should produce zero-mean output
- `test_zscore_unit_std()` — `zscore()` should produce unit-std output
- `test_zscore_raises_on_non_2d()` — `zscore()` should raise on bad input
- `test_megsample_rejects_wrong_shape()` — `MEGSample` should validate shapes

## 6. What to test

### Always test:

- ✅ Shape / dtype contracts of public functions
- ✅ Edge cases: empty input, single element, large input
- ✅ Error paths: bad input raises specific exception
- ✅ Numerical stability: NaN / Inf handling
- ✅ Reproducibility: same seed → same output

### Don't bother testing:

- ❌ PyTorch / numpy internals (they have their own tests)
- ❌ Configuration loading (test the underlying library)
- ❌ One-line getters / setters

## 7. Fixtures

Use fixtures for shared test data:

```python
# tests/conftest.py
import pytest
import numpy as np

@pytest.fixture
def fake_meg_sample():
    """Standard MEG sample with realistic shapes."""
    return MEGSample(
        x=np.random.randn(100, 306, 256).astype(np.float32),
        pos=np.random.randn(100, 306, 6).astype(np.float32),
        sensor_type=np.zeros((100, 306), dtype=np.int32),
        story_id=1,
        subject_id=1,
        word_times=np.linspace(0, 100, 100, dtype=np.float32),
    )
```

Use it in tests:

```python
def test_zscore_megsample(fake_meg_sample):
    result = zscore(fake_meg_sample.x)
    assert result.shape == fake_meg_sample.x.shape
    assert np.abs(result.mean(axis=-1)).max() < 1e-6
```

## 8. Markers

Mark tests that need special resources:

```python
import pytest

@pytest.mark.gpu
def test_model_forward_cuda():
    ...

@pytest.mark.slow
def test_full_epoch():
    ...

@pytest.mark.integration
def test_trainer_with_fake_data():
    ...
```

Default run (`make test`) skips `@pytest.mark.gpu` and `@pytest.mark.slow`. To run them:

```bash
pytest -m gpu          # only GPU tests
pytest -m "not slow"   # everything except slow
```

## 9. Reproducibility

Tests must be deterministic:

```python
def test_something():
    np.random.seed(42)
    torch.manual_seed(42)
    ...
```

If a test is flaky, it MUST be fixed before merge. Use `pytest-repeat` to verify:

```bash
pytest tests/unit/test_flaky.py --count=10
```

## 10. Coverage targets

- **Line coverage**: ≥ 70% overall, ≥ 80% for `recon.data` and `recon.engine`
- **Branch coverage**: ≥ 60% for `recon.engine`
- **Critical paths**: 100% (loss computation, data validation)

Coverage is reported by `make test-coverage` and uploaded to W&B.

## 11. Mocking the cluster

For tests that touch cluster-specific code (paths, GPU detection), use mocking:

```python
from unittest.mock import patch

@patch("torch.cuda.is_available", return_value=False)
def test_no_gpu_warning(mock_cuda):
    with pytest.warns(UserWarning, match="No GPU"):
        Trainer(cfg)
```

## 12. Anti-patterns

```python
# ❌ Bad: test depends on filesystem order
def test_load_all():
    for f in Path("data/").glob("*.npy"):
        sample = load(f)
        assert sample is not None  # depends on which files exist

# ✅ Good: test is self-contained
def test_load_specific():
    sample = load("data/sub-01/story-01.npy")
    assert sample.subject_id == 1

# ❌ Bad: sleeps in tests
def test_async():
    do_thing_async()
    time.sleep(5)
    assert state.is_done()  # flaky on slow machines

# ✅ Good: event-based
def test_async():
    event = threading.Event()
    do_thing_async(on_done=event.set)
    assert event.wait(timeout=10)

# ❌ Bad: order-dependent
def test_a():
    global STATE
    STATE = "A"
def test_b():
    assert STATE == "B"  # fails if test_a didn't run first
```

## 13. See also

- [Standard 01: Python style](01-python-style.md)
- [ADR-0005: No local GPU testing](../decisions/0005-no-local-gpu-test.md)
- [`pyproject.toml`](../../pyproject.toml) (pytest config)
- [Guide 04: Run training](../guides/04-run-training.md)

---

Maintained by owner. Update when test conventions evolve.