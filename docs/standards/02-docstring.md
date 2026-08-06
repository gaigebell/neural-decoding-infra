# Standard 02: Docstring Style

> **Audience**: Anyone writing docstrings.
> **Status**: Active.

---

## 1. Convention

We use **Google-style docstrings**. Sphinx / mkdocstrings parses them.

## 2. Module docstring

Every module file starts with a docstring:

```python
"""One-line summary.

Longer description if needed. Can span multiple lines and reference
external concepts, papers, or other modules.

Example:
    >>> from recon.data import MEGSample
    >>> sample = MEGSample(...)
"""
```

## 3. Class docstring

```python
class MEGSample(BaseModel):
    """A single MEG recording window.

    Represents the brain data for one story for one subject, after
    preprocessing and alignment to word onsets.

    Attributes:
        x: MEG signal tensor of shape (T, C, time_samples).
        pos: Sensor positions of shape (T, C, 6).
        sensor_type: Integer sensor type code, shape (T, C).
        story_id: Which story (1-60).
        subject_id: Which subject (1-12).
        word_times: Word onset timestamps in seconds, shape (T,).

    Example:
        >>> sample = MEGSample(x=arr, pos=pos, ...)
    """
```

## 4. Function docstring

```python
def zscore(data: np.ndarray, return_stats: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Z-score normalize a 2D matrix column-wise.

    Each column is normalized independently. If ``return_stats`` is True,
    the per-column mean and std are returned alongside the normalized data.

    Args:
        data: Input 2D array of shape (T, D).
        return_stats: If True, also return (mean, std) used for normalization.

    Returns:
        Normalized array of the same shape as ``data``.
        If ``return_stats`` is True, returns ``(normalized, stats)`` where
        ``stats`` is shape (D, 2) with ``[std, mean]`` per column.

    Raises:
        ValueError: If ``data`` is not 2D.

    Example:
        >>> z = zscore(raw)
    """
```

## 5. Sections

Standard sections (in order):

| Section | When to use |
|---|---|
| `Args:` | All function arguments (except `self`, `cls`) |
| `Returns:` | Function return value |
| `Yields:` | If generator (overrides Returns) |
| `Raises:` | Exceptions that may be raised |
| `Example:` | One short usage example (optional but recommended for public APIs) |
| `Note:` | Non-obvious caveats |
| `Todo:` | Known limitations / future work |

Sections are introduced by a line ending in `:` followed by an indented block.

## 6. Style rules

1. **First line**: imperative mood, capitalized, ends in period.
   - ✅ `"""Load a sample from disk."""`
   - ❌ `"""This function loads a sample from disk."""` (passive)

2. **Multi-paragraph**: separate with blank line.

3. **Cross-references**: use backticks for code, full paths for modules.

   ```python
   """See :class:`recon.data.MEGSample` for the return type."""
   ```

4. **Math**: use single backticks for inline, code-block for display.

   ```python
   """Computes cosine similarity: `cos(theta) = a . b / (||a|| ||b||)`."""
   ```

5. **Type hints in signature, not in docstring**. Don't repeat types in `Args:`.
   - ✅ One exception: shape information for tensors

   ```python
   """Args:
       x: MEG signal tensor. Shape (T, C, time_samples).
   """
   ```

## 7. Public vs private

- **Public** (no leading underscore): full docstring required
- **Private** (`_name`): one-line docstring is OK
- **Internal helper modules**: docstrings still required but can be brief

## 8. Anti-patterns

```python
# ❌ Bad: empty docstring
def foo(x):
    """"""
    return x

# ❌ Bad: docstring describes the obvious
def add(a, b):
    """Add a and b."""
    return a + b  # even worse: docstring adds no info

# ❌ Bad: types in docstring conflict with signature
def foo(x: int) -> str:
    """Args:
        x: A string.   # wrong! x is int
    """

# ❌ Bad: rambling, no structure
def complex_thing(...):
    """This function does a lot of stuff. First it does X, then Y, and also Z if A is true. There's also some logic for B. Note that you need to call it after doing C..."""
```

## 9. See also

- [Standard 01: Python style](01-python-style.md)
- [Google style guide](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)
- [`pyproject.toml`](../../pyproject.toml) (ruff config)

---

Maintained by owner. Update if we adopt a different docstring style.