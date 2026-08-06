# Guide 02: Write a data adapter

> **Audience**: Anyone adding a new dataset to the project.

---

## What is a "data adapter"?

A data adapter is the bridge between **raw data on disk** and the in-memory **Pydantic schemas** the rest of the system uses.

```
[Raw files] → [Adapter] → [Pydantic schema] → [PyTorch Dataset] → [DataLoader]
```

## When you need a new adapter

- Adding a new public dataset (e.g., BOLD5000, Zuco, Pereira2018)
- Adding a new preprocessing variant for an existing dataset
- Adding a new modality (e.g., ECoG if we ever expand)

## Steps

### 1. Define the schema

If the new dataset's shapes differ from existing ones, add a new schema in `recon/data/schema.py`:

```python
class Bold5000Sample(BaseModel):
    """One trial from the BOLD5000 dataset.
    
    Attributes:
        volume: 4D fMRI volume. Shape (T, X, Y, Z).
        mask: Brain mask. Shape (X, Y, Z), bool.
        label: Stimulus category (e.g., COCO image class).
        trial_id: Trial index within a session.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    volume: np.ndarray
    mask: np.ndarray
    label: str
    trial_id: int
```

### 2. Write the adapter

Create `recon/data/bold5000.py`:

```python
"""Adapter for the BOLD5000 dataset.

Loads preprocessed BOLD5000 .nii volumes and per-trial stimulus labels
into a sequence of Pydantic-validated samples.
"""
from pathlib import Path
import numpy as np
import nibabel as nib

from .schema import Bold5000Sample


class Bold5000Adapter:
    """Loads BOLD5000 data from disk."""

    def __init__(self, data_root: str | Path):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"BOLD5000 root not found: {self.data_root}")

    def list_trials(self) -> list[trial_id]:
        """Return all trial IDs."""
        return sorted([
            int(p.stem.split("-")[1])
            for p in self.data_root.glob("sub-*/trial-*")
        ])

    def load_trial(self, trial_id: int) -> Bold5000Sample:
        """Load one trial.

        Args:
            trial_id: The trial index.

        Returns:
            A validated Bold5000Sample.

        Raises:
            FileNotFoundError: If trial files are missing.
        """
        vol_path = self.data_root / f"trial-{trial_id:04d}/bold.nii.gz"
        mask_path = self.data_root / f"trial-{trial_id:04d}/mask.nii.gz"
        label_path = self.data_root / f"trial-{trial_id:04d}/label.txt"

        if not vol_path.exists():
            raise FileNotFoundError(f"Missing volume: {vol_path}")

        volume = nib.load(str(vol_path)).get_fdata().astype(np.float32)
        mask = nib.load(str(mask_path)).get_fdata().astype(bool)
        label = label_path.read_text().strip()

        return Bold5000Sample(
            volume=volume,
            mask=mask,
            label=label,
            trial_id=trial_id,
        )
```

### 3. Wrap in a PyTorch Dataset

Create `recon/data/datasets/bold5000.py`:

```python
"""PyTorch Dataset for BOLD5000."""
from torch.utils.data import Dataset

from ..bold5000 import Bold5000Adapter
from ..schema import Bold5000Sample


class Bold5000Dataset(Dataset):
    def __init__(self, data_root: str, trial_ids: list[int] | None = None):
        self.adapter = Bold5000Adapter(data_root)
        self.trial_ids = trial_ids or self.adapter.list_trials()

    def __len__(self) -> int:
        return len(self.trial_ids)

    def __getitem__(self, idx: int) -> Bold5000Sample:
        return self.adapter.load_trial(self.trial_ids[idx])
```

### 4. Register in Hydra config

Create `configs/data/bold5000.yaml`:

```yaml
name: bold5000
data_root: ${paths.data_root}/bold5000
train_split: 0.8
batch_size: 8
```

Add to `configs/train.yaml` defaults (or let user override):

```yaml
defaults:
  - data: bold5000   # user can override with data=drdr
```

### 5. Write a unit test

Create `tests/unit/test_bold5000_adapter.py`:

```python
"""Tests for the BOLD5000 adapter."""
import pytest
from recon.data.bold5000 import Bold5000Adapter
from recon.data.datasets.bold5000 import Bold5000Dataset


def test_adapter_init_missing_root(tmp_path):
    """Adapter should raise if root doesn't exist."""
    with pytest.raises(FileNotFoundError):
        Bold5000Adapter(tmp_path / "nonexistent")


def test_dataset_len_empty(tmp_path):
    """Empty dataset should have length 0."""
    (tmp_path / "sub-01").mkdir()
    dataset = Bold5000Dataset(str(tmp_path), trial_ids=[])
    assert len(dataset) == 0
```

### 6. Wire into the registry (if applicable)

If the dataset plugs into a different training pipeline than `drdr`, register it:

```python
# recon/data/__init__.py
DATASET_REGISTRY = {
    "drdr": build_drdr_dataset,
    "bold5000": build_bold5000_dataset,
}
```

## Checklist

Before submitting your PR:

- [ ] Schema defined in `recon/data/schema.py`
- [ ] Adapter class in `recon/data/<name>.py`
- [ ] PyTorch Dataset wrapper in `recon/data/datasets/<name>.py`
- [ ] Hydra config in `configs/data/<name>.yaml`
- [ ] Unit tests in `tests/unit/test_<name>*.py`
- [ ] Added entry to [Data card](../research/02-data-card.md)
- [ ] (Optional) ADR if a significant design decision was made

## See also

- [Data card](../research/02-data-card.md)
- [Architecture: Data pipeline](../architecture/02-data-pipeline.md)
- [ADR-0002: Pydantic schema](../decisions/0002-pydantic-schema.md)
- [Standard 02: Docstring style](../standards/02-docstring.md)

---

Maintained by owner.