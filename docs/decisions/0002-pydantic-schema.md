# ADR-0002: Pydantic schemas for data

- **Status**: Accepted
- **Date**: 2026-07-28
- **Deciders**: owner

## Context and problem statement

Brain data comes in many shapes and modalities (MEG, fMRI, zstim, brainomni inputs). Today the codebase uses 3 different storage formats (`.npy`, `.pt`, `.json`) with shapes hardcoded in 5+ places. Adding a new dataset requires grep-ing through all model files to update shapes.

We need a single canonical representation that:
- Documents the shape and dtype of every tensor
- Validates at load time (catches preprocessing bugs early)
- Serializes to/from disk predictably
- Is independent of the training loop

## Decision drivers

- Multiple data formats coexist; migration will be gradual
- Shape mismatches cause silent bugs in DDP training
- Future: integrate new public datasets (each with different schemas)
- Single-owner project; prefer fewer dependencies

## Considered options

### Option A: Plain `dataclass` + numpy arrays

- **Pro**: Zero dependencies; familiar Python
- **Con**: No runtime validation; can't serialize to JSON

### Option B: `pydantic` v2

- **Pro**: Built-in validation, JSON-serializable, IDE type-hints
- **Pro**: Plays well with numpy via `model_validate`
- **Con**: Heavyweight; some numpy friction (must use `arbitrary_types_allowed`)

### Option C: `TypedDict`

- **Pro**: Zero deps
- **Con**: No validation; just documentation

### Option D: `numpy.typing.NDArray` annotations only

- **Pro**: Lighter; works in mypy
- **Con**: Still no runtime validation

## Decision outcome

**Chosen option**: **Option B (Pydantic v2)**.

Pydantic v2 is fast (Rust core), has excellent numpy support via `model_validate`, and catches shape/dtype bugs at the earliest possible point. The dependency is well-justified: data validation is the primary defense against the hardest class of bugs.

### Consequences

- ✅ Good: Shape/dtype errors caught at `load_sample()` time, not deep in the model
- ✅ Good: Schemas double as documentation (one file per data type)
- ✅ Good: Future-proof for adding new datasets — just add a new schema + adapter
- ❌ Bad: Pydantic v2 has a learning curve (especially around numpy + generics)
- ❌ Bad: Slight overhead (~1 ms per sample validation) — acceptable for our scale
- ❓ Risk: Pydantic v1→v2 had breaking changes; pin major version.

## Implementation sketch

```python
from pydantic import BaseModel, ConfigDict
import numpy as np

class MEGSample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    x: np.ndarray            # (T, C, time_samples)
    pos: np.ndarray          # (T, C, 6)
    sensor_type: np.ndarray  # (T, C)
    story_id: int
    subject_id: int
    word_times: np.ndarray   # (T,)

class BrainStimPair(BaseModel):
    brain: MEGSample  # or FMRI / BrainOmni variants
    stim: StimSample
```

## Links

- [Data card](../research/02-data-card.md)
- [Architecture: Data pipeline](02-data-pipeline.md)