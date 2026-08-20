"""PyTorch Dataset wrappers around the recon data layer.

Each dataset:
- Takes a ``DRDRIndex`` (or similar)
- Implements ``__len__`` and ``__getitem__``
- Returns ``BrainStimPair`` (or its brain / stim components)
- Is compatible with PyTorch's DataLoader (incl. DistributedSampler)

Datasets do NOT:
- Apply data augmentation (that's a transform)
- Batch samples (DataLoader does)
- Move to GPU (Trainer does)
"""