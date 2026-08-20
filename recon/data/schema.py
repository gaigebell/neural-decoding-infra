"""Pydantic schemas for all data flowing through the recon pipeline.

These schemas are the **single source of truth** for what data looks like
at each layer. Adapters read raw files and emit these; downstream code
(Trainer, Decoder, Evaluator) only knows these shapes.

See ADR-0002 for the rationale and docs/standards/07-dependencies.md for
how to add new sample types.
"""
from __future__ import annotations

import logging
from typing import Literal, Union, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

# ───────────────────── Type aliases ─────────────────────

# Standard float types for tensors
Float32Array = NDArray[np.float32]
Int32Array = NDArray[np.int32]
BoolArray = NDArray[np.bool_]


# ───────────────────── Stimulus (semantic features) ─────────────────────


class StimSample(BaseModel):
    """A single time step's stimulus (semantic features) for one subject.

    Attributes:
        zstim: Semantic target vector — the delay-weighted combination of
            the 4 GPT-2 delay vectors (weights ``[0.1, 0.7, 0.5, 0.3]``,
            see ``recon.data.drdr.weight_delays``). Shape (768,).
        zstim_raw: Optional raw concatenated delays, shape (4*768,).
        layer: GPT-2 layer used (default 10).
        delays: Tuple of delay values (in time steps) used.
        story_id: Which story this sample belongs to.
        subject_id: Which subject this sample belongs to.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    zstim: Float32Array
    zstim_raw: Float32Array | None = None
    layer: int = 10
    delays: tuple[int, ...] = (1, 2, 3, 4)
    story_id: int
    subject_id: int

    @field_validator("zstim")
    @classmethod
    def _check_zstim_shape(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 1:
            raise ValueError(f"zstim must be 1D, got shape {v.shape}")
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        return v


# ───────────────────── MEG ─────────────────────


class MEGSample(BaseModel):
    """A single time step's MEG data for one subject.

    Attributes:
        x: MEG signal. Shape (channels, time_samples). For example,
            (306, 256) for 306 channels at 256 Hz over 1 second.
        pos: Sensor positions. Shape (channels, 6) — xyz + direction.
        sensor_type: Integer sensor type code per channel. Shape (channels,).
            Encoding: 0=EEG, 1=MAG (magnetometer), 2=GRAD (gradiometer).
        word_times: Optional word onset timestamps in seconds. Shape ().
        story_id: Which story this sample belongs to.
        subject_id: Which subject this sample belongs to.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    modality: Literal["meg"] = "meg"
    x: Float32Array
    pos: Float32Array
    sensor_type: Int32Array
    word_times: Float32Array | None = None
    story_id: int
    subject_id: int

    @field_validator("x")
    @classmethod
    def _check_x(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2:
            raise ValueError(f"MEG x must be 2D (channels, time), got {v.shape}")
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        return v

    @field_validator("pos")
    @classmethod
    def _check_pos(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 2 or v.shape[1] != 6:
            raise ValueError(f"MEG pos must be (channels, 6), got {v.shape}")
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        return v

    @field_validator("sensor_type")
    @classmethod
    def _check_sensor_type(cls, v: np.ndarray) -> np.ndarray:
        valid = {0, 1, 2}
        unique = set(np.unique(v).tolist())
        if not unique.issubset(valid):
            raise ValueError(f"Invalid sensor_type values: {unique - valid}")
        if v.dtype != np.int32:
            v = v.astype(np.int32)
        return v


class MEGChunkedSample(MEGSample):
    """MEG sample with a context window of past time steps.

    Used by models that take a temporal context of MEG (e.g., MEG_model_A
    with `n_context=5`).

    Attributes:
        x: MEG signal with context. Shape (context_len, channels, time_per_step).
            Default: (5, 306, 256).
        context_len: Number of past time steps included.
    """

    # pyright: ignore[reportIncompatibleVariableOverride]
    # pydantic discriminator tag narrowing; runtime value is a plain str default.
    modality: Literal["meg_chunked"] = "meg_chunked"
    context_len: int = 5

    @field_validator("x")
    @classmethod
    def _check_x(cls, v: np.ndarray) -> np.ndarray:
        # Overrides MEGSample._check_x (2D) — chunked MEG is 3D.
        if v.ndim != 3:
            raise ValueError(
                f"Chunked MEG x must be 3D (context, channels, time), got {v.shape}"
            )
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        return v


# ───────────────────── fMRI ─────────────────────


class fMRISample(BaseModel):
    """A single time step's fMRI data for one subject.

    Attributes:
        volume: 3D BOLD volume. Shape (X, Y, Z). For example, (53, 63, 52)
            for MNI-2mm template resolution.
        mask: Brain mask. Shape (X, Y, Z), bool. Voxels outside the mask
            should be ignored.
        story_id: Which story this sample belongs to.
        subject_id: Which subject this sample belongs to.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    modality: Literal["fmri"] = "fmri"
    volume: Float32Array
    mask: BoolArray
    story_id: int
    subject_id: int

    @field_validator("volume")
    @classmethod
    def _check_volume(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 3:
            raise ValueError(f"fMRI volume must be 3D, got shape {v.shape}")
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        return v

    @field_validator("mask")
    @classmethod
    def _check_mask(cls, v: np.ndarray) -> np.ndarray:
        if v.dtype != bool:
            v = v.astype(bool)
        return v


# ───────────────────── BrainOmni features ─────────────────────


class BrainOmniSample(BaseModel):
    """BrainOmni-encoded features (output of BrainOmni encoder).

    This is what comes out of ``recon.encoders.brainomni`` and gets fed to
    a downstream alignment head.

    Attributes:
        features: BrainOmni features. Shape (n_neurons, seq_len, n_dim).
        subject_id: Which subject this sample belongs to.
        story_id: Which story this sample belongs to.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    modality: Literal["brainomni"] = "brainomni"
    features: Float32Array
    subject_id: int
    story_id: int

    @field_validator("features")
    @classmethod
    def _check_features(cls, v: np.ndarray) -> np.ndarray:
        if v.ndim != 3:
            raise ValueError(
                f"BrainOmni features must be 3D (n_neurons, seq_len, n_dim), got {v.shape}"
            )
        if v.dtype != np.float32:
            v = v.astype(np.float32)
        return v


# ───────────────────── Combined pairs ─────────────────────


class BrainStimPair(BaseModel):
    """A (brain, stimulus) pair for one time step.

    The brain data can be from any modality. The stimulus is always
    semantic features.

    Attributes:
        brain: One of MEG, MEGChunked, fMRI, or BrainOmni sample.
        stim: The corresponding stimulus sample.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    brain: Union[MEGSample, MEGChunkedSample, fMRISample, BrainOmniSample] = Field(
        discriminator="modality"
    )
    stim: StimSample

    @property
    def modality(self) -> Literal["meg", "meg_chunked", "fmri", "brainomni"]:
        """Return the modality name."""
        return cast(
            Literal["meg", "meg_chunked", "fmri", "brainomni"], self.brain.modality
        )


# ───────────────────── Public API ─────────────────────


__all__ = [
    "BoolArray",
    "Float32Array",
    "Int32Array",
    "MEGChunkedSample",
    "MEGSample",
    "BrainOmniSample",
    "BrainStimPair",
    "StimSample",
    "fMRISample",
]