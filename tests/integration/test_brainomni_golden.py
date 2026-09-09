"""Integration smoke tests for the BrainOmni pipeline stages.

The segments stage runs MNE filtering on a real fif (~1-2 min); the
encode stage loads the BrainOmni checkpoints from the owner's local
repo and encodes one story. Both are marked ``slow`` — run with
``-m slow`` on demand (encode prefers GPU via RECON_USE_CUDA).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from recon.preprocessing import brainomni

pytestmark = pytest.mark.slow

DATA_ROOT = Path(os.environ.get("RECON_DATA_ROOT", "E:/reconstruction"))
FIF = DATA_ROOT / "mydata/derivatives/preprocessed_data/sub-01/MEG/sub-01_task-RDR_run-1_meg.fif"
TIME_ALIGN = Path(os.environ.get("RECON_PROCESSED_ROOT", "E:/results")) / "MEG" / "downsample" / "time_1_10.npy"
BRAINOMNI_REPO = Path("D:/allforwork/Liu_Lab/_Reconstruction/BrainOmni-main")


@pytest.mark.skipif(not FIF.exists(), reason="raw fif not present")
class TestBrainOmniSegments:
    def test_segments_story_1(self, tmp_path: Path):
        out = brainomni.extract_segments_story(
            FIF, TIME_ALIGN, tmp_path, subject_id=1, story_id=1,
            sample_rate=256, segment_length=2,
        )
        assert out.exists()
        import torch

        data = torch.load(out, weights_only=False)
        assert data["x"].shape[0] == data["pos"].shape[0] == data["sensor_type"].shape[0]
        assert data["x"].shape[1] == data["pos"].shape[1] == 306
        assert data["x"].shape[2] == 256 * 2
        assert data["pos"].shape[2] == 6
        assert not np.isnan(data["x"].numpy()).any()
        # Sidecar + word-time meta written
        assert out.with_suffix(".pt.meta.json").exists()
        assert (out.parent / "word_time_meta_1_1.json").exists()

    def test_matches_golden_pos_and_sensor_type(self, tmp_path: Path):
        """vs golden seg4 artifact: pos and sensor_type are BIT-EXACT.

        x differs in scale: our fif stores physical units (fT, ~1e-15)
        while the golden was generated from a differently-scaled fif copy
        (range ±0.5). Shape/structure must match; per-segment correlation
        is scale-invariant and should be ~1 for a same-content check only
        when the input copies match — skip content compare here.
        """
        golden_path = (
            Path("E:/results/MEG/zresp/brainomni_sample_rate_256_segment_length_4")
            / "brainomni_zresp1_1.pt"
        )
        if not golden_path.exists():
            pytest.skip("golden brainomni artifact missing")
        out = brainomni.extract_segments_story(
            FIF, TIME_ALIGN, tmp_path, subject_id=1, story_id=1,
            sample_rate=256, segment_length=4,
        )
        import torch

        mine = torch.load(out, weights_only=False)
        golden = torch.load(golden_path, weights_only=False)
        assert mine["x"].shape == golden["x"].shape
        assert torch.equal(mine["pos"], golden["pos"])  # bit-exact
        assert torch.equal(mine["sensor_type"], golden["sensor_type"])  # bit-exact
        # x: same shape, real signal (not zeros), NaN-free
        assert not np.isnan(mine["x"].numpy()).any()
        assert float(mine["x"].abs().max()) > 0


@pytest.mark.skipif(
    not BRAINOMNI_REPO.exists(), reason="BrainOmni repo not present"
)
class TestBrainOmniEncode:
    def test_encode_small_slice_tiny(self, tmp_path: Path):
        """Encode a SMALL slice of segments with the tiny checkpoint.

        The full story (T≈1586 × C306 × 512) needs a big-memory GPU
        (cluster PH402); a 4-segment slice validates the stage end-to-end
        locally. Slice the .pt dict after the segments stage.
        """
        import torch

        if not TIME_ALIGN.exists():
            pytest.skip("time align missing")
        seg = brainomni.extract_segments_story(
            FIF, TIME_ALIGN, tmp_path, subject_id=1, story_id=1,
            sample_rate=256, segment_length=2,
        )
        data = torch.load(seg, weights_only=False)
        sliced = {k: v[:4] for k, v in data.items()}
        seg_small = tmp_path / "brainomni_zresp1_1_small.pt"
        torch.save(sliced, seg_small)

        device = "cuda" if os.environ.get("RECON_USE_CUDA") else "cpu"
        out = brainomni.encode_story(
            seg_small, tmp_path, subject_id=1, story_id=1,
            brainomni_repo=BRAINOMNI_REPO,
            brainomni_ckpt=BRAINOMNI_REPO / "ckpt_collection" / "tiny",
            tokenizer_ckpt=BRAINOMNI_REPO / "ckpt_collection" / "braintokenizer",
            device=device,
        )
        assert out.exists()
        features = torch.load(out, map_location="cpu", weights_only=False)
        assert features.ndim == 4  # (batch, n_neurons, seq_len, n_dim)
        assert out.with_suffix(".pt.meta.json").exists()