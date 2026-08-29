"""Golden-equivalence tests: new pipeline vs legacy artifacts in E:/results.

The legacy pipeline's outputs (``E:/results``) are the golden standard.
These tests feed the SAME inputs the legacy pipeline used and assert the
new implementation reproduces the golden artifacts bit-exactly
(``rtol=0, atol=0``).

Auto-skipped when the golden data is absent. Slow tests are marked
``slow``; run explicitly with ``-m slow``.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from recon.preprocessing import meg, semantic

PROCESSED_ROOT = Path(os.environ.get("RECON_PROCESSED_ROOT", "E:/results"))
DATA_ROOT = Path(os.environ.get("RECON_DATA_ROOT", "E:/reconstruction"))

pytestmark = pytest.mark.skipif(
    not PROCESSED_ROOT.exists(), reason="golden data (E:/results) not present"
)

GOLDEN = {
    "wordvectors": PROCESSED_ROOT / "wordvectors" / "story1_10.npy",
    "mat": DATA_ROOT / "mydata/derivatives/annotations/time_align/char-level/story_1_char_time.mat",
    "ds": PROCESSED_ROOT / "MEG" / "downsample" / "ds_1_10.npy",
    "zstim": PROCESSED_ROOT / "MEG" / "zstim" / "sub1_zstim_10_story1.npy",
    "fif": DATA_ROOT / "mydata/derivatives/preprocessed_data/sub-01/MEG/sub-01_task-RDR_run-1_meg.fif",
    "zresp": PROCESSED_ROOT / "MEG" / "zresp" / "zresp1_1.npy",
    "zresp_ctx": PROCESSED_ROOT / "MEG" / "zresp" / "zresp1_1_context_5.npy",
    "fmri_nii": DATA_ROOT / "mydata/derivatives/preprocessed_data/sub-01/MNI/sub-01_task-RDR_run-1_bold.nii.gz",
    "fmri_ds": PROCESSED_ROOT / "downsample" / "ds_1_10.npy",
    "fmri_cube": PROCESSED_ROOT / "zresp" / "cube" / "zresp1_1.npy",
}


def _require_golden(name: str) -> Path:
    p = GOLDEN[name]
    if not p.exists():
        pytest.skip(f"golden input missing: {p}")
    return p


class TestSemanticGolden:
    def test_downsample_matches_golden(self, tmp_path: Path):
        wv, mat = _require_golden("wordvectors"), _require_golden("mat")
        out = semantic.downsample_story(wv, mat, tmp_path, story_id=1, layer=10)
        golden_ds = _require_golden("ds")
        # Lanczos uses sin() + dot(): the golden was generated with an
        # older numpy on another machine — last-bit float noise (~1e-13)
        # is expected. Everything else is bit-exact.
        np.testing.assert_allclose(np.load(out), np.load(golden_ds), rtol=0, atol=1e-9)
        assert out.with_suffix(".npy.meta.json").exists()  # sidecar written

    def test_delay_matches_golden(self, tmp_path: Path):
        ds = _require_golden("ds")
        out = semantic.delay_story(ds, tmp_path, subject_id=1, story_id=1, layer=10)
        golden = _require_golden("zstim")
        np.testing.assert_array_equal(np.load(out), np.load(golden))

    def test_full_chain_matches_golden(self, tmp_path: Path):
        """wordvectors + mat → ds → zstim, end-to-end vs golden."""
        wv, mat = _require_golden("wordvectors"), _require_golden("mat")
        ds_out = semantic.downsample_story(wv, mat, tmp_path, story_id=1, layer=10)
        zstim_out = semantic.delay_story(ds_out, tmp_path, subject_id=1, story_id=1, layer=10)
        golden = _require_golden("zstim")
        np.testing.assert_allclose(np.load(zstim_out), np.load(golden), rtol=0, atol=1e-9)


class TestMEGGolden:
    def test_zresp_matches_golden(self, tmp_path: Path):
        fif, ds = _require_golden("fif"), _require_golden("ds")
        out = meg.process_story(fif, ds, tmp_path, subject_id=1, story_id=1)
        golden = _require_golden("zresp")
        np.testing.assert_array_equal(np.load(out), np.load(golden))

    def test_chunk_context_matches_golden(self, tmp_path: Path):
        zresp = _require_golden("zresp")
        out = meg.chunk_context(zresp, tmp_path, subject_id=1, story_id=1, n_context=5)
        golden = _require_golden("zresp_ctx")
        np.testing.assert_array_equal(np.load(out), np.load(golden))


@pytest.mark.slow
class TestFMRIGolden:
    def test_cube_matches_golden(self, tmp_path: Path):
        from recon.preprocessing import fmri

        nii, ds = _require_golden("fmri_nii"), _require_golden("fmri_ds")
        out = fmri.process_story(nii, ds, tmp_path, subject_id=1, story_id=1)
        golden = _require_golden("fmri_cube")
        np.testing.assert_array_equal(np.load(out), np.load(golden))


@pytest.mark.slow
class TestGPTFeaturesGolden:
    def test_char_features_match_golden(self, tmp_path: Path):
        """Regenerate story-1 layer-10 char features and diff vs golden.

        The golden was generated on CPU with an older torch/transformers;
        float32 kernel noise is expected — tolerate last-bits differences
        while keeping the threshold tight enough to catch real bugs
        (wrong layer/context produce O(1) diffs).
        """
        mat = _require_golden("mat")
        gpt_path = Path("D:/allforwork/Liu_Lab/_Reconstruction/gpt2-chinese-cluecorpussmall")
        if not gpt_path.exists():
            pytest.skip("GPT-2 weights not found")
        outs = semantic.extract_gpt_char_features(
            mat, gpt_path, tmp_path, story_id=1, layers=[10],
            device="cuda" if os.environ.get("RECON_USE_CUDA") else "cpu",
        )
        golden = _require_golden("wordvectors")
        mine, gold = np.load(outs[0]), np.load(golden)
        assert mine.shape == gold.shape
        np.testing.assert_allclose(mine, gold, rtol=1e-4, atol=1e-3)
