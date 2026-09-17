"""Integration smoke: aligned decode flow (D1-A) on synthetic data.

Builds a synthetic story tensor + char-time mat, decodes a few characters
with a random-init brain model (pipeline smoke — output is meaningless
text, but the alignment plumbing is exercised end-to-end).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

pytestmark = pytest.mark.slow


@pytest.fixture()
def synthetic(tmp_path: Path):
    # story: (T=100, ctx=5, C=20)
    rng = np.random.default_rng(0)
    story = rng.standard_normal((100, 5, 20)).astype(np.float32)
    story_path = tmp_path / "zresp1_1_context_5.npy"
    np.save(story_path, story)

    chars = ["今", "天", "很", "好", "，"]
    ends = [12.0, 12.4, 12.8, 13.2, 13.6]
    mat_path = tmp_path / "story_1_char_time.mat"
    cell = np.array(chars, dtype="<U10").reshape(1, -1)
    savemat(str(mat_path), {"char": cell, "end": np.array(ends).reshape(1, -1)})
    return story_path, mat_path


class TestAlignedDecode:
    def test_decode_story_smoke(self, synthetic):
        import torch

        from recon.cli.decode import decode_story
        from recon.decoders.beam import DecodingConfig
        from recon.models.meg.meg_model_a import MEGModelA, MEGModelAConfig

        story_path, mat_path = synthetic
        story = np.load(story_path)

        model = MEGModelA(MEGModelAConfig(n_channels=20, n_context=5, semantic_dim=768))
        model.eval()

        from transformers import BertTokenizer, GPT2LMHeadModel
        gpt_path = "D:/allforwork/Liu_Lab/_Reconstruction/gpt2-chinese-cluecorpussmall"
        if not Path(gpt_path).exists():
            pytest.skip("GPT-2 weights not present")
        tokenizer = BertTokenizer.from_pretrained(gpt_path)
        gpt_model = GPT2LMHeadModel.from_pretrained(gpt_path)
        gpt_model.eval()

        cfg = DecodingConfig(beam_width=4, extensions=3, max_chars=5, device="cpu")
        text, reference = decode_story(
            model, gpt_model, tokenizer, story, Path(mat_path),
            n_context=5, decoding_cfg=cfg)

        assert reference == "今天很好，"  # valid chars (grid rows 0..4)
        assert len(text) == 5  # one char per decode step
        # cold-start + valid-char vocab guarantee printable single chars
        assert all(len(c) == 1 and c.isprintable() for c in text)

    def test_length_override_truncates(self, synthetic):
        import torch

        from recon.cli.decode import decode_story
        from recon.decoders.beam import DecodingConfig
        from recon.models.meg.meg_model_a import MEGModelA, MEGModelAConfig

        story_path, mat_path = synthetic
        model = MEGModelA(MEGModelAConfig(n_channels=20, n_context=5))
        model.eval()
        from transformers import BertTokenizer, GPT2LMHeadModel
        gpt_path = "D:/allforwork/Liu_Lab/_Reconstruction/gpt2-chinese-cluecorpussmall"
        if not Path(gpt_path).exists():
            pytest.skip("GPT-2 weights not present")
        tokenizer = BertTokenizer.from_pretrained(gpt_path)
        gpt_model = GPT2LMHeadModel.from_pretrained(gpt_path)
        gpt_model.eval()

        cfg = DecodingConfig(beam_width=4, extensions=3, max_chars=5, device="cpu")
        text, reference = decode_story(
            model, gpt_model, tokenizer, np.load(story_path), Path(mat_path),
            n_context=5, decoding_cfg=cfg, length=3)
        assert reference == "今天很"
        assert len(text) == 3
