"""Beam search decoder for Chinese character generation.

Faithful port of the original decoding loop (``2026-7-22/run_decoder.py`` +
``decode/Decoder.py`` + ``decode/StimulusModel.py``), simplified:

- **Batched brain encoding**: the brain model runs once over the whole
  story, producing (T, 768) semantic predictions — instead of re-running
  it per beam extension (the original's biggest cost).
- **Contextualized candidate features**: each candidate character is
  embedded by the GPT-2 hidden state at ``select_layer`` (default 10,
  same as the stimulus features), conditioned on the previous
  ``context_words`` characters — matching the original ``LMFeatures``.
- **Proper beam**: ``beam_width`` hypotheses are maintained; per step each
  hypothesis proposes ``extensions`` nucleus-sampled characters; the global
  pool is pruned to the top ``beam_width`` by combined score
  ``logprob + sim_ratio * cosine_sim``.
- **Constant-size LM context**: each step forwards the LM only over the
  last ``context_words + 1`` characters (batched across all candidates),
  so per-step cost does not grow with story length. (KV-cache reuse is a
  planned optimization, not implemented yet.)

Deliberate simplifications vs the original:

- No word-rate (WR) model: the original downsampled word-level stimulus
  features to TRs via a lanczos matrix and decoded word-by-word. Here the
  per-step brain predictions align 1:1 with decoded characters (one char
  per time step). Porting the WR model is planned (P2).
- Combined score ``logprob + sim_ratio * cos`` instead of the original's
  softmax-similarity ranking with separately accumulated logprobs.

Usage:
    >>> from recon.decoders.beam import BeamSearchDecoder, DecodingConfig
    >>> config = DecodingConfig(beam_width=200, max_chars=1000)
    >>> decoder = BeamSearchDecoder(brain_encoder, gpt_model, tokenizer, config)
    >>> text = decoder.decode(brain_signals)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from ..utils.logging import get_logger

logger = get_logger(__name__)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


# Cold-start candidates when the beam has no context yet.
_COLD_START_CHARS = ["的", "是", "我", "你", "他", "一", "不", "了", "人", "在"]

# Punctuation allowed as a single-character candidate (in addition to
# Chinese characters and ASCII alphanumerics).
_ALLOWED_PUNCT = set("，。！？、；：""''（）《》…—·")


def _is_valid_char(s: str) -> bool:
    """True if ``s`` is a single printable character worth proposing.

    Excludes [UNK]/special tokens, whitespace, BPE continuation pieces
    (they decode to multi-char or replacement-char garbage), and control
    characters. This mirrors the original pipeline's restricted decode
    vocabulary (``decode_vocab`` built from story characters / the GPT
    vocab), which we rebuild here from the tokenizer itself.
    """
    if len(s) != 1:
        return False
    c = s[0]
    if c in ("�", "[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"):
        return False
    if "一" <= c <= "鿿":  # CJK unified ideograph
        return True
    if c.isascii() and (c.isalnum() or c in _ALLOWED_PUNCT):
        return True
    if c in _ALLOWED_PUNCT:
        return True
    return False


@dataclass
class DecodingConfig:
    """Configuration for :class:`BeamSearchDecoder`."""

    beam_width: int = 200
    extensions: int = 5  # candidates proposed per hypothesis per step
    lm_mass: float = 0.9  # nucleus sampling top-p mass
    sim_ratio: float = 0.15  # weight of cosine sim vs LM logprob
    max_chars: int = 2000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    semantic_dim: int = 768
    select_layer: int = 10  # GPT-2 layer for candidate features
    context_words: int = 5  # chars of left context for features
    log_interval: int = 50


@dataclass
class Hypothesis:
    """One beam hypothesis: a partial character sequence + cumulative score."""

    words: list[str] = field(default_factory=list)  # characters
    logprobs: list[float] = field(default_factory=list)  # per-step log probs
    score: float = 0.0  # cumulative score (logprob + sim_ratio * cos)

    def extend(self, char: str, logprob: float) -> "Hypothesis":
        return Hypothesis(
            words=self.words + [char],
            logprobs=self.logprobs + [logprob],
            score=self.score + logprob,
        )


class BeamSearchDecoder:
    """Beam-search decoder over GPT-2 semantic features.

    Args:
        brain_encoder: Trained model (callable: brain_input -> semantic
            prediction (B, 768), possibly as a tuple). Eval mode is set.
        gpt_model: A HuggingFace GPT-2 model (used for nucleus proposal
            and layer-``select_layer`` feature extraction).
        gpt_tokenizer: Corresponding tokenizer (must have a pad token;
            ``[PAD]`` id 0 in the Chinese cluecorpus vocab).
        config: :class:`DecodingConfig`.
    """

    def __init__(
        self,
        brain_encoder: Any,
        gpt_model: Any,
        gpt_tokenizer: Any,
        config: DecodingConfig | None = None,
    ):
        self.brain_encoder = brain_encoder
        self.gpt_model = gpt_model
        self.tokenizer = gpt_tokenizer
        self.config = config or DecodingConfig()
        self.device = torch.device(self.config.device)
        # Lazily built in _valid_token_mask(): bool mask over the vocab of
        # tokens that decode to a single printable character.
        self._valid_mask: torch.Tensor | None = None

        if hasattr(self.brain_encoder, "to"):
            self.brain_encoder = self.brain_encoder.to(self.device)
        if hasattr(self.brain_encoder, "eval"):
            self.brain_encoder.eval()
        if hasattr(self.gpt_model, "to"):
            self.gpt_model = self.gpt_model.to(self.device)
        if hasattr(self.gpt_model, "eval"):
            self.gpt_model.eval()

    # ───────────────────── Public API ─────────────────────

    @torch.no_grad()
    def decode(self, brain_input: torch.Tensor) -> str:
        """Decode a story's worth of brain signals to text.

        Args:
            brain_input: Brain signal tensor. Shape depends on the model:
                - MEG chunked: (T, n_context, n_channels)
                - fMRI: (T, X, Y, Z) or (T, 1, X, Y, Z)

        Returns:
            Decoded string (concatenation of the best beam hypothesis).
        """
        brain_input = brain_input.to(self.device)

        logger.info("Step 1: batched brain encoding...")
        brain_features = self._encode_brain(brain_input)  # (T, 768)
        n_steps = brain_features.shape[0]
        logger.info("Encoded %d time steps, semantic dim=%d", n_steps, brain_features.shape[-1])

        beam = [Hypothesis()]
        for t in tqdm(range(min(n_steps, self.config.max_chars)), desc="Decoding", leave=False):
            pool: list[tuple[float, Hypothesis]] = []
            for hyp in beam:
                # 1. Nucleus proposal from this hypothesis's recent context
                context = "".join(hyp.words[-self.config.context_words :])
                chars, logprobs = self._nucleus_propose(context)
                if not chars:
                    continue

                # 2. Contextualized features of candidate chars (layer L)
                cand_feats = self._candidate_features(hyp.words, chars)  # (K, 768)

                # 3. Cosine similarity with the brain prediction at step t
                brain_t = brain_features[t]  # (768,)
                sims = F.cosine_similarity(
                    brain_t.unsqueeze(0), cand_feats, dim=-1
                )  # (K,)

                # 4. Combined score per extension
                for char, lp, sim in zip(chars, logprobs, sims.tolist()):
                    score = hyp.score + lp + self.config.sim_ratio * sim
                    pool.append((score, hyp.extend(char, lp)))

            if not pool:
                break
            pool.sort(key=lambda x: -x[0])
            beam = [hyp for _, hyp in pool[: self.config.beam_width]]

            if (t + 1) % self.config.log_interval == 0:
                best = beam[0]
                logger.info(
                    "step %d: best='%s' score=%.3f",
                    t + 1,
                    best.words[-1] if best.words else "",
                    best.score,
                )

        return "".join(beam[0].words)

    # ───────────────────── Internal helpers ─────────────────────

    def _encode_brain(self, brain_input: torch.Tensor) -> torch.Tensor:
        """Run the brain encoder over the full story.

        Chunked MEG arrives as (T, n_context, n_channels): each t is a
        window of past context, so steps are encoded one at a time.
        """
        if brain_input.ndim == 3:
            features = []
            for t in range(brain_input.shape[0]):
                out = self.brain_encoder(brain_input[t].unsqueeze(0))
                if isinstance(out, tuple):
                    out = out[0]
                features.append(out)
            return torch.cat(features, dim=0)
        out = self.brain_encoder(brain_input)
        if isinstance(out, tuple):
            out = out[0]
        return out

    def _valid_token_mask(self) -> torch.Tensor:
        """Bool mask (model vocab_size,) over tokens that decode to single chars.

        Sized by the MODEL's vocab (``config.vocab_size``), not the
        tokenizer's ``len()``: vocab.txt can carry an extra added token
        that the embedding matrix does not have.
        """
        if self._valid_mask is None:
            n = int(self.gpt_model.config.vocab_size)
            mask = [
                _is_valid_char(self.tokenizer.decode([i])) for i in range(n)
            ]
            self._valid_mask = torch.tensor(mask, device=self.device)
            n_valid = int(self._valid_mask.sum().item())
            logger.info("Decode vocabulary: %d/%d tokens are single chars", n_valid, n)
        return self._valid_mask

    def _nucleus_propose(self, context: str) -> tuple[list[str], list[float]]:
        """Nucleus-sample candidate next characters from the LM.

        Sampling is restricted to tokens that decode to a single printable
        character (see ``_valid_token_mask``) — the original pipeline's
        restricted decode vocabulary. This keeps [UNK], special tokens,
        and BPE fragments out of the beam.

        Returns:
            (candidate characters, their log probabilities). Cold start
            (empty context) returns a fixed set of common characters.
        """
        if not context:
            return list(_COLD_START_CHARS), [-1.0] * len(_COLD_START_CHARS)

        input_ids = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        if input_ids.shape[1] == 0:
            return [], []
        logits = self.gpt_model(input_ids).logits[0, -1, :]
        # Restrict to single-character tokens before the nucleus cutoff
        mask = self._valid_token_mask()
        logits = logits[: mask.shape[0]].masked_fill(~mask, -float("inf"))
        probs = F.softmax(logits, dim=-1)

        sorted_probs, sorted_idx = probs.sort(descending=True)
        cutoff = (sorted_probs.cumsum(dim=-1) <= self.config.lm_mass).sum().item() + 1
        nucleus_probs = sorted_probs[:cutoff]
        nucleus_probs = nucleus_probs / nucleus_probs.sum()
        nucleus_idx = sorted_idx[:cutoff]

        k = min(self.config.extensions, len(nucleus_idx))
        sampled = torch.multinomial(nucleus_probs, k, replacement=False)
        chosen_logprobs = torch.log(nucleus_probs[sampled] + 1e-12)
        chars = [self.tokenizer.decode([int(i)]) for i in nucleus_idx[sampled]]
        return chars, chosen_logprobs.tolist()

    def _candidate_features(
        self, history: list[str], chars: list[str]
    ) -> torch.Tensor:
        """Layer-``select_layer`` GPT-2 features of candidate characters.

        Port of the original ``LMFeatures.extend``: the feature of a
        candidate char is the GPT-2 hidden state at ``select_layer`` at the
        char's position, conditioned on the previous ``context_words``
        chars of the hypothesis. All candidates are batched with left
        padding (pad token id 0), so one forward serves the whole batch.
        """
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        contexts = [
            history[-(self.config.context_words) :] + [c] for c in chars
        ]
        max_len = max(len(ctx) for ctx in contexts)
        ids = []
        for ctx in contexts:
            tok = self.tokenizer.convert_tokens_to_ids(ctx)
            ids.append([pad_id] * (max_len - len(tok)) + tok)
        input_ids = torch.tensor(ids, device=self.device)
        attention_mask = (input_ids != pad_id).long()

        outputs = self.gpt_model(
            input_ids, attention_mask=attention_mask, output_hidden_states=True
        )
        hidden = outputs.hidden_states[self.config.select_layer]  # (B, L, 768)
        last_pos = attention_mask.sum(dim=1) - 1  # position of each char
        feats = hidden[torch.arange(hidden.shape[0], device=self.device), last_pos]
        return feats


__all__ = ["BeamSearchDecoder", "DecodingConfig", "Hypothesis"]
