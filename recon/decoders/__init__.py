"""Decoders: convert brain-predicted semantic vectors to text.

The key decoder is :mod:`recon.decoders.beam`, which implements a
batched beam search with nucleus sampling and GPT-2 KV-cache support
for ~60x speedup over the original implementation.

See ``docs/architecture/05-decoding-engine.md`` (TBD) and
``docs/guides/06-run-decoding.md``.
"""