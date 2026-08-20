"""Command-line entry points.

Three subcommands:
    - train: Run training (single-GPU or DDP)
    - decode: Run inference on one story
    - eval: Compute metrics on decoded outputs

Each is implemented as a separate module under ``recon.cli`` and exposes
a ``main()`` function for ``python -m recon.cli.<name>``.

Hydra config is composed from ``configs/`` — see ``docs/architecture/01-overview.md``
and ``docs/standards/05-configuration.md``.
"""