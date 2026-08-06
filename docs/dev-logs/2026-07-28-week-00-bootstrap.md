# Week 00 — 2026-07-28 ~ 2026-07-28

> **Theme**: Bootstrap — repo setup, architecture design, and documentation foundation.

## Goals

- [x] Create the new repo `neural-decoding-infra`
- [x] Establish industry-standard docs/ structure
- [x] Write all 6 architecture decisions as ADRs
- [x] Write 6 development standards
- [x] Write 8 operational guides
- [x] Write 3 research context docs (project, data, cluster)
- [ ] Write supporting files (LICENSE, CONTRIBUTING, Makefile, .gitignore, pyproject.toml)

## Completed ✅

- [x] Created GitHub private repo `neural-decoding-infra` ([d:\allforwork\Liu_Lab\_Reconstruction\recon\neural-decoding-infra](../../))
- [x] Wrote `README.md` with quickstart + docs map
- [x] Wrote `docs/README.md` as documentation index (Diátaxis-based)
- [x] Wrote `docs/research/01-project-overview.md` (research context)
- [x] Wrote `docs/research/02-data-card.md` (12 subjects, 3 modalities, BIDS structure)
- [x] Wrote `docs/research/03-cluster-card.md` (4 working nodes, 8 GPU total)
- [x] Wrote `docs/architecture/01-overview.md` (system architecture + data flow)
- [x] Wrote 6 ADRs:
  - [ADR-0001](../decisions/0001-why-hydra.md) — Why Hydra
  - [ADR-0002](../decisions/0002-pydantic-schema.md) — Pydantic schemas
  - [ADR-0003](../decisions/0003-drop-slurm.md) — Drop SLURM
  - [ADR-0004](../decisions/0004-manual-ssh-launch.md) — Manual ssh launch
  - [ADR-0005](../decisions/0005-no-local-gpu-test.md) — No local GPU testing
  - [ADR-0006](../decisions/0006-unified-trainer.md) — Unified Trainer
- [x] Wrote 6 standards (Python, docstring, git, testing, config, docs)
- [x] Wrote 8 guides (setup, data adapter, model, training, multi-node, decoding, eval, debug)
- [x] Wrote first dev log (this file)

## In Progress 🚧

- [ ] Supporting files (LICENSE, CONTRIBUTING, Makefile, etc.)
- [ ] `recon/` package skeleton
- [ ] First commit push

## Blockers 🚨

- None at this point. Documenting the plan took time but the path forward is clear.

## Next Week

- [ ] Week 1 (Day 1-2): Groundwork — package skeleton, config structure, pydantic schema
- [ ] Week 1 (Day 3-4): Model registry + first fMRI model + first MEG model
- [ ] Week 1 (Day 5): Decoding speedup (batched + KV cache)
- [ ] Week 1 (Day 6-7): Evaluator + first cluster smoke test (4 nodes manual ssh)
- [ ] First baseline 12-subject training run

## Notes / Lessons 💡

- **SLURM is not coming back.** Diagnostic in July 2026 showed `slurmd` started but `slurmctld` rejects it (munge key / config issue). System too old to reinstall. Pivoted to manual ssh launches via thin wrapper script. See [ADR-0003](../decisions/0003-drop-slurm.md) and [ADR-0004](../decisions/0004-manual-ssh-launch.md).
- **Local GPU testing is dead.** Development machines are personal laptops with heterogeneous OS/Python/GPU. Owner's RTX 4060 can't run fMRI 3D CNN smoke (OOM). Pivoted to "all smoke tests on cluster". See [ADR-0005](../decisions/0005-no-local-gpu-test.md).
- **4 working nodes, not 6 or 10.** Of the 11 nodes in `/etc/hosts`, only `cn3`, `gn14`, `gn15`, `gn16` are reachable. Total compute: 8 PH402 GPUs (Pascal, 32G each). No InfiniBand → NCCL over TCP.
- **Data location finally clarified.** All data and code live on `test@mgmt:/home/reconstruction/` (NFS-exported to compute nodes). Development machine has its own copy on `E:/reconstruction/` for browsing only — not for training.
- **The team is me, alone.** 4 new members (psych/life-sci/AI/math) don't have the HPC expertise to deal with the cluster. They will eventually contribute to model variants / evaluation, but the infra build is solo.
- **Documentation first, code second.** After this week, every future code change has a doc home. This is the difference between "research scripts" and "industrial framework".

## Metrics / Artifacts

- Documentation files written: **22**
- Lines of docs: ~3,500
- ADRs: 6
- Standards: 6
- Guides: 8
- Architecture docs: 1 (overview; rest pending)
- Research docs: 3
- Code files written: **0** (next week)

## Reflections

This week was deliberately slow on code, fast on documentation. The reason:

- Future-me will look at this repo in 6 months and ask "why is it this way?"
- New team members (when they join the actual code review) need a clear entry point
- Decisions made under stress (during debugging) are best captured when calm

The cost is one week without code. The benefit is a foundation that scales.

---

End of week 00. Next: [Week 01 →](2026-08-04-week-01-data-schema.md) (to be written next week).