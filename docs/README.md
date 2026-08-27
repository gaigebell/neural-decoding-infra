# Documentation Index

> **Audience**: Anyone working on this project (current or future).
> **Principle**: Every important piece of knowledge lives in exactly one place. This file is the entry point.

---

## How to read this documentation

This documentation follows the [Diátaxis framework](https://diataxis.fr/):

| Section | Question it answers | When to read |
|---|---|---|
| **[tutorials/](tutorials/)** | "How do I learn the design method?" | When you want to master a skill, not just do a task |
| **[architecture/](architecture/)** | "Why is the system designed this way?" | When you need to understand a component |
| **[decisions/](decisions/)** | "What did we decide and what were the alternatives?" | When you question a design choice |
| **[standards/](standards/)** | "How should I write code/docs?" | Before writing code or docs |
| **[guides/](guides/)** | "How do I do task X?" | When you need to perform a task |
| **[dev-logs/](dev-logs/)** | "What happened recently?" | Weekly review |
| **[research/](research/)** | "What is the scientific context?" | When context is needed |

---

## Quick links

### 🎓 Tutorials
- [Data pipeline design: from assumptions to industrial scale](tutorials/01-data-pipeline-design.md)

### 🏗️ Architecture
- [System overview](architecture/01-overview.md)
- [Data pipeline](architecture/02-data-pipeline.md)
- [Model registry](architecture/03-model-registry.md)
- [Training engine](architecture/04-training-engine.md)
- [Decoding engine](architecture/05-decoding-engine.md)
- [Evaluation](architecture/06-evaluation.md)

### ⚖️ Decisions (ADRs)
- [All ADRs](decisions/)
- [ADR template](decisions/README.md)

### 📐 Standards
- [Python style](standards/01-python-style.md)
- [Docstring style](standards/02-docstring.md)
- [Git workflow](standards/03-git-workflow.md)
- [Testing](standards/04-testing.md)
- [Configuration](standards/05-configuration.md)
- [Documentation](standards/06-documentation.md)
- [Dependencies](standards/07-dependencies.md)

### 📖 Guides
- [Set up dev/cluster environment](guides/01-setup-env.md)
- [Write a data adapter](guides/02-write-data-adapter.md)
- [Write a new model](guides/03-write-model.md)
- [Run training](guides/04-run-training.md)
- [Launch multi-node training](guides/05-launch-multi-node.md)
- [Run decoding](guides/06-run-decoding.md)
- [Run evaluation](guides/07-run-evaluation.md)
- [Debug checklist](guides/08-debug-checklist.md)
- [**用户手册：训练与解码全流程**](guides/10-user-manual.md)（当前用法的唯一入口，含配置字段大全）

### 📅 Development logs
- [Week 0 (2026-07-28): Bootstrap](dev-logs/2026-07-28-week-00-bootstrap.md)

### 🔬 Research context
- [Project overview](research/01-project-overview.md)
- [Data card](research/02-data-card.md)
- [Cluster card](research/03-cluster-card.md)

---

## Conventions for THIS documentation

1. **One concept per file.** If a file is doing two things, split it.
2. **Use links liberally.** Prefer `[link text](relative/path.md)` over copy-paste.
3. **Diátaxis discipline.** Don't put "how-to" content in "reference" files or vice versa.
4. **Files start with a status block** (see [standards/06-documentation.md](standards/06-documentation.md)).
5. **Cross-link related docs.** If a guide references an architecture, link it.

---

## Adding new documentation

When adding a new doc:
1. Decide which section it belongs to (architecture / decisions / standards / guides / dev-logs / research).
2. Use the appropriate template (see each section's README.md).
3. Update this index file with the new link.
4. Reference it from any related docs.

---

Last updated: 2026-08-20 by owner.