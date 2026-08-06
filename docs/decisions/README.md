# Architecture Decision Records (ADRs)

> **Audience**: Everyone. Read before questioning a design choice.

---

## What is an ADR?

An **Architecture Decision Record** captures a single significant design decision: its context, what was chosen, and what the consequences are. They are immutable once accepted — superseded decisions link to their replacements.

This project follows [Michael Nygard's ADR template](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions), lightly adapted.

## When to write an ADR

Write an ADR when:

- ✅ Choosing between two or more plausible approaches (any "we could do X or Y").
- ✅ Adopting a new framework, library, or major dependency.
- ✅ Changing a previously-accepted design decision.
- ✅ Disagreeing with an industry-standard practice.

You do NOT need an ADR for:

- ❌ Bug fixes.
- ❌ Pure refactors with no behavioral change.
- ❌ Decisions reversible in <1 day.

## Template

Copy this into a new file `NNNN-short-title.md`:

```markdown
# ADR-NNNN: <Short Title>

- **Status**: [Proposed | Accepted | Deprecated | Superseded by ADR-XXXX]
- **Date**: YYYY-MM-DD
- **Deciders**: owner (and anyone else involved)

## Context and problem statement

[Describe the forces at play, including the technological, business, political, social,
and local context. What is the issue we're seeing? What constraints exist?]

## Decision drivers

- [Driver 1: e.g., performance, simplicity, team familiarity]
- [Driver 2]
- ...

## Considered options

1. **Option A** — [brief description]
2. **Option B** — [brief description]
3. **Option C** — [brief description]

## Decision outcome

**Chosen option**: "[Option X]", because [reasoning].

### Consequences

- ✅ Good: [1-3 bullets]
- ❌ Bad: [1-3 bullets]
- ❓ Unknown / risks: [1-3 bullets]

## Pros and cons of the options

### Option A
- ✅ Pro: ...
- ❌ Con: ...

### Option B
- ✅ Pro: ...
- ❌ Con: ...

## Links

- Related ADRs: [ADR-XXXX](./XXXX-...)
- Related docs: [link]
- External references: [URLs]
```

## Index

| ADR | Title | Status |
|---|---|---|
| [0001](0001-why-hydra.md) | Why Hydra over plain YAML | Accepted |
| [0002](0002-pydantic-schema.md) | Pydantic schemas for data | Accepted |
| [0003](0003-drop-slurm.md) | Drop SLURM as the scheduler | Accepted |
| [0004](0004-manual-ssh-launch.md) | Manual ssh for multi-node launch | Accepted |
| [0005](0005-no-local-gpu-test.md) | No local GPU testing | Accepted |
| [0006](0006-unified-trainer.md) | Unified Trainer (over per-model scripts) | Accepted |

## Conventions

- **Filename**: `NNNN-kebab-case-title.md` (zero-padded 4-digit number)
- **Numbering**: monotonic, never reuse
- **Immutability**: accepted ADRs do NOT change. Create a new ADR to supersede.
- **Date**: use the date the decision was made, not when discussed

---

Last updated: 2026-07-28 by owner.