# Architecture Decision Records

Significant design decisions in this repository are recorded here, following the
[Nygard ADR format](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions):
**Context** (the forces at play), **Decision** (what we chose and why), and
**Consequences** (what becomes easier/harder), plus a mandatory **Sources** section
citing the papers, documentation, or measurements that justify the choice.

Conventions:

- Sequential numbering: `NNNN-short-kebab-title.md`.
- Status is one of `Proposed`, `Accepted`, `Superseded by NNNN`. Decisions recorded
  after the fact (already in code before this log existed) are marked
  `Accepted (backfilled)`.
- Rejected alternatives are listed with the reason they lost — that's usually the
  most useful part later.
- ADRs are written in the same change set as the decision they record.
