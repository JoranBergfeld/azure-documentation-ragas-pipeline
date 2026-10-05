# Faithfulness Calibration + Drift Canary Implementation Plan

**Goal:** Harden the online faithfulness gate (issue #10): pin RAGAS, add pure
calibration machinery that tracks false-pass vs false-abstain separately, add a
frozen drift canary re-scored by both judges on a schedule, log per-claim
verdicts, and document that faithfulness is a grounding heuristic — without
changing the live gate decision.

**Architecture:** Two pure, deterministic modules (`ragpipe.calibration`,
`ragpipe.canary`) hold all logic and are the tested seam; live judge wiring lives
in `scripts/` behind `# pragma: no cover`. Frozen JSON/JSONL artifacts under
`data/` pin the operating point and the canary set. A scheduled workflow runs the
canary. `guardrail.decide_next` / `workflow.run_pipeline` are untouched.

**Tech Stack:** Python 3.11, RAGAS 0.4.3 (pinned), pytest (`asyncio_mode=auto`),
`uv`, ruff (line-length 100). Repo runs under WSL (Linux); Windows `uv` fails on
native build deps.

**Spec:** `docs/superpowers/specs/2026-06-24-faithfulness-calibration-design.md`
**Decision:** `docs/adr/0018-calibrated-faithfulness-gate.md`

**Conventions to honor:**
- `from __future__ import annotations` at the top of every new module.
- Pure logic is unit-tested with fakes/no network; live judge calls stay `# pragma: no cover`.
- Commit trailer: `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`.

---

## File Structure

**Create:**
- `src/ragpipe/calibration.py` — threshold sweep, separate FP/FA rates, recommender.
- `src/ragpipe/canary.py` — drift verdict + per-claim verdict parsing/extraction.
- `data/faithfulness_canary.jsonl` — frozen canary set (4 faithful + 4 unfaithful).
- `data/faithfulness_calibration.json` — pinned versions/threshold artifact.
- `data/faithfulness_calibration_set.example.jsonl` — labeled-set schema example.
- `scripts/calibrate_threshold.py` — live: fit + pin the threshold.
- `scripts/faithfulness_canary.py` — live: re-score canary, exit non-zero on drift.
- `.github/workflows/faithfulness-canary.yml` — scheduled, gated on `CANARY_ENABLED`.
- `tests/test_calibration.py`, `tests/test_canary.py`.
- `docs/adr/0018-calibrated-faithfulness-gate.md`, this spec + plan.

**Modify:**
- `pyproject.toml` — pin `ragas==0.4.3`.
- `.gitignore` — ignore `faithfulness_canary_report.json`.
- `README.md`, `.env.example` — grounding-heuristic note + calibration/canary docs.

---

## Tasks

- [x] Pin `ragas==0.4.3` in `pyproject.toml`.
- [x] `ragpipe.calibration` + `tests/test_calibration.py` (FP/FA separation, recommender, budget).
- [x] `ragpipe.canary` + `tests/test_canary.py` (drift verdict, claim parsing, frozen-file load).
- [x] Frozen data artifacts (canary, calibration pin, labeled example).
- [x] Live scripts (`calibrate_threshold.py`, `faithfulness_canary.py`).
- [x] Scheduled canary workflow gated on `CANARY_ENABLED`.
- [x] ADR-0018 + README + `.env.example` docs.
- [x] `uv run pytest -q` + `uv run ruff check .` green (via WSL).
