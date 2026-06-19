"""Render docs/eval-results.svg from the committed per-mode eval files.

Reads each eval_results_<mode>.json (the standalone, committed reference result
for that substrate; ADR-0016) and draws one grouped bar chart of the four core
RAGAS metrics across the evaluated modes. matplotlib emits SVG directly, so no
rasterizer is needed. Agentic modes are retrieval wrappers with no eval files
and are therefore not shown.

Run: uv run python scripts/plot_eval_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
EVAL_MODES = ["contextual", "baseline", "raptor_sac", "graphrag", "combined"]
METRIC_LABELS = {
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer relevancy",
    "context_precision": "Context precision",
    "context_recall": "Context recall",
}
METRIC_COLORS = {
    "faithfulness": "#2E9E4B",
    "answer_relevancy": "#0078D4",
    "context_precision": "#8661C5",
    "context_recall": "#E3B341",
}


def load_means(modes: list[str], results_dir: Path) -> dict[str, dict[str, float | None]]:
    """For each mode with an eval_results_<mode>.json, return its 4 core RAGAS means.

    Modes whose file is absent are skipped. A metric missing from a file's `means`
    maps to None so the caller can decide how to render it.
    """
    out: dict[str, dict[str, float | None]] = {}
    for mode in modes:
        path = results_dir / f"eval_results_{mode}.json"
        if not path.exists():
            continue
        means = json.loads(path.read_text()).get("means", {})
        out[mode] = {m: means.get(m) for m in RAGAS_METRICS}
    return out


def build_figure(means_by_mode: dict[str, dict[str, float | None]]):  # pragma: no cover - rendering
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = [m for m in EVAL_MODES if m in means_by_mode]
    width = 0.8 / len(RAGAS_METRICS)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for i, metric in enumerate(RAGAS_METRICS):
        offsets = [xi - 0.4 + width * (i + 0.5) for xi in range(len(modes))]
        values = [means_by_mode[m].get(metric) or 0.0 for m in modes]
        bars = ax.bar(
            offsets, values, width=width, label=METRIC_LABELS[metric], color=METRIC_COLORS[metric]
        )
        for rect, v in zip(bars, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                v + 0.012,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#1B2733",
            )

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score (higher is better)")
    ax.set_title("RAGAS evaluation across retrieval modes", fontweight="bold", color="#1B2733")
    ax.yaxis.grid(True, color="#E3E9EF")
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return fig


def main() -> None:  # pragma: no cover - file IO entrypoint
    root = Path(__file__).resolve().parent.parent
    means = load_means(EVAL_MODES, root)
    if not means:
        raise SystemExit("No eval_results_<mode>.json files found; run the eval harness first.")
    out = root / "docs" / "eval-results.svg"
    build_figure(means).savefig(out, format="svg", bbox_inches="tight")
    print(f"wrote {out} for modes: {', '.join(means)}")


if __name__ == "__main__":
    main()
