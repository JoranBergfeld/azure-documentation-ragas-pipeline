from __future__ import annotations

import json

from scripts.plot_eval_results import RAGAS_METRICS, load_means


def test_load_means_selects_only_ragas_metrics(tmp_path):
    (tmp_path / "eval_results_contextual.json").write_text(
        json.dumps(
            {
                "mode": "contextual",
                "means": {
                    "faithfulness": 0.9,
                    "answer_relevancy": 0.8,
                    "context_precision": 0.86,
                    "context_recall": 0.81,
                    "mrr@fused": 0.5,
                    "abstained": 0.03,
                },
            }
        )
    )
    means = load_means(["contextual", "baseline"], tmp_path)

    # baseline file is absent -> skipped, not an error
    assert set(means) == {"contextual"}
    # only the 4 core RAGAS metrics are kept, retrieval/abstention dropped
    assert set(means["contextual"]) == set(RAGAS_METRICS)
    assert means["contextual"]["faithfulness"] == 0.9


def test_load_means_tolerates_missing_metric(tmp_path):
    (tmp_path / "eval_results_graphrag.json").write_text(
        json.dumps({"mode": "graphrag", "means": {"faithfulness": 0.74}})
    )
    means = load_means(["graphrag"], tmp_path)
    assert means["graphrag"]["faithfulness"] == 0.74
    assert means["graphrag"]["context_recall"] is None
