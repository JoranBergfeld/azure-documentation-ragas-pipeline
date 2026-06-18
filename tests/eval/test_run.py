import json

from ragpipe.eval.run import build_per_mode_payload, per_mode_filename


def test_per_mode_filename_uses_mode_value_suffix():
    assert per_mode_filename("contextual") == "eval_results_contextual.json"
    assert per_mode_filename("raptor_sac") == "eval_results_raptor_sac.json"
    assert per_mode_filename("combined") == "eval_results_combined.json"


def test_build_per_mode_payload_is_self_contained_and_tagged():
    mode_result = {
        "means": {"faithfulness": 0.9},
        "means_by_tag": {"original": {"faithfulness": 0.9}},
        "coverage": {"faithfulness": {"valid": 33, "total": 33}},
        "records": [{"query": "q"}],
    }
    payload = build_per_mode_payload("graphrag", mode_result)

    assert payload["mode"] == "graphrag"
    assert payload["means"] == {"faithfulness": 0.9}
    assert payload["means_by_tag"] == {"original": {"faithfulness": 0.9}}
    assert payload["coverage"] == {"faithfulness": {"valid": 33, "total": 33}}
    assert payload["records"] == [{"query": "q"}]
    # the source dict must not be mutated (no stray "mode" key leaks back)
    assert "mode" not in mode_result


def test_per_mode_file_round_trips_back_to_modes_map_entry(tmp_path):
    # The resume path reloads a per-mode file and strips the top-level "mode" key
    # to reconstruct that mode's entry in the combined file's `modes` map. That
    # reconstruction must be loss-free, otherwise a resumed run would emit a
    # different combined eval_results.json than an uninterrupted one.
    mode_result = {
        "means": {"faithfulness": 0.9, "answer_relevancy": 0.8},
        "means_by_tag": {"original": {"faithfulness": 0.9}},
        "coverage": {"faithfulness": {"valid": 33, "total": 33}},
        "records": [{"query": "q", "metrics": {"faithfulness": 0.9}}],
    }
    path = tmp_path / per_mode_filename("graphrag")
    path.write_text(json.dumps(build_per_mode_payload("graphrag", mode_result)))

    cached = json.loads(path.read_text())
    reconstructed = {k: v for k, v in cached.items() if k != "mode"}

    assert reconstructed == mode_result
    # means_by_mode in the combined file is exactly each mode's means.
    assert reconstructed["means"] == mode_result["means"]
