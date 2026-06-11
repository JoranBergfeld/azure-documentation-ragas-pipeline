from ragpipe.eval.synthetic import (
    content_word_overlap,
    make_candidates,
    parse_candidates,
)

DOC = (
    "Semantic ranking in Azure AI Search re-scores an initial result set "
    "using deep learning models to improve relevance of the top results."
)


def test_overlap_high_for_verbatim_question():
    q = "How does semantic ranking re-score the initial result set?"
    assert content_word_overlap(q, DOC) > 0.7


def test_overlap_low_for_user_phrased_question():
    q = "Can the engine make my best hits float upward automatically?"
    assert content_word_overlap(q, DOC) < 0.4


def test_parse_candidates_tolerates_fenced_json():
    raw = 'Here you go:\n```json\n[{"question": "q1", "ground_truth": "a1"}]\n```'
    assert parse_candidates(raw) == [{"question": "q1", "ground_truth": "a1"}]


def test_parse_candidates_drops_malformed_entries():
    raw = '[{"question": "q1"}, {"question": "q2", "ground_truth": "a2"}]'
    assert parse_candidates(raw) == [{"question": "q2", "ground_truth": "a2"}]


def test_make_candidates_screens_verbatim_and_stamps_provenance():
    def fake_complete(prompt):
        return (
            '[{"question": "How does semantic ranking re-score the initial '
            'result set using deep learning?", "ground_truth": "It re-scores."},'
            ' {"question": "Can my best hits float upward automatically?",'
            ' "ground_truth": "Yes, via semantic ranking."}]'
        )

    rows = make_candidates(fake_complete, url="http://learn/sem", document=DOC, n=2)
    # the verbatim-echo question is screened out; the user-phrased one survives
    assert len(rows) == 1
    assert rows[0]["question"].startswith("Can my best hits")
    assert rows[0]["ground_truth_context"] == "http://learn/sem"
    assert rows[0]["tags"] == ["synthetic"]
