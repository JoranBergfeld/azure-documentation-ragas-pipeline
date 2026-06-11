from ragpipe.guardrail import LoopDecision, decide_next


def test_passes_when_score_meets_threshold():
    d = decide_next(score=0.8, threshold=0.7, attempt=0, max_retries=2)
    assert d is LoopDecision.PASS


def test_retries_when_below_threshold_and_attempts_remain():
    d = decide_next(score=0.5, threshold=0.7, attempt=0, max_retries=2)
    assert d is LoopDecision.RETRY


def test_exhausted_when_below_threshold_and_no_attempts_left():
    d = decide_next(score=0.5, threshold=0.7, attempt=2, max_retries=2)
    assert d is LoopDecision.EXHAUSTED


def test_judge_failure_exhausts_immediately():
    # fail-closed AND fail-fast: regeneration cannot fix judge infrastructure,
    # so a missing score abstains without burning generator/rerank retries.
    assert decide_next(score=None, threshold=0.7, attempt=0, max_retries=2) is LoopDecision.EXHAUSTED
    assert decide_next(score=None, threshold=0.7, attempt=2, max_retries=2) is LoopDecision.EXHAUSTED
