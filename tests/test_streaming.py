from __future__ import annotations

import pytest

from ragpipe.progress import ProgressEvent
from ragpipe.streaming import pipeline_event_stream


@pytest.mark.asyncio
async def test_stream_yields_progress_then_result():
    async def fake_pipeline(query, *, on_event=None):
        on_event(ProgressEvent(phase="retrieve", status="start"))
        on_event(ProgressEvent(phase="generate", status="complete"))
        return {"state": query}

    items = [item async for item in pipeline_event_stream(fake_pipeline, "q")]
    assert [kind for kind, _ in items] == ["progress", "progress", "result"]
    assert items[-1][1] == {"state": "q"}
    assert items[0][1].phase == "retrieve"


@pytest.mark.asyncio
async def test_stream_yields_error_on_pipeline_failure():
    async def boom(query, *, on_event=None):
        raise RuntimeError("down")

    items = [item async for item in pipeline_event_stream(boom, "q")]
    assert items[-1][0] == "error"
    assert isinstance(items[-1][1], RuntimeError)
