from __future__ import annotations

import asyncio
from typing import AsyncIterator


async def pipeline_event_stream(pipeline_fn, query: str) -> AsyncIterator[tuple[str, object]]:
    """Bridge ``run_pipeline``'s synchronous ``on_event`` sink to an async iterator.

    Yields ``("progress", ProgressEvent)`` for each emitted event, then exactly one
    terminal ``("result", state)`` on success or ``("error", exc)`` on failure.
    The queue and sink are created per call, so concurrent callers are isolated."""
    queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    def sink(event) -> None:
        queue.put_nowait(("progress", event))

    async def runner() -> None:
        try:
            state = await pipeline_fn(query, on_event=sink)
            queue.put_nowait(("result", state))
        except Exception as exc:  # noqa: BLE001 - surfaced to the client as an error frame
            queue.put_nowait(("error", exc))
        finally:
            queue.put_nowait(_DONE)

    task = asyncio.create_task(runner())
    try:
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            yield item
    finally:
        await task
