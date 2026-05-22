"""
Lightweight asyncio event bus that bridges agent6's trace into the SSE stream.

agent6.run() calls events.emit(run_id, payload) at each trace point.
agent_server.py registers a queue per run and reads from it for SSE.

emit() is a silent no-op when no queue is registered for a run_id, so
agent6 continues to work as a standalone CLI tool without any changes
to its external behaviour.
"""

from __future__ import annotations

import asyncio
from typing import Any

# run_id → asyncio.Queue
_queues: dict[str, asyncio.Queue] = {}


def create_queue(run_id: str) -> asyncio.Queue:
    """Create and register a new event queue for a run.
    Called by agent_server before launching the agent task."""
    q: asyncio.Queue = asyncio.Queue()
    _queues[run_id] = q
    return q


def emit(run_id: str, event: dict[str, Any]) -> None:
    """Emit an event for a run. Silent no-op if no queue is registered."""
    q = _queues.get(run_id)
    if q is not None:
        q.put_nowait(event)


def get_queue(run_id: str) -> asyncio.Queue | None:
    """Return the event queue for a run, or None if not registered."""
    return _queues.get(run_id)


def close_queue(run_id: str) -> None:
    """Deregister and discard the event queue for a completed run."""
    _queues.pop(run_id, None)
