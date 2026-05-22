"""
Minimal FastAPI server for Agent6's web trace UI.

Run from the code/ directory:
    uvicorn agent_server:app --port 8200

Or from the project root:
    uvicorn code.agent_server:app --port 8200

Browser auto-opens at http://localhost:8200 on startup.

Routes:
  GET  /                        serve agent_ui.html
  POST /run  {"query": "..."}   start agent run, returns {"run_id": "..."}
  GET  /run/{run_id}/stream     SSE stream of trace events
"""

from __future__ import annotations

import asyncio
import json
import uuid
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

import agent6
import events as ev

HTML_PATH = Path(__file__).parent / "agent_ui.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    webbrowser.open("http://localhost:8200")
    yield


app = FastAPI(title="Agent6 UI", lifespan=lifespan)


class RunRequest(BaseModel):
    query: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PATH.read_text(encoding="utf-8")


@app.post("/run")
async def start_run(req: RunRequest):
    """Start an agent run and return a run_id for SSE streaming."""
    run_id = uuid.uuid4().hex[:8]
    ev.create_queue(run_id)
    asyncio.create_task(agent6.run(req.query, run_id=run_id))
    return {"run_id": run_id}


@app.get("/run/{run_id}/stream")
async def stream_run(run_id: str):
    """SSE endpoint: streams trace events until the run completes."""
    q = ev.get_queue(run_id)
    if q is None:
        raise HTTPException(404, f"run '{run_id}' not found or already completed")

    async def generate():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=120.0)
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'agent timed out'})}\n\n"
                    break
                yield f"data: {json.dumps(event, default=str)}\n\n"
                if event.get("type") == "done":
                    break
        finally:
            ev.close_queue(run_id)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
