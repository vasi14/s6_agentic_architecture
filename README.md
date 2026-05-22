# Session 6 — Agentic Architecture

A from-scratch four-role agent that answers multi-step queries through a typed
Perception → Decision → Action loop backed by durable memory and an artifact
store. No third-party agent framework is used; every boundary between roles is
enforced by Pydantic v2 models. All LLM calls are routed through **LLM Gateway
V3** running locally on port 8101.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Layout](#project-layout)
4. [Setup](#setup)
5. [Running the Agent](#running-the-agent)
6. [Target Queries](#target-queries)
   - [Query A — Claude Shannon (artifact attach)](#query-a--claude-shannon-artifact-attach)
   - [Query B — Tokyo weekend activities (multi-goal + memory carryover)](#query-b--tokyo-weekend-activities-multi-goal--memory-carryover)
   - [Query C — Mom's birthday (durable memory across runs)](#query-c--moms-birthday-durable-memory-across-runs)
   - [Query D — Python asyncio best practices (multi-source synthesis)](#query-d--python-asyncio-best-practices-multi-source-synthesis)
7. [Output Logs](#output-logs)
8. [Resetting State](#resetting-state)

---

## Overview

The agent receives a free-text query and works through it iteratively. Each
iteration runs the four cognitive roles in sequence:

```
Memory.read  →  Perception  →  Decision  →  Action  →  Memory.record_outcome
```

Iteration continues until Perception marks every goal as `done` or the hard cap
of **10 iterations** is reached. Large tool results are stored in a content-hash
artifact store and attached to the relevant goal prompt rather than being
re-fetched.

---

## Architecture

### Four Cognitive Roles

| Role | Module | Responsibility | LLM call? |
|---|---|---|---|
| **Memory** | `code/memory.py` | Typed persistence — facts, preferences, tool outcomes, scratchpad. Exposes `read`, `filter`, `relevant`, `remember`, `record_outcome`. | Only for ambiguous `remember` classifications |
| **Perception** | `code/perception.py` | Reads query + memory hits + history, emits a ranked goal list with `done` flags and optional `attach_artifact_id`. | Yes — routed `auto_route="perception"` |
| **Decision** | `code/decision.py` | Picks the next action for the first open goal: either `ANSWER` (plain text) or a single `TOOL_CALL`. | Yes — routed `auto_route="decision"` |
| **Action** | `code/action.py` | Dispatches MCP tool calls via stdio transport. Stores large payloads in the artifact store; returns a short descriptor. | None |

### Pydantic Schemas (`code/schemas.py`)

All data crossing role boundaries is typed:

| Model | Used by |
|---|---|
| `MemoryItem` | Memory ↔ Perception, Memory ↔ Decision |
| `Artifact` | Action → ArtifactStore, Perception attach |
| `Goal` | Perception → Decision |
| `Observation` | Perception output |
| `ToolCall` | Decision → Action |
| `DecisionOutput` | Decision output |
| `PerceptionLLMOutput` | Typed parse of Perception LLM response |
| `DecisionLLMOutput` | Typed parse of Decision LLM response |
| `MemoryClassifyOutput` | Typed parse of Memory classification response |

### MCP Tool Server (`code/mcp_server.py`)

Nine tools exposed over **stdio transport**:

| Tool | Description |
|---|---|
| `web_search` | Tavily primary, DuckDuckGo fallback; capped at 5 results |
| `fetch_url` | crawl4ai — returns clean markdown via headless Chromium |
| `get_time` | Current time with timezone support |
| `currency_convert` | Live exchange rates |
| `read_file` | Read files sandboxed under `code/sandbox/` |
| `list_dir` | List directory contents under sandbox |
| `create_file` | Create new files in sandbox |
| `update_file` | Overwrite a file in sandbox |
| `edit_file` | Patch file contents in sandbox |

File tools are sandboxed to `code/sandbox/` — paths that escape the sandbox
raise a `ValueError`.

### LLM Gateway V3 (`llm_gatewayV3/`)

A local FastAPI service on **port 8101** that routes calls across seven free
provider tiers (Gemini, Groq, NVIDIA NIM, Cerebras, OpenRouter, GitHub Models,
Ollama) with automatic failover. The `auto_route` field enables per-cognitive-
layer routing:

- `auto_route="perception"` → pinned to Gemini
- `auto_route="decision"` → capability-aware worker pool
- `auto_route="memory"` → Gemini (classification tasks)

### Artifact Store (`code/artifacts.py`)

Large or raw tool results (HTML pages, JSON blobs, search result sets) are
stored as content-hash files under `code/state/artifacts/`. Each artifact has:

- `{id}.bin` — raw bytes
- `{id}.meta.json` — `Artifact` metadata (content type, size, source, descriptor)

IDs have the form `art:<sha256-prefix>`. Perception detects when an open
synthesis goal has a relevant artifact in memory and sets `attach_artifact_id`,
causing the loop to load and inject the bytes into Decision's prompt.

### Iteration Loop (`code/agent6.py`)

```
run_id = uuid4 short
memory.remember(query)                  # classify user input once

for iteration in 1..MAX_ITERATIONS:
    hits = memory.read(query, history)  # keyword-overlap search, no LLM
    obs  = perception.observe(...)      # Perception LLM call
    if all goals done → break

    goal = first open goal
    if goal.attach_artifact_id:
        attached = artifact_store.get(...)
    
    out = decision.next_step(goal, hits, attached, history, tools)

    if out.answer:
        history.append(answer); continue
    
    descriptor, artifact_id = await action.execute(session, out.tool_call)
    memory.record_outcome(...)          # no LLM, pure keyword index
    history.append(action_record)

return final_answer_from(history, query)
```

---

## Project Layout

```
s6_agentic_architecture/
├── code/
│   ├── agent6.py          # Orchestration loop
│   ├── memory.py          # Memory role
│   ├── perception.py      # Perception role
│   ├── decision.py        # Decision role
│   ├── action.py          # Action role
│   ├── artifacts.py       # Content-hash artifact store
│   ├── gateway.py         # LLM gateway client + MCP session adapters
│   ├── mcp_server.py      # FastMCP stdio tool server (9 tools)
│   ├── schemas.py         # All Pydantic v2 models
│   ├── agent_server.py    # FastAPI SSE server for the web UI
│   ├── agent_ui.html      # Browser UI
│   ├── events.py          # SSE event types
│   ├── state/
│   │   ├── memory.json    # Durable memory (persists across runs)
│   │   └── artifacts/     # Artifact blobs
│   ├── sandbox/           # Sandboxed file I/O for MCP file tools
│   └── runs/              # Per-run JSONL trace logs
├── llm_gatewayV3/
│   ├── main.py            # FastAPI app (port 8101)
│   ├── client.py          # Python LLM client
│   ├── router.py          # Auto-routing logic
│   ├── providers.py       # Provider adapters
│   └── run.sh             # Start gateway
├── notes/
│   ├── problem_statement.md
│   ├── target_queries.md
│   └── run2/              # Recorded HTML output traces
├── pyproject.toml
└── README.md
```

---

## Setup

### Prerequisites

- Python ≥ 3.13
- [uv](https://docs.astral.sh/uv/) package manager
- A `.env` file in the project root (or `code/`) with provider API keys

**Minimum `.env` for the four target queries:**

```env
TAVILY_API_KEY=tvly-...          # web_search primary
GEMINI_API_KEY=...               # Perception + Memory LLM calls
# At least one more for Decision routing:
GROQ_API_KEY=...
# or NVIDIA_API_KEY, GITHUB_TOKEN, etc.
```

### Install dependencies

```bash
# From the project root
uv sync
```

### Start LLM Gateway V3

In a separate terminal:

```bash
cd llm_gatewayV3
./run.sh          # Windows: uv run python main.py
```

Verify it is running:

```bash
curl http://localhost:8101/health
```

---

## Running the Agent

All commands are run from the project root with `uv run`.

### Python API

```python
import asyncio
from code.agent6 import run

answer = asyncio.run(run("Your query here"))
print(answer)
```

### Command line (agent_server + UI)

Start the SSE server:

```bash
uv run uvicorn code.agent_server:app --port 8000 --reload
```

Open `code/agent_ui.html` in a browser or navigate to
`http://localhost:8000`.

---

## Target Queries

### Query A — Claude Shannon (artifact attach)

**Query:**
```
Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date,
death date, and three key contributions to information theory.
```

**What it exercises:** The artifact attach path. The fetched Wikipedia page
(~250 KB of markdown) is stored as an artifact. Perception identifies that the
extraction goal needs the bytes, sets `attach_artifact_id`, and the loop injects
the content into Decision's prompt — no re-fetch.

**Expected iteration count:** 3

**How to run:**
```bash
uv run python -c "
import asyncio, sys
sys.path.insert(0, 'code')
from agent6 import run
print(asyncio.run(run('Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory.')))
"
```

**Recorded output log:** [Agent6 · Live Trace 1](notes/run2/Agent6%20·%20Live%20Trace%201.html)

---

### Query B — Tokyo weekend activities (multi-goal + memory carryover)

**Query:**
```
Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's
weather forecast there and tell me which one is most appropriate.
```

**What it exercises:** Three logical goals — search activities, fetch Saturday's
weather, select the best activity. The weather fact written by Action in
iteration 2 is carried into Decision's context in iteration 3 through the memory
keyword search.

**Expected iteration count:** 3–4 (up to 6 with search refinement)

**How to run:**
```bash
uv run python -c "
import asyncio, sys
sys.path.insert(0, 'code')
from agent6 import run
print(asyncio.run(run('Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday\'s weather forecast there and tell me which one is most appropriate.')))
"
```

**Recorded output logs:**
- [Agent6 · Live Trace 2](notes/run2/Agent6%20·%20Live%20Trace%202.html)

---

### Query C — Mom's birthday (durable memory across runs)

**Run 1 query:**
```
My mom's birthday is 15 May 2026. Remember that and give me a calendar reminder
for two weeks before and on the day.
```

**Run 2 query (separate execution):**
```
When is mom's birthday?
```

**What it exercises:** Durable memory. Run 1 calls `memory.remember(...)` which
classifies the statement and persists a `fact` item with keywords `[mom,
birthday, may, 2026]` to `code/state/memory.json`. Run 2 (cold start, same
`state/` directory) finds the fact through keyword search and answers without
any tool call.

**Expected iteration count:** 4 for run 1, 2 for run 2

**How to run:**

Run 1:
```bash
uv run python -c "
import asyncio, sys
sys.path.insert(0, 'code')
from agent6 import run
print(asyncio.run(run('My mom\'s birthday is 15 May 2026. Remember that and give me a calendar reminder for two weeks before and on the day.')))
"
```

Run 2 (do not clear `state/` between runs):
```bash
uv run python -c "
import asyncio, sys
sys.path.insert(0, 'code')
from agent6 import run
print(asyncio.run(run('When is mom\'s birthday?')))
"
```

**Recorded output logs:**
- Run 1: [Agent6 · Live Trace 3a](notes/run2/Agent6%20·%20Live%20Trace%203a.html)
- Run 2: [Agent6 · Live Trace 3b](notes/run2/Agent6%20·%20Live%20Trace%203b.html)

---

### Query D — Python asyncio best practices (multi-source synthesis)

**Query:**
```
Search for 'Python asyncio best practices', read the top 3 results, and give me
a short numbered list of the advice they agree on.
```

**What it exercises:** Multi-artifact synthesis. The agent performs one
`web_search` call followed by three `fetch_url` calls (one per result),
producing three artifacts. Perception's synthesis-keyword safety net detects the
open "synthesise" goal and attaches the most relevant artifact. Decision reads
the attached content and produces a consolidated numbered list.

**Expected iteration count:** 5–7

**How to run:**
```bash
uv run python -c "
import asyncio, sys
sys.path.insert(0, 'code')
from agent6 import run
print(asyncio.run(run(\"Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on.\")))
"
```

**Recorded output logs:**
- [Agent6 · Live Trace 4](notes/run2/Agent6%20·%20Live%20Trace%204.html)
- [Agent6 · Live Trace 4 (GCP variant)](notes/run2/Agent6%20·%20Live%20Trace%204%20gcp.html)

---

## Output Logs

All recorded live traces are stored as self-contained HTML files in
[notes/run2/](notes/run2/). Open them directly in a browser — no server
required.

| File | Query |
|---|---|
| [Agent6 · Live Trace 1](notes/run2/Agent6%20·%20Live%20Trace%201.html) | Query A — Claude Shannon |
| [Agent6 · Live Trace 2](notes/run2/Agent6%20·%20Live%20Trace%202.html) | Query B — Tokyo activities |
| [Agent6 · Live Trace 3a](notes/run2/Agent6%20·%20Live%20Trace%203a.html) | Query C — Mom's birthday (run 1) |
| [Agent6 · Live Trace 3b](notes/run2/Agent6%20·%20Live%20Trace%203b.html) | Query C — Mom's birthday (run 2) |
| [Agent6 · Live Trace 4](notes/run2/Agent6%20·%20Live%20Trace%204.html) | Query D — asyncio synthesis |
| [Agent6 · Live Trace 4 (GCP)](notes/run2/Agent6%20·%20Live%20Trace%204%20gcp.html) | Query D — asyncio synthesis (GCP run) |

Per-run JSONL machine logs are written to `code/runs/<run_id>.jsonl` during
every execution.

---

## Resetting State

To clear durable memory and all artifacts between assignment attempts:

```bash
# Remove memory and artifacts only — keeps sandbox files
Remove-Item code/state/memory.json -ErrorAction SilentlyContinue
Remove-Item code/state/artifacts/* -Recurse -ErrorAction SilentlyContinue
```

Or delete the entire state directory:

```bash
Remove-Item code/state -Recurse -Force
```

The agent recreates `state/memory.json` (empty `[]`) and `state/artifacts/` on
the next run.
