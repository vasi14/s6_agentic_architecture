"""
Four-role agent orchestration loop.

Implements the iterative perception-decision-action cycle with:
- Memory carryover between goals and across runs
- Artifact attachment for synthesis goals
- Iteration cap guard for convergence
- Final answer synthesis
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import action
import artifacts
import decision
import memory
import perception
from gateway import ensure_gateway, load_tools, mcp_session, mcp_tools_for_decision
from rich.console import Console
from rich.panel import Panel
from schemas import Goal, Observation
import events

# Configuration
MAX_ITERATIONS = 15  # Hard cap to prevent infinite loops
DEFAULT_EXPECTED_ITERATIONS = 5  # Default expected iteration count

# Query-specific expected iteration bounds
QUERY_BOUNDS = {
    # Query A: Shannon info extraction - fetch + extract
    "shannon": 3,
    "claude shannon": 3,
    # Query B: Tokyo activities + weather - 2 tool calls + answer
    "tokyo": 4,
    # Query C: Memory test - classify + answer
    "mom": 2,
    "birthday": 2,
    # Query D: Consensus search - search + 3 fetches + synthesis
    "consensus": 6,
    "health tip": 6,
}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Rich console for human-readable trace output
console = Console()

# Runs directory for logs
RUNS_DIR = Path(__file__).parent / "runs"
RUNS_DIR.mkdir(exist_ok=True)


def _short(text: str, limit: int = 120) -> str:
    """Return a one-line, truncated preview of text."""
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _trace_iter_header(it: int) -> None:
    """Render iteration banner in expected-trace style."""
    console.print(f"\n[bold cyan]--- iter {it} ---[/bold cyan]")


def _trace_memory(hits: list) -> None:
    """Render memory read step."""
    console.print(f"[bold yellow][memory.read][/bold yellow] {len(hits)} hits")
    for hit in hits[:2]:
        # MemoryItem attributes
        kind = getattr(hit, "kind", "memory")
        descriptor = getattr(hit, "descriptor", "")
        if descriptor:
            console.print(f"  - {kind}: {_short(descriptor, 90)}")


def _trace_perception(obs: Observation) -> None:
    """Render goals emitted by perception."""
    console.print("[bold magenta][perception][/bold magenta]")
    for g in obs.goals:
        status = "done" if g.done else "open"
        console.print(f"  [{status}] {g.text}")
        if g.attach_artifact_id:
            console.print(f"    attach={g.attach_artifact_id}")


def _trace_decision_answer(answer: str) -> None:
    """Render decision answer path."""
    console.print(f"[bold blue][decision][/bold blue] ANSWER: {_short(answer, 220)}")


def _trace_decision_tool(name: str, arguments: dict[str, Any]) -> None:
    """Render decision tool call path."""
    args_text = json.dumps(arguments, ensure_ascii=False)
    console.print(
        f"[bold blue][decision][/bold blue] TOOL_CALL: {name}({_short(args_text, 180)})"
    )


def _trace_action(result_text: str, artifact_id: str | None) -> None:
    """Render action result path."""
    if artifact_id:
        console.print(
            f"[bold green][action][/bold green] -> [artifact {artifact_id}] preview: {_short(result_text, 140)}"
        )
    else:
        console.print(f"[bold green][action][/bold green] -> {_short(result_text, 140)}")


def _trace_layer_flow() -> None:
    """Render four-layer communication path."""
    console.print(
        "[dim]flow: memory -> perception -> decision -> action -> memory (feedback)[/dim]"
    )


def _get_expected_bound(query: str) -> int:
    """Get the expected iteration bound for a query."""
    query_lower = query.lower()
    for key, bound in QUERY_BOUNDS.items():
        if key in query_lower:
            return bound * 2  # Allow 2x expected
    return DEFAULT_EXPECTED_ITERATIONS * 2


def _log_iteration(run_id: str, iteration: int, data: dict) -> None:
    """Log iteration data to the runs directory."""
    log_file = RUNS_DIR / f"{run_id}.jsonl"
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iteration": iteration,
        **data,
    }
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def final_answer_from(history: list[dict], query: str) -> str:
    """
    Synthesize a final answer from the action history.
    
    Collects all intermediate answers and tool results to produce
    a coherent final response.
    """
    # Collect answers and key results
    answers = []
    tool_results = []
    
    for entry in history:
        if entry.get("kind") == "answer":
            answers.append(entry.get("text", ""))
        elif entry.get("kind") == "action":
            result = entry.get("result_descriptor", "")
            if result:
                tool_results.append(f"[{entry.get('tool', '?')}] {result}")
    
    # If we have direct answers, combine them
    if answers:
        # Filter out error/fallback answers
        good_answers = [a for a in answers if not a.startswith("I encountered")]
        if good_answers:
            if len(good_answers) == 1:
                return good_answers[0]
            else:
                return "\n\n".join(good_answers)
    
    # Fall back to summarizing tool results
    if tool_results:
        return f"Based on my research:\n" + "\n".join(tool_results[:5])
    
    return "I was unable to find a complete answer to your query."


async def run(query: str, run_id: str | None = None) -> str:
    """
    Run the agent on a query.

    Args:
        query:  The user's query to answer
        run_id: Optional pre-generated ID (used by agent_server for SSE correlation)

    Returns:
        The final answer string
    """
    # Ensure gateway is available
    ensure_gateway()
    
    # Generate run ID (accept pre-generated id from agent_server for SSE correlation)
    run_id = run_id or uuid.uuid4().hex[:8]
    logger.info(f"Starting run {run_id}: {query[:50]}...")
    console.print(
        Panel.fit(
            f"Run: [bold]{run_id}[/bold]\nQuery: {query}",
            title="Agent6 Rich Trace",
            border_style="cyan",
        )
    )
    events.emit(run_id, {"type": "start", "run_id": run_id, "query": query})

    # Initialize state
    history: list[dict] = []
    prior_goals: list[Goal] = []
    
    # Get expected iteration bound
    max_bound = _get_expected_bound(query)
    
    # Store the query in memory for classification
    memory.remember(query, source="user_query", run_id=run_id)
    
    async with mcp_session() as session:
        # Load available tools
        mcp_tools = await load_tools(session)
        tools = mcp_tools_for_decision(mcp_tools)
        logger.info(f"Loaded {len(tools)} tools: {[t['name'] for t in tools]}")
        console.print(
            f"[bold]Tools:[/bold] {', '.join(t['name'] for t in tools) if tools else 'none'}"
        )
        
        for it in range(1, MAX_ITERATIONS + 1):
            logger.info(f"[{run_id}] Iteration {it}/{max_bound}")
            _trace_iter_header(it)
            events.emit(run_id, {"type": "iter_start", "iter": it})
            _trace_layer_flow()
            
            # Check iteration bound
            if it > max_bound:
                logger.warning(f"[{run_id}] Exceeded iteration bound {max_bound}")
                _log_iteration(run_id, it, {
                    "status": "exceeded_bound",
                    "max_bound": max_bound,
                })
                break
            
            # Read relevant memories
            hits = memory.read(query, history)
            logger.debug(f"[{run_id}] Found {len(hits)} relevant memories")
            _trace_memory(hits)
            _first_hit_preview = _short(getattr(hits[0], "descriptor", ""), 80) if hits else ""
            events.emit(run_id, {"type": "memory", "layer": "memory", "hits": len(hits), "preview": _first_hit_preview})

            # Perception: observe and update goals
            obs = perception.observe(query, hits, history, prior_goals, run_id)
            prior_goals = obs.goals
            _trace_perception(obs)
            events.emit(run_id, {"type": "perception", "layer": "perception", "goals": [{"text": g.text, "done": g.done} for g in obs.goals]})

            _log_iteration(run_id, it, {
                "phase": "perception",
                "goals": [{"id": g.id, "text": g.text, "done": g.done, "artifact": g.attach_artifact_id} for g in obs.goals],
                "all_done": obs.all_done,
            })
            
            # Check if all goals are done
            if obs.all_done:
                logger.info(f"[{run_id}] All goals complete at iteration {it}")
                console.print("[bold green][done][/bold green] all goals satisfied")
                events.emit(run_id, {"type": "complete"})
                break
            
            # Get the next unfinished goal
            goal = obs.next_unfinished()
            if not goal:
                logger.info(f"[{run_id}] No unfinished goals")
                break
            
            logger.info(f"[{run_id}] Working on goal: {goal.text[:50]}")
            
            # Load attached artifact content if present
            attached: list[tuple[str, bytes]] = []
            if goal.attach_artifact_id and artifacts.exists(goal.attach_artifact_id):
                try:
                    content = artifacts.get_bytes(goal.attach_artifact_id)
                    attached.append((goal.attach_artifact_id, content))
                    logger.debug(f"[{run_id}] Attached artifact {goal.attach_artifact_id}")
                    console.print(
                        f"[bold white][attach][/bold white] {goal.attach_artifact_id} ({len(content)} bytes)"
                    )
                    events.emit(run_id, {"type": "attach", "layer": "attach", "artifact_id": goal.attach_artifact_id, "bytes": len(content)})
                except Exception as e:
                    logger.warning(f"[{run_id}] Failed to load artifact: {e}")
            
            # Decision: decide next action
            out = decision.next_step(goal, hits, attached, history, tools)
            
            _log_iteration(run_id, it, {
                "phase": "decision",
                "goal_id": goal.id,
                "is_answer": out.is_answer,
                "answer": out.answer[:200] if out.answer else None,
                "tool_call": {"name": out.tool_call.name, "arguments": out.tool_call.arguments} if out.tool_call else None,
            })
            
            if out.is_answer:
                _trace_decision_answer(out.answer or "")
                events.emit(run_id, {"type": "decision_answer", "layer": "decision", "preview": _short(out.answer or "", 200)})
                # Record the answer in history
                history.append({
                    "iter": it,
                    "kind": "answer",
                    "goal_id": goal.id,
                    "text": out.answer,
                })
                logger.info(f"[{run_id}] Answered goal {goal.id[:8]}")
                continue
            
            # Action: execute tool call
            logger.info(f"[{run_id}] Calling tool: {out.tool_call.name}")
            _trace_decision_tool(out.tool_call.name, out.tool_call.arguments)
            events.emit(run_id, {"type": "decision_tool", "layer": "decision", "tool": out.tool_call.name, "args_preview": _short(json.dumps(out.tool_call.arguments, ensure_ascii=False), 120)})
            result_text, art_id = await action.execute(session, out.tool_call)
            
            # Record outcome in memory
            memory.record_outcome(
                tool_call=out.tool_call,
                result_text=result_text,
                artifact_id=art_id,
                run_id=run_id,
                goal_id=goal.id,
            )
            
            # Record in history
            history.append({
                "iter": it,
                "kind": "action",
                "goal_id": goal.id,
                "tool": out.tool_call.name,
                "arguments": out.tool_call.arguments,
                "result_descriptor": result_text[:300],
                "artifact_id": art_id,
            })
            
            _log_iteration(run_id, it, {
                "phase": "action",
                "tool": out.tool_call.name,
                "result_preview": result_text[:200],
                "artifact_id": art_id,
            })
            _trace_action(result_text, art_id)
            events.emit(run_id, {"type": "action", "layer": "action", "tool": out.tool_call.name, "preview": _short(result_text, 150), "artifact_id": art_id})

            logger.info(f"[{run_id}] Tool result: {result_text[:100]}")
    
    # Synthesize final answer
    answer = final_answer_from(history, query)
    
    _log_iteration(run_id, -1, {
        "phase": "complete",
        "total_iterations": len([h for h in history if h.get("kind") == "action"]),
        "answer_preview": answer[:200],
    })
    console.print("\n[bold green]FINAL:[/bold green]")
    console.print(answer)
    events.emit(run_id, {"type": "done", "answer": answer})
    events.close_queue(run_id)

    logger.info(f"[{run_id}] Complete. Answer: {answer[:100]}...")
    return answer


async def main():
    """CLI entry point for testing."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python agent6.py 'your query here'")
        print("\nExample queries:")
        print("  - 'Tell me about Claude Shannon's life and contributions'")
        print("  - 'What are fun activities in Tokyo? Check the weather for Saturday.'")
        print("  - 'Remember that my mom's birthday is March 15th'")
        print("  - 'What is the health tip consensus from top experts?'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    answer = await run(query)
    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())