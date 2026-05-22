"""
Perception layer for the four-role agent.

Observes the current state and produces goals with done flags.
Includes artifact attachment logic for synthesis-type goals.
"""

from __future__ import annotations

import json
import re
from typing import Any

import artifacts
from gateway import get_llm
from schemas import (
    Goal,
    MemoryItem,
    Observation,
    PerceptionLLMOutput,
    deterministic_goal_id,
)


# Keywords indicating a synthesis/extraction goal that needs artifact context
SYNTHESIS_KEYWORDS = {
    "synthesize", "synthesise", "extract", "list", "compare", "decide",
    "summarize", "summarise", "analyze", "analyse", "combine", "compile",
    "identify", "determine", "conclude", "evaluate", "recommend",
}


def observe(
    query: str,
    hits: list[MemoryItem],
    history: list[dict],
    prior_goals: list[Goal],
    run_id: str,
) -> Observation:
    """
    Observe the current state and produce goals.
    
    Uses auto_route=perception for the LLM call.
    
    Args:
        query: The original user query
        hits: Relevant memory items
        history: Action history from this run
        prior_goals: Goals from the previous iteration
        run_id: Current run ID
    
    Returns:
        Observation with updated goals
    """
    # Build context for the LLM
    memory_context = _format_memory(hits)
    history_context = _format_history(history)
    prior_context = _format_prior_goals(prior_goals)
    
    prompt = f"""You are the Perception layer of an AI agent. Analyze the current state and produce a list of goals.

    ORIGINAL QUERY: {query}

    MEMORY CONTEXT (relevant facts and prior tool outcomes):
    {memory_context or "(no relevant memories)"}

    ACTION HISTORY (what has been done this run):
    {history_context or "(no actions yet)"}

    PRIOR GOALS:
    {prior_context or "(no prior goals)"}

    TASK:
    1. Identify what goals are needed to answer the query
    2. Mark goals as done=true if they have been completed (evidence exists in history or memory)
    3. Keep goals focused and actionable
    4. Typically 1-3 goals are sufficient

    IMPORTANT:
    - A goal is done if the required information has been obtained
    - If a tool has been called and the result is in history, that goal is likely done
    - If facts needed are in memory, the goal may be done
    - Always include at least one goal
    - Goals should be short imperative phrases

    Respond with JSON:
    {{
        "goals": [
            {{"id": "...", "text": "...", "done": true/false}},
            ...
        ]
    }}
    """

    try:
        llm = get_llm()
        resp = llm.chat(
            prompt=prompt,
            auto_route="perception",
            temperature=0.0,
            max_tokens=500,
        )
        text = resp.get("text", "{}")
        
        # Parse JSON from response
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            data = json.loads(match.group())
            goals_data = data.get("goals", [])
            
            goals = []
            for g in goals_data:
                goal = Goal(
                    id=g.get("id") or deterministic_goal_id(g.get("text", "")),
                    text=g.get("text", ""),
                    done=g.get("done", False),
                    attach_artifact_id=None,
                )
                goals.append(goal)
        else:
            raise ValueError("No JSON found in response")
    
    except Exception as e:
        # Fallback: create a single goal from the query
        goals = [
            Goal(
                id=deterministic_goal_id(query),
                text=f"Answer: {query}",
                done=False,
            )
        ]
    
    # Apply artifact attachment logic
    goals = _attach_artifacts(goals, hits, history)
    
    return Observation(goals=goals)


def _format_memory(hits: list[MemoryItem]) -> str:
    """Format memory items for the prompt."""
    if not hits:
        return ""
    
    lines = []
    for hit in hits[:10]:  # Limit to 10
        artifact_note = f" [artifact:{hit.artifact_id}]" if hit.artifact_id else ""
        lines.append(
            f"- [{hit.kind}] {hit.descriptor}{artifact_note}"
        )
    return "\n".join(lines)


def _format_history(history: list[dict]) -> str:
    """Format action history for the prompt."""
    if not history:
        return ""
    
    lines = []
    for entry in history[-10:]:  # Last 10 entries
        kind = entry.get("kind", "unknown")
        if kind == "action":
            tool = entry.get("tool", "?")
            args = entry.get("arguments", {})
            result = entry.get("result_descriptor", "")[:100]
            art_id = entry.get("artifact_id", "")
            artifact_note = f" [artifact:{art_id}]" if art_id else ""
            lines.append(f"- ACTION: {tool}({_summarize_args(args)}) -> {result}{artifact_note}")
        elif kind == "answer":
            text = entry.get("text", "")[:100]
            lines.append(f"- ANSWER: {text}")
    return "\n".join(lines)


def _format_prior_goals(prior_goals: list[Goal]) -> str:
    """Format prior goals for the prompt."""
    if not prior_goals:
        return ""
    
    lines = []
    for g in prior_goals:
        status = "DONE" if g.done else "pending"
        artifact_note = f" [attached:{g.attach_artifact_id}]" if g.attach_artifact_id else ""
        lines.append(f"- [{status}] {g.text}{artifact_note}")
    return "\n".join(lines)


def _summarize_args(args: dict[str, Any]) -> str:
    """Summarize tool arguments for display."""
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            v_str = v[:20] + "..." if len(v) > 20 else v
            parts.append(f'{k}="{v_str}"')
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)[:60]


def _is_synthesis_goal(text: str) -> bool:
    """Check if a goal text indicates a synthesis/extraction task."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in SYNTHESIS_KEYWORDS)


def _attach_artifacts(
    goals: list[Goal],
    hits: list[MemoryItem],
    history: list[dict],
) -> list[Goal]:
    """
    Apply artifact attachment logic to goals.
    
    For the first unfinished synthesis goal, attach the most recent
    relevant artifact if one exists.
    """
    # Find the first unfinished goal
    target_goal = None
    for g in goals:
        if not g.done:
            target_goal = g
            break
    
    if not target_goal:
        return goals
    
    # Check if it's a synthesis-type goal
    if not _is_synthesis_goal(target_goal.text):
        return goals
    
    # Find the most recent artifact from history or memory
    artifact_id = None
    
    # Check history first (most recent)
    for entry in reversed(history):
        art_id = entry.get("artifact_id")
        if art_id and artifacts.exists(art_id):
            artifact_id = art_id
            break
    
    # Fall back to memory hits
    if not artifact_id:
        for hit in hits:
            if hit.artifact_id and artifacts.exists(hit.artifact_id):
                artifact_id = hit.artifact_id
                break
    
    # Update the target goal with artifact attachment
    if artifact_id:
        updated_goals = []
        for g in goals:
            if g.id == target_goal.id:
                updated_goals.append(Goal(
                    id=g.id,
                    text=g.text,
                    done=g.done,
                    attach_artifact_id=artifact_id,
                ))
            else:
                updated_goals.append(g)
        return updated_goals
    
    return goals