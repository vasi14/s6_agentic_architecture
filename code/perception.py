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
    
    prompt = f"""You are the Perception Layer of an autonomous AI agent system.

    Your responsibility is to analyze the current execution state and determine the minimal set of goals required to answer the user's request.

    You do NOT execute tools or solve the task itself.
    You ONLY identify, track, and update goals.

    ========================
    ORIGINAL USER QUERY
    ========================
    {query}

    ========================
    MEMORY CONTEXT
    ========================
    Relevant stored memories, known facts, and prior validated outcomes:

    {memory_context or "(no relevant memories)"}

    ========================
    ACTION HISTORY
    ========================
    Actions already performed during the current run, including tool calls and outcomes:

    {history_context or "(no actions yet)"}

    ========================
    PRIOR GOALS
    ========================
    Previously generated goals and their completion states:

    {prior_context or "(no prior goals)"}

    ========================
    OBJECTIVE
    ========================
    Determine what goals are necessary to fully answer the user query.

    A goal should:
    - Represent a single actionable objective
    - Be short, concrete, and outcome-oriented
    - Avoid implementation details
    - Be phrased as an imperative action

    Examples:
    - "Find current weather in Chennai"
    - "Summarize uploaded document"
    - "Verify model accuracy metrics"

    ========================
    REASONING PROCESS
    ========================
    Before producing goals, reason step-by-step internally using this sequence:

    1. Understand the user’s intent
    2. Identify missing information required to answer
    3. Check whether required information already exists in:
    - memory context
    - action history
    - prior completed goals
    4. Decide which goals are already complete
    5. Determine the minimum remaining goals needed

    ========================
    GOAL COMPLETION RULES
    ========================
    Mark a goal as done=true ONLY if:
    - The required information already exists in memory/history, OR
    - A previous tool/action successfully produced the needed result, OR
    - A prior goal already satisfies the requirement

    Otherwise mark done=false.

    Avoid duplicate or redundant goals.

    ========================
    REASONING TYPE TAGS
    ========================
    For each goal, identify the dominant reasoning type:

    - "lookup" → retrieving information
    - "analysis" → interpreting or comparing information
    - "planning" → decomposing or sequencing tasks
    - "verification" → validating correctness or consistency
    - "generation" → producing new content
    - "decision" → selecting among alternatives

    ========================
    SELF-CHECK REQUIREMENTS
    ========================
    Before finalizing:
    - Ensure every goal directly supports answering the query
    - Ensure no completed goals are incorrectly marked incomplete
    - Ensure there are no duplicate goals
    - Ensure goals are minimal (typically 1–3 goals)
    - Ensure at least one goal exists

    ========================
    FAILURE / UNCERTAINTY HANDLING
    ========================
    If the available context is ambiguous or insufficient:
    - Infer the most likely user intent conservatively
    - Create exploratory goals when necessary
    - Do NOT hallucinate completed work
    - Prefer incomplete goals over falsely completed ones

    ========================
    OUTPUT FORMAT
    ========================
    Respond ONLY with valid JSON.

    Schema:
    {
    "goals": [
        {
        "id": "goal_1",
        "text": "Short imperative goal",
        "done": true,
        "reasoning_type": "lookup",
        "evidence": "Found completed weather lookup in action history"
        }
    ]
    }

    Rules:
    - Output must be machine-parseable JSON
    - Do not include markdown
    - Do not include explanations outside JSON
    - Goal IDs must be unique
    - Evidence should briefly justify the done state
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