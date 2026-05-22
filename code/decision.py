"""
Decision layer for the four-role agent.

Decides the next action: either provide an answer or call a tool.
Uses anti-loop constraints to ensure convergence.
"""

from __future__ import annotations

import json
import re
from typing import Any

import artifacts
from gateway import get_llm, tools_summary
from schemas import (
    DecisionLLMOutput,
    DecisionOutput,
    Goal,
    MemoryItem,
    ToolCall,
)


def next_step(
    goal: Goal,
    hits: list[MemoryItem],
    attached: list[tuple[str, bytes]],
    history: list[dict],
    tools: list[dict],
) -> DecisionOutput:
    """
    Decide the next action for a goal.
    
    Uses auto_route=decision for the LLM call.
    
    Args:
        goal: The current goal to work on
        hits: Relevant memory items
        attached: List of (artifact_id, content) tuples for attached artifacts
        history: Action history from this run
        tools: Available MCP tools
    
    Returns:
        DecisionOutput with either an answer or a tool_call
    """
    # Build context
    memory_context = _format_memory(hits)
    artifact_context = _format_artifacts(attached)
    history_context = _format_history(history)
    tools_context = tools_summary(tools)
    
    # Check for duplicate tool calls
    recent_tools = _get_recent_tool_calls(history)
    
    prompt = f"""You are the Decision layer of an AI agent. Decide the next action for the current goal.

CURRENT GOAL: {goal.text}
Goal ID: {goal.id}

MEMORY CONTEXT (relevant facts and prior tool outcomes):
{memory_context or "(no relevant memories)"}

ATTACHED ARTIFACT CONTENT:
{artifact_context or "(no artifacts attached)"}

ACTION HISTORY (what has been done):
{history_context or "(no actions yet)"}

AVAILABLE TOOLS:
{tools_context}

DECISION RULES:
1. If you have sufficient information to answer the goal, provide an answer
2. If you need more information, call ONE appropriate tool
3. DO NOT call a tool with the same arguments if it was already called (check history)
4. Prefer answering once required facts are present
5. Be concise but complete in answers

RECENT TOOL CALLS (avoid duplicates):
{_format_recent_tools(recent_tools)}

Respond with JSON (exactly one of answer or tool_call):
{{
    "reasoning": "brief explanation of decision",
    "answer": "your answer text" OR null,
    "tool_call": {{"name": "tool_name", "arguments": {{...}}}} OR null
}}"""

    try:
        llm = get_llm()
        resp = llm.chat(
            prompt=prompt,
            auto_route="decision",
            temperature=0.1,  # Low temperature for consistency
            max_tokens=1024,
        )
        text = resp.get("text", "{}")
        
        # Parse JSON from response
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            data = json.loads(match.group())
            
            # Validate and construct output
            if data.get("answer"):
                return DecisionOutput(answer=data["answer"])
            elif data.get("tool_call"):
                tc = data["tool_call"]
                # Validate tool exists
                tool_names = {t["name"] for t in tools}
                if tc.get("name") not in tool_names:
                    # Fall back to answer if tool doesn't exist
                    return DecisionOutput(
                        answer=f"I cannot find the tool '{tc.get('name')}'. Based on available information: {data.get('reasoning', 'Unable to proceed.')}"
                    )
                
                # Check for duplicate call
                call_key = _tool_call_key(tc["name"], tc.get("arguments", {}))
                if call_key in recent_tools:
                    # Already called this, force an answer
                    return DecisionOutput(
                        answer=f"I've already retrieved this information. Based on what I found: {data.get('reasoning', 'Please check the history.')}"
                    )
                
                return DecisionOutput(
                    tool_call=ToolCall(
                        name=tc["name"],
                        arguments=tc.get("arguments", {}),
                    )
                )
            else:
                raise ValueError("Neither answer nor tool_call provided")
        else:
            raise ValueError("No JSON found in response")
    
    except Exception as e:
        # Fallback: provide a generic answer
        return DecisionOutput(
            answer=f"I encountered an issue while processing: {str(e)[:100]}. Please try rephrasing your question."
        )


def _format_memory(hits: list[MemoryItem]) -> str:
    """Format memory items for the prompt."""
    if not hits:
        return ""
    
    lines = []
    for hit in hits[:8]:  # Limit to 8
        value_preview = ""
        if hit.value:
            raw = hit.value.get("raw", hit.value.get("result_preview", ""))
            if raw:
                value_preview = f"\n    Content: {str(raw)[:200]}"
        
        lines.append(
            f"- [{hit.kind}] {hit.descriptor}{value_preview}"
        )
    return "\n".join(lines)


def _format_artifacts(attached: list[tuple[str, bytes]]) -> str:
    """Format attached artifact content."""
    if not attached:
        return ""
    
    lines = []
    for art_id, content in attached[:3]:  # Limit to 3 artifacts
        try:
            text = content.decode("utf-8", errors="replace")
            # Truncate large content
            if len(text) > 3000:
                text = text[:3000] + f"\n... [truncated, {len(text)} total chars]"
            lines.append(f"=== Artifact {art_id} ===\n{text}\n")
        except Exception:
            lines.append(f"=== Artifact {art_id} === (binary, {len(content)} bytes)\n")
    
    return "\n".join(lines)


def _format_history(history: list[dict]) -> str:
    """Format action history for the prompt."""
    if not history:
        return ""
    
    lines = []
    for entry in history[-8:]:  # Last 8 entries
        kind = entry.get("kind", "unknown")
        if kind == "action":
            tool = entry.get("tool", "?")
            args = json.dumps(entry.get("arguments", {}))[:60]
            result = entry.get("result_descriptor", "")[:150]
            lines.append(f"- CALLED {tool}({args}) -> {result}")
        elif kind == "answer":
            goal_id = entry.get("goal_id", "?")
            text = entry.get("text", "")[:100]
            lines.append(f"- ANSWERED [{goal_id}]: {text}")
    return "\n".join(lines)


def _get_recent_tool_calls(history: list[dict]) -> set[str]:
    """Extract recent tool call signatures to prevent duplicates."""
    calls = set()
    for entry in history[-10:]:
        if entry.get("kind") == "action":
            tool = entry.get("tool", "")
            args = entry.get("arguments", {})
            calls.add(_tool_call_key(tool, args))
    return calls


def _tool_call_key(name: str, arguments: dict[str, Any]) -> str:
    """Create a unique key for a tool call."""
    # Sort arguments for consistent hashing
    sorted_args = json.dumps(arguments, sort_keys=True)
    return f"{name}:{sorted_args}"


def _format_recent_tools(calls: set[str]) -> str:
    """Format recent tool calls for display."""
    if not calls:
        return "(none)"
    
    lines = []
    for call in list(calls)[:5]:
        lines.append(f"- {call[:80]}")
    return "\n".join(lines)
