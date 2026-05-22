"""
Durable memory system for the four-role agent.

Provides persistent storage at code/state/memory.json with methods:
- read(query, history, ...) - pure-Python keyword overlap ranking
- filter(kinds, goal_id, recent) - filter by criteria
- relevant(query, kinds, top_k) - LLM-routed relevance ranking
- remember(raw_text, ...) - LLM classification and storage
- record_outcome(...) - non-LLM tool outcome recording
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from schemas import MemoryItem, MemoryClassifyOutput

# Lazy import to avoid circular dependency
_llm = None


def _get_llm():
    """Lazy load LLM client."""
    global _llm
    if _llm is None:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "llm_gatewayV3"))
        from client import LLM
        _llm = LLM()
    return _llm


# State file path
STATE_DIR = Path(__file__).parent / "state"
MEMORY_FILE = STATE_DIR / "memory.json"


def _ensure_state() -> None:
    """Create state directory and memory file if they don't exist."""
    STATE_DIR.mkdir(exist_ok=True)
    if not MEMORY_FILE.exists():
        MEMORY_FILE.write_text("[]", encoding="utf-8")


def _load_all() -> list[MemoryItem]:
    """Load all memory items from disk."""
    _ensure_state()
    try:
        data = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
        return [MemoryItem.model_validate(item) for item in data]
    except (json.JSONDecodeError, Exception):
        return []


def _save_all(items: list[MemoryItem]) -> None:
    """Save all memory items to disk."""
    _ensure_state()
    data = [item.model_dump(mode="json") for item in items]
    MEMORY_FILE.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _add_item(item: MemoryItem) -> None:
    """Add a single item to memory."""
    items = _load_all()
    items.append(item)
    _save_all(items)


def _tokenize(text: str) -> list[str]:
    """Simple tokenization for keyword matching."""
    # Lowercase and extract word tokens
    words = re.findall(r'\b[a-z0-9]+\b', text.lower())
    # Filter stopwords
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                 'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                 'from', 'as', 'into', 'through', 'during', 'before', 'after',
                 'above', 'below', 'between', 'under', 'again', 'further',
                 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                 'how', 'all', 'each', 'few', 'more', 'most', 'other', 'some',
                 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                 'than', 'too', 'very', 's', 't', 'just', 'don', 'now', 'and',
                 'or', 'but', 'if', 'because', 'until', 'while', 'this', 'that',
                 'these', 'those', 'what', 'which', 'who', 'whom', 'it', 'its'}
    return [w for w in words if w not in stopwords and len(w) > 1]


def _keyword_score(query_tokens: list[str], item: MemoryItem) -> float:
    """Score an item by keyword overlap with query."""
    item_text = f"{item.descriptor} {' '.join(item.keywords)}"
    item_tokens = set(_tokenize(item_text))
    
    if not query_tokens or not item_tokens:
        return 0.0
    
    query_set = set(query_tokens)
    overlap = len(query_set & item_tokens)
    
    # Jaccard-like score with query weighting
    return overlap / (len(query_set) + 0.5 * len(item_tokens - query_set))


def read(
    query: str,
    history: list[dict] | None = None,
    kinds: list[str] | None = None,
    top_k: int = 10,
) -> list[MemoryItem]:
    """
    Read relevant memories using pure-Python keyword overlap ranking.
    
    Args:
        query: The search query
        history: Conversation history (used to extract additional keywords)
        kinds: Filter to specific memory kinds
        top_k: Maximum number of results
    
    Returns:
        List of relevant MemoryItem objects, sorted by relevance
    """
    items = _load_all()
    
    # Filter by kinds if specified
    if kinds:
        items = [i for i in items if i.kind in kinds]
    
    if not items:
        return []
    
    # Build query tokens from query and recent history
    query_tokens = _tokenize(query)
    if history:
        # Add tokens from recent history entries
        for entry in history[-5:]:
            if isinstance(entry, dict):
                for v in entry.values():
                    if isinstance(v, str):
                        query_tokens.extend(_tokenize(v)[:10])
    
    # Score and sort
    scored = [(item, _keyword_score(query_tokens, item)) for item in items]
    scored.sort(key=lambda x: (-x[1], x[0].created_at), reverse=False)
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Filter out zero-score items and return top_k
    return [item for item, score in scored[:top_k] if score > 0]


def filter(
    kinds: list[str] | None = None,
    goal_id: str | None = None,
    recent: int | None = None,
) -> list[MemoryItem]:
    """
    Filter memories by criteria.
    
    Args:
        kinds: Filter to specific memory kinds
        goal_id: Filter to a specific goal
        recent: Return only the N most recent items
    
    Returns:
        Filtered list of MemoryItem objects
    """
    items = _load_all()
    
    if kinds:
        items = [i for i in items if i.kind in kinds]
    
    if goal_id:
        items = [i for i in items if i.goal_id == goal_id]
    
    # Sort by creation time (newest first) for recent filtering
    items.sort(key=lambda x: x.created_at, reverse=True)
    
    if recent:
        items = items[:recent]
    
    return items


def relevant(
    query: str,
    kinds: list[str] | None = None,
    top_k: int = 5,
) -> list[MemoryItem]:
    """
    Find relevant memories using LLM-based ranking.
    
    Uses auto_route=memory for the LLM call.
    
    Args:
        query: The relevance query
        kinds: Filter to specific memory kinds
        top_k: Maximum number of results
    
    Returns:
        List of relevant MemoryItem objects
    """
    items = _load_all()
    
    if kinds:
        items = [i for i in items if i.kind in kinds]
    
    if not items:
        return []
    
    # For small sets, use keyword-based ranking
    if len(items) <= top_k:
        return items
    
    # Build prompt for LLM ranking
    items_desc = "\n".join([
        f"{i}: {item.descriptor} (kind={item.kind}, keywords={item.keywords})"
        for i, item in enumerate(items)
    ])
    
    prompt = f"""Given the query: "{query}"

Rank these memory items by relevance (most relevant first). Return only the indices.

Items:
{items_desc}

Return a JSON array of indices, e.g. [2, 0, 5, 1]. Return at most {top_k} indices."""

    try:
        llm = _get_llm()
        resp = llm.chat(prompt=prompt, auto_route="memory", temperature=0.0, max_tokens=200)
        text = resp.get("text", "[]")
        
        # Parse indices from response
        match = re.search(r'\[[\d,\s]+\]', text)
        if match:
            indices = json.loads(match.group())
            result = []
            for idx in indices[:top_k]:
                if 0 <= idx < len(items):
                    result.append(items[idx])
            return result
    except Exception:
        pass
    
    # Fallback to keyword ranking
    return read(query, kinds=kinds, top_k=top_k)


def remember(
    raw_text: str,
    *,
    source: str = "unknown",
    run_id: str = "",
    goal_id: str | None = None,
    artifact_id: str | None = None,
    value: dict[str, Any] | None = None,
) -> MemoryItem:
    """
    Classify and store a piece of information in memory.
    
    Uses auto_route=memory for the LLM classification call.
    
    Args:
        raw_text: The text to classify and remember
        source: Source of the information
        run_id: Current run ID
        goal_id: Associated goal ID
        artifact_id: Associated artifact ID
        value: Optional structured value
    
    Returns:
        The created MemoryItem
    """
    prompt = f"""Classify this information for memory storage:

Text: {raw_text[:1000]}

Respond with JSON:
{{
    "kind": "fact" | "preference" | "tool_outcome" | "scratchpad",
    "keywords": ["keyword1", "keyword2", ...],  // 3-7 relevant keywords
    "descriptor": "one short sentence describing the key information",
    "confidence": 0.0-1.0  // how confident this is useful/accurate
}}

Guidelines:
- "fact": verifiable information about the world or a person
- "preference": user preferences, likes, dislikes
- "tool_outcome": results from tool executions
- "scratchpad": temporary working notes"""

    try:
        llm = _get_llm()
        resp = llm.chat(prompt=prompt, auto_route="memory", temperature=0.0, max_tokens=300)
        text = resp.get("text", "{}")
        
        # Parse JSON from response
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            classified = MemoryClassifyOutput.model_validate(data)
        else:
            raise ValueError("No JSON found")
    except Exception:
        # Fallback classification
        classified = MemoryClassifyOutput(
            kind="scratchpad",
            keywords=_tokenize(raw_text)[:5],
            descriptor=raw_text[:100],
            confidence=0.5,
        )
    
    item = MemoryItem(
        id=uuid.uuid4().hex[:12],
        kind=classified.kind,
        keywords=classified.keywords,
        descriptor=classified.descriptor,
        value=value or {"raw": raw_text[:500]},
        artifact_id=artifact_id,
        source=source,
        run_id=run_id,
        goal_id=goal_id,
        confidence=classified.confidence,
        created_at=datetime.utcnow(),
    )
    
    _add_item(item)
    return item


def record_outcome(
    tool_call: Any,  # ToolCall from schemas
    result_text: str,
    artifact_id: str | None,
    run_id: str,
    goal_id: str | None,
) -> MemoryItem:
    """
    Record a tool outcome without LLM classification.
    
    This is a fast, non-LLM path for storing tool results.
    
    Args:
        tool_call: The ToolCall that was executed
        result_text: Short descriptor of the result
        artifact_id: Associated artifact ID (if result was stored)
        run_id: Current run ID
        goal_id: Associated goal ID
    
    Returns:
        The created MemoryItem
    """
    # Extract keywords from tool name and arguments
    keywords = [tool_call.name]
    for k, v in tool_call.arguments.items():
        if isinstance(v, str):
            keywords.extend(_tokenize(v)[:3])
    keywords = list(set(keywords))[:7]
    
    item = MemoryItem(
        id=uuid.uuid4().hex[:12],
        kind="tool_outcome",
        keywords=keywords,
        descriptor=result_text[:300],
        value={
            "tool": tool_call.name,
            "arguments": tool_call.arguments,
            "result_preview": result_text[:500],
        },
        artifact_id=artifact_id,
        source=f"tool:{tool_call.name}",
        run_id=run_id,
        goal_id=goal_id,
        confidence=1.0,
        created_at=datetime.utcnow(),
    )
    
    _add_item(item)
    return item


def clear() -> int:
    """Clear all memory. Returns count of deleted items."""
    items = _load_all()
    count = len(items)
    _save_all([])
    return count


def reset() -> dict:
    """
    Reset to a completely clean slate.

    Clears memory.json *and* deletes every artifact in state/artifacts/.
    Call this before re-running a query to avoid stale state contamination.

    Returns a dict with keys ``memory_cleared`` and ``artifacts_cleared``.
    """
    from artifacts import get_store

    memory_count = clear()
    artifact_count = get_store().clear_all()
    return {"memory_cleared": memory_count, "artifacts_cleared": artifact_count}
