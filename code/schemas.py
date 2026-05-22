"""
Pydantic v2 schemas for the four-role agent system.

Defines contracts for:
- Memory items (facts, preferences, tool outcomes, scratchpad)
- Artifacts (stored blobs with metadata)
- Goals and Observations (Perception layer output)
- ToolCall and DecisionOutput (Decision layer output)
- LLM output models for typed parsing
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Memory
# ─────────────────────────────────────────────────────────────────────────────

class MemoryItem(BaseModel):
    """A single memory entry persisted across runs."""
    id: str
    kind: Literal["fact", "preference", "tool_outcome", "scratchpad"]
    keywords: list[str] = Field(default_factory=list)
    descriptor: str  # one short human-readable line
    value: dict[str, Any] = Field(default_factory=dict)  # structured payload
    artifact_id: str | None = None  # handle into the artifact store
    source: str = ""
    run_id: str = ""
    goal_id: str | None = None
    confidence: float = 1.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Artifacts
# ─────────────────────────────────────────────────────────────────────────────

class Artifact(BaseModel):
    """Metadata for a stored artifact blob."""
    id: str  # "art:<sha256-prefix>"
    content_type: str
    size_bytes: int
    source: str
    descriptor: str


# ─────────────────────────────────────────────────────────────────────────────
# Perception (Goals / Observation)
# ─────────────────────────────────────────────────────────────────────────────

class Goal(BaseModel):
    """A single goal produced by Perception."""
    id: str
    text: str  # short imperative description
    done: bool = False
    attach_artifact_id: str | None = None


class Observation(BaseModel):
    """Output of the Perception layer — current goal state."""
    goals: list[Goal] = Field(default_factory=list)

    @property
    def all_done(self) -> bool:
        """True when every goal is marked done."""
        return len(self.goals) > 0 and all(g.done for g in self.goals)

    def next_unfinished(self) -> Goal | None:
        """Return the first goal that is not done, or None if all done."""
        for g in self.goals:
            if not g.done:
                return g
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Decision (ToolCall / DecisionOutput)
# ─────────────────────────────────────────────────────────────────────────────

class ToolCall(BaseModel):
    """A single tool invocation."""
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class DecisionOutput(BaseModel):
    """Output of the Decision layer — either an answer or a tool call."""
    answer: str | None = None
    tool_call: ToolCall | None = None

    @model_validator(mode="after")
    def exactly_one_populated(self) -> "DecisionOutput":
        """Enforce that exactly one of answer/tool_call is set."""
        has_answer = self.answer is not None
        has_tool = self.tool_call is not None
        if has_answer == has_tool:
            raise ValueError("DecisionOutput must have exactly one of answer or tool_call")
        return self

    @property
    def is_answer(self) -> bool:
        """True if this decision is a final answer (no tool call)."""
        return self.answer is not None


# ─────────────────────────────────────────────────────────────────────────────
# LLM Output Models (for typed JSON parsing)
# ─────────────────────────────────────────────────────────────────────────────

class PerceptionLLMOutput(BaseModel):
    """Structured output from the Perception LLM call."""
    goals: list[Goal]


class DecisionLLMOutput(BaseModel):
    """Structured output from the Decision LLM call."""
    reasoning: str = ""
    answer: str | None = None
    tool_call: ToolCall | None = None

    @model_validator(mode="after")
    def exactly_one_populated(self) -> "DecisionLLMOutput":
        has_answer = self.answer is not None
        has_tool = self.tool_call is not None
        if has_answer == has_tool:
            raise ValueError("Must have exactly one of answer or tool_call")
        return self


class MemoryClassifyOutput(BaseModel):
    """Structured output from memory classification LLM call."""
    kind: Literal["fact", "preference", "tool_outcome", "scratchpad"]
    keywords: list[str]
    descriptor: str
    confidence: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def deterministic_goal_id(text: str) -> str:
    """Generate a stable goal ID from normalized goal text."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()[:12]


def artifact_id_from_bytes(blob: bytes) -> str:
    """Generate deterministic artifact ID from content hash."""
    return f"art:{hashlib.sha256(blob).hexdigest()[:16]}"