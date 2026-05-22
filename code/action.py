"""
Action execution for the four-role agent.

Dispatches MCP tool calls and handles result storage:
- Large/raw payloads are stored as artifacts
- Returns short descriptor text for history
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp import ClientSession

import artifacts
from schemas import ToolCall

# Maximum descriptor length before storing as artifact
MAX_DESCRIPTOR_LENGTH = 300

# Content types that should always be stored as artifacts
ARTIFACT_CONTENT_TYPES = {"text/markdown", "text/html", "application/json"}

# Tools whose output should always be artifacted (regardless of size)
ARTIFACT_TOOLS = {"fetch_url", "web_search"}


async def execute(
    session: ClientSession,
    tool_call: ToolCall,
) -> tuple[str, str | None]:
    """
    Execute a tool call and return (descriptor, artifact_id).
    
    Args:
        session: Active MCP ClientSession
        tool_call: The ToolCall to execute
    
    Returns:
        Tuple of:
        - descriptor: Short human-readable result description (<=300 chars)
        - artifact_id: ID of stored artifact if result was large, else None
    """
    # Execute the tool (60 s hard cap — prevents the agent blocking on a
    # hung MCP server tool such as a stalled headless browser).
    result = await asyncio.wait_for(
        session.call_tool(tool_call.name, tool_call.arguments),
        timeout=60.0,
    )
    
    # Extract content from MCP result
    raw_content = _extract_content(result)
    
    # Convert to string for processing
    if isinstance(raw_content, (dict, list)):
        content_str = json.dumps(raw_content, indent=2, ensure_ascii=False)
        content_type = "application/json"
    else:
        content_str = str(raw_content)
        content_type = "text/plain"
    
    # Determine if we need to store as artifact
    should_artifact = (
        len(content_str) > MAX_DESCRIPTOR_LENGTH * 2
        or tool_call.name in ARTIFACT_TOOLS
        or content_type in ARTIFACT_CONTENT_TYPES
    )
    
    artifact_id = None
    if should_artifact:
        # Build a source descriptor
        source = f"{tool_call.name}({_args_summary(tool_call.arguments)})"
        
        # Create a short descriptor
        descriptor = _make_descriptor(tool_call.name, raw_content, content_str)
        
        # Store artifact
        artifact_id = artifacts.put(
            content_str.encode("utf-8"),
            content_type=content_type,
            source=source,
            descriptor=descriptor,
        )
    
    # Build the short descriptor for history
    descriptor = _make_descriptor(tool_call.name, raw_content, content_str)
    
    return descriptor, artifact_id


def _extract_content(result: Any) -> Any:
    """Extract content from MCP tool result."""
    if hasattr(result, 'content') and result.content:
        # MCP returns content as a list of content blocks
        contents = []
        for block in result.content:
            if hasattr(block, 'text'):
                try:
                    # Try to parse as JSON
                    contents.append(json.loads(block.text))
                except (json.JSONDecodeError, TypeError):
                    contents.append(block.text)
            elif hasattr(block, 'data'):
                contents.append(block.data)
        
        if len(contents) == 1:
            return contents[0]
        return contents
    
    return result


def _args_summary(args: dict[str, Any]) -> str:
    """Create a short summary of tool arguments."""
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            v_str = v[:30] + "..." if len(v) > 30 else v
            parts.append(f'{k}="{v_str}"')
        else:
            parts.append(f"{k}={v}")
    return ", ".join(parts)[:100]


def _make_descriptor(tool_name: str, raw: Any, content_str: str) -> str:
    """Create a short descriptor for a tool result."""
    if tool_name == "fetch_url":
        # For fetch, include URL and content length
        length = len(content_str)
        if isinstance(raw, dict):
            status = raw.get("status", "?")
            return f"Fetched page ({status}), {length:,} chars of markdown"
        return f"Fetched content, {length:,} chars"
    
    elif tool_name == "web_search":
        # For search, count results
        if isinstance(raw, list):
            count = len(raw)
            titles = [r.get("title", "")[:40] for r in raw[:3] if isinstance(r, dict)]
            return f"Found {count} results: {', '.join(titles)}"[:MAX_DESCRIPTOR_LENGTH]
        return f"Search results: {content_str[:100]}"
    
    elif tool_name == "get_time":
        if isinstance(raw, dict):
            return f"Current time: {raw.get('human', raw.get('iso', str(raw)))}"
        return f"Time: {content_str[:100]}"
    
    elif tool_name == "currency_convert":
        if isinstance(raw, dict):
            return f"Converted {raw.get('amount')} {raw.get('from')} = {raw.get('converted')} {raw.get('to')}"
        return f"Currency: {content_str[:100]}"
    
    elif tool_name in ("read_file", "list_dir"):
        if isinstance(raw, dict) and "content" in raw:
            return f"File content ({raw.get('size_bytes', '?')} bytes): {raw['content'][:100]}..."
        return f"File result: {content_str[:100]}"
    
    elif tool_name in ("create_file", "update_file", "edit_file"):
        if isinstance(raw, dict) and raw.get("ok"):
            return f"File operation successful: {raw.get('path', 'unknown')}"
        return f"File result: {content_str[:100]}"
    
    else:
        # Generic descriptor
        preview = content_str[:MAX_DESCRIPTOR_LENGTH - 20]
        if len(content_str) > MAX_DESCRIPTOR_LENGTH - 20:
            preview += "..."
        return f"{tool_name}: {preview}"
