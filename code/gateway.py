"""
Gateway and MCP session adapters for the four-role agent.

Provides:
- ensure_gateway() - verify LLM gateway is available
- mcp_session() - async context manager for MCP stdio connection
- load_tools(session) - list available MCP tools
- mcp_tools_for_decision(tools) - convert to llm_gatewayV3 tool schema
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Add llm_gatewayV3 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "llm_gatewayV3"))
from client import LLM

# MCP server script path
MCP_SERVER = Path(__file__).parent / "mcp_server.py"

# Gateway URL
GATEWAY_URL = os.getenv("LLM_GATEWAY_V3_URL", "http://localhost:8101")

# Cached LLM client
_llm: LLM | None = None


def get_llm() -> LLM:
    """Get a cached LLM client instance."""
    global _llm
    if _llm is None:
        _llm = LLM(base_url=GATEWAY_URL)
    return _llm


def ensure_gateway(timeout: float = 5.0) -> bool:
    """
    Verify the LLM gateway is available.
    
    Returns True if gateway is healthy, raises RuntimeError otherwise.
    """
    try:
        resp = httpx.get(f"{GATEWAY_URL}/health", timeout=timeout)
        if resp.status_code == 200:
            return True
    except httpx.RequestError:
        pass
    
    # Try capabilities endpoint as fallback
    try:
        llm = get_llm()
        caps = llm.capabilities()
        if caps:
            return True
    except Exception:
        pass
    
    raise RuntimeError(
        f"LLM Gateway not available at {GATEWAY_URL}. "
        "Start it with: cd llm_gatewayV3 && python main.py"
    )


@asynccontextmanager
async def mcp_session():
    """
    Async context manager for MCP stdio connection to the tool server.
    
    Usage:
        async with mcp_session() as session:
            tools = await load_tools(session)
            result = await session.call_tool("web_search", {"query": "test"})
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER)],
        env={**os.environ},
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def load_tools(session: ClientSession) -> list[dict]:
    """
    Load available tools from an MCP session.
    
    Returns a list of tool definitions with name, description, and input_schema.
    """
    result = await session.list_tools()
    tools = []
    
    for tool in result.tools:
        tools.append({
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.inputSchema if hasattr(tool, 'inputSchema') else {},
        })
    
    return tools


def mcp_tools_for_decision(mcp_tools: list[dict]) -> list[dict]:
    """
    Convert MCP tool definitions to the llm_gatewayV3 tool schema format.
    
    The gateway expects tools in this format:
    {
        "name": "tool_name",
        "description": "what it does",
        "input_schema": { JSON Schema }
    }
    """
    converted = []
    
    for tool in mcp_tools:
        # Ensure input_schema has required fields
        schema = tool.get("input_schema", {})
        if not schema:
            schema = {"type": "object", "properties": {}}
        
        converted.append({
            "name": tool["name"],
            "description": tool.get("description", ""),
            "input_schema": schema,
        })
    
    return converted


def tools_summary(tools: list[dict]) -> str:
    """Generate a compact summary of available tools for prompts."""
    lines = []
    for t in tools:
        desc = t.get("description", "")[:80]
        lines.append(f"- {t['name']}: {desc}")
    return "\n".join(lines)


async def call_tool(session: ClientSession, name: str, arguments: dict[str, Any]) -> Any:
    """
    Call an MCP tool and return the result.
    
    Args:
        session: Active MCP ClientSession
        name: Tool name
        arguments: Tool arguments
    
    Returns:
        The tool result content
    """
    result = await session.call_tool(name, arguments)
    
    # Extract content from result
    if hasattr(result, 'content') and result.content:
        # MCP returns content as a list of content blocks
        contents = []
        for block in result.content:
            if hasattr(block, 'text'):
                contents.append(block.text)
            elif hasattr(block, 'data'):
                contents.append(block.data)
        
        if len(contents) == 1:
            # Try to parse as JSON
            try:
                return json.loads(contents[0])
            except (json.JSONDecodeError, TypeError):
                return contents[0]
        return contents
    
    return result


def chat_with_tools(
    prompt: str,
    tools: list[dict],
    *,
    system: str | None = None,
    auto_route: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> dict:
    """
    Make an LLM chat call with tools available.
    
    Args:
        prompt: User prompt
        tools: List of tool definitions
        system: Optional system prompt
        auto_route: Auto-routing role ("perception", "memory", "decision")
        temperature: Sampling temperature
        max_tokens: Maximum output tokens
    
    Returns:
        Gateway response dict with text and/or tool_calls
    """
    llm = get_llm()
    return llm.chat(
        prompt=prompt,
        system=system,
        tools=tools if tools else None,
        auto_route=auto_route,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def chat_json(
    prompt: str,
    schema: dict[str, Any],
    *,
    system: str | None = None,
    auto_route: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> dict:
    """
    Make an LLM chat call expecting JSON output.
    
    Args:
        prompt: User prompt
        schema: JSON schema for the expected output
        system: Optional system prompt
        auto_route: Auto-routing role
        temperature: Sampling temperature
        max_tokens: Maximum output tokens
    
    Returns:
        Gateway response dict with parsed JSON in 'parsed' field
    """
    llm = get_llm()
    return llm.chat(
        prompt=prompt,
        system=system,
        response_format={
            "type": "json_schema",
            "schema": schema,
            "name": "output",
            "strict": True,
        },
        auto_route=auto_route,
        temperature=temperature,
        max_tokens=max_tokens,
    )
