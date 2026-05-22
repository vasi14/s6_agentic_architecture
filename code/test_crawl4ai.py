"""
Standalone test for the crawl4ai fetch functionality.

Mirrors the _crawl4ai_fetch / fetch_url logic from mcp_server.py
without any MCP dependencies.

Run:  python test_crawl4ai.py [URL]
"""

from __future__ import annotations

import asyncio
import os
import sys


async def _crawl4ai_fetch(url: str, suppress_output: bool = False) -> dict:
    """Fetch clean markdown from a URL via crawl4ai (headless Chromium).

    Parameters
    ----------
    url:
        The URL to fetch.
    suppress_output:
        Redirect crawl4ai's stdout noise to stderr so terminal output stays
        clean (mirrors the MCP-server fd-redirect trick).
    """
    from crawl4ai import AsyncWebCrawler

    if suppress_output:
        # crawl4ai uses Rich which bypasses contextlib.redirect_stdout.
        # Redirect at the file-descriptor level.
        saved_fd = os.dup(1)
        os.dup2(2, 1)

    try:
        async with AsyncWebCrawler(verbose=False) as crawler:
            r = await crawler.arun(url=url)
    finally:
        if suppress_output:
            os.dup2(saved_fd, 1)
            os.close(saved_fd)

    # r.markdown is a str subclass (StringCompatibleMarkdown) whose real
    # field is private — getattr fallbacks extract the raw string.
    md = r.markdown
    raw = (
        getattr(md, "raw_markdown", None)
        or getattr(md, "fit_markdown", None)
        or md
        or r.cleaned_html
        or r.html
        or ""
    )
    text = str(raw)
    return {
        "status": int(getattr(r, "status_code", None) or 200),
        "content_type": "text/markdown",
        "length_bytes": len(text.encode("utf-8")),
        "text": text,
    }


async def fetch_url(url: str, timeout: int = 20, suppress_output: bool = False) -> dict:
    """Fetch clean markdown from a URL.  Mirrors the MCP tool fetch_url."""
    return await asyncio.wait_for(
        _crawl4ai_fetch(url, suppress_output=suppress_output),
        timeout=timeout,
    )


async def main() -> None:
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"

    print(f"Fetching: {test_url}")
    print("-" * 60)

    result = await fetch_url(test_url, suppress_output=True)

    print(f"Status       : {result['status']}")
    print(f"Content-Type : {result['content_type']}")
    print(f"Length       : {result['length_bytes']:,} bytes")
    print("-" * 60)

    # Print first 2 000 characters of content so the terminal stays readable
    preview = result["text"][:2000]
    if len(result["text"]) > 2000:
        preview += f"\n\n... [{result['length_bytes']:,} bytes total, truncated for display]"
    print(preview)


if __name__ == "__main__":
    asyncio.run(main())
