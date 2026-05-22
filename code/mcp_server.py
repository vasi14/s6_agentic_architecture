"""
MCP server for EAGV3 Session 6.

Nine tools, stdio transport:
    web_search, fetch_url, get_time, currency_convert,
    read_file, list_dir, create_file, update_file, edit_file

web_search:  Tavily primary, DuckDuckGo fallback. Hard-capped at 5 results.
fetch_url:   crawl4ai only — clean markdown via headless Chromium.
Usage for tavily and duckduckgo is logged to ./usage.json with monthly
rollover and a soft cap of 950/1000 on Tavily.

File tools are sandboxed under ./sandbox/. Run:  python mcp_server.py
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import threading
from datetime import datetime, timezone as dt_timezone
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from ddgs import DDGS
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

MAX_SEARCH_RESULTS = 5  # hard cap — Tavily prices per result

load_dotenv(Path(__file__).parent / ".env")

mcp = FastMCP("eagv3-s6-server")

SANDBOX = Path(__file__).parent / "sandbox"
SANDBOX.mkdir(exist_ok=True)

USAGE_PATH = Path(__file__).parent / "usage.json"
MONTHLY_CAP = 950  # leave 50/mo headroom on Tavily
_usage_lock = threading.Lock()


def _safe(path: str) -> Path:
    p = (SANDBOX / path).resolve()
    base = SANDBOX.resolve()
    if p != base and base not in p.parents:
        raise ValueError(f"Path '{path}' escapes the sandbox")
    return p


def _empty_usage(month: str) -> dict:
    return {
        "month": month,
        "tavily": {"count": 0, "errors": 0},
        "duckduckgo": {"count": 0, "errors": 0},
    }


def _load_usage() -> dict:
    month = datetime.now().strftime("%Y-%m")
    if not USAGE_PATH.exists():
        return _empty_usage(month)
    try:
        data = json.loads(USAGE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_usage(month)
    if data.get("month") != month:
        return _empty_usage(month)
    for k in ("tavily", "duckduckgo"):
        data.setdefault(k, {"count": 0, "errors": 0})
    return data


def _save_usage(data: dict) -> None:
    USAGE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _bump(provider: str, field: str = "count") -> None:
    with _usage_lock:
        data = _load_usage()
        data[provider][field] = data[provider].get(field, 0) + 1
        _save_usage(data)


def _under_cap(provider: str) -> bool:
    return _load_usage()[provider]["count"] < MONTHLY_CAP


# Timezone aliases for common city/region names
TIMEZONE_ALIASES = {
    "tokyo": "Asia/Tokyo",
    "delhi": "Asia/Kolkata",
    "kolkata": "Asia/Kolkata",
    "london": "Europe/London",
    "new york": "America/New_York",
    "newyork": "America/New_York",
    "la": "America/Los_Angeles",
    "los angeles": "America/Los_Angeles",
    "sydney": "Australia/Sydney",
    "dubai": "Asia/Dubai",
    "singapore": "Asia/Singapore",
    "toronto": "America/Toronto",
    "moscow": "Europe/Moscow",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
}


def _resolve_timezone(tz_name: str) -> str:
    """Resolve a timezone name to IANA format.
    
    Handles aliases and suggests close matches on failure.
    Returns the resolved IANA timezone name.
    """
    key = (tz_name or "UTC").strip().lower()
    original = tz_name.strip()
    
    # Check for UTC variants
    if key in {"utc", "etc/utc", "z"}:
        return "UTC"
    
    # Check aliases first
    if key in TIMEZONE_ALIASES:
        return TIMEZONE_ALIASES[key]
    
    # Try as-is (might be valid IANA) — validate by attempting to construct ZoneInfo
    try:
        ZoneInfo(original)
        return original
    except (ZoneInfoNotFoundError, OSError, ValueError):
        # OSError: occurs on Windows when tzdata is missing
        # ValueError: can occur from some system implementations
        pass
    
    # Suggest close matches from aliases
    suggestions = difflib.get_close_matches(key, TIMEZONE_ALIASES.keys(), n=3, cutoff=0.6)
    sugg_text = " Or try: " + ", ".join(f"'{TIMEZONE_ALIASES[s]}'" for s in suggestions) if suggestions else ""
    raise ValueError(
        f"Unknown timezone '{tz_name}'. Use an IANA name like 'Asia/Tokyo', 'UTC', or try an alias.{sugg_text}"
    )


def _tavily_search(query: str, max_results: int) -> list[dict]:
    from tavily import TavilyClient

    client = TavilyClient(os.environ["TAVILY_API_KEY"])
    resp = client.search(query=query, max_results=max_results, search_depth="advanced")
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
        }
        for r in resp.get("results", [])
    ]


def _ddg_search(query: str, max_results: int, timeout: int = 10) -> list[dict]:
    hits: list[dict] = []
    with DDGS(timeout=timeout) as ddgs:
        for backend in ("auto", "html", "lite"):
            try:
                hits = list(ddgs.text(query, max_results=max_results, backend=backend))
            except Exception:
                hits = []
            if hits:
                break
    return [
        {
            "title": h.get("title", ""),
            "url": h.get("href", ""),
            "snippet": h.get("body", ""),
        }
        for h in hits
    ]


def _html_to_text(html_str: str) -> str:
    """Convert HTML to readable plain text using stdlib html.parser."""
    from html.parser import HTMLParser

    class _Extractor(HTMLParser):
        _SKIP = frozenset({"script", "style", "noscript", "svg", "head"})
        _BLOCK = frozenset({
            "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6",
            "tr", "td", "th", "br", "hr", "article", "section", "header", "footer",
        })

        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self._depth = 0
            self._buf: list[str] = []

        def handle_starttag(self, tag, attrs) -> None:
            t = tag.lower()
            if t in self._SKIP:
                self._depth += 1
            elif t in self._BLOCK:
                self._buf.append("\n")

        def handle_endtag(self, tag) -> None:
            if tag.lower() in self._SKIP and self._depth > 0:
                self._depth -= 1

        def handle_data(self, data: str) -> None:
            if self._depth == 0 and data.strip():
                self._buf.append(data)

        def text(self) -> str:
            lines = [ln.strip() for ln in "".join(self._buf).splitlines() if ln.strip()]
            return "\n".join(lines)

    ex = _Extractor()
    try:
        ex.feed(html_str)
    except Exception:
        pass
    return ex.text()


async def _crawl4ai_fetch(url: str, timeout: int = 20) -> dict:
    """
    Fetch URL content.

    Strategy:
      1. Fast path — httpx plain HTTP GET + HTML-to-text (~2-5 s, no browser).
         Sufficient for static / server-rendered pages.
      2. crawl4ai fallback — headless Chromium for JS-rendered pages.
         Only used when the httpx result is too short (< 500 chars).
    """
    # ── 1. Fast path: httpx ──────────────────────────────────────────────
    httpx_text = ""
    httpx_status = 0
    try:
        async with httpx.AsyncClient(
            timeout=min(timeout, 10),
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1)"},
        ) as client:
            resp = await client.get(url)
            httpx_status = resp.status_code
            # For definitive client errors (4xx) there is nothing to gain from
            # launching a headless browser — return immediately.
            if 400 <= resp.status_code < 500:
                return {
                    "status": resp.status_code,
                    "content_type": "text/plain",
                    "length_bytes": 0,
                    "text": f"HTTP {resp.status_code}: {url}",
                }
            resp.raise_for_status()
            httpx_text = _html_to_text(resp.text)
    except Exception:
        pass

    if len(httpx_text) >= 500:
        return {
            "status": 200,
            "content_type": "text/markdown",
            "length_bytes": len(httpx_text.encode("utf-8")),
            "text": httpx_text,
        }

    # ── 2. crawl4ai fallback (JS-heavy pages) ────────────────────────────
    # crawl4ai uses Rich which writes via its own captured stdout reference, so
    # contextlib.redirect_stdout doesn't catch it. Redirect at the file-descriptor
    # level — crawl4ai's banner / [FETCH] / [SCRAPE] markers would otherwise
    # corrupt the MCP stdio JSON-RPC stream.
    saved_fd = os.dup(1)
    os.dup2(2, 1)
    crawl_text = ""
    # Wrap the ENTIRE crawl4ai lifecycle (browser startup + page crawl +
    # browser shutdown) in a single wait_for.  The previous code only timed
    # out crawler.arun(), leaving Playwright/Chromium startup uncovered; a
    # cold-start browser launch could block indefinitely and halt the agent.
    async def _do_crawl() -> str:
        from crawl4ai import AsyncWebCrawler
        async with AsyncWebCrawler(verbose=False) as crawler:
            r = await crawler.arun(url=url)
        # r.markdown is a str subclass (StringCompatibleMarkdown) that Pydantic
        # serializes as {} because its real field is private. Pull the raw string
        # out and force a plain str so FastMCP serializes correctly.
        md = r.markdown
        raw = (
            getattr(md, "raw_markdown", None)
            or getattr(md, "fit_markdown", None)
            or md
            or r.cleaned_html
            or r.html
            or ""
        )
        return str(raw)

    try:
        crawl_text = await asyncio.wait_for(_do_crawl(), timeout=timeout)
    except (asyncio.TimeoutError, Exception):
        crawl_text = httpx_text  # retain whatever httpx managed to get
    finally:
        os.dup2(saved_fd, 1)
        os.close(saved_fd)

    text = crawl_text or httpx_text or "(could not retrieve page content)"
    return {
        "status": 200,
        "content_type": "text/markdown",
        "length_bytes": len(text.encode("utf-8")),
        "text": text,
    }


@mcp.tool()
def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Search the web (Tavily primary, DDG fallback). Hard-capped at 5 results. Example: web_search("python asyncio tutorial", 3)."""
    max_results = max(1, min(max_results, MAX_SEARCH_RESULTS))
    if os.environ.get("TAVILY_API_KEY") and _under_cap("tavily"):
        try:
            results = _tavily_search(query, max_results)
            if results:
                _bump("tavily")
                return results
        except Exception:
            _bump("tavily", "errors")
    results = _ddg_search(query, max_results)
    _bump("duckduckgo")
    return results


@mcp.tool()
async def fetch_url(url: str, timeout: int = 20) -> dict:
    """Fetch clean markdown from a URL via crawl4ai (headless Chromium). Example: fetch_url("https://example.com")."""
    return await _crawl4ai_fetch(url, timeout=timeout)


@mcp.tool()
def get_time(timezone: str = "UTC") -> dict:
    """Current time in a named IANA timezone. Supports aliases like 'Tokyo', 'Delhi'. Example: get_time("Asia/Kolkata")."""
    # Resolve timezone name (handles aliases, suggestions, UTC fallback)
    resolved_tz = _resolve_timezone(timezone)
    
    # Use built-in UTC for robustness on systems without full tzdata
    if resolved_tz.upper() == "UTC":
        tz = dt_timezone.utc
    else:
        tz = ZoneInfo(resolved_tz)
    
    now = datetime.now(tz)
    offset = now.utcoffset()
    offset_hours = offset.total_seconds() / 3600 if offset else 0.0
    return {
        "iso": now.isoformat(),
        "human": now.strftime("%A, %d %B %Y %H:%M:%S %Z"),
        "timezone": resolved_tz,
        "offset_hours": offset_hours,
    }


@mcp.tool()
def currency_convert(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert money between ISO-3 currencies via frankfurter.dev. Example: currency_convert(100, "USD", "INR")."""
    f = from_currency.upper()
    t = to_currency.upper()
    url = f"https://api.frankfurter.dev/v1/latest?amount={amount}&base={f}&symbols={t}"
    with httpx.Client(timeout=20, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        data = r.json()
    converted = data["rates"][t]
    return {
        "amount": amount,
        "from": f,
        "to": t,
        "rate": converted / amount if amount else 0.0,
        "converted": converted,
        "date": data["date"],
        "source": "frankfurter.dev",
    }


@mcp.tool()
def read_file(path: str) -> dict:
    """Read a UTF-8 text file from the sandbox. Example: read_file("notes.txt")."""
    p = _safe(path)
    text = p.read_text(encoding="utf-8")
    return {
        "path": path,
        "size_bytes": p.stat().st_size,
        "content": text,
        "encoding": "utf-8",
    }


@mcp.tool()
def list_dir(path: str = ".") -> list[dict]:
    """List a directory inside the sandbox. Example: list_dir(".")."""
    p = _safe(path)
    out = []
    for child in sorted(p.iterdir()):
        is_dir = child.is_dir()
        out.append({
            "name": child.name,
            "type": "dir" if is_dir else "file",
            "size_bytes": 0 if is_dir else child.stat().st_size,
        })
    return out


@mcp.tool()
def create_file(path: str, content: str) -> dict:
    """Create a new file in the sandbox; errors if it exists. Example: create_file("hello.txt", "hi")."""
    p = _safe(path)
    if p.exists():
        raise ValueError(f"File '{path}' already exists")
    if not p.parent.exists():
        raise ValueError(f"Parent directory of '{path}' does not exist")
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "size_bytes": p.stat().st_size}


@mcp.tool()
def update_file(path: str, content: str) -> dict:
    """Overwrite an existing sandbox file. Example: update_file("hello.txt", "new body")."""
    p = _safe(path)
    if not p.exists():
        raise ValueError(f"File '{path}' does not exist")
    p.write_text(content, encoding="utf-8")
    return {"ok": True, "path": path, "size_bytes": p.stat().st_size}


@mcp.tool()
def edit_file(path: str, find: str, replace: str, replace_all: bool = False) -> dict:
    """Find-and-replace inside a sandbox file. Example: edit_file("hello.txt", "foo", "bar")."""
    p = _safe(path)
    text = p.read_text(encoding="utf-8")
    count = text.count(find)
    if count == 0:
        raise ValueError(f"'{find}' not found in '{path}'")
    if count > 1 and not replace_all:
        raise ValueError(
            f"'{find}' occurs {count} times in '{path}'; pass replace_all=True"
        )
    new_text = text.replace(find, replace) if replace_all else text.replace(find, replace, 1)
    p.write_text(new_text, encoding="utf-8")
    replacements = count if replace_all else 1
    return {
        "ok": True,
        "path": path,
        "replacements": replacements,
        "size_bytes": p.stat().st_size,
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
