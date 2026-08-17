"""Keyless web search, for providers with no search of their own.

The provider's own search is the better path wherever it exists, and llm.py uses it
first; see the table in the README for which ones have one. This is the floor under
that: DuckDuckGo through the `ddgs` package, no key, no signup, no card. It is scrapier
and less current than a real search API, and that is the tradeoff for costing nothing.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)


def _search_sync(query: str, max_results: int) -> list[dict[str, Any]]:
    try:
        from ddgs import DDGS
    except ImportError:
        log.warning("ddgs is not installed, so there is no web search")
        return []
    try:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as exc:  # noqa: BLE001 - scraped search fails in many ways
        log.warning("Web search failed: %s", exc)
        return []


async def search(query: str, max_results: int = 6) -> str:
    """Return numbered result snippets, or an empty string if nothing came back."""
    results = await asyncio.to_thread(_search_sync, query, max_results)
    if not results:
        return ""
    return "\n\n".join(
        f"[{i + 1}] {r.get('title', '')}\n{(r.get('body') or '')[:600]}\n{r.get('href', '')}"
        for i, r in enumerate(results)
    )
