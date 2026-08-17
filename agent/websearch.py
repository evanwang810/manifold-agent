"""Web search for providers with no search of their own.

The provider's own search is the better path wherever it exists and llm.py uses it
first; see the table in the README for which ones have one. This is what the rest fall
back to, and what a forecasting agent lives on, so it takes a real search API when you
give it a key and scrapes DuckDuckGo when you do not.

Four backends, all speaking the same three-field shape. `duckduckgo` needs no key and
is the default. The others need one key each and are meaningfully better at the thing
that actually matters here, which is returning something written this week rather than
the best-ranked page of all time.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx

from .config import SearchConfig

log = logging.getLogger(__name__)

BACKENDS = ("duckduckgo", "tavily", "brave", "serper")


@dataclass
class Result:
    title: str
    body: str
    url: str
    date: str = ""

    def render(self, index: int) -> str:
        stamp = f" ({self.date})" if self.date else ""
        return f"[{index}] {self.title}{stamp}\n{self.body[:600]}\n{self.url}"


def _bucket(days: int) -> str:
    """Collapse a day count into the coarse recency band every backend speaks."""
    if days <= 0:
        return ""
    if days <= 1:
        return "day"
    if days <= 7:
        return "week"
    if days <= 31:
        return "month"
    return "year"


def _ddg_sync(query: str, count: int, band: str) -> list[Result]:
    try:
        from ddgs import DDGS
    except ImportError:
        log.warning("ddgs is not installed, so there is no keyless web search")
        return []
    try:
        with DDGS() as ddgs:
            rows = list(ddgs.text(
                query,
                max_results=count,
                timelimit={"day": "d", "week": "w", "month": "m", "year": "y"}.get(band),
            ))
    except Exception as exc:  # noqa: BLE001 - scraped search fails in many ways
        log.warning("DuckDuckGo search failed: %s", exc)
        return []
    return [
        Result(r.get("title", ""), r.get("body") or "", r.get("href", ""))
        for r in rows
    ]


async def _duckduckgo(query: str, cfg: SearchConfig) -> list[Result]:
    return await asyncio.to_thread(
        _ddg_sync, query, cfg.max_results, _bucket(cfg.recency_days)
    )


async def _tavily(query: str, cfg: SearchConfig) -> list[Result]:
    body: dict[str, Any] = {
        "query": query,
        "max_results": cfg.max_results,
        "search_depth": "basic",
        "include_answer": False,
    }
    band = _bucket(cfg.recency_days)
    if band:
        body["time_range"] = band
    data = await _post("https://api.tavily.com/search", body,
                       {"Authorization": f"Bearer {cfg.api_key}"})
    return [
        Result(r.get("title", ""), r.get("content") or "", r.get("url", ""),
               r.get("published_date") or "")
        for r in (data.get("results") or [])
    ]


async def _brave(query: str, cfg: SearchConfig) -> list[Result]:
    params: dict[str, Any] = {"q": query, "count": cfg.max_results}
    band = _bucket(cfg.recency_days)
    if band:
        params["freshness"] = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}[band]
    data = await _get(
        "https://api.search.brave.com/res/v1/web/search", params,
        {"X-Subscription-Token": cfg.api_key, "Accept": "application/json"},
    )
    return [
        Result(r.get("title", ""), r.get("description") or "", r.get("url", ""),
               r.get("page_age") or r.get("age") or "")
        for r in ((data.get("web") or {}).get("results") or [])
    ]


async def _serper(query: str, cfg: SearchConfig) -> list[Result]:
    body: dict[str, Any] = {"q": query, "num": cfg.max_results}
    band = _bucket(cfg.recency_days)
    if band:
        body["tbs"] = {"day": "qdr:d", "week": "qdr:w",
                       "month": "qdr:m", "year": "qdr:y"}[band]
    data = await _post("https://google.serper.dev/search", body,
                       {"X-API-KEY": cfg.api_key})
    return [
        Result(r.get("title", ""), r.get("snippet") or "", r.get("link", ""),
               r.get("date") or "")
        for r in (data.get("organic") or [])
    ]


async def _post(url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
        resp = await client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        return resp.json()


async def _get(url: str, params: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(30)) as client:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


_RUNNERS = {
    "duckduckgo": _duckduckgo,
    "tavily": _tavily,
    "brave": _brave,
    "serper": _serper,
}


async def search_many(queries: list[str], cfg: SearchConfig) -> str:
    """Run every query, merge by URL, and render numbered snippets.

    Queries run concurrently because they are independent and the agent is already
    waiting on this. Deduping by URL matters more than it sounds: two queries about the
    same event routinely return the same three articles, and without this the model
    reads one source three times and treats it as three.
    """
    queries = [q.strip() for q in queries if q and q.strip()][:4]
    if not queries:
        return ""

    runner = _RUNNERS.get(cfg.backend, _duckduckgo)
    if cfg.backend != "duckduckgo" and not cfg.api_key:
        log.warning("Search backend %s has no key, using DuckDuckGo", cfg.backend)
        runner = _duckduckgo

    async def one(query: str) -> list[Result]:
        try:
            return await runner(query, cfg)
        except Exception as exc:  # noqa: BLE001 - research is best effort
            log.warning("Search failed for %r: %s", query, exc)
            return []

    batches = await asyncio.gather(*(one(q) for q in queries))

    seen: set[str] = set()
    merged: list[Result] = []
    for batch in batches:
        for result in batch:
            key = result.url or result.title
            if key and key not in seen:
                seen.add(key)
                merged.append(result)

    merged = merged[: cfg.max_results * 2]
    return "\n\n".join(r.render(i + 1) for i, r in enumerate(merged))


async def search(query: str, max_results: int = 6) -> str:
    """Single keyless query. Kept for `run.py --check`, which probes before config."""
    return await search_many([query], SearchConfig(max_results=max_results))
